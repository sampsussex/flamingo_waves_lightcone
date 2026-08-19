#!/bin/env python
"""
make_mock_catalog.py

Build a galaxy mock catalogue from the FLAMINGO lightcone using Synthesizer.

Inputs:
  * The halo-tagged lightcone particle files produced by the
    lightcone->halo matching script (per-particle IndexInHaloLightcone +
    copied halo properties: Redshift, SnapshotNumber, HaloCatalogueIndex,
    SOAPIndex, TotalMass, HaloCentre).
  * The ORIGINAL lightcone particle files (same basenames), which hold the
    star/gas particle properties (masses, birth scale factors,
    metallicities, velocities, SFRs).

For every halo (= "galaxy") with at least --min-star-particles star
particles the pipeline computes:
  * apparent + absolute AB magnitudes in VISTA Z,Y,J,H,Ks and
    VST/OmegaCAM u,g,r,i,z (Synthesizer, BPASS + Pacman dust screen by
    default),
  * z_cos (halo lightcone redshift), z_obs (including the line-of-sight
    peculiar velocity of the stars),
  * stellar mass, gas mass, instantaneous SFR (from gas particles),
  * galaxy_id (= IndexInHaloLightcone, unique in the halo lightcone),
    group_id (host FOF group via SOAP join if --soap-filenames given,
    else HaloCatalogueIndex), plus snapshot / SOAPIndex bookkeeping,
  * position (comoving halo centre), RA/Dec, halo mass.

Output: one parquet "part" file per MPI rank in --output-dir; read the
directory as a single dataset with pandas/pyarrow/dask.

Parallelism (designed for COSMA):
  1. Files are assigned round-robin to MPI ranks; each rank streams its
     files in chunks, keeping only particles with a halo id.
  2. Particles are exchanged between ranks by hashing the halo id
     (dest = id % nranks), so each rank ends up with complete galaxies
     and, with ~1e5+ galaxies per rank, a well balanced load.
  3. Gas particles are never held in memory: per-chunk sums of mass and
     SFR per halo id are accumulated and merged in the same exchange.
  4. Each rank loops over its galaxies calling Synthesizer, then writes
     its own parquet part file. No communication in the main loop except
     the optional collective SOAP group join.

Example (see submit_mock.slurm):

  mpirun python3 -u -m mpi4py make_mock_catalog.py \\
      /path/to/tagged_particles \\
      /cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/lightcones \\
      lightcone0 \\
      --grid-name bpass-2.2.1-bin_chabrier03-0.1,300.0 \\
      --grid-dir /path/to/grids \\
      --filters-file /path/to/filters_vista_vst.hdf5 \\
      --soap-filenames "/path/SOAP/halo_properties_%(snap_nr)04d.hdf5" \\
      --output-dir /path/to/mock_out

Run with --inspect first to check dataset names/units in your lightcone.
"""

import os
import sys
import time
import gc

t0 = time.time()

import numpy as np
import h5py

from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

import virgo.mpi.parallel_sort as psort
import virgo.mpi.parallel_hdf5 as phdf5

from photometry_core import (
    FILTER_DEF,
    FILTER_NAMES,
    load_filters,
    build_emission_model,
    get_grid_age_limits_yr,
    get_grid_metallicity_limits,
    compute_galaxy_photometry,
    build_age_interpolator,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_KMS = 299792.458
CGS_MSUN = 1.98841e33
CGS_KMS = 1.0e5
YR_IN_S = 3.1556952e7
GROUP_ID_SNAP_STRIDE = 10**12  # group_id = snap*stride + host_fof_id


def message(m):
    if comm_rank == 0:
        print(f"{time.time()-t0:8.1f}s: {m}", flush=True)


def fatal(m):
    if comm_rank == 0:
        print(f"FATAL: {m}", flush=True)
    comm.barrier()
    comm.Abort(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(
        description="Make a Synthesizer mock catalogue from the FLAMINGO lightcone.",
        comm=comm,
    )
    parser.add_argument("tagged_dir",
                        help="Directory with halo-tagged particle files "
                             "(output_dir of the matching script)")
    parser.add_argument("lightcone_dir",
                        help="Directory with the ORIGINAL lightcone outputs "
                             "(containing <base>_index.hdf5 and <base>_particles/)")
    parser.add_argument("lightcone_base", help="Base name, e.g. lightcone0")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write the parquet catalogue parts")
    # Synthesizer inputs
    parser.add_argument("--grid-name",
                        default="bpass-2.2.1-bin_chabrier03-0.1,300.0",
                        help="Synthesizer SPS grid name")
    parser.add_argument("--grid-dir", required=False, default=None,
                        help="Directory holding the grid HDF5 file")
    parser.add_argument("--filters-file", required=False, default=None,
                        help="FilterCollection HDF5 written by prepare_inputs.py "
                             "(required unless --inspect)")
    parser.add_argument("--emission", default="pacman",
                        choices=("pacman", "reprocessed", "incident"),
                        help="Emission model (default pacman: nebular + dust screen)")
    parser.add_argument("--tau-v", type=float, default=0.33,
                        help="V-band optical depth of the dust screen (pacman)")
    parser.add_argument("--fesc", type=float, default=0.0,
                        help="LyC escape fraction")
    parser.add_argument("--dust-slope", type=float, default=-1.0,
                        help="Power-law dust curve slope (pacman)")
    parser.add_argument("--no-igm", action="store_true",
                        help="Disable Inoue+14 IGM absorption in apparent mags")
    parser.add_argument("--lam-min", type=float, default=1000.0,
                        help="Truncate grid below this rest wavelength [Angstrom] "
                             "(memory/speed; 0 = no truncation)")
    parser.add_argument("--lam-max", type=float, default=30000.0,
                        help="Truncate grid above this rest wavelength [Angstrom] "
                             "(must cover reddest band * (1+z_max); 0 = none)")
    parser.add_argument("--nthreads", type=int, default=1,
                        help="Threads per rank for Synthesizer C extensions")
    # Cosmology (FLAMINGO D3A / DES Y3 by default)
    parser.add_argument("--h0", type=float, default=0.681)
    parser.add_argument("--omega-m", type=float, default=0.306)
    parser.add_argument("--omega-b", type=float, default=0.0486)
    # Dataset names in the ORIGINAL lightcone files (check with --inspect!)
    parser.add_argument("--initial-mass-dataset", default="InitialMasses",
                        help="Star initial mass dataset (falls back to Masses)")
    parser.add_argument("--star-metal-dataset",
                        default="SmoothedMetalMassFractions",
                        help="Star metal mass fraction dataset")
    parser.add_argument("--birth-a-dataset", default="BirthScaleFactors",
                        help="Star birth scale factor dataset")
    parser.add_argument("--gas-sfr-dataset", default="StarFormationRates",
                        help="Gas SFR dataset ('' to skip SFR)")
    # Optional SOAP join for proper group ids
    parser.add_argument("--soap-filenames", default=None,
                        help="Format string for SOAP files, e.g. "
                             "'.../halo_properties_%%(snap_nr)04d.hdf5'. "
                             "If given, group_id is built from --group-dataset.")
    parser.add_argument("--group-dataset",
                        default="InputHalos/HBTplus/HostFOFId",
                        help="SOAP dataset giving the host group id")
    # Selection / performance
    parser.add_argument("--min-star-particles", type=int, default=1,
                        help="Minimum star particles for a galaxy to be output")
    parser.add_argument("--read-chunk", type=int, default=20_000_000,
                        help="Particles per HDF5 read chunk")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Debug: only process the first N files (0 = all)")
    parser.add_argument("--inspect", action="store_true",
                        help="Print datasets/attrs of the first file pair and exit")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# File list (mirrors the matching script's index logic)
# ---------------------------------------------------------------------------
def list_particle_files(args):
    index_file = os.path.join(args.lightcone_dir,
                              f"{args.lightcone_base}_index.hdf5")
    if comm_rank == 0:
        with h5py.File(index_file, "r") as index:
            lc = index["Lightcone"]
            nr_mpi_ranks = int(lc.attrs["nr_mpi_ranks"])
            final_file_on_rank = np.asarray(lc.attrs["final_particle_file_on_rank"])
    else:
        nr_mpi_ranks = None
        final_file_on_rank = None
    nr_mpi_ranks, final_file_on_rank = comm.bcast((nr_mpi_ranks, final_file_on_rank))

    pairs = []
    for rank_nr in range(nr_mpi_ranks):
        for file_nr in range(final_file_on_rank[rank_nr] + 1):
            basename = f"{args.lightcone_base}_{file_nr:04d}.{rank_nr}.hdf5"
            orig = os.path.join(args.lightcone_dir,
                                f"{args.lightcone_base}_particles", basename)
            tagged = os.path.join(args.tagged_dir, basename)
            pairs.append((orig, tagged))
    if args.max_files > 0:
        pairs = pairs[: args.max_files]
    message(f"Have {len(pairs)} particle file pairs")
    return pairs


def inspect_files(pairs):
    if comm_rank == 0:
        for label, path in (("ORIGINAL", pairs[0][0]), ("TAGGED", pairs[0][1])):
            print(f"\n===== {label}: {path}")
            with h5py.File(path, "r") as f:
                def show(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"  {name}  shape={obj.shape} dtype={obj.dtype}")
                        for k, v in obj.attrs.items():
                            if "CGS" in k or "exponent" in k or "Units" in k:
                                print(f"      attr: {k} = {v}")
                f.visititems(show)
    comm.barrier()


# ---------------------------------------------------------------------------
# Units: use the SWIFT/SOAP "Conversion factor to CGS" attributes if present
# ---------------------------------------------------------------------------
def cgs_factor(dset):
    """Return the CGS conversion factor from dataset attrs, or None."""
    best = None
    for key, val in dset.attrs.items():
        if "Conversion factor to CGS" in key or "Conversion factor to physical CGS" in key:
            v = float(np.ravel(val)[0])
            if "including cosmological corrections" in key:
                return v
            best = v
    return best


def unit_factor(dset, target, default, warned, label):
    """Factor converting dataset values to target units, with fallback."""
    cgs = cgs_factor(dset)
    if cgs is None:
        if comm_rank == 0 and label not in warned:
            print(f"WARNING: no CGS attrs on {label}; assuming factor {default}",
                  flush=True)
            warned.add(label)
        return default
    if target == "msun":
        return cgs / CGS_MSUN
    if target == "kms":
        return cgs / CGS_KMS
    if target == "msun_per_yr":
        return cgs * YR_IN_S / CGS_MSUN
    raise ValueError(target)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_local_particles(args, my_pairs):
    """
    Stream this rank's files, keeping only in-halo particles.

    Returns:
      star : dict of concatenated arrays over in-halo star particles
      gas_ids, gas_mass, gas_sfr : per-halo-id partial sums from local files
    """
    warned = set()
    star_cols = {
        "id": [], "im": [], "m": [], "a_birth": [], "zmet": [],
        "vel": [], "z_cos": [], "snap": [], "hci": [], "soapi": [],
        "hmass": [], "centre": [],
    }
    gas_id_l, gas_m_l, gas_sfr_l = [], [], []
    used_mass_fallback = False

    for orig_path, tagged_path in my_pairs:
        with h5py.File(orig_path, "r") as f, h5py.File(tagged_path, "r") as ft:

            # ---------------- stars ----------------
            if "Stars" in ft and "Stars" in f:
                g, gt = f["Stars"], ft["Stars"]
                n = gt["IndexInHaloLightcone"].shape[0]
                if g["Masses"].shape[0] != n:
                    fatal(f"Star count mismatch in {orig_path} vs {tagged_path}")
                if args.initial_mass_dataset in g:
                    im_name = args.initial_mass_dataset
                else:
                    im_name = "Masses"
                    used_mass_fallback = True
                mfac = unit_factor(g["Masses"], "msun", 1.0e10, warned,
                                   "Stars/Masses")
                imfac = unit_factor(g[im_name], "msun", 1.0e10, warned,
                                    f"Stars/{im_name}")
                vfac = unit_factor(g["Velocities"], "kms", 1.0, warned,
                                   "Stars/Velocities")
                hmfac = unit_factor(gt["TotalMass"], "msun", 1.0, warned,
                                    "tagged TotalMass")

                for c0 in range(0, n, args.read_chunk):
                    c1 = min(c0 + args.read_chunk, n)
                    ids = gt["IndexInHaloLightcone"][c0:c1]
                    sel = ids >= 0
                    if not np.any(sel):
                        continue
                    star_cols["id"].append(ids[sel].astype(np.int64))
                    del ids
                    star_cols["im"].append(
                        (g[im_name][c0:c1][sel] * imfac).astype(np.float32))
                    star_cols["m"].append(
                        (g["Masses"][c0:c1][sel] * mfac).astype(np.float32))
                    star_cols["a_birth"].append(
                        g[args.birth_a_dataset][c0:c1][sel].astype(np.float32))
                    star_cols["zmet"].append(
                        g[args.star_metal_dataset][c0:c1][sel].astype(np.float32))
                    star_cols["vel"].append(
                        (g["Velocities"][c0:c1][sel] * vfac).astype(np.float32))
                    star_cols["z_cos"].append(
                        gt["Redshift"][c0:c1][sel].astype(np.float32))
                    star_cols["snap"].append(
                        gt["SnapshotNumber"][c0:c1][sel].astype(np.int32))
                    star_cols["hci"].append(
                        gt["HaloCatalogueIndex"][c0:c1][sel].astype(np.int64))
                    star_cols["soapi"].append(
                        gt["SOAPIndex"][c0:c1][sel].astype(np.int64))
                    star_cols["hmass"].append(
                        (gt["TotalMass"][c0:c1][sel] * hmfac).astype(np.float32))
                    star_cols["centre"].append(
                        gt["HaloCentre"][c0:c1][sel].astype(np.float64))

            # ---------------- gas (aggregated per halo id) ----------------
            if "Gas" in ft and "Gas" in f:
                g, gt = f["Gas"], ft["Gas"]
                n = gt["IndexInHaloLightcone"].shape[0]
                if g["Masses"].shape[0] != n:
                    fatal(f"Gas count mismatch in {orig_path} vs {tagged_path}")
                mfac = unit_factor(g["Masses"], "msun", 1.0e10, warned,
                                   "Gas/Masses")
                have_sfr = (args.gas_sfr_dataset != "" and
                            args.gas_sfr_dataset in g)
                if have_sfr:
                    sfac = unit_factor(g[args.gas_sfr_dataset], "msun_per_yr",
                                       1.0, warned, "Gas/SFR")
                for c0 in range(0, n, args.read_chunk):
                    c1 = min(c0 + args.read_chunk, n)
                    ids = gt["IndexInHaloLightcone"][c0:c1]
                    sel = ids >= 0
                    if not np.any(sel):
                        continue
                    ids = ids[sel].astype(np.int64)
                    m = g["Masses"][c0:c1][sel].astype(np.float64) * mfac
                    if have_sfr:
                        sfr = g[args.gas_sfr_dataset][c0:c1][sel].astype(np.float64)
                        sfr = np.where(sfr > 0, sfr * sfac, 0.0)  # SWIFT: <0 = last SF a
                    else:
                        sfr = np.zeros_like(m)
                    uid, inv = np.unique(ids, return_inverse=True)
                    gas_id_l.append(uid)
                    gas_m_l.append(np.bincount(inv, weights=m))
                    gas_sfr_l.append(np.bincount(inv, weights=sfr))
                    del ids, m, sfr, uid, inv

    # Concatenate stars
    star = {}
    for key, lst in star_cols.items():
        if lst:
            star[key] = np.concatenate(lst)
        else:
            shape = (0, 3) if key in ("vel", "centre") else (0,)
            dtypes = {"id": np.int64, "snap": np.int32, "hci": np.int64,
                      "soapi": np.int64, "centre": np.float64}
            star[key] = np.zeros(shape, dtype=dtypes.get(key, np.float32))
        lst.clear()

    # Combine local gas partial sums
    if gas_id_l:
        gid = np.concatenate(gas_id_l)
        gm = np.concatenate(gas_m_l)
        gs = np.concatenate(gas_sfr_l)
        uid, inv = np.unique(gid, return_inverse=True)
        gas_ids = uid
        gas_mass = np.bincount(inv, weights=gm)
        gas_sfr = np.bincount(inv, weights=gs)
        del gid, gm, gs
    else:
        gas_ids = np.zeros(0, dtype=np.int64)
        gas_mass = np.zeros(0)
        gas_sfr = np.zeros(0)

    any_fallback = comm.allreduce(used_mass_fallback, op=MPI.LOR)
    if any_fallback:
        message(f"WARNING: '{args.initial_mass_dataset}' not found; "
                "using Masses as initial masses (photometry will be "
                "slightly underestimated for old populations)")
    gc.collect()
    return star, gas_ids, gas_mass, gas_sfr


# ---------------------------------------------------------------------------
# Exchange particles/aggregates so each halo id lives on exactly one rank
# ---------------------------------------------------------------------------
def exchange_by_id(ids, arrays):
    """
    Send element i to rank (ids[i] % comm_size). Returns (ids, arrays)
    after the exchange. 2D arrays with trailing dim k are supported.
    """
    dest = (ids % comm_size).astype(np.int32)
    order = np.argsort(dest, kind="stable")
    dest = dest[order]
    send_count = np.bincount(dest, minlength=comm_size).astype(np.int64)
    send_offset = np.cumsum(send_count) - send_count
    recv_count = np.asarray(comm.alltoall(send_count), dtype=np.int64)
    recv_offset = np.cumsum(recv_count) - recv_count
    nr_recv = int(np.sum(recv_count))

    def xchg(arr):
        arr = arr[order]
        if arr.ndim == 1:
            recv = np.empty(nr_recv, dtype=arr.dtype)
            psort.my_alltoallv(np.ascontiguousarray(arr),
                               send_count, send_offset,
                               recv, recv_count, recv_offset, comm=comm)
            return recv
        k = arr.shape[1]
        flat = np.ascontiguousarray(arr).reshape(-1)
        recv = np.empty(nr_recv * k, dtype=arr.dtype)
        psort.my_alltoallv(flat, send_count * k, send_offset * k,
                           recv, recv_count * k, recv_offset * k, comm=comm)
        return recv.reshape(-1, k)

    new_ids = xchg(ids)
    new_arrays = [xchg(a) for a in arrays]
    return new_ids, new_arrays


# ---------------------------------------------------------------------------
# Optional SOAP join for host group ids (collective!)
# ---------------------------------------------------------------------------
def fetch_group_ids(args, gal_snap, gal_soapi):
    """Return host group id per galaxy via a collective SOAP read."""
    n = len(gal_snap)
    host = np.full(n, -1, dtype=np.int64)
    local_min = int(gal_snap.min()) if n else 2**30
    local_max = int(gal_snap.max()) if n else -1
    snap_min = comm.allreduce(local_min, op=MPI.MIN)
    snap_max = comm.allreduce(local_max, op=MPI.MAX)
    if snap_max < 0:
        return host
    for snap in range(snap_min, snap_max + 1):
        fname = args.soap_filenames % {"snap_nr": snap}
        message(f"Group join: reading {args.group_dataset} from {fname}")
        mf = phdf5.MultiFile(fname, file_idx=(0,), comm=comm)
        data = mf.read((args.group_dataset,))
        sel = gal_snap == snap
        ptr = gal_soapi[sel].astype(np.int64)
        fetched = psort.fetch_elements(
            data[args.group_dataset].astype(np.int64), ptr, comm=comm)
        host[sel] = fetched
        del data, mf
        gc.collect()
    return host


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    message(f"Starting on {comm_size} MPI ranks")

    pairs = list_particle_files(args)
    if args.inspect:
        inspect_files(pairs)
        return

    if args.filters_file is None:
        fatal("--filters-file is required (make it with prepare_inputs.py)")

    if comm_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    comm.barrier()

    # ---------------- read ----------------
    my_pairs = pairs[comm_rank::comm_size]
    message("Reading particle files (round-robin over ranks)")
    star, gas_ids, gas_mass, gas_sfr = read_local_particles(args, my_pairs)
    nstar_tot = comm.allreduce(len(star["id"]))
    message(f"Total in-halo star particles: {nstar_tot}")
    if nstar_tot == 0:
        fatal("No star particles found in any halo - check inputs")

    # ---------------- exchange ----------------
    message("Exchanging star particles by halo id")
    keys = [k for k in star if k != "id"]
    ids, arrays = exchange_by_id(star["id"], [star[k] for k in keys])
    star = dict(zip(keys, arrays))
    star["id"] = ids
    del ids, arrays
    gc.collect()

    message("Exchanging gas aggregates by halo id")
    gas_ids, (gas_mass, gas_sfr) = exchange_by_id(gas_ids, [gas_mass, gas_sfr])
    if len(gas_ids):
        uid, inv = np.unique(gas_ids, return_inverse=True)
        gas_mass = np.bincount(inv, weights=gas_mass)
        gas_sfr = np.bincount(inv, weights=gas_sfr)
        gas_ids = uid
        del uid, inv

    # ---------------- build galaxy segments ----------------
    message("Sorting local star particles by halo id")
    order = np.argsort(star["id"], kind="stable")
    for k in star:
        star[k] = star[k][order]
    del order
    gc.collect()

    uid, start, count = np.unique(star["id"], return_index=True,
                                  return_counts=True)
    ngal_all = comm.allreduce(len(uid))
    message(f"Total galaxies with >=1 star particle: {ngal_all}")

    keep = count >= args.min_star_particles
    uid_k, start_k, count_k = uid[keep], start[keep], count[keep]
    ngal = len(uid_k)
    ngal_tot = comm.allreduce(ngal)
    message(f"Galaxies passing min-star-particles={args.min_star_particles}: "
            f"{ngal_tot}")

    # Galaxy-level metadata (constant within a halo -> take first particle)
    gal_id = uid_k
    gal_z_cos = star["z_cos"][start_k].astype(np.float64)
    gal_snap = star["snap"][start_k]
    gal_hci = star["hci"][start_k]
    gal_soapi = star["soapi"][start_k]
    gal_hmass = star["hmass"][start_k]
    gal_centre = star["centre"][start_k]

    # Stellar mass and mass-weighted LOS velocity (all via reduceat on the
    # full segment structure, then selected)
    seg_mstar = np.add.reduceat(star["m"].astype(np.float64), start)[keep] \
        if len(uid) else np.zeros(0)
    seg_msum = seg_mstar.copy()
    vlos = np.zeros(ngal)
    if len(uid):
        mv = star["m"].astype(np.float64)[:, None] * star["vel"].astype(np.float64)
        mv_sum = np.stack(
            [np.add.reduceat(mv[:, i], start)[keep] for i in range(3)], axis=1)
        del mv
        r = np.sqrt(np.sum(gal_centre**2, axis=1))
        r = np.where(r > 0, r, 1.0)
        nhat = gal_centre / r[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            vlos = np.sum(mv_sum * nhat, axis=1) / np.where(seg_msum > 0,
                                                            seg_msum, 1.0)
        del mv_sum
    gal_mstar = seg_mstar
    gal_z_obs = (1.0 + gal_z_cos) * (1.0 + vlos / C_KMS) - 1.0

    # RA/Dec from the comoving halo centre
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = np.sqrt(np.sum(gal_centre**2, axis=1))
        gal_dec = np.degrees(np.arcsin(
            np.where(rr > 0, gal_centre[:, 2] / np.where(rr > 0, rr, 1.0), 0.0)))
        gal_ra = np.degrees(np.arctan2(gal_centre[:, 1], gal_centre[:, 0])) % 360.0

    # Join gas aggregates
    gal_mgas = np.zeros(ngal)
    gal_sfr = np.zeros(ngal)
    if len(gas_ids):
        pos = np.searchsorted(gas_ids, gal_id)
        pos = np.clip(pos, 0, len(gas_ids) - 1)
        hit = gas_ids[pos] == gal_id
        gal_mgas[hit] = gas_mass[pos[hit]]
        gal_sfr[hit] = gas_sfr[pos[hit]]
    del gas_ids, gas_mass, gas_sfr

    # ---------------- group ids ----------------
    if args.soap_filenames is not None:
        host = fetch_group_ids(args, gal_snap, gal_soapi)
        gal_group = np.where(
            host >= 0,
            gal_snap.astype(np.int64) * GROUP_ID_SNAP_STRIDE + host,
            np.int64(-1))
        gal_host_raw = host
    else:
        message("No --soap-filenames given: group_id = HaloCatalogueIndex "
                "(subhalo catalogue index, NOT a FOF group link)")
        gal_group = gal_hci.copy()
        gal_host_raw = np.full(ngal, -1, dtype=np.int64)

    # ---------------- Synthesizer setup ----------------
    message("Loading SPS grid and filters")
    from synthesizer.grid import Grid

    lam_lims = ()
    if args.lam_min > 0 and args.lam_max > 0:
        from unyt import angstrom
        lam_lims = (args.lam_min * angstrom, args.lam_max * angstrom)
    grid = Grid(args.grid_name, grid_dir=args.grid_dir, ignore_lines=True,
                lam_lims=lam_lims)
    filters = load_filters(args.filters_file)
    model = build_emission_model(grid, mode=args.emission, tau_v=args.tau_v,
                                 fesc=args.fesc, dust_slope=args.dust_slope)
    igm = None
    if not args.no_igm:
        from synthesizer.emission_models.attenuation import Inoue14
        igm = Inoue14()

    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=100.0 * args.h0, Om0=args.omega_m,
                          Ob0=args.omega_b)
    age_of_a = build_age_interpolator(cosmo)
    age_lo, age_hi = get_grid_age_limits_yr(grid)
    zmet_lo, zmet_hi = get_grid_metallicity_limits(grid)

    # ---------------- photometry loop ----------------
    message("Computing photometry")
    app = np.full((ngal, len(FILTER_NAMES)), np.nan, dtype=np.float32)
    absm = np.full((ngal, len(FILTER_NAMES)), np.nan, dtype=np.float32)
    t_loop = time.time()
    for i in range(ngal):
        s0 = start_k[i]
        s1 = s0 + count_k[i]
        a_obs = 1.0 / (1.0 + gal_z_cos[i])
        ages_yr = np.clip(age_of_a(a_obs)
                          - age_of_a(star["a_birth"][s0:s1].astype(np.float64)),
                          age_lo, age_hi)
        zmet = np.clip(star["zmet"][s0:s1].astype(np.float64),
                       zmet_lo, zmet_hi)
        app[i], absm[i] = compute_galaxy_photometry(
            grid, model, filters, cosmo,
            star["im"][s0:s1].astype(np.float64),
            ages_yr, zmet, float(gal_z_cos[i]),
            igm=igm, nthreads=args.nthreads)
        if comm_rank == 0 and (i + 1) % 1000 == 0:
            rate = (i + 1) / (time.time() - t_loop)
            eta = (ngal - i - 1) / max(rate, 1e-9)
            message(f"  rank 0: {i+1}/{ngal} galaxies "
                    f"({rate:.1f}/s, ETA {eta/60:.1f} min)")

    nan_frac = comm.allreduce(int(np.isnan(app).any(axis=1).sum())) / max(ngal_tot, 1)
    message(f"Photometry done (galaxies with any NaN mag: {nan_frac:.2%})")

    # ---------------- write parquet ----------------
    message("Writing parquet parts")
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {
        "galaxy_id": gal_id,
        "group_id": gal_group,
        "host_fof_id": gal_host_raw,
        "halo_catalogue_index": gal_hci,
        "soap_index": gal_soapi,
        "snapshot": gal_snap,
        "ra": gal_ra,
        "dec": gal_dec,
        "pos_x": gal_centre[:, 0].astype(np.float32),
        "pos_y": gal_centre[:, 1].astype(np.float32),
        "pos_z": gal_centre[:, 2].astype(np.float32),
        "z_cos": gal_z_cos.astype(np.float32),
        "z_obs": gal_z_obs.astype(np.float32),
        "stellar_mass": gal_mstar.astype(np.float32),
        "gas_mass": gal_mgas.astype(np.float32),
        "sfr": gal_sfr.astype(np.float32),
        "halo_mass": gal_hmass.astype(np.float32),
        "n_star_particles": count_k.astype(np.int32),
    }
    for j, name in enumerate(FILTER_NAMES):
        cols[f"app_{name}"] = app[:, j]
    for j, name in enumerate(FILTER_NAMES):
        cols[f"abs_{name}"] = absm[:, j]

    table = pa.table(cols)
    out_path = os.path.join(args.output_dir,
                            f"mock_catalog.{comm_rank:05d}.parquet")
    pq.write_table(table, out_path, compression="zstd")
    comm.barrier()
    message(f"Done: {ngal_tot} galaxies written to {args.output_dir}")


if __name__ == "__main__":
    main()