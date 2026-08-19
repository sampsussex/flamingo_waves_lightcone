# FLAMINGO lightcone → Synthesizer mock catalogue

Turns the halo-tagged lightcone particle files (from your lightcone→halo
matching script) into a galaxy mock catalogue with VISTA Z,Y,J,H,Ks and
VST/OmegaCAM u,g,r,i,z apparent + absolute AB magnitudes, computed with
[Synthesizer](https://synthesizer-project.github.io/synthesizer/) (tested
against **cosmos-synthesizer v1.2.0** — pin this version, the API moves).

## Files

`photometry_core.py` holds everything that touches Synthesizer (filter
definitions, emission model, per-galaxy photometry) and is importable
without MPI. `make_mock_catalog.py` is the MPI pipeline. `prepare_inputs.py`
fetches filters (and points you at the grid download) on a login node.
`submit_mock.slurm` is a COSMA8 template. `test_smoke.py` is a serial smoke
test using a synthetic grid — run it anywhere to sanity-check the install.

## Setup (once, on a COSMA login node — compute nodes have no internet)

Into the same Python environment you use for the matching script (it already
has mpi4py, virgo, parallel h5py):

    pip install cosmos-synthesizer==1.2.0 astropy unyt pyarrow

Get the BPASS grid `bpass-2.2.1-bin_chabrier03-0.1,300.0.hdf5` from the
synthesizer grids collection (see the Grids page of the synthesizer docs;
`synthesizer-download --test-grids -d <dir>` gives you a small `test_grid`
to trial the pipeline with first). Then build the filter file:

    python3 prepare_inputs.py \
        --grid-name bpass-2.2.1-bin_chabrier03-0.1,300.0 \
        --grid-dir /cosma8/data/.../synthesizer_grids \
        --out filters_vista_vst.hdf5

## Before the first real run: check your dataset names

The star/gas properties needed for SEDs live in the *original* lightcone
particle files, and SWIFT lightcone contents are configurable per run, so
verify what's there:

    mpirun -np 1 python3 -m mpi4py make_mock_catalog.py \
        $TAGGED_DIR $LIGHTCONE_DIR lightcone0 --output-dir /tmp/x --inspect

The pipeline needs, per star particle: `Masses`, `InitialMasses` (falls
back to `Masses` with a warning), `BirthScaleFactors`,
`SmoothedMetalMassFractions` (or set `--star-metal-dataset`), `Velocities`;
per gas particle: `Masses` and `StarFormationRates` (set
`--gas-sfr-dataset ''` if absent — SFR column will be 0). Unit conversions
are taken from the SWIFT "Conversion factor to CGS" attributes when
present; otherwise documented fallbacks are used (masses assumed 1e10 Msun,
velocities km/s) with a warning.

A cheap end-to-end test before the full run:

    ... make_mock_catalog.py ... --max-files 4 --grid-name test_grid

## Running

Edit `submit_mock.slurm` and `sbatch` it. The parallelisation:

1. File pairs (original + tagged) are read round-robin by rank, streamed
   in chunks; only in-halo particles are kept. Gas is reduced on the fly
   to per-halo (mass, SFR) sums and never held in memory.
2. Star particles are exchanged between ranks by `halo_id % nranks`, so
   every rank holds complete galaxies and the load is balanced by galaxy
   count.
3. Each rank runs Synthesizer over its galaxies independently (~5–50 ms
   per galaxy depending on grid resolution) and writes its own parquet
   part — no communication in the hot loop.

Memory notes: each rank loads its own copy of the SPS grid, so
`--lam-min/--lam-max` truncation (default 1000–30000 Å, ample for u→Ks at
z ≤ 1) keeps that small; 64 ranks/node on COSMA8 is comfortable. The star
particle working set is ~80 bytes/particle spread over all ranks.

## Output

`--output-dir` contains one `mock_catalog.NNNNN.parquet` per rank; read
the directory as one dataset:

    import pandas as pd
    df = pd.read_parquet("/path/to/mock_out")

Columns: `galaxy_id` (= IndexInHaloLightcone, unique per halo-lightcone
entry), `group_id`, `host_fof_id`, `halo_catalogue_index`, `soap_index`,
`snapshot`, `ra`, `dec` [deg], `pos_x/y/z` (comoving halo centre, SOAP
units), `z_cos`, `z_obs`, `stellar_mass`, `gas_mass` [Msun], `sfr`
[Msun/yr, instantaneous from gas], `halo_mass` (BoundSubhalo/TotalMass,
Msun), `n_star_particles`, and `app_*` / `abs_*` AB magnitudes for
VISTA_Z, VISTA_Y, VISTA_J, VISTA_H, VISTA_Ks, VST_u, VST_g, VST_r, VST_i,
VST_z.

## Modelling choices and caveats

Photometry: BPASS 2.2.1 (Chabrier, 0.1–300 Msun) through a Pacman emission
model — nebular reprocessing plus a power-law dust screen with
`--tau-v 0.33` (a z~0 average; there is no per-galaxy dust model here) and
`--fesc 0`. Swap with `--emission incident|reprocessed` or change
`--tau-v`. Apparent magnitudes use the luminosity distance at `z_cos` and
include Inoue+14 IGM absorption (`--no-igm` to disable — irrelevant at
z<1 anyway). Absolute magnitudes are rest-frame, dust *included*
(emergent light).

Ages: star ages are t(z_cos) − t(a_birth) with the FLAMINGO D3A cosmology
(h=0.681, Ωm=0.306, Ωb=0.0486; override with `--h0/--omega-m/--omega-b`),
clipped to the grid age range.

Redshifts: `z_cos` is the halo's lightcone redshift; `z_obs = (1+z_cos)
(1+v_los/c) − 1` with v_los the stellar-mass-weighted peculiar velocity
projected on the line of sight (assumes lightcone `Velocities` are
peculiar velocities — check with `--inspect`).

Galaxy definition: one "galaxy" per halo-lightcone subhalo, i.e. every
star particle your matching script assigned to that subhalo (within
`radius_multiplier × EncloseRadius`), with `--min-star-particles` (default
1) as the floor. At FLAMINGO resolution (~1.1e9 Msun/particle for
L1000N1800) single-particle "galaxies" are extremely noisy — consider
`--min-star-particles 10` for science cuts, or filter on `n_star_particles`
afterwards.

group_id: with `--soap-filenames` given, each galaxy's host FOF group is
fetched from SOAP (`--group-dataset`, default
`InputHalos/HBTplus/HostFOFId` — change if your SOAP uses a different halo
finder) and `group_id = snapshot*1e12 + host_fof_id`, so centrals and
satellites of one FOF group share a group_id (unique across snapshots;
hostless objects get −1). Without it, `group_id` falls back to
`HaloCatalogueIndex`, which identifies the subhalo, not the group, and is
only unique within a snapshot.

Objects crossing the lightcone near shell/snapshot boundaries can appear
in the halo lightcone more than once (once per snapshot); `galaxy_id` is
unique but you may wish to de-duplicate on (`halo_catalogue_index`,
position) for clustering work.
