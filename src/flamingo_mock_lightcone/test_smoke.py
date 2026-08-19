"""
Smoke test for photometry_core using a synthetic SPS grid and top-hat
filters (no internet needed). Not part of the production pipeline.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Build a fake SPS grid file that synthesizer can load
# ---------------------------------------------------------------------------
grid_dir = "/tmp/fake_grids"
grid_name = "fake_sps"
import os

os.makedirs(grid_dir, exist_ok=True)

nage, nZ, nlam = 8, 5, 2000
ages = np.logspace(6, 10.1, nage)  # yr
mets = np.logspace(-4, np.log10(0.04), nZ)
lam = np.logspace(2.6, 4.9, nlam)  # 400 A - 8e4 A

# A vaguely blackbody-ish Lnu per Msun (erg/s/Hz/Msun), declining with age
lnu = np.zeros((nage, nZ, nlam))
for i in range(nage):
    for j in range(nZ):
        T = 3.0e4 / (1 + i)  # hotter when younger
        # crude Planck-like shape in lambda, arbitrary normalisation
        x = 1.43877e8 / (lam * T)  # hc/(lam k T), lam in A
        b = 1.0 / (lam**3 * np.expm1(np.clip(x, 1e-6, 500)))
        lnu[i, j] = 1e21 * b / b.max() * (ages[0] / ages[i]) ** 0.5

with h5py.File(f"{grid_dir}/{grid_name}.hdf5", "w") as hf:
    hf.attrs["axes"] = ["ages", "metallicities"]
    hf.attrs["WeightVariable"] = "initial_masses"
    hf.attrs["cloudy_version"] = "c17.03"  # mark as cloudy-reprocessed
    ax = hf.create_group("axes")
    d = ax.create_dataset("ages", data=ages)
    d.attrs["Units"] = "yr"
    d.attrs["log_on_read"] = True
    d = ax.create_dataset("metallicities", data=mets)
    d.attrs["Units"] = "dimensionless"
    d.attrs["log_on_read"] = False
    sp = hf.create_group("spectra")
    d = sp.create_dataset("wavelength", data=lam)
    d.attrs["Units"] = "angstrom"
    for key, arr in [
        ("incident", lnu),
        ("transmitted", 0.9 * lnu),
        ("nebular", 0.15 * lnu),
        ("linecont", 0.05 * lnu),
    ]:
        d = sp.create_dataset(key, data=arr)
        d.attrs["Units"] = "erg/s/Hz"

from synthesizer.grid import Grid

grid = Grid(grid_name, grid_dir=grid_dir, ignore_lines=True)
print("Grid loaded. reprocessed =", grid.reprocessed)

import photometry_core as pc

lo, hi = pc.get_grid_age_limits_yr(grid)
print("age limits (yr):", lo, hi)
zlo, zhi = pc.get_grid_metallicity_limits(grid)
print("Z limits:", zlo, zhi)

# Top-hat stand-ins for the real VISTA/VST filters (same codes)
from synthesizer.instruments import FilterCollection
from unyt import angstrom

tophats = {}
for (name, code), cen in zip(pc.FILTER_DEF, np.linspace(3800, 21500, 10)):
    tophats[code] = {"lam_eff": cen * angstrom, "lam_fwhm": 900 * angstrom}
filters = FilterCollection(tophat_dict=tophats, new_lam=grid.lam)
print("Filters:", len(filters.filter_codes))

# check writing/reading filters round-trip
filters.write_filters("/tmp/filters_test.hdf5")
filters2 = pc.load_filters("/tmp/filters_test.hdf5")
print("Filter round-trip OK:", len(filters2.filter_codes))

from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=68.1, Om0=0.306, Ob0=0.0486)
age_of_a = pc.build_age_interpolator(cosmo)
print("age(a=1) Gyr:", age_of_a(1.0) / 1e9)

for mode in ("incident", "reprocessed", "pacman"):
    model = pc.build_emission_model(grid, mode=mode, tau_v=0.33, fesc=0.0)

    rng = np.random.default_rng(42)
    n = 50
    im = np.full(n, 1.0e9)  # FLAMINGO-like particle masses, Msun
    a_birth = rng.uniform(0.2, 0.83, n)
    z_gal = 0.1
    a_obs = 1.0 / (1 + z_gal)
    ages_yr = np.clip(age_of_a(a_obs) - age_of_a(a_birth), lo, hi)
    zmet = np.clip(rng.lognormal(np.log(0.01), 0.3, n), zlo, zhi)

    from synthesizer.emission_models.attenuation import Inoue14

    app, ab = pc.compute_galaxy_photometry(
        grid, model, filters, cosmo, im, ages_yr, zmet, z_gal, igm=Inoue14()
    )
    print(f"[{mode}] app mags:", np.round(app, 2))
    print(f"[{mode}] abs mags:", np.round(ab, 2))
    assert np.all(np.isfinite(app)) and np.all(np.isfinite(ab))
    # sanity: distance modulus ~ app - abs (K-corr aside)
    from astropy.cosmology import z_at_value

    dm = cosmo.distmod(z_gal).value
    print(f"[{mode}] mean (app-abs) = {np.mean(app - ab):.2f}, DM = {dm:.2f}")

print("SMOKE TEST PASSED")