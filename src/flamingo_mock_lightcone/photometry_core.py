"""
photometry_core.py

Serial (non-MPI) core of the FLAMINGO lightcone -> mock catalogue pipeline:
everything that touches Synthesizer lives here so it can be imported and
tested without mpi4py / virgo.

Tested against synthesizer (cosmos-synthesizer) v1.2.0.
"""

import numpy as np
from unyt import Msun, yr

# ----------------------------------------------------------------------------
# Filters: (short name used in output columns, SVO filter code)
# ----------------------------------------------------------------------------
FILTER_DEF = [
    ("VISTA_Z",  "Paranal/VISTA.Z"),
    ("VISTA_Y",  "Paranal/VISTA.Y"),
    ("VISTA_J",  "Paranal/VISTA.J"),
    ("VISTA_H",  "Paranal/VISTA.H"),
    ("VISTA_Ks", "Paranal/VISTA.Ks"),
    ("VST_u",    "Paranal/OmegaCAM.u"),
    ("VST_g",    "Paranal/OmegaCAM.g"),
    ("VST_r",    "Paranal/OmegaCAM.r"),
    ("VST_i",    "Paranal/OmegaCAM.i"),
    ("VST_z",    "Paranal/OmegaCAM.z"),
]
FILTER_NAMES = [n for n, _ in FILTER_DEF]
FILTER_CODES = [c for _, c in FILTER_DEF]

# AB magnitude constants
_FOUR_PI_D10_SQ = 4.0 * np.pi * (3.0856775814913673e19) ** 2  # (10 pc in cm)^2
_AB_ZP_CGS = -48.60  # AB zero point for f_nu in erg/s/cm^2/Hz


def load_filters(path):
    """Load a FilterCollection previously written with write_filters()."""
    from synthesizer.instruments import FilterCollection

    return FilterCollection(path=path)


def make_filters(new_lam=None):
    """Create the FilterCollection from SVO (needs internet: login node)."""
    from synthesizer.instruments import FilterCollection

    return FilterCollection(filter_codes=FILTER_CODES, new_lam=new_lam)


def build_emission_model(grid, mode="pacman", tau_v=0.33, fesc=0.0,
                         dust_slope=-1.0):
    """
    Build the emission model.

    mode:
      "incident"    : pure stellar
      "reprocessed" : stellar + nebular (no dust)
      "pacman"      : stellar + nebular + power-law dust screen (tau_v).
                      With fesc=0 and no dust emission the root spectrum is
                      "attenuated"; with fesc>0 it is "emergent".
    """
    if mode == "incident":
        from synthesizer.emission_models import IncidentEmission

        return IncidentEmission(grid)
    elif mode == "reprocessed":
        from synthesizer.emission_models import ReprocessedEmission

        return ReprocessedEmission(grid, fesc=fesc)
    elif mode == "pacman":
        from synthesizer.emission_models import PacmanEmission
        from synthesizer.emission_models.attenuation import PowerLaw

        return PacmanEmission(
            grid,
            tau_v=tau_v,
            dust_curve=PowerLaw(slope=dust_slope),
            fesc=fesc,
            fesc_ly_alpha=1.0,
        )
    else:
        raise ValueError(f"Unknown emission model mode: {mode}")


def get_grid_age_limits_yr(grid):
    """Return (min, max) grid age in yr, robust to axis naming."""
    for attr in ("ages", "age"):
        vals = getattr(grid, attr, None)
        if vals is not None:
            v = np.asarray(vals)
            return float(v.min()), float(v.max())
    return 1.0e6, 1.5e10  # conservative fallback


def get_grid_metallicity_limits(grid):
    for attr in ("metallicities", "metallicity"):
        vals = getattr(grid, attr, None)
        if vals is not None:
            v = np.asarray(vals)
            return float(v.min()), float(v.max())
    return 1.0e-5, 0.04  # BPASS-like fallback


def compute_galaxy_photometry(
    grid,
    model,
    filters,
    cosmo,
    initial_masses_msun,
    ages_yr,
    metallicities,
    redshift,
    igm=None,
    nthreads=1,
):
    """
    Compute apparent and absolute AB magnitudes for one galaxy.

    Parameters
    ----------
    initial_masses_msun, ages_yr, metallicities : arrays over star particles
        (already clipped to the grid ranges by the caller).
    redshift : float, cosmological redshift used for luminosity distance
        and the observer-frame SED.
    igm : IGM absorption class instance (e.g. Inoue14()) or None.

    Returns (app_mags, abs_mags): float32 arrays of len(filters), NaN where
    the flux is non-positive.
    """
    from synthesizer.particle import Stars

    stars = Stars(
        initial_masses=np.ascontiguousarray(initial_masses_msun) * Msun,
        ages=np.ascontiguousarray(ages_yr) * yr,
        metallicities=np.ascontiguousarray(metallicities, dtype=np.float64),
        redshift=redshift,
    )

    # The root Sed of the model (e.g. "attenuated"/"emergent" for Pacman)
    sed = stars.get_spectra(model, verbose=False, nthreads=nthreads)

    # ---- absolute (rest-frame) magnitudes -------------------------------
    photo_lnu = sed.get_photo_lnu(filters, verbose=False)
    abs_mags = np.empty(len(FILTER_CODES), dtype=np.float32)
    for i, code in enumerate(FILTER_CODES):
        lnu = float(photo_lnu[code].to("erg/s/Hz").value)
        if lnu > 0:
            abs_mags[i] = -2.5 * np.log10(lnu / _FOUR_PI_D10_SQ) + _AB_ZP_CGS
        else:
            abs_mags[i] = np.nan

    # ---- apparent (observer-frame) magnitudes ---------------------------
    # Guard against z ~ 0 (d_L -> 0): floor at z=1e-4 (~0.4 Mpc)
    z_use = max(float(redshift), 1.0e-4)
    sed.get_fnu(cosmo, z_use, igm=igm)
    photo_fnu = sed.get_photo_fnu(filters, verbose=False)
    app_mags = np.empty(len(FILTER_CODES), dtype=np.float32)
    for i, code in enumerate(FILTER_CODES):
        fnu = float(photo_fnu[code].to("nJy").value)
        if fnu > 0:
            # m_AB = -2.5 log10(fnu/nJy) + 31.4
            app_mags[i] = -2.5 * np.log10(fnu) + 31.4
        else:
            app_mags[i] = np.nan

    return app_mags, abs_mags


def build_age_interpolator(cosmo, a_min=1.0e-3, n=4096):
    """
    Return a function a -> age of universe in yr (linear interp in a).
    cosmo is an astropy cosmology.
    """
    a_grid = np.linspace(a_min, 1.0, n)
    z_grid = 1.0 / a_grid - 1.0
    t_grid = cosmo.age(z_grid).to_value("yr")

    def age_of_a(a):
        return np.interp(np.clip(a, a_min, 1.0), a_grid, t_grid)

    return age_of_a