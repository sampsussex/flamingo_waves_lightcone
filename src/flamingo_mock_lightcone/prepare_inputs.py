#!/bin/env python
"""
prepare_inputs.py  --  run this ONCE on a COSMA *login* node (needs internet).

COSMA compute nodes have no outbound network access, so the SPS grid and
the SVO filter curves must be fetched beforehand:

1. The SPS grid. Pre-computed production grids (including
   bpass-2.2.1-bin_chabrier03-0.1,300.0) are available from the
   synthesizer grids collection -- see
   https://synthesizer-project.github.io/synthesizer/  (Grids section)
   or generate one with the synthesizer-grids package. Put the HDF5 file
   in your grid directory. (`synthesizer-download --test-grids -d <dir>`
   fetches a small test grid you can use to try the pipeline first:
   test_grid.hdf5, load it with --grid-name test_grid.)

2. The filter curves. This script downloads the VISTA Z,Y,J,H,Ks and
   VST/OmegaCAM u,g,r,i,z transmission curves from SVO, resamples them
   onto the grid wavelength axis, and writes them to a single HDF5 file
   that make_mock_catalog.py loads with --filters-file (no internet
   needed at run time).

Usage:
  python3 prepare_inputs.py \
      --grid-name bpass-2.2.1-bin_chabrier03-0.1,300.0 \
      --grid-dir /cosma8/data/.../synthesizer_grids \
      --out filters_vista_vst.hdf5
"""

import argparse

from photometry_core import FILTER_CODES, make_filters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-name", default=None,
                        help="If given, resample filters onto this grid's "
                             "wavelength axis (recommended)")
    parser.add_argument("--grid-dir", default=None)
    parser.add_argument("--out", default="filters_vista_vst.hdf5")
    args = parser.parse_args()

    new_lam = None
    if args.grid_name is not None:
        from synthesizer.grid import Grid

        grid = Grid(args.grid_name, grid_dir=args.grid_dir,
                    ignore_lines=True)
        new_lam = grid.lam
        print(f"Resampling filters onto grid axis ({len(new_lam)} points)")

    print("Fetching filters from SVO:")
    for code in FILTER_CODES:
        print(f"  {code}")
    filters = make_filters(new_lam=new_lam)
    filters.write_filters(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()