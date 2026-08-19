#!/bin/bash -l
#
# COSMA8 submission script for make_mock_catalog.py
# Adjust account/partition/modules to match whatever you already use to run
# the lightcone->halo matching script (same virgo/mpi4py environment works).
#
#SBATCH -J syth_mock_lightcone
#SBATCH --output=logs/log_mock_%j.txt
#SBATCH -p cosma-analyse
#SBATCH -A do012
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

module purge
# Use the same module stack as for the halo-matching run, e.g.:
module load gnu_comp/14.1.0 openmpi hdf5 python

# Environment with: mpi4py, virgo, h5py, cosmos-synthesizer, astropy,
# unyt, pyarrow  (see README)
source /cosma/home/do012/dc-phil3/envs/main_env/bin/activate

# Keep numpy/BLAS from oversubscribing cores
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BASEDIR=/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL #LGTM
TAGGED=/cosma8/data/do012/dc-phil3/FLAMINGO/ScienceRuns/HYDRO_FIDUCIAL/lightcone_particle_halo_ids/lightcone0/   # output of matching script
GRIDDIR=/cosma8/data/dp012/${USER}/synthesizer_grids #LGTM
FILTERS=/cosma8/data/dp012/${USER}/synthesizer_grids/filters_vista_vst.hdf5 #LGTM
OUTDIR=/snap8/scratch/dp012/${USER}/mocks/L1000N1800/HYDRO_FIDUCIAL/lightcone0 #LGTM

mkdir -p logs "${OUTDIR}"

mpirun -- python3 -u -m mpi4py make_mock_catalog.py \
    "${TAGGED}" \
    "${BASEDIR}/lightcones" \
    lightcone0 \
    --output-dir "${OUTDIR}" \
    --grid-name  "bpass-2.2.1-bin_chabrier03-0.1,300.0" \
    --grid-dir   "${GRIDDIR}" \
    --filters-file "${FILTERS}" \
    --emission pacman --tau-v 0.33 --fesc 0.0 \
    --soap-filenames "${BASEDIR}/SOAP-HBT/halo_properties_%(snap_nr)04d.hdf5" \ 
    --group-dataset "InputHalos/HBTplus/HostFOFId" \
    --min-star-particles 3