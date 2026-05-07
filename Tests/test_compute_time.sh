#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh

# echo "fenics-2019-py310 (x86)" > test_compute_time.txt
# conda activate fenics-2019-py310
# echo CONDA_PREFIX=$CONDA_PREFIX
# unset OMP_NUM_THREADS
# echo OMP_NUM_THREADS=$OMP_NUM_THREADS
# rm -rf $CONDA_PREFIX/.cache/dijitso
# gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
# gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
# conda deactivate

# echo "fenics-2019-py310 (x86) (OMP_NUM_THREADS=1)" >> test_compute_time.txt
# conda activate fenics-2019-py310
# echo CONDA_PREFIX=$CONDA_PREFIX
# export OMP_NUM_THREADS=1
# echo OMP_NUM_THREADS=$OMP_NUM_THREADS
# rm -rf $CONDA_PREFIX/.cache/dijitso
# gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
# gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
# conda deactivate

# echo "dolfin_mech_arm (arm)" >> test_compute_time.txt
# conda activate dolfin_mech_arm
# echo CONDA_PREFIX=$CONDA_PREFIX
# # unset OMP_NUM_THREADS
# # echo OMP_NUM_THREADS=$OMP_NUM_THREADS
# rm -rf $CONDA_PREFIX/.cache/dijitso
# gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
# gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
# conda deactivate

echo "arm (fenics)" >> test_compute_time.txt
conda activate fenics
echo CONDA_PREFIX=$CONDA_PREFIX
unset OMP_NUM_THREADS
echo OMP_NUM_THREADS=$OMP_NUM_THREADS
rm -rf $CONDA_PREFIX/.cache/dijitso
gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
gtime -f "Time: %e s | User: %U s | Syst: %S s | CPU: %P | Max RAM: %M KB" -a -o test_compute_time.txt make
conda deactivate
