#!/bin/bash

#estimated number of cores:

#SBATCH -n 64				# Number of cores requested
#SBATCH -N 1 				# Ensure that all cores are on one machine
#SBATCH -t 60 				# Runtime in minutes
#SBATCH -p serial_requeue 	#	 Partition to submit to
#SBATCH --mem=14 			# Memory in GB (see also --mem-per-cpu)
#SBATCH -o output_%j.out 	# Standard out goes to this file
#SBATCH -e error_%j.err 	# Standard err goes to this file

# 16 tests, 8 cores each. Then we have the cv loop, requesting four cores per run.
# 16 * 8

#install the customized version of Reinier's reservoir package
cd ..
chmod a+x ./reinstall.sh
./reinstall.sh
cd MARIOS

chmod a+x ./build_file_system.sh
./build_filesystem.sh

chmod a+x ./execute_test.py
python execute_test.py

#python PyFiles/test.py



#module load Anaconda3/2019.10
