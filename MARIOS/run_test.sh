#!/bin/bash

#SBATCH -n 30		    # Number of cores requested
#SBATCH -N 1 				# Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=6
#SBATCH -t 1440 		    # Runtime in minutes
#SBATCH -p serial_requeue 	#	 Partition to submit to
#SBATCH --mem=200gb 			# Memory in GB (see also --mem-per-cpu)
#SBATCH -o output_%j.out 	# Standard out goes to this file
#SBATCH -e error_%j.err 	# Standard err goes to this file

#cv_samples = 3 ; n_cores = 2 ==> 6 tasks ; n_experiments = 5 ==> 30

# Labtop uses 8 gb per core, try to go way above that. Let's say, 20 gb per core. 20*8 = 160gb, let's go for 200gb.

echo "Running bayesRC on 20 CPU cores"

# n-tasks per node: n_cores * cv_samples
# -n or num_cores: len(experiment_set) * n-tasks

# 16 tests, 8 cores each. Then we have the cv loop, requesting four cores per run.
# 16 * 8

#install the customized version of Reinier's reservoir package
cd ..; ./reinstall.sh; cd MARIOS; 
#chmod a+x ./reinstall.sh
# 
# ##### asfSBATCH	--cpus-per-task=8

#chmod a+x ./build_file_system.sh
#./build_filesystem.sh

python execute_test.py

#python PyFiles/test.py
