#!/bin/bash

#SBATCH --job-name=parallel_job      # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hnjoy@mac.com   # Where to send mail	
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=10                   # Run a single task		
#SBATCH --cpus-per-task=10            # Number of CPU cores per task
#SBATCH --mem=128000                   # Job memory request
#SBATCH --time=00:05:00              # Time limit hrs:min:sec
#SBATCH --output=parallel_%j.log     # Standard output and error log

echo "Running bayesRC on 20 CPU cores"



# 16 tests, 8 cores each. Then we have the cv loop, requesting four cores per run.
# 16 * 8

#install the customized version of Reinier's reservoir package
#cd ..
#chmod a+x ./reinstall.sh
# ./reinstall.sh
#cd MARIOS ##### asfSBATCH	--cpus-per-task=8

#chmod a+x ./build_file_system.sh
#./build_filesystem.sh

python execute_test.py

#python PyFiles/test.py
