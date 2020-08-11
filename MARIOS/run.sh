#!/bin/bash


#SBATCH --job-name=parallel_job      # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hnjoy@mac.com   # Where to send mail	

#SBATCH --output=parallel_%j.log     # Standard output and error log

#SBATCH --job-name=slurm-test    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks>1         # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64000        # memory per cpu-core
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

#install the customized version of Reinier's reservoir package
cd ..
chmod a+x ./reinstall.sh
./reinstall.sh
cd MARIOS

chmod a+x ./build_file_system.sh
./build_filesystem.sh

chmod a+x ./execute.py
python execute.py

#python PyFiles/test.py