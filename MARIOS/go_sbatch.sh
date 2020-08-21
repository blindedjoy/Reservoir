#!/bin/bash

#SBATCH -n=64
#SBATCH --continuous
#SBATCH -p shared# partition (queue)
#SBATCH --mem 250 # memory
#SBATCH -t 3-0:00 # time (D-HH:MM)
#SBATCH -o myscript_%j_output.out # STDOUT
#SBATCH --mail-type=END # notifications for job done
#SBATCH --cpus-per-task=15 # notifications for job done
#SBATCH -c=30 # notifications for job done


module load Anaconda3/2019.10; 
python execute.py '$1'


#--cpus-per-task=32
#-t 5760 -p shared -n 12 -c 30 --mem-per-cpu=11gb  bash -c "python execute.py '$1'" & #--cpus-per-task=24 
# 2 nodes for main tasks, 5 cv samples so 2 + (5*2)


#How do I figure out how much memory and time my script took?
#sacct -j jobnumber --format=MaxRSS,elapsed,reqmem,timelimit