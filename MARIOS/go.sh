#!/bin/bash

#module load Anaconda3/2019.10; 
# -t 5760 -p bigmem -N 4 --cpus-per-task=32 bash -c "python execute.py '$1'" 

module load Anaconda3/2019.10; 
srun -t 5760 -p shared -n 12 bash -c "python execute.py '$1'" & #--cpus-per-task=24 --mem-per-cpu=11gb 
# 2 nodes for main tasks, 5 cv samples so 2 + (5*2)