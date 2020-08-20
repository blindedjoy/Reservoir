#!/bin/bash

#module load Anaconda3/2019.10; 
# 
#-c, --cpus-per-task=
module load Anaconda3/2019.10; 
srun -t 10000 --mem 300000 -p bigmem -n 192 bash -c "python execute.py '$1'" & #--cpus-per-task=32
#-t 5760 -p shared -n 12 -c 30 --mem-per-cpu=11gb  bash -c "python execute.py '$1'" & #--cpus-per-task=24 
# 2 nodes for main tasks, 5 cv samples so 2 + (5*2)