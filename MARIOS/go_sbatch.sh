#!/bin/bash

module load Anaconda3/2019.10; 



#--cpus-per-task=32
srun -t 5760 -p shared -n 30  --mem 64000 --mem-per-cpu=11gb  bash -c "python execute.py '$1'" & #--cpus-per-task=24 
# 2 nodes for main tasks, 5 cv samples so 2 + (5*2)


#How do I figure out how much memory and time my script took?
#sacct -j jobnumber --format=MaxRSS,elapsed,reqmem,timelimit -c 30


# -n=64
# --continuous
# -p shared# partition (queue)
# --mem 250 # memory
# -t 3-0:00 # time (D-HH:MM)
# -o myscript_%j_output.out # STDOUT
# --mail-type=END # notifications for job done
# --cpus-per-task=15 # notifications for job done
# -c=30 # notifications for job done

