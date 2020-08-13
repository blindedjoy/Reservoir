#!/bin/bash

module load Anaconda3/2019.10; 
srun -t 2440 -n 1 --cpus-per-task=8 --mem-per-cpu=25gb python execute.py "$1" #--core-spec=10 