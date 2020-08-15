#!/bin/bash

module load Anaconda3/2019.10; 
srun -t 2440 -n 1 --cpus-per-task=20 --mem-per-cpu=10gb  python execute.py "$1" &
