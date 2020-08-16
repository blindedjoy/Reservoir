#!/bin/bash

module load Anaconda3/2019.10; 
srun -t 5760 -n 1 --cpus-per-task=20 --mem-per-cpu=10gb bash -c "python execute.py '$1'" &
