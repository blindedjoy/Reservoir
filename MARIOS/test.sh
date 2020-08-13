#!/bin/bash

module load Anaconda3/2019.10; 
srun --mem=200gb -t 1440 -n 1 --cpus-per-task=30  python execute.py 4 #--core-spec=10 