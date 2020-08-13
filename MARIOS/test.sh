#!/bin/bash

module load Anaconda3/2019.10; 
srun --mem=300gb -t 1440 -n 1 --cpus-per-task=30  python execute.py "$1" #--core-spec=10 