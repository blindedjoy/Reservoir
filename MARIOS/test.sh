#!/bin/bash
srun --mem=200gb -t 1440 -n 2 --cpus-per-task=15  python execute.py 4 #--core-spec=10 