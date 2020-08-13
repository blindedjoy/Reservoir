#!/bin/bash
srun --mem=200gb -t 1440 -n 16 --threads-per-core=5 -N 1 python execute.py 4