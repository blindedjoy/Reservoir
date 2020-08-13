#!/bin/bash
srun --mem=200gb -t 1440 --cpus-per-task=3  --ntasks-per-node=2 --threads-per-core=5 -N 1 python execute.py 4