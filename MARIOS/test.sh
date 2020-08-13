#!/bin/bash
srun --mem=200gb -t 1440 -n 16 --threads-per-core=5 python execute.py 4