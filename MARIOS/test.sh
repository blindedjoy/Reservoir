#!/bin/bash
srun --mem=200gb -t 1440 -n 30 --core-spec=10 python execute.py 4