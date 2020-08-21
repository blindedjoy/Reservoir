#!/bin/bash
# --export=ALL
# -N 1 
# -p shared                          # partition (queue)
# -n 25                              # number of cores                 
# --mem 150gb                        # memory
# -t 7-0:00                          # time (D-HH:MM)
# -o myscript_%j_output.out          # STDOUT
# --mail-type=END                    # notifications for job done

#module load Anaconda3/2019.10
# 
#-c, --cpus-per-task=
#module load Anaconda3/2019.10; 
for x in {0..8}
do
	echo $x
	srun -N 1 -t 9000 --mem 124gb -p shared -n 20 bash -c "python execute.py '$x'" & #--cpus-per-task=32
done
#python execute.py 1 & #'$1'
#-t 5760 -p shared -n 12 -c 30 --mem-per-cpu=11gb  bash -c "python execute.py '$1'" & #--cpus-per-task=24 
# 2 nodes for main tasks, 5 cv samples so 2 + (5*2)