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
module load Anaconda3/2019.10; 
for x in {0..1}
do
	echo $x
	srun  -t 5760 --cpus-per-task=8 -n 16 -p shared --mem=64gb bash -c "python execute.py '$x'" & #-N 1 -t 9000 --mem 124gb -n 20 bash -c "python execute.py '$x'" & #--cpus-per-task=32 -p shared 
done
#python execute.py 1 & #'$1' --continuous -c 30 -  -N 1  -mem-per-cpu=11gb 
# #
# 2 nodes for main tasks, 5 cv samples so 2 + (5*2)