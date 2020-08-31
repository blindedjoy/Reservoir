#!/bin/bash
ifdir()
{
	for cmdd in "$@"
	do
		if  test -d "$cmdd";
		then
			echo the $cmdd directory exists 
		else
			mkdir $cmdd
		fi
	done 
}

# spectogram/ (ft means fourier transform, cqt is constant-Q transform)
#	type/ power, amplitude, log-normalized ect.
#		target_size/ (0.5 to 1 kilohertz)
#			{model}_{n_obs}.txt ([exp, unif], [0.5 kh to 1 kh])
spectograms=("18th_ft" "18th_cqt" "free_ft" "free_cqt")

build_all="T" 
### currently in use:
if [ "$build_all" == "T" ];
then
	curr_directory="results";  ifdir "${curr_directory}"
	for spectogram in ${spectograms[@]}; do
		ifdir "${curr_directory}/${spectogram}/"

		for type in "power" "db"; do
			ifdir "${curr_directory}/${spectogram}/${type}"

			for flat in "flat" "untouched"; do
				ifdir "${curr_directory}/${spectogram}/${type}/${flat}"

				for split in "0.5" "0.9"; do
					ifdir "${curr_directory}/${spectogram}/${type}/${flat}/split_${split}"
				done
			done
		done
	done
fi
tree "results"
# then you will, for this first experimental pass, create 
#four files: 
#	exp__1kh_obs.txt, unif__1kh_obs.txt 
#   and 
#	exp__.5kh_obs.txt, unif__0.5kh_obs.txt 
