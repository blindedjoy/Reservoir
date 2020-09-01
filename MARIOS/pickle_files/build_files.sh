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

# spectrogram/ (ft means fourier transform, cqt is constant-Q transform)
#	type/ power, amplitude, log-normalized ect.
#		target_size/ (0.5 to 1 kilohertz)
#			{model}_{n_obs}.txt ([exp, unif], [0.5 kh to 1 kh])
spectrograms=("18th_ft_high" "18th_cqt_high" "18th_ft_low" "18th_cqt_low" "free_ft" "free_cqt")

build_all="T" 
### currently in use:
if [ "$build_all" == "T" ];
then
	curr_directory="results";  ifdir "${curr_directory}"
	for spectrogram in ${spectrograms[@]}; do
		ifdir "${curr_directory}/${spectrogram}/"

		for type in "power" "db"; do
			ifdir "${curr_directory}/${spectrogram}/${type}"

			for flat in "flat" "untouched"; do
				ifdir "${curr_directory}/${spectrogram}/${type}/${flat}"

				for split in "0.5" "0.9"; do
					ifdir "${curr_directory}/${spectrogram}/${type}/${flat}/split_${split}"
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
