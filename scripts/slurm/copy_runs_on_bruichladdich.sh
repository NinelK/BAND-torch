repo_home=/home/${USER}/experiments/BAND-torch
#src_path=${repo_home}/runs/pbt-band-paper
#dest_path=bruichladdich:/disk/scratch2/nkudryas/BAND-torch/runs/pbt-band-paper-slurm
#rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

src_path=${repo_home}/runs/band-paper-slurm
dest_path=bruichladdich:/disk/scratch2/nkudryas/BAND-torch/runs/band-paper-slurm
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}
