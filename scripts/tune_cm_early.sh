model=kl1_gauss
bs=256
fac_dim=100
co_dim=4
bw=0.1 # band bw

dataset=mihili_02_03
n_all=114
n_m1=34
n_pmd=80
T=92

python scripts/run_pbt.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/ablate_controls.py ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/band_performance.py ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} 0.

python scripts/run_pbt.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/ablate_controls.py ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/band_performance.py ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} ${bw}

python scripts/run_pbt.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/ablate_controls.py ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/band_performance.py ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} 0.

python scripts/run_pbt.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/ablate_controls.py ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/band_performance.py ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} ${bw}

python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/ablate_controls.py ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/band_performance.py ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} 0.

python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/ablate_controls.py ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/band_performance.py ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} ${bw}

dataset=mihili_02_17
n_all=148
n_m1=44
n_pmd=104
T=85

python scripts/run_pbt.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/ablate_controls.py ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/band_performance.py ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} 0.

python scripts/run_pbt.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/ablate_controls.py ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/band_performance.py ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} ${bw}

python scripts/run_pbt.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/ablate_controls.py ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/band_performance.py ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} 0.

python scripts/run_pbt.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/ablate_controls.py ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/band_performance.py ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} ${bw}

python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/ablate_controls.py ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/band_performance.py ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} 0.

python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/ablate_controls.py ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/band_performance.py ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} ${bw}


dataset=chewie_09_15
n_all=309
n_m1=76
n_pmd=233
T=102

python scripts/run_pbt.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/ablate_controls.py ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/band_performance.py ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} 0.

python scripts/run_pbt.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/ablate_controls.py ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/band_performance.py ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} ${bw}

python scripts/run_pbt.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/ablate_controls.py ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/band_performance.py ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} 0.

python scripts/run_pbt.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/ablate_controls.py ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/band_performance.py ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} ${bw}

python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/ablate_controls.py ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/band_performance.py ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} 0.

python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/ablate_controls.py ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/band_performance.py ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} ${bw}

dataset=chewie_09_21
n_all=303
n_m1=72
n_pmd=231
T=103

python scripts/run_pbt.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/ablate_controls.py ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/band_performance.py ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} 0.

python scripts/run_pbt.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/ablate_controls.py ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/band_performance.py ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_m1} ${bw}

python scripts/run_pbt.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/ablate_controls.py ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/band_performance.py ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} 0.

python scripts/run_pbt.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/ablate_controls.py ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/band_performance.py ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_pmd} ${bw}

python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/ablate_controls.py ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/band_performance.py ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} 0.

python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/ablate_controls.py ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/band_performance.py ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${fac_dim} ${co_dim} ${n_all} ${bw}