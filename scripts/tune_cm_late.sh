model=kl1_gauss
bs=256
fac_dim=100
co_dim=4
bw=0.1 # band bw

dataset=mihili_02_18
n_all=159
n_m1=38
n_pmd=121
T=86

# python scripts/run_pbt.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/ablate_controls.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/band_performance.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.

# python scripts/run_pbt.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/ablate_controls.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/band_performance.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}

# python scripts/run_pbt.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/ablate_controls.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/band_performance.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.

# python scripts/run_pbt.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/ablate_controls.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/band_performance.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}

# python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/ablate_controls.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/band_performance.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

# python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/ablate_controls.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/band_performance.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}

dataset=mihili_03_07
n_all=92
n_m1=26
n_pmd=66
T=87

# python scripts/run_pbt.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/ablate_controls.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/band_performance.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.

# python scripts/run_pbt.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/ablate_controls.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/band_performance.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}

# python scripts/run_pbt.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/ablate_controls.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/band_performance.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.

# python scripts/run_pbt.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/ablate_controls.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/band_performance.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}

# python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/ablate_controls.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/band_performance.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

# python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/ablate_controls.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/band_performance.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}


dataset=chewie_10_05
n_all=244
n_m1=81
n_pmd=163
T=100

python scripts/run_pbt.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/ablate_controls.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/band_performance.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.

python scripts/run_pbt.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/ablate_controls.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/band_performance.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}

python scripts/run_pbt.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/ablate_controls.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/band_performance.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.

python scripts/run_pbt.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/ablate_controls.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/band_performance.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}

python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/ablate_controls.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/band_performance.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/ablate_controls.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/band_performance.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}

dataset=chewie_10_07
n_all=207
n_m1=70
n_pmd=137
T=101

python scripts/run_pbt.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/ablate_controls.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
python scripts/band_performance.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.

python scripts/run_pbt.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/ablate_controls.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
python scripts/band_performance.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}

python scripts/run_pbt.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/ablate_controls.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
python scripts/band_performance.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.

python scripts/run_pbt.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/ablate_controls.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
python scripts/band_performance.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}

python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/ablate_controls.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/band_performance.py ${model} ${dataset} lfads_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

# python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# python scripts/ablate_controls.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# python scripts/band_performance.py ${model} ${dataset} band_both_${fac_dim}f_${model}_bs${bs} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}

sh scripts/tune_cm_late_8f.sh