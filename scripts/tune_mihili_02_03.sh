project='pbt-causal-go'
model=kl1_gauss_bd_d20_causal_ci
fac_dim=100
co_dim=4
causal=True

dataset=mihili_02_03
n_all=114
n_m1=34
n_pmd=80
T=92

# python scripts/run_pbt.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}

# python scripts/run_pbt.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}

# python scripts/run_pbt.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}

# python scripts/run_pbt.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}

# python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

# python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

dataset=mihili_02_17
n_all=148
n_m1=44
n_pmd=104
T=85

# python scripts/run_pbt.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}

# python scripts/run_pbt.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}

# python scripts/run_pbt.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}

# python scripts/run_pbt.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}

# python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

# python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

dataset=mihili_02_18
n_all=159
n_m1=38
n_pmd=121
T=86

# python scripts/run_pbt.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}

# python scripts/run_pbt.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}

# python scripts/run_pbt.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}

# python scripts/run_pbt.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}

# python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

# python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

dataset=mihili_03_07
n_all=92
n_m1=26
n_pmd=66
T=87

# python scripts/run_pbt.py ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}

# python scripts/run_pbt.py ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_m1} ${causal}

# python scripts/run_pbt.py ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}

# python scripts/run_pbt.py ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${causal}

# python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

# python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
