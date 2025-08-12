project='pbt-causal-go'
model=kl1_gauss_bd_d20_causal_ci
fac_dim=100
co_dim=4
causal=True

dataset=chewie_10_05
n_all=244
n_m1=81
n_pmd=163
T=100

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

dataset=chewie_10_07
n_all=207
n_m1=70
n_pmd=137
T=101

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
