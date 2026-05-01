project='pbt-pendulum'
model=pendulum
fac_dim=20
co_dim=1
causal=True

dataset=pendulum
n_all=200
T=100

python scripts/run_pbt.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/ablate_controls.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

python scripts/run_pbt.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/ablate_controls.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}


project='pbt-lorenz'
model=lorenz
fac_dim=20
co_dim=4
causal=True

dataset=lorenz
n_all=200
T=100

python scripts/run_pbt.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/ablate_controls.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

python scripts/run_pbt.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/ablate_controls.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/band_performance.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
