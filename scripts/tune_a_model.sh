# python scripts/run_pbt.py

fac_dim=8
co_dim=1
for sample in 0 1
do
    # python scripts/ablate_controls.py chewie_10_07_small lfads_${fac_dim}f_${co_dim}c_minloss_KL1_normal_sample${sample} ${fac_dim} ${co_dim} 207 0. 0
    python scripts/band_performance.py chewie_10_07_small lfads_${fac_dim}f_${co_dim}c_minloss_KL1_normal_sample${sample} ${fac_dim} ${co_dim} 207 0. 0

    # python scripts/ablate_controls.py chewie_10_07_small band_${fac_dim}f_${co_dim}c_minloss_KL1_normal_sample${sample} ${fac_dim} ${co_dim} 207 0.1 0
    python scripts/band_performance.py chewie_10_07_small band_${fac_dim}f_${co_dim}c_minloss_KL1_normal_sample${sample} ${fac_dim} ${co_dim} 207 0.1 0
done
${fac_dim}