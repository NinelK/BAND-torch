model=kl1_gauss_bd
fac_dim=100
co_dim=4
bw=0.1 # band bw #TODO substitute with a binary flag

# dataset=chewie_09_15
# n_all=309
# n_m1=76
# n_pmd=233
# T=102

# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.

# dataset=chewie_09_21
# n_all=303
# n_m1=72
# n_pmd=231
# T=103

# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.

# dataset=chewie_10_05
# n_all=244
# n_m1=81
# n_pmd=163
# T=100

# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.

dataset=chewie_10_07
n_all=207
n_m1=70
n_pmd=137
T=101

python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.

# dataset=mihili_02_03
# n_all=114
# n_m1=34
# n_pmd=80
# T=92

# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.

# dataset=mihili_02_17
# n_all=148
# n_m1=44
# n_pmd=104
# T=85

# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.

# dataset=mihili_02_18
# n_all=159
# n_m1=38
# n_pmd=121
# T=86

# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.

# dataset=mihili_03_07
# n_all=92
# n_m1=26
# n_pmd=66
# T=87

# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# # python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
# # python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 band_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_M1 lfads_M1_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_m1} 0.

# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd band_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} ${bw}
# python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
# python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_PMd lfads_PMd_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_pmd} 0.
