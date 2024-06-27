python scripts/run_pbt_longitudinal.py

fac_dim=100
co_dim=0
if [ $co_dim -gt 0 ]; then
    python scripts/ablate_controls.py longitudinal_data1 lfads_${fac_dim}f_${co_dim}c ${fac_dim} ${co_dim} 192 0. 0
fi
python scripts/band_performance.py longitudinal_data1 lfads_${fac_dim}f_${co_dim}c ${fac_dim} ${co_dim} 192 0. 0

if [ $co_dim -gt 0 ]; then
    python scripts/ablate_controls.py longitudinal_data1 band_${fac_dim}f_${co_dim}c ${fac_dim} ${co_dim} 192 0.1 0
fi
python scripts/band_performance.py longitudinal_data1 band_${fac_dim}f_${co_dim}c ${fac_dim} ${co_dim} 192 0.1 0
