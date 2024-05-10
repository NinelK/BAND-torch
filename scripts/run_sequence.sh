python scripts/run_single.py chewie_10_05_M1 lfads_M1_100f_kl1_studentT_bs256 81 0.
python scripts/ablate_controls.py chewie_10_05_M1 lfads_M1_100f_kl1_studentT_bs256 81 0.
python scripts/band_performance.py chewie_10_05_M1 lfads_M1_100f_kl1_studentT_bs256 81 0.

python scripts/run_single.py chewie_10_05_M1 band_M1_100f_kl1_studentT_bs256 81 0.1
python scripts/ablate_controls.py chewie_10_05_M1 band_M1_100f_kl1_studentT_bs256 81 0.1
python scripts/band_performance.py chewie_10_05_M1 band_M1_100f_kl1_studentT_bs256 81 0.1

python scripts/run_single.py chewie_10_05_PMd lfads_PMd_100f_kl1_studentT_bs256 163 0.
python scripts/ablate_controls.py chewie_10_05_PMd lfads_PMd_100f_kl1_studentT_bs256 163 0.
python scripts/band_performance.py chewie_10_05_PMd lfads_PMd_100f_kl1_studentT_bs256 163 0.

python scripts/run_single.py chewie_10_05_PMd band_PMd_100f_kl1_studentT_bs256 163 0.1
python scripts/ablate_controls.py chewie_10_05_PMd band_PMd_100f_kl1_studentT_bs256 163 0.1
python scripts/band_performance.py chewie_10_05_PMd band_PMd_100f_kl1_studentT_bs256 163 0.1

python scripts/run_single.py chewie_10_05 lfads_both_100f_kl1_studentT_bs256 244 0.
python scripts/ablate_controls.py chewie_10_05 lfads_both_100f_kl1_studentT_bs256 244 0.
python scripts/band_performance.py chewie_10_05 lfads_both_100f_kl1_studentT_bs256 244 0.

python scripts/run_single.py chewie_10_05 band_both_100f_kl1_studentT_bs256 244 0.1
python scripts/ablate_controls.py chewie_10_05 band_both_100f_kl1_studentT_bs256 244 0.1
python scripts/band_performance.py chewie_10_05 band_both_100f_kl1_studentT_bs256 244 0.1
