#! /bin/bash
set -e

python -u ssem/run_test.py \
  --data_path /home/bcl/wanghongyu/wanghongyu_humx/dynamic/ssem/data/CREMI/sample_C_20160501.hdf \
  --output_dir /home/bcl/wanghongyu/wanghongyu_humx/dynamic/ssem/outputs \
  --r 3 \
  --weights 0.05,0.15,0.30,0.30,0.15,0.05 \
  --loss_func w2 \
  --epochs 1000 \
  --lr 1e-2 \
  --reg_grad 1e-4 \
  --reg_time 1e-4 \
  --reg_l2 1e-4 \
  --w2_eps 1e-8 \
  --w2_maxiter 100 \
  --time_steps 10 \
  --smooth_sigma 0.5 \
  --interp_mode bicubic \
  --padding_mode border \
  --align_corners \
  --down_sample \
  --early_stop_delta 1e-3 \
  --k 5

# CUDA_VISIBLE_DEVICES=1 nohup ssem/runs/run.sh > ssem/runs/run.log 2>&1 &
# 1288382
