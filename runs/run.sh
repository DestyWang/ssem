#! /bin/bash
set -e

python -u ssem/run_test.py \
  --data_path /home/bcl/wanghongyu/wanghongyu_humx/dynamic/ssem/data/sampleA/sample_A+_20160601.hdf \
  --output_dir /home/bcl/wanghongyu/wanghongyu_humx/dynamic/ssem/outputs \
  --r 3 \
  --weights 0.05,0.15,0.3,0.3,0.15,0.05 \
  --loss_func w2 \
  --epochs 200 \
  --lr 1e-2 \
  --reg_grad 1e-3 \
  --reg_time 1e-3 \
  --reg_l2 1e-3 \
  --w2_eps 1e-8 \
  --w2_maxiter 100 \
  --time_steps 10 \
  --smooth_sigma 0.5 \
  --interp_mode bicubic \
  --padding_mode border \
  --align_corners \
  --down_sample \
  --k 5 > ssem/outputs/run_w2.log 2>&1

  python -u ssem/run_test.py \
  --data_path /home/bcl/wanghongyu/wanghongyu_humx/dynamic/ssem/data/sampleA/sample_A+_20160601.hdf \
  --output_dir /home/bcl/wanghongyu/wanghongyu_humx/dynamic/ssem/outputs \
  --r 3 \
  --weights 0.05,0.15,0.3,0.3,0.15,0.05 \
  --loss_func l2 \
  --epochs 200 \
  --lr 1e-2 \
  --reg_grad 1e-3 \
  --reg_time 1e-3 \
  --reg_l2 1e-3 \
  --w2_eps 1e-8 \
  --w2_maxiter 100 \
  --time_steps 10 \
  --smooth_sigma 0.5 \
  --interp_mode bicubic \
  --padding_mode border \
  --align_corners \
  --down_sample \
  --k 5 > ssem/outputs/run_l2.log 2>&1

# CUDA_VISIBLE_DEVICES=7 nohup ssem/runs/run.sh > ssem/runs/run.log 2>&1 &
# 2388882
