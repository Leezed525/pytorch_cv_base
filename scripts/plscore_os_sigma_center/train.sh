#!/bin/bash
cd /home/lzd/workspace/LeeNet
export CUDA_VISIBLE_DEVICES=1,2,3
nohup python -m torch.distributed.launch --nproc_per_node=3 --use_env /home/lzd/workspace/LeeNet/train/plscore_os_sigma_center.py  >/home/lzd/workspace/LeeNet/logs/LeeNet_plScore_OS_sigma_CENTER/run.log 2>&1 &
echo $! > ./scripts/plscore_os_sigma_center/pid.txt
