#!/bin/bash
cd /home/lzd/workspace/LeeNet
nohup python -u /home/lzd/workspace/LeeNet/train/plscore_pureRMT_sigma_center.py  >/home/lzd/workspace/LeeNet/logs/LeeNet_plScore_RMT_sigma_CENTER/run.log 2>&1 &
echo $! > ./scripts/plscore_pureRMT_sigma_center/pid.txt
