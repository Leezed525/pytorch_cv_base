#!/bin/bash
cd /home/lzd/workspace/LeeNet
nohup python -u /home/lzd/workspace/LeeNet/train/score_pureRMT_mlp.py  >/home/lzd/workspace/LeeNet/logs/LeeNet_score_RMT_MLP00001/run.log 2>&1 &
echo $! > ./scripts/score_pureRMT_mlp_00001/pid.txt
