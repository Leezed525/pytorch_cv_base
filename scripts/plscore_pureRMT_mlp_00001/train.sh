#!/bin/bash
cd /home/lzd/workspace/LeeNet
nohup python -u /home/lzd/workspace/LeeNet/train/plscore_pureRMT_mlp00001.py  >/home/lzd/workspace/LeeNet/logs/LeeNet_plScore_RMT_MLP00001//run.log 2>&1 &
echo $! > ./scripts/plscore_pureRMT_mlp_00001/pid.txt
