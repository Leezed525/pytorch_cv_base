#!/bin/bash
cd /home/lzd/workspace/LeeNet
nohup python -u /home/lzd/workspace/LeeNet/train/plscore_RMT_mix_center00001.py  >/home/lzd/workspace/LeeNet/logs/LeeNet_plScore_RMT_mix_CENTER/run.log 2>&1 &
echo $! > ./scripts/plscore_RMT_mix_center_00001/pid.txt
