#!/bin/bash
cd /home/lzd/workspace/LeeNet
nohup python -u /home/lzd/workspace/LeeNet/train/score_pureRMT_mlp.py  >/home/lzd/workspace/LeeNet/logs/run_log.log 2>&1 &
echo $! > ./scripts/pid.txt
