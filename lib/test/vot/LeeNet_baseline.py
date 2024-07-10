import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.vot.LeeNet_class import run_vot_exp

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

run_vot_exp('LeeNet', 'plscore_os_sigma_RGBD', vis=False, out_conf=True, channel_type='rgbd')
