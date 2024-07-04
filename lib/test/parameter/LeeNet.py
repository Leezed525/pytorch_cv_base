from lib.test.utils import TrackerParams
import os
from lib.config.cfg_loader import env_setting


def parameters(yaml_name: str, epoch=None):
    cfg = env_setting(yaml_name)
    params = TrackerParams()
    # prj_dir = env_settings().prj_dir
    # save_dir = env_settings().save_dir
    # update default config from yaml file
    # yaml_file = os.path.join(prj_dir, 'experiments/vipt/%s.yaml' % yaml_name)
    # update_config_from_file(yaml_file)
    params.cfg = cfg
    # print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.test.template.factor
    params.template_size = cfg.test.template.size
    params.search_factor = cfg.test.search.factor
    params.search_size = cfg.test.search.size

    # Network checkpoint path
    params.checkpoint = cfg.test.checkpoint
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
