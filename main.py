import hydra
import other_utils
import torch
import logging
from core import train
import os
from pyvirtualdisplay import Display


logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    log_dict = other_utils.get_log_dict(cfg.agent._target_)
    for seed in cfg.seeds:
        train(cfg, seed, log_dict, -1, logger, None)


if __name__ == "__main__":
    # print(os.chdir())
    os.chdir("/home/liuan/workspace/drl_project/LIGENT")
    with Display(visible=False) as disp:
        main()
