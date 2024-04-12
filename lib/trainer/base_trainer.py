"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/12 12:39
"""

import torch
from lib.config.cfg_loader import CfgLoader


class BaseTrainer:
    def __init__(self, actor, loaders, optimizer, cfg: CfgLoader, lr_scheduler=None):
        """

        :param actor: the actor for training the network
        :param loaders:  list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one epoch for each loader.
        :param optimizer: the optimizer used for training e.g. Adam
        :param cfg: training settings
        :param lr_scheduler: Learning rate schedule
        """
        self.actor = actor
        self.optimizier = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.epoch = 0
        self.stats = {}

        self.device = cfg.train.device
        # 添加多卡检测，暂时没有实现
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        self.cfg = cfg

    def train(self, max_epochs, load_latest=False, fail_safe=True, load_previous_ckpt=False, distill=False):
        """
        Do training for the given number of epochs.
        :param max_epochs: Max number of training epochs,
        :param load_latest: Bool indicating whether to resume from latest epoch.
        :param fail_safe: Bool indicating whether the training to automatically restart in case of any crashes.
        :param load_previous_ckpt:
        :param distill:
        :return:
        """

        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            try:
                # if load_latest:
                #     self.load_checkpoint()
                # if load_previous_ckpt:
                #     directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_prv)
                #     self.load_state_dict(directory)
                for epoch in range(self.epoch + 1, max_epochs + 1):
                    self.epoch = epoch

                    self.train_epoch()

                    self.lr_scheduler.step()

                    # if self.lr_scheduler is not None:
                    #     if self.settings.scheduler_type != 'cosine':
                    #         self.lr_scheduler.step()
                    #     else:
                    #         self.lr_scheduler.step(epoch - 1)
                    # only save the last 10 checkpoints
                    # save_epoch_interval = getattr(self.settings, "save_epoch_interval", 1)
                    # save_last_n_epoch = getattr(self.settings, "save_last_n_epoch", 1)
                    # if epoch > (max_epochs - save_last_n_epoch) or epoch % save_epoch_interval == 0:
                    #     if self._checkpoint_dir:
                    #         if self.settings.local_rank in [-1, 0]:
                    #             self.save_checkpoint()
            except:
                print('Training crashed at epoch {}'.format(epoch))
                raise
                # if fail_safe:
                #     self.epoch -= 1
                #     load_latest = True
                #     print('Traceback for the error!')
                #     print(traceback.format_exc())
                #     print('Restarting training from last epoch ...')
                # else:
                #     raise

        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError
