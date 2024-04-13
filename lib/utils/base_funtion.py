"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/13 15:51
"""

from lib.config.cfg_loader import CfgLoader
import lib.data.transforms as tfm
from lib.dataset import names_to_datasets
from lib.data import processing, sampler, image_loader, loader


def build_dataloaders(cfg: CfgLoader):
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.data.mean, std=cfg.data.std))
    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.data.mean, std=cfg.data.std))

    output_size = {
        'template': cfg.data.template.size,
        'search': cfg.data.search.size
    }
    search_area_factor = {
        'template': cfg.data.template.factor,
        'search': cfg.data.search.factor
    }
    center_jitter_factor = {
        'template': cfg.data.template.center_jitter,
        'search': cfg.data.search.center_jitter
    }
    scale_jitter_factor = {
        'template': cfg.data.template.scale_jitter,
        'search': cfg.data.search.scale_jitter
    }

    data_processing_train = processing.ViPTProcessing(search_area_factor=search_area_factor,
                                                      output_sz=output_size,
                                                      center_jitter_factor=center_jitter_factor,
                                                      scale_jitter_factor=scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    data_processing_val = processing.ViPTProcessing(search_area_factor=search_area_factor,
                                                    output_sz=output_size,
                                                    center_jitter_factor=center_jitter_factor,
                                                    scale_jitter_factor=scale_jitter_factor,
                                                    mode='sequence',
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)

    sampler_mode = cfg.data.sampler_mode
    train_cls = cfg.train.train_cls

    dataset_train = sampler.TrackingSampler(datasets=names_to_datasets(cfg.data.train.datasets_name, cfg, image_loader.opencv_loader),
                                            p_datasets=cfg.data.train.datasets_ratio,
                                            samples_per_epoch=cfg.data.train.samples_per_epoch,
                                            max_gap=cfg.data.max_sample_interval,
                                            num_search_frames=cfg.data.search.number,
                                            num_template_frames=cfg.data.template.number,
                                            processing=data_processing_train,
                                            frame_sample_mode=sampler_mode,
                                            train_cls=train_cls)

    # train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    # shuffle = False if cfg.local_rank != -1 else True
    train_sampler = None
    shuffle = False

    loader_train = loader.LTRLoader(name='train',
                                    dataset=dataset_train,
                                    training=True,
                                    batch_size=cfg.train.batch_size,
                                    shuffle=shuffle,
                                    num_workers=8,
                                    drop_last=True,
                                    stack_dim=1,
                                    sampler=train_sampler)

    if cfg.data.val.datasets_name[0] is None:
        loader_val = None
    else:
        dataset_val = sampler.TrackingSampler(datasets=names_to_datasets(cfg.data.val.datasets_name, cfg, image_loader.opencv_loader),
                                              p_datasets=cfg.data.val.datasets_ratio,
                                              samples_per_epoch=cfg.data.val.samples_per_epoch,
                                              max_gap=cfg.data.max_sample_interval,
                                              num_search_frames=cfg.data.search.number,
                                              num_template_frames=cfg.data.template.number,
                                              processing=data_processing_val,
                                              frame_sample_mode=sampler_mode,
                                              train_cls=train_cls)
        # val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
        val_sampler = None
        loader_val = loader.LTRLoader('val', dataset_val,
                                      training=False,
                                      batch_size=cfg.train.batch_size,
                                      num_workers=cfg.train.num_worker,
                                      drop_last=True,
                                      stack_dim=1,
                                      sampler=val_sampler,
                                      epoch_interval=cfg.train.val_epoch_interval)
    return loader_train, loader_val
