"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/3/19 19:34
"""
import random
import torch.utils.data
from lib.common.tensor import TensorDict


def no_processing(data):
    """
    默认数据处理函数，就是不处理
    :param data: 原始数据
    :return:  原始数据
    """
    return data


class TrackingSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap, num_search_frames, num_template_frames=1, processing=no_processing,
                 frame_sample_mode='causal', train_cls=False, pos_prob=0.5):
        """

        :param datasets: 训练要用的数据集
        :param p_datasets: 每个数据集被使用的可能性
        :param samples_per_epoch: 每个epoch采样数
        :param max_gap: 采样是随机选择一帧，然后前gap帧作为train帧，后gap帧为test帧
        :param num_search_frames:
        :param num_template_frames:
        :param processing:数据预处理函数
        :param frame_sample_mode:
        :param train_cls: 是否训练分类
        :param pos_prob:分类时抽样正样本的概率
        """

        self.datasets = datasets
        self.train_cls = train_cls
        self.pos_prob = pos_prob

        # 如果没有指定数据集概率，就同一概率
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        p_total = sum(p_datasets)

        # 生成概率
        self.p_datasets = [p / p_total for p in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
        else:
            return self.getitem()

    def getitem(self):

        valid = False

        while not valid:
            # 选择一个数据集
            dataset = random.choices(self.datasets, weights=self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # 从数据集中获取一个有足够帧数的序列

            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frames_ids = None
                search_frames_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    # 以因果方式对帧进行采样测试和训练，即search_frame_ids>template_frame_ids
                    while search_frames_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase, max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue

                        template_frame_ids = base_frame_id + prev_frame_ids  # todo 这里得检查一下顺序有没有出问题
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                    max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                    num_ids=self.num_search_frames)

                        gap_increase += 5
                elif self.frame_sample_mode == 'trident' or self.frame_sample_mode == 'trident_pro':
                    template_frame_ids, search_frame_ids = self.get_frames_ids_trident(visible)
                elif self.frame_sample_mode == ' start':
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames

            try:
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})
                # make data augmentation

                data = self.processing(data)

                # check whether data is valid
                valid = data['valid']
            except:
                valid = False
        return data

    def get_frames_ids_trident(self, visible):
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]

                if self.frame_sample_mode == 'trident_pro':
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id, allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)

                if f_id is None:
                    template_frame_ids_extra = [None]
                else:
                    template_frame_ids_extra += f_id

            template_frame_ids = template_frame_id1 + template_frame_ids_extra
            return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict
