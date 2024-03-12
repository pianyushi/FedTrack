from got10k.trackers import Tracker
import cv2
import numpy as np
import torch

def scale_to_fit(image, size):
    """
    缩放图像以适合指定大小。
    """
    h, w = image.shape[:2]
    if max(h, w) > size:
        if h > w:
            new_h = size
            new_w = round(w * new_h / h)
        else:
            new_w = size
            new_h = round(h * new_w / w)
        return cv2.resize(image, (new_w, new_h))
    return image

class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(
            name='IdentityTracker',  # tracker name
            is_deterministic=True  # stochastic (False) or deterministic (True)
        )

    def init(self, data):
        """
        初始化跟踪器状态。
        """
        image, bbox = data[0], data[1]
        search_scale = 1.0

        # 缩放图像以适应网络输入大小，并提取初始目标框内的特征
        input_size = self.conv_layers[0].kernel_size[0] + 2
        scaled_image = scale_to_fit(image, input_size)
        features = self.extract_features(scaled_image, bbox)

        # 对每个搜索尺度计算响应图
        response_maps = []
        for scale in [search_scale, search_scale / 2, search_scale * 2]:
            scaled_bbox = bbox.copy()
        scaled_bbox[:2] = np.round(scaled_bbox[:2] * scale).astype(np.int32)
        scaled_bbox[2:] = np.round(scaled_bbox[2:] * scale).astype(np.int32)
        scaled_features = self.extract_features(image, scaled_bbox)
        filter_pos = [input_size // 2, input_size // 2]
        response_map = self.calc_response_map(scaled_features.reshape(-1, input_size, input_size),
                                              filter_pos)  # 对特征进行卷积
        response_maps.append(response_map)

        # 将响应图相加并根据最大响应值确定目标框在该尺度下的位置
        max_response = sum(response_maps)
        max_position = np.unravel_index(max_response.argmax(), max_response.shape)
        target_bbox = bbox.copy()
        target_bbox[:2] = np.round((np.array(max_position) + 0.5) * search_scale).astype(np.int32) - target_bbox[2:] / 2

        return target_bbox

    def update(self, data):
        """
            更新跟踪器状态，并返回最新的目标框位置。
        """
        image, prev_bbox = data[0], data[1]
        search_scale = 2.0

        # 推断当前搜索区域
        input_size = self.conv_layers[0].kernel_size[0] + 2
        curr_bbox = prev_bbox.copy()

        curr_bbox[:2] = prev_bbox[:2] - prev_bbox[2:] * (search_scale - 1) / 2
        curr_bbox[2:] = prev_bbox[2:] * search_scale

        # 缩放图像以适应网络输入大小，并提取当前目标框内的特征
        scaled_image = scale_to_fit(image, input_size)
        features = self.extract_features(scaled_image, curr_bbox)

        # 计算响应图
        filter_pos = [input_size // 2, input_size // 2]
        response_map = self.calc_response_map(features.reshape(-1, input_size, input_size), filter_pos)

        # 根据最大响应值确定目标框在该尺度下的位置
        max_position = np.unravel_index(response_map.argmax(), response_map.shape)
        target_bbox = curr_bbox.copy()
        target_bbox[:2] += (np.array(max_position) + 0.5) / response_map.shape * target_bbox[2:]

        return target_bbox.astype(np.int32)

from got10k.experiments import ExperimentGOT10k

# ... tracker definition ...

# instantiate a tracker
tracker = IdentityTracker()

# setup experiment (validation subset)
experiment = ExperimentGOT10k(
    root_dir='../pysot/datasets/GOT-10k',    # GOT-10k's root directory
    subset='val',               # 'train' | 'val' | 'test'
    result_dir='results',       # where to store tracking results
    report_dir='reports'        # where to store evaluation reports
)
experiment.run(tracker, visualize=True)

# report tracking performance
experiment.report([tracker.name])
