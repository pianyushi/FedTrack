import cv2
import numpy as np
import torch

from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k


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


class MDNet(torch.nn.Module):
    def __init__(self, input_size=107, num_layers=3, feature_dim=512,
                 init_cnn=False, dropout=0.5):
        super(MDNet, self).__init__()

        # 定义卷积层、池化层、全连接层等组件
        self.conv_layers = []
        self.pool_layers = []
        in_channels = 3
        for i in range(num_layers):
            out_channels = feature_dim // (2 ** i)
            conv_layer = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
            pool_layer = torch.nn.MaxPool2d(2, 2)
            relu_layer = torch.nn.ReLU(inplace=True)
            self.conv_layers.append(conv_layer)
            self.pool_layers.append(pool_layer)
            self.add_module('conv{}'.format(i + 1), conv_layer)
            self.add_module('pool{}'.format(i + 1), pool_layer)
            self.add_module('relu{}'.format(i + 1), relu_layer)
            in_channels = out_channels

        self.fc_layers = []
        fc_dim = feature_dim * (input_size // (2 ** num_layers)) * (input_size // (2 ** num_layers))
        for i in range(2):
            fc_layer = torch.nn.Linear(fc_dim, feature_dim)
            dropout_layer = torch.nn.Dropout(dropout)
            relu_layer = torch.nn.ReLU(inplace=True)
            self.fc_layers.append(fc_layer)
            self.add_module('fc{}'.format(i + 1), fc_layer)
            self.add_module('dropout{}'.format(i + 1), dropout_layer)
            self.add_module('relu_fc{}'.format(i + 1), relu_layer)
            fc_dim = feature_dim

        # 初始化卷积层权重
        if init_cnn:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))

    def forward(self, x):
        """
        前向传播。
        """
        h = x
        for i in range(len(self.conv_layers)):
            h = self.conv_layers[i](h)
            h = self.pool_layers[i](h)
            h = getattr(self, 'relu{}'.format(i + 1))(h)

        h = h.view(h.size(0), -1)
        for i in range(len(self.fc_layers)):
            h = self.fc_layers[i](h)
            h = getattr(self, 'relu_fc{}'.format(i + 1))(h)
            h = getattr(self, 'dropout{}'.format(i + 1))(h)

        return h

    def extract_features(self, image, bbox, search_scale=2.0):
        """
        提取输入图像内的特征。
        """
        # 截取目标框内的图像区域
        x1, y1, w, h = bbox
        x2 = x1 + w - 1
        y2 = y1 + h - 1
        cx, cy = x1 + w / 2, y1 + h / 2
        size = round((w + h) / 2 * search_scale)
        x1 = round(cx - size / 2)
        y1 = round(cy - size / 2)
        x2 = round(cx + size / 2)
        y2 = round(cy + size / 2)
        cropped_image = image[y1:y2, x1:x2]

        # 缩放图像以适应网络输入大小
        input_size = self.conv_layers[0].kernel_size[0] + 2
        scaled_image = scale_to_fit(cropped_image, input_size)

        # 将缩放后的图像作为模型的输入
        tensor_image = torch.from_numpy(scaled_image).float().permute(2, 0, 1).unsqueeze(0)
        if torch.cuda.is_available():
            tensor_image = tensor_image.cuda()
        features = self(tensor_image)

        return features.data.cpu().numpy().flatten()

    def calc_response_map(self, features, filter_pos):
        """
        计算响应图。
        """
        c, h, w = features.shape
        template = features[:, filter_pos[1]:filter_pos[1] + 1, filter_pos[0]:filter_pos[0] + 1]
        assert template.shape == (c, 1, 1)
        response_map = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                patch = features[:, i:i+1, j:j+1]
                assert patch.shape == (c, 1, 1)
                score = (template * patch).sum()
                response_map[i][j] = score

        return response_map

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
        for scale in[search_scale, search_scale / 2, search_scale * 2]:
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

class MDNetTracker(Tracker):
    def init(self):
        super(MDNetTracker, self).init('MDNet')

    # 加载预训练模型权重
        model_path = 'path/to/mdnet.pth'
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model = MDNet()
        self.model.load_state_dict(state_dict['state_dict'])
        if torch.cuda.is_available():
            self.model.cuda()
    """
        def init(self, image, bbox):
        """
    # 初始化跟踪器状态。
    """
        return self.model.init((image, bbox))
    """


    def update(self, image):
        """
        更新跟踪器状态，并返回最新的目标框位置。
        """
        return self.model.update((image, self._state['bbox']))

    def set_state(self, state):
        """
        设置跟踪器状态。
        """
        self._state = state

    def set_meta_data(self, meta_data):
        """
        设置跟踪器元数据。
        """
        self._meta_data = meta_data

    def set_visualization(self, flag):
        """
        设置是否进行可视化。
        """
        self._visualize = flag

    @staticmethod
    def get_default_hyper_params():
        """
        获取默认超参数。
        """
        return {'num_layers': 3, 'feature_dim': 512, 'init_cnn': False, 'dropout': 0.5}

    @staticmethod
    def name():
        """
        获取跟踪器名称。
        """
        return 'MDNet'

net_file = 'mdnet_vot-otb.pth'
tracker = MDNetTracker(net_file)

# 运行测试
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
