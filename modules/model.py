import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def append_params(params, module, prefix):  # 向参数字典中添加参数
    # params 为参数字典
    # module 为具体的模型
    # prefix 为模型中具体某一层的名称 如：conv1，fc4，fc6_0
    for child in module.children():  # 遍历模型的网络结构
        # child为每一层网络的每一个部分
        for k,p in child._parameters.items():
            # k为每一层的参数名称，如weight，bias
            # p为参数的具体数值
            if p is None:
                continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError('Duplicated param name: {:s}'.format(name))


def set_optimizer(model, lr_base, lr_mult, train_all=False, momentum=0.9, w_decay=0.0005):  # 设置优化器
    if train_all:
        params = model.get_all_params()
    else:
        params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)  # 随机梯度下降法训练
    return optimizer  # 返回优化器


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        # 共享层
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True))),
                ('fc4',   nn.Sequential(nn.Linear(512 * 3 * 3, 512),
                                        nn.ReLU(inplace=True))),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU(inplace=True)))]))
        # 分支层
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if model_path is not None:  # 加载训练好的模型参数
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path))
        self.build_param_dict()

    def build_param_dict(self):  # 建立参数字典
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            # name 为每一层的名称 如：conv1，fc4
            # module 为该层的具体网络结构，如：
            # Sequential(
            #   (0): Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2))
            #   (1): ReLU()
            #   (2): LRN()
            #   (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
            # )
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_{:d}'.format(k))

    def set_learnable_params(self, layers):  # 设置以layers开头名称的层的参数均要求梯度,默认参数均不求梯度
        for k, p in self.params.items():
            # k 为每一层的参数名称 如:conv1_weight conv1_bias fc4_weight
            # p 为相对应参数的具体数值
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):  # 得到网络中需要求解梯度的参数
        params = OrderedDict()
        for k, p in self.params.items():
            # k为参数名称 如:conv1_weight conv1_bias fc4_weight fc6_0_bias
            # p为该参数具体的数值
            if p.requires_grad:
                params[k] = p
        return params  # 返回需要求解梯度参数字典
    
    def get_all_params(self):  # 得到网络中的所有参数
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        # forward model from in_layer to out_layer
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':
                    x = x.contiguous().view(x.size(0), -1)
                if name == out_layer:
                    return x

        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x, dim=1)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])


class BCELoss(nn.Module):
    def forward(self, pos_score, neg_score, average=True):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]

        loss = pos_loss.sum() + neg_loss.sum()
        if average:
            loss /= (pos_loss.size(0) + neg_loss.size(0))
        return loss


class Accuracy():  # 计算检测正样本和负样本各自的准确率
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()
        acc = (pos_correct + neg_correct) / (pos_score.size(0) + neg_score.size(0) + 1e-8)
        return acc.item()


class Precision():  # 计算精度
    def __call__(self, pos_score, neg_score):
        #  torch.cat([shape=(3,4),shape=(3,4)],0).shape = (6,4)
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]  # 返回前pos_score.size(0) = 32 大的元素的索引
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8) # 正样本中判断为目标的个数/正样本的个数
        return prec.item()
