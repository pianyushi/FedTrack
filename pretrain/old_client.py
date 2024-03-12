import pickle
import prepro_vot
import train_mdnet
import os, sys
import pickle
import yaml
import time
import argparse
import numpy as np
import torch
sys.path.insert(0,'.')
from data_prov import RegionDataset
from modules.model import MDNet, set_optimizer, BCELoss, Precision


class Client(object):

    def __init__(self, conf, opts, id=-1):
        self.conf = conf
        self.client_id = id
        self.global_epochs = 0
        self.local_model = MDNet(opts['init_model_path'], self.get_K(opts))  # Init model# K个分支
        # 得到各个客户端相应的数据
        prepro_vot.preprocess(id, conf["k"])

    def get_K(self, opts):
        data_path = "data/vot-otb{}.pkl".format(self.client_id)
        with open(data_path, 'rb') as fp:
            data = pickle.load(fp)
        # 视频序列的个数,data是一个字典，key是视频序列的相对路径，value也是一个字典，key1是images，value1是image_name,
        # key2是gt，value是对应帧的gt: target bbox (min_x,min_y,w,h)
        K = len(data)
        return K

    def local_train(self, global_model, opts, id):  # 用视频序列训练网络

        # Init dataset # 初始化数据集
        data_path = "data/vot-otb{}.pkl".format(id)
        with open(data_path, 'rb') as fp:
            data = pickle.load(fp)
        # 视频序列的个数,data是一个字典，key是视频序列的相对路径，value也是一个字典，key1是images，value1是image_name,
        # key2是gt，value是对应帧的gt: target bbox (min_x,min_y,w,h)
        K = len(data)
        dataset = [None] * K
        for k, seq in enumerate(data.values()):
            # 随机提取8帧，抽取32个正样本和96个负样本，生成第k个序列训练的mini-batch
            dataset[k] = RegionDataset(seq['images'], seq['gt'], opts)

        for name, param in global_model.state_dict().items():
            if not name.startswith("branches"):
                self.local_model.state_dict()[name].copy_(param.clone())
        # model = self.local_model
        if opts['use_gpu']:
            self.local_model = self.local_model.cuda()
        self.local_model.set_learnable_params(opts['ft_layers'])  # 为卷基层和全连接层设置参数

        # Init criterion and optimizer # 初始化损失函数和优化器
        criterion = BCELoss()
        evaluator = Precision()  # 正样本中判断为目标的个数/正样本的个数
        optimizer = set_optimizer(self.local_model, opts['lr'], opts['lr_mult'])

        # Main trainig loop
        for i in range(5):  # 迭代次数
            # print('==== Start Cycle {:d}/{:d} ===='.format(i + 1, opts['n_cycles']))

            if i in opts.get('lr_decay', []):
                # print('decay learning rate')
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= opts.get('gamma', 0.1)

            # Training
            self.local_model.train()
            prec = np.zeros(K)
            k_list = np.random.permutation(K)  # 随机梯度下降，需要打乱数据顺序
            for j, k in enumerate(k_list):
                tic = time.time()
                # training # 每次迭代都随机抽取8帧，从中抽取32个正样本和96个负样本bbox，然后从原图中抽取出对应的区域
                pos_regions, neg_regions = dataset[k].next()
                if opts['use_gpu']:
                    pos_regions = pos_regions.cuda()
                    neg_regions = neg_regions.cuda()

                # 计算正样本分数和负样本分数
                # pos_score.shape (32,2) ,neg_score.shape (96,2)分别是 目标区域的分数和背景区域的得分
                pos_score = self.local_model(pos_regions, k)  # 执行前向传播计算得分
                neg_score = self.local_model(neg_regions, k)  # 前向传播计算得分

                loss = criterion(pos_score, neg_score)  # 计算总损失

                batch_accum = opts.get('batch_accum', 1)
                if j % batch_accum == 0:
                    self.local_model.zero_grad()  # 梯度归零
                loss.backward()  # 反向传播
                if j % batch_accum == batch_accum - 1 or j == len(k_list) - 1:
                    if 'grad_clip' in opts:
                        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), opts['grad_clip'])  # 梯度裁剪，防止梯度爆炸和梯度消失
                    optimizer.step()  # 更新梯度

                prec[k] = evaluator(pos_score, neg_score)  # 计算精度

                toc = time.time() - tic
                # print('Cycle {:2d}/{:2d}, Iter {:2d}/{:2d} (Domain {:2d}), Loss {:.3f}, Precision {:.3f}, Time {:.3f}'
                #        .format(i, opts['n_cycles'], j, len(k_list), k, loss.item(), prec[k], toc))

            print('Client{:2d}, local Epoch {:2d}/{:2d}, Mean Precision: {:.3f}'.format(self.client_id+1, i+1, 5, prec.mean()))  # 本次迭代后的平均精度
            # print('Save model to {:s}'.format(opts['model_path']))
            '''
            if opts['use_gpu']:
                model = model.cpu()
            '''
            states = {'shared_layers': self.local_model.layers.state_dict()}

            model_path = "../models/mdnet_vot-otb{}.pth".format(id)
            torch.save(states, model_path)
            if opts['use_gpu']:
                self.local_model = self.local_model.cuda()


        diff = dict()

        for name, data in self.local_model.state_dict().items():
            if not name.startswith("branches"):
                diff[name] = (data - global_model.state_dict()[name])

        return diff  # params