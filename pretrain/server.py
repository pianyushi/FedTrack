import torch
from modules.model import MDNet, BCELoss, set_optimizer
from tracking import gen_config
import eval
import json
import argparse
import numpy as np
import os
import sys
import time
import argparse
import yaml
from PIL import Image

import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.optim as optim

sys.path.insert(0, '.')
from modules.sample_generator import SampleGenerator
from modules.utils import overlap_ratio
from tracking.data_prov import RegionExtractor
from tracking.bbreg import BBRegressor
from scripts import *
from config import *
from client import Client
class Server(object): # Client

    def __init__(self, conf, opts, model):
        # Client.__init__(self, conf, opts, id=id)
        self.opts1 = yaml.safe_load(open('options.yaml','r'))
        self.global_model = model  # self.local_model # MDNet(opts['model_path'])
        self.global_acc = 0
        self.conf = conf
        self.opts = opts
        self.temp(opts)

    def model_aggregate(self, weight_accumulator):
        # print(weight_accumulator)
        for name, data in self.global_model.state_dict().items():
            if not name.startswith("branches"):
                update_per_layer = weight_accumulator[name] * self.conf["lambda"]

                if data.type() != update_per_layer.type():
                    data.add_(update_per_layer.to(torch.int64))
                else:
                    data.add_(update_per_layer)
        self.global_model.set_learnable_params(['fc'])  # 为卷基层和全连接层设置参数
        states = {'shared_layers': self.global_model.layers.state_dict()}
        torch.save(states, self.opts['model_path'])

    def temp(self,opts):
        if opts['use_gpu']:
            self.global_model = self.global_model.cuda()
        self.global_model.set_learnable_params(['fc'])  # 为卷基层和全连接层设置参数

    def forward_samples(self, model, image, samples, out_layer='conv3'):
        model.eval()  # 测试模型
        extractor = RegionExtractor(image, samples, self.opts1)
        for i, regions in enumerate(extractor):
            if self.opts1['use_gpu']:
                regions = regions.cuda()
            with torch.no_grad():
                feat = model(regions, out_layer=out_layer)
            if i == 0:
                feats = feat.detach().clone()
            else:
                feats = torch.cat((feats, feat.detach().clone()), 0)
        return feats

    def train(self, model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
        model.train()  # 训练模型

        batch_pos = self.opts1['batch_pos']  # mini-batch上正样本的个数32
        batch_neg = self.opts1['batch_neg']  # mini-batch上负样本的个数96
        batch_test = self.opts1['batch_test']  # 每一帧的候选区域个数256
        batch_neg_cand = max(self.opts1['batch_neg_cand'], batch_neg)  # 困难负样本挖掘时生成的1024个负样本，我们要从中取出96个

        pos_idx = np.random.permutation(pos_feats.size(0))  # 打乱正样本索引
        neg_idx = np.random.permutation(neg_feats.size(0))  # 打乱负样本索引
        while (len(pos_idx) < batch_pos * maxiter):  # 每次迭代从正样本中挑选出32个
            pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
        while (len(neg_idx) < batch_neg_cand * maxiter):  # 每次迭代从负样本中挑选出96个
            neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
        pos_pointer = 0  # 正样本索引
        neg_pointer = 0  # 负样本索引

        for i in range(maxiter):  # 迭代maxiter次

            # select pos idx # 挑选正样本idx
            pos_next = pos_pointer + batch_pos  # 32个
            pos_cur_idx = pos_idx[pos_pointer:pos_next]
            pos_cur_idx = pos_feats.new(pos_cur_idx).long()
            pos_pointer = pos_next

            # select neg idx # 挑选负样本idx
            neg_next = neg_pointer + batch_neg_cand  # 1024个
            neg_cur_idx = neg_idx[neg_pointer:neg_next]
            neg_cur_idx = neg_feats.new(neg_cur_idx).long()
            neg_pointer = neg_next

            # create batch # 创建mini-batch
            batch_pos_feats = pos_feats[pos_cur_idx]
            batch_neg_feats = neg_feats[neg_cur_idx]

            # hard negative mining # 困难负样本挖掘
            if batch_neg_cand > batch_neg:
                model.eval()  # 测试模型
                for start in range(0, batch_neg_cand, batch_test):
                    end = min(start + batch_test, batch_neg_cand)
                    with torch.no_grad():
                        score = model(batch_neg_feats[start:end], in_layer=in_layer)  # 前向传播计算得分
                    if start == 0:  # 记录得分，只需要目标的得分
                        neg_cand_score = score.detach()[:, 1].clone()
                    else:
                        neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

                _, top_idx = neg_cand_score.topk(batch_neg)  # 挑选出前96大的，即为困难负样本
                batch_neg_feats = batch_neg_feats[top_idx]
                model.train()

            # forward 前向传播，计算正样本的得分和负样本的得分
            pos_score = model(batch_pos_feats, in_layer=in_layer)
            neg_score = model(batch_neg_feats, in_layer=in_layer)

            # optimize
            loss = criterion(pos_score, neg_score)
            model.zero_grad()
            loss.backward()
            if 'grad_clip' in self.opts1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.opts1['grad_clip'])
            optimizer.step()

    def run_mdnet(self, img_list, init_bbox, gt=None, savefig_dir='', display=False):

        # Init bbox # 初始化包围盒
        target_bbox = np.array(init_bbox)
        result = np.zeros((len(img_list), 4))
        result_bb = np.zeros((len(img_list), 4))
        result[0] = target_bbox
        result_bb[0] = target_bbox

        if gt is not None:
            overlap = np.zeros(len(img_list))
            overlap[0] = 1

        if self.opts1['use_gpu']:
            self.global_model = self.global_model.cuda()

        # Init criterion and optimizer
        criterion = BCELoss()
        self.global_model.set_learnable_params(self.opts1['ft_layers'])  # 设置全连接层的参数都需要求解梯度
        init_optimizer = set_optimizer(self.global_model, self.opts1['lr_init'], self.opts1['lr_mult'])  # 初始的优化器
        update_optimizer = set_optimizer(self.global_model, self.opts1['lr_update'], self.opts1['lr_mult'])  # 更新的优化器

        tic = time.time()
        # Load first image
        image = Image.open(img_list[0]).convert('RGB')

        # Draw pos/neg samples
        pos_examples = SampleGenerator('gaussian', image.size, self.opts1['trans_pos'], self.opts1['scale_pos'])(
            target_bbox, self.opts1['n_pos_init'], self.opts1['overlap_pos_init'])

        neg_examples = np.concatenate([
            SampleGenerator('uniform', image.size, self.opts1['trans_neg_init'], self.opts1['scale_neg_init'])(
                target_bbox, int(self.opts1['n_neg_init'] * 0.5), self.opts1['overlap_neg_init']),
            SampleGenerator('whole', image.size)(
                target_bbox, int(self.opts1['n_neg_init'] * 0.5), self.opts1['overlap_neg_init'])])
        neg_examples = np.random.permutation(neg_examples)

        # Extract pos/neg features
        pos_feats = self.forward_samples(self.global_model, image, pos_examples)
        neg_feats = self.forward_samples(self.global_model, image, neg_examples)

        # Initial training
        self.train(self.global_model, criterion, init_optimizer, pos_feats, neg_feats, self.opts1['maxiter_init'])
        del init_optimizer, neg_feats
        torch.cuda.empty_cache()

        # Train bbox regressor # 生成1000个符合overlab要求和scale要求的bbox
        bbreg_examples = SampleGenerator('uniform', image.size, self.opts1['trans_bbreg'], self.opts1['scale_bbreg'],
                                         self.opts1['aspect_bbreg'])(
            target_bbox, self.opts1['n_bbreg'], self.opts1['overlap_bbreg'])
        bbreg_feats = self.forward_samples(self.global_model, image, bbreg_examples)
        bbreg = BBRegressor(image.size)
        bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
        del bbreg_feats
        torch.cuda.empty_cache()

        # Init sample generators for update
        sample_generator = SampleGenerator('gaussian', image.size, self.opts1['trans'], self.opts1['scale'])
        pos_generator = SampleGenerator('gaussian', image.size, self.opts1['trans_pos'], self.opts1['scale_pos'])
        neg_generator = SampleGenerator('uniform', image.size, self.opts1['trans_neg'], self.opts1['scale_neg'])

        # Init pos/neg features for update
        neg_examples = neg_generator(target_bbox, self.opts1['n_neg_update'], self.opts1['overlap_neg_init'])
        neg_feats = self.forward_samples(self.global_model, image, neg_examples)
        pos_feats_all = [pos_feats]
        neg_feats_all = [neg_feats]

        spf_total = time.time() - tic
        '''
        # Display
        savefig = savefig_dir != ''
        if display or savefig:
            dpi = 80.0
            figsize = (image.size[0] / dpi, image.size[1] / dpi)

            fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            im = ax.imshow(image, aspect='auto')

            if gt is not None:
                gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                        linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
                ax.add_patch(gt_rect)

            rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                                 linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
            ax.add_patch(rect)

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)
        '''
        # Main loop
        for i in range(1, len(img_list)):  # 跟踪每一帧

            tic = time.time()
            # Load image
            image = Image.open(img_list[i]).convert('RGB')

            # Estimate target bbox # 估计目标包围盒
            samples = sample_generator(target_bbox, self.opts1['n_samples'])  # 每一帧随机生成256个候选区域
            sample_scores = self.forward_samples(self.global_model, image, samples, out_layer='fc6')  # 计算这256个候选窗口的得分

            top_scores, top_idx = sample_scores[:, 1].topk(5)  # 挑选出得分最高的5个候选窗口和索引
            top_idx = top_idx.cpu()
            target_score = top_scores.mean()  # 这一帧的候选框的得分为前5个得分的平均值
            target_bbox = samples[top_idx]  # 更新target_bbox用于下一帧的跟踪
            if top_idx.shape[0] > 1:
                target_bbox = target_bbox.mean(axis=0)
            success = target_score > 0

            # Expand search area at failure
            if success:
                sample_generator.set_trans(self.opts1['trans'])
            else:
                sample_generator.expand_trans(self.opts1['trans_limit'])

            # Bbox regression
            if success:
                bbreg_samples = samples[top_idx]
                if top_idx.shape[0] == 1:
                    bbreg_samples = bbreg_samples[None, :]
                bbreg_feats = self.forward_samples(self.global_model, image, bbreg_samples)
                bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
                bbreg_bbox = bbreg_samples.mean(axis=0)
            else:
                bbreg_bbox = target_bbox

            # Save result
            result[i] = target_bbox
            result_bb[i] = bbreg_bbox

            # Data collect
            if success:
                pos_examples = pos_generator(target_bbox, self.opts1['n_pos_update'], self.opts1['overlap_pos_update'])
                pos_feats = self.forward_samples(self.global_model, image, pos_examples)
                pos_feats_all.append(pos_feats)
                if len(pos_feats_all) > self.opts1['n_frames_long']:
                    del pos_feats_all[0]

                neg_examples = neg_generator(target_bbox, self.opts1['n_neg_update'], self.opts1['overlap_neg_update'])
                neg_feats = self.forward_samples(self.global_model, image, neg_examples)
                neg_feats_all.append(neg_feats)
                if len(neg_feats_all) > self.opts1['n_frames_short']:
                    del neg_feats_all[0]

            # Short term update # 跟踪失败时进行短期更新
            if not success:
                nframes = min(self.opts1['n_frames_short'], len(pos_feats_all))
                pos_data = torch.cat(pos_feats_all[-nframes:], 0)
                neg_data = torch.cat(neg_feats_all, 0)
                self.train(self.global_model, criterion, update_optimizer, pos_data, neg_data, self.opts1['maxiter_update'])

            # Long term update # 每10帧进行一次长期更新
            elif i % self.opts1['long_interval'] == 0:
                pos_data = torch.cat(pos_feats_all, 0)
                neg_data = torch.cat(neg_feats_all, 0)
                self.train(self.global_model, criterion, update_optimizer, pos_data, neg_data, self.opts1['maxiter_update'])

            torch.cuda.empty_cache()
            spf = time.time() - tic
            spf_total += spf
            '''
            # Display
            if display or savefig:
                im.set_data(image)

                if gt is not None:
                    gt_rect.set_xy(gt[i, :2])
                    gt_rect.set_width(gt[i, 2])
                    gt_rect.set_height(gt[i, 3])

                rect.set_xy(result_bb[i, :2])
                rect.set_width(result_bb[i, 2])
                rect.set_height(result_bb[i, 3])

                if display:
                    plt.pause(.01)
                    plt.draw()
                if savefig:
                    fig.savefig(os.path.join(savefig_dir, '{:04d}.jpg'.format(i)), dpi=dpi)
                    '''
            if i>=len(gt):
                break

            if gt is None:
                overlap = overlap
            else:
                overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]

        if gt is not None:
            # print('meanIOU: {:.3f}'.format(overlap.mean()))
            meanIOU = overlap.mean()
        fps = len(img_list) / spf_total
        # print('avg_time: {:.5f}'.format(fps))
        return result, result_bb, fps, meanIOU

    def model_eval(self, i):
        temp_path = "../results/result/{}.txt".format(i)

        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--seq', default=i, help='input seq')
        parser.add_argument('-j', '--json', default='', help='input json')
        parser.add_argument('-f', '--savefig', action='store_true')
        parser.add_argument('-d', '--display', action='store_true')

        args = parser.parse_args()
        assert args.seq != '' or args.json != ''

        np.random.seed(0)
        torch.manual_seed(0)

                # Generate sequence config
        img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config.gen_config(args)

                # Run tracker
        result, result_bb, fps, meanIOU = self.run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)

                # Save result
        res = {}
        res['res'] = result_bb.round().tolist()
        res['type'] = 'rect'
        res['fps'] = fps
            # print(res['res'][0])
        # self.global_acc = meanIOU

        if os.path.exists(temp_path):
            os.remove(temp_path)
        for j in res['res']:
            f = open(temp_path, "a")
            temp = str(j)
            temp1 = temp.replace('[', '')
            content = temp1.replace(']', '')
                # print(content)
            f.writelines(content + "\n")
            f.close()
        json.dump(res, open(result_path, 'w'), indent=2)
        return fps, meanIOU