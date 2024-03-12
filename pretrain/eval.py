import numpy as np
import os
import sys
import time
import argparse
import yaml, json
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim

sys.path.insert(0, '.')
from modules.model import MDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from modules.utils import overlap_ratio
from tracking.data_prov import RegionExtractor
from tracking.bbreg import BBRegressor
from tracking.gen_config import gen_config

opts = yaml.safe_load(open('options.yaml', 'r'))


def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()  # 测试模型
    extractor = RegionExtractor(image, samples, opts)
    for i, regions in enumerate(extractor):
        if opts['use_gpu']:
            regions = regions.cuda()
        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)
        if i == 0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), 0)
    return feats


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()  # 训练模型

    batch_pos = opts['batch_pos']  # mini-batch上正样本的个数32
    batch_neg = opts['batch_neg']  # mini-batch上负样本的个数96
    batch_test = opts['batch_test']  # 每一帧的候选区域个数256
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)  # 困难负样本挖掘时生成的1024个负样本，我们要从中取出96个

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
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()


def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=False):
    # Init bbox # 初始化包围盒
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    if gt is not None:
        overlap = np.zeros(len(img_list))
        overlap[0] = 1

    # Init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()

    # Init criterion and optimizer
    criterion = BCELoss()
    model.set_learnable_params(opts['ft_layers'])  # 设置全连接层的参数都需要求解梯度
    init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])  # 初始的优化器
    update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])  # 更新的优化器

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')

    # Draw pos/neg samples
    pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
        target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
        SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
            target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
        SampleGenerator('whole', image.size)(
            target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples)
    neg_feats = forward_samples(model, image, neg_examples)

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
    del init_optimizer, neg_feats
    torch.cuda.empty_cache()

    # Train bbox regressor # 生成1000个符合overlab要求和scale要求的bbox
    bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'],
                                     opts['aspect_bbreg'])(
        target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
    del bbreg_feats
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
    pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
    neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

    # Init pos/neg features for update
    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_feats = forward_samples(model, image, neg_examples)
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]

    spf_total = time.time() - tic

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

    # Main loop
    for i in range(1, len(img_list)):  # 跟踪每一帧

        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')

        # Estimate target bbox # 估计目标包围盒
        samples = sample_generator(target_bbox, opts['n_samples'])  # 每一帧随机生成256个候选区域
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')  # 计算这256个候选窗口的得分

        top_scores, top_idx = sample_scores[:, 1].topk(5)  # 挑选出得分最高的5个候选窗口和索引
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()  # 这一帧的候选框的得分为前5个得分的平均值
        target_bbox = samples[top_idx]  # 更新target_bbox用于下一帧的跟踪
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0

        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            pos_feats = forward_samples(model, image, pos_examples)
            pos_feats_all.append(pos_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]

            neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
            neg_feats = forward_samples(model, image, neg_examples)
            neg_feats_all.append(neg_feats)
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update # 跟踪失败时进行短期更新
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.cat(pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update # 每10帧进行一次长期更新
        elif i % opts['long_interval'] == 0:
            pos_data = torch.cat(pos_feats_all, 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf

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

        if gt is not None:
            overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]

    states = {'shared_layers': model.layers.state_dict()}
    torch.save(states, opts['model_path'])

    return overlap.mean()


