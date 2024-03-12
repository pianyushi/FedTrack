import os
import numpy as np
import pickle
from collections import OrderedDict

import pdb
def preprocess(id, num):
    seq_home = '../datasets/VOT'
    seqlist_path = '../datasets/list/vot-otb.txt'
    output_path = 'data/vot-otb{}.pkl'.format(id)

    with open(seqlist_path,'r') as fp:
        seq_list = fp.read().splitlines()
    seq_list = seq_list  # [0:40]
    split_num = len(seq_list) // num
    seq_list_temp = seq_list[(split_num * id):(split_num * (id + 1) - 1)]
    # Construct db
    data = OrderedDict()
    for i, seq in enumerate(seq_list_temp):
        # pdb.set_trace()
        img_list = sorted([p for p in os.listdir(os.path.join(seq_home, seq)) if os.path.splitext(p)[1] == '.jpg'])
        # gt = np.loadtxt(os.path.join(seq_home, seq, 'groundtruth.txt'), delimiter=',')
        gt = np.loadtxt(os.path.join(seq_home, seq.split('/')[0], 'annotations', seq.split('/')[1], 'groundtruth.txt'), delimiter=',')
        # print(gt)
        if seq == 'vot2014/ball':
            img_list = img_list[1:]

        assert len(img_list) == len(gt), "Lengths do not match!!"

        if gt.shape[1] == 8:
            x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
            y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
            x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
            y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
            gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

        img_list = [os.path.join(seq_home, seq, img) for img in img_list]
        # print(img_list)
        data[seq] = {'images': img_list, 'gt': gt}
        # print(data)

    # Save db
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'wb') as fp:
        pickle.dump(data, fp)
