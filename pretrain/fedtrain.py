import argparse, json
import yaml
import datetime
import os
import logging
import random

from old_server import *
from old_client import *
import datasets
import tracking
import config

if __name__ == "__main__":
    with open("conf.json", 'r') as f:
        conf = json.load(f)
    if SETUP_SEQ:
        print('Setup sequences ...')
        """
        Doing in setup_seqs:
        1. check integrity of data, if not, download all
        2. write cfg.json and attrs.txt in each seq directory
        """
        loadSeqs = 'tb100'
        butil.setup_seqs(loadSeqs)
    opts = yaml.safe_load(open('options_vot.yaml', 'r'))
    # 初始化服务器端
    init_model = MDNet(opts['init_model_path'], 1)
    server = Server(conf, opts, init_model)
    clients = []
    params = []
    for c in range(conf["k"]):
        clients.append(Client(conf, opts, c))
    for e in range(10):
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            if not name.startswith("branches"):
                weight_accumulator[name] = torch.zeros_like(params)

        for client in clients:
            diff = client.local_train(server.global_model, opts, client.client_id)
            for name, params in server.global_model.state_dict().items():
                if not name.startswith("branches"):
                    weight_accumulator[name].add_(diff[name].to("cuda"))
        server.model_aggregate(weight_accumulator)

        fps, meanIOU = server.model_eval("DragonBaby")
        # server.model_eval()
        print("Global Epoch %d, meanIOU: %f avg_time: %f\n" % (e, meanIOU, fps))
    loadSeqs = 'tb100'
    evalType = 'OPE'
    seqNames = butil.get_seq_names(loadSeqs)

    for i in seqNames:
        temp_path = "../results/result/{}.txt".format(i)
        if os.path.exists(temp_path):
            print(i, "already exists!")
        else:
            print(i, " is Ready!")
            server.model_eval(i)
            print(i, " is OK!")
