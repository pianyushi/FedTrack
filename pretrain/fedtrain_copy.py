import argparse, json
import yaml
import datetime
import os
import logging
import random

from server import *
from client import *
import datasets
import tracking
import config
import select_nodes
import time

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
    loadSeqs = 'tb100'
    evalType = 'OPE'
    seqNames = butil.get_seq_names(loadSeqs)

    # random.seed(0)
    samples = ["DragonBaby"]# ["Dog", "DragonBaby", "Man"]  # random.sample(seqNames, 3)
    # 初始化服务器端
    t5 = time.time()
    init_model = MDNet(opts['init_model_path'], 1)
    # server = Server(conf, opts, init_model)
    clients = []
    params = []
    final_server = Server(conf, opts, init_model)
    t6 = time.time()
    for c in range(conf["k"]):
        clients.append(Client(conf, opts, c))
    for client in clients:
        client.get_data()
    Locals = select_nodes.init_Local(clients)
    # print("Locals:", Locals)
    # print("Clients:", clients)

    # 任期循环从这里写
    sum_time = t6 - t5
    for term in range(5):

        candidates, Locals = select_nodes.candidate_select(clients, Locals)
        print("The number of candidates is ", len(candidates))
        # print(candidates)
        server_temp = select_nodes.server_select(candidates)
        # candidates = candidates.remove(server_temp)
        # temp = candidates.index(server_temp)
        # candidates.pop(temp)
        # print(candidates)
        # server = Server(server_temp, opts, server_temp.client_id)
        server = Server(server_temp.conf, opts, server_temp.local_model)
        print("The server is Client{:2d}".format(server_temp.client_id))
        for epoch in range(2):
            real_epoch = term * 2 + epoch
            weight_accumulator = {}
            for name, params in server.global_model.state_dict().items():
                if not name.startswith("branches"):
                    weight_accumulator[name] = torch.zeros_like(params)
            # print(weight_accumulator)
            # for candidate in candidates:


            # print(weights)
            diffs = []

            for num, candidate in enumerate(candidates):
                # print(candidate)  * weights[num] * len(candidates)
                if candidate.client_id!=1 and candidate.client_id!=2 and candidate.client_id!=3:
                    diff = candidate.local_train(server.global_model, opts, candidate.client_id)
                candidate.model_eval(samples)
                t1 = time.time()
                diffs.append(diff)
                t2 = time.time()
                temp_time = (t2 - t1)
                sum_time += temp_time
            t3 = time.time()
            Locals = select_nodes.calculate_Local(clients, Locals, real_epoch, server.global_acc)
            # weights = select_nodes.cal_weight(candidates, real_epoch, Locals)
            weights = select_nodes.cal_weight(candidates)
            for num, diff in enumerate(diffs):
                for name, params in server.global_model.state_dict().items():
                    if not name.startswith("branches"):
                        weight_accumulator[name].add_((diff[name].to("cuda")*0.4) * weights[num] * len(candidates))
            # print(weight_accumulator)
            # print(server.global_model.state_dict().items())
            server.model_aggregate(weight_accumulator)

            # print(Locals)
            for candidate in candidates:
                if candidate.client_id!=1 and candidate.client_id!=2 and candidate.client_id!=3:
                    candidate.local_model = server.global_model
            t4 = time.time()
            sum_time += (t4 - t3)
            print(sum_time)
            fps = 0
            meanIOU = 0
            for sample in samples:
                fps_temp, meanIOU_temp = server.model_eval(sample)
                fps += fps_temp
                meanIOU += meanIOU_temp
            final_server = server
            # server.model_eval()
            fps = fps / 3
            meanIOU = meanIOU / 3
            server.global_acc = meanIOU
            print("Global Epoch %d, meanIOU: %f avg_time: %f\n" % (real_epoch, meanIOU, fps))


    for i in seqNames:
        temp_path = "../results/result/{}.txt".format(i)
        print(i, " is Ready!")
        final_server.model_eval(i)
        print(i, " is OK!")
