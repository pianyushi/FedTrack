def candidate_select(clients, Locals):
    candidates = []
    for client in clients:
        if len(Locals[client.client_id])==0:
            candidates.append(client)
        else:
            if Locals[client.client_id]['dec_chance']<2:
                candidates.append(client)
    return candidates, Locals

def supervise(client, Locals, epoch):
    if Locals[client.client_id][str(epoch)]['score']<0:
        Locals[client.client_id]['dec_chance']+=1
    return Locals

def init_Local(clients):
    Locals = {}
    for client in clients:
        Locals[client.client_id] = {}
    return Locals

def calculate_Local(clients, Locals, epoch, global_acc):
    count = 0
    accs = 0
    for client in clients:
        count += 1
        accs += client.loc_acc
            # Locals[num][str(epoch)]['score']
    acc_score = accs / count
    for client in clients:
        temp = 0.5 * (client.loc_acc - acc_score) + 0.5 * (client.loc_acc - global_acc)
        Locals[client.client_id][str(epoch)] = {}
        temp_epoch = Locals[client.client_id]
        if 'dec_chance' not in temp_epoch.keys():
            Locals[client.client_id]['dec_chance'] = 0
        #print("!!!!!!!",temp_epoch.keys())
        if str(epoch-1) in temp_epoch.keys():
            Locals[client.client_id][str(epoch)]['score'] = 0.1 * Locals[client.client_id][str(epoch-1)]['score'] + 0.9 * temp
        else:
            Locals[client.client_id][str(epoch)]['score'] = temp
        Locals = supervise(client, Locals, epoch)
    return Locals

def cal_weight_(candidates, epoch, Locals):
    sum_scores = 0
    weight = []
    for candidate in candidates:
        sum_scores += Locals[candidate.client_id][str(epoch)]['score']
    for candidate in candidates:
        temp = Locals[candidate.client_id][str(epoch)]['score'] / sum_scores
        weight.append(temp)
    return weight

def cal_weight(candidates):
    sum_scores = 0
    weight = []
    for candidate in candidates:
        sum_scores += candidate.loc_acc
    for candidate in candidates:
        temp = candidate.loc_acc / sum_scores
        weight.append(temp)
    return weight

def server_select(candidates):
    temp = []
    for i in range(len(candidates)):
        score = calculate()
        temp.append(score)
    index = max(temp)
    server = candidates[temp.index(index)]
    return server

def calculate():
    score = get_route_num() * (1 - get_cpu_usage()) + (1 - get_memory())
    return score
# 选择聚合节点的辅助函数
"""
import os, psutil

def get_info(metric):
    metric_cmd_map = {
        "cpu_usage_rate": "wmic cpu get loadpercentage",
        "mem_total": "wmic ComputerSystem get TotalPhysicalMemory",
        "mem_free": "wmic OS get FreePhysicalMemory"
    }
    out = os.popen("{}".format(metric_cmd_map.get(metric)))
    value = out.read().split("\n")[2]
    out.close()
    return float(value)

def get_cpu():
    cpu_usage_rate = get_info('cpu_usage_rate')
    return cpu_usage_rate

def get_memory():
    mem_total = get_info('mem_total') / 1024
    mem_free = get_info('mem_free')
    mem_usage_rate = (1 - mem_free / mem_total) * 100
    return mem_usage_rate

def get_cpu_num():
    result = psutil.cpu_count(False)
    return result

def get_route_num():
    result = psutil.cpu_count()
    return result

# cpu使用率

print("windows的CPU使用率是{}%".format(get_cpu()))

# 内存使用率
print("windows的内存使用率是{}%".format(get_memory()))

# CPU内核个数
print("windows的CPU内核个数是{}".format(get_cpu_num()))

# CPU线程个数
print("windows的CPU线程个数是{}".format(get_route_num()))
"""

# Linux 系统
import os, time, psutil
import logging
 
logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='monitor.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )
last_worktime = 0
last_idletime = 0
 
 
def get_cpu(): # CPU使用率
    global last_worktime, last_idletime
    f = open("/proc/stat", "r")
    line = ""
    while not "cpu " in line: line = f.readline()
    f.close()
    spl = line.split(" ")
    worktime = int(spl[2]) + int(spl[3]) + int(spl[4])
    idletime = int(spl[5])
    dworktime = (worktime - last_worktime)
    didletime = (idletime - last_idletime)
    if (didletime + dworktime) == 0:
        rate = 0
    else:
        rate = float(dworktime) / (didletime + dworktime)
    last_worktime = worktime
    last_idletime = idletime
    if (last_worktime == 0): return 0
    return rate
 
 
def get_mem_usage_percent(): # 内存利用率
    try:
        f = open('/proc/meminfo', 'r')
        for line in f:
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1])
            elif line.startswith('MemFree:'):
                mem_free = int(line.split()[1])
            elif line.startswith('Buffers:'):
                mem_buffer = int(line.split()[1])
            elif line.startswith('Cached:'):
                mem_cache = int(line.split()[1])
            elif line.startswith('SwapTotal:'):
                vmem_total = int(line.split()[1])
            elif line.startswith('SwapFree:'):
                vmem_free = int(line.split()[1])
            else:
                continue
        f.close()
    except:
        return None
    physical_percent = usage_percent(mem_total - (mem_free + mem_buffer + mem_cache), mem_total)
    virtual_percent = 0
    if vmem_total > 0:
        virtual_percent = usage_percent((vmem_total - vmem_free), vmem_total)
    return physical_percent, virtual_percent
 
 
def usage_percent(use, total):
    try:
        ret = (float(use) / total) * 100
    except ZeroDivisionError:
        raise Exception("ERROR - zero division error")
    return ret
 
 
def go(logfile=""):
    statvfs = os.statvfs('/')
 
    total_disk_space = statvfs.f_frsize * statvfs.f_blocks
    free_disk_space = statvfs.f_frsize * statvfs.f_bfree
    disk_usage = (total_disk_space - free_disk_space) * 100.0 / total_disk_space
    disk_usage = int(disk_usage)
    disk_tip = "硬盘空间使用率（最大100%）：" + str(disk_usage) + "%"
    print(disk_tip)
    # logging.info(disk_tip)
 
 
    mem_usage = get_mem_usage_percent()
    mem_usage = int(mem_usage[0])
    mem_tip = "物理内存使用率（最大100%）：" + str(mem_usage) + "%"
    print(mem_tip)
    # logging.info(mem_tip)
 
    cpu_usage = int(get_cpu() * 100)
    cpu_tip = "CPU使用率（最大100%）：" + str(cpu_usage) + "%"
    print(cpu_tip)
    # logging.info(cpu_tip)
 
    load_average = os.getloadavg()
    load_tip = "系统负载（三个数值中有一个超过3就是高）：" + str(load_average)
    print(load_tip)
    # logging.info(load_tip)

def get_cpu_num():
    result = psutil.cpu_count(False)
    return result

def get_route_num():
    result = psutil.cpu_count()
    return result

def get_memory():
    mem_usage = get_mem_usage_percent()
    mem_usage = int(mem_usage[0]) / 100
    return mem_usage

    # logging.info(mem_tip)
def get_cpu_usage():
    cpu_usage = int(get_cpu() * 100) / 100
    return cpu_usage
'''
if __name__ == '__main__':
    # logfile = "$HOME"
    # while True:
    # go(logfile)
        # time.sleep(30) # 每30s统计一次
    # CPU内核个数
    print("Linux的CPU内核个数是{}".format(get_cpu_num()))

    # CPU线程个数
    print("Linux的CPU线程个数是{}".format(get_route_num()))

    # CPU
    print("Linux的{}".format(get_cpu_usage()))

    # CPU
    print("Linux的{}".format(get_memory()))
'''


