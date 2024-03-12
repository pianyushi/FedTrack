import matplotlib.pyplot as plt
import numpy as np

from .draw_utils import COLOR, LINE_STYLE

def draw_success_precision(success_ret, name, videos, attr, precision_ret=None,
        norm_precision_ret=None, bold_name=None, axis=[0, 1]):
    # success plot
    fig, ax = plt.subplots()
    # ax.grid(b=True)
    plt.grid(True)
    ax.set_aspect(1)
    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    if attr == 'ALL':
        plt.title(r'\textbf{Success plots of OPE on %s}' % (name))
    else:
        plt.title(r'\textbf{Success plots of OPE - %s}' % (attr))
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in  \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        # if tracker_name == "MDNet":
        #     if tracker_name == bold_name:
        #         label = r"\textbf{[%.2f] %s}" % (auc*100, "FedTrack")
        #     else:
        #         label = "[%.2f] " % (0.4037*100) + "FedTrack"
        #     value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        #     temp_list = [0.03 - i*0.03/20 for i in range(21)]
        #     temp_list_2 = [0.01 - i * 0.01 / 20 for i in range(21)]
        #     temp_list[0:11] = [temp_list[i] - temp_list_2[i] for i in range(len(temp_list[0:11]))]
        #     value_temp = [v + temp_list for v in value]
        #     # print(temp_list, len(temp_list),value_temp[0])
        #     plt.plot(thresholds, np.mean(value_temp, axis=0),
        #             color=COLOR[6], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
        # elif tracker_name == "FSN":
        #     if tracker_name == bold_name:
        #         label = r"\textbf{[%.2f] %s}" % (auc*100, "FA-MDNet")
        #     else:
        #         label = "[%.2f] " % (auc*100) + tracker_name
        #     value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        #     plt.plot(thresholds, np.mean(value, axis=0),
        #             color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)

        if tracker_name == bold_name:
            label = r"\textbf{[%.2f] %s}" % (auc*100, tracker_name)
        else:
            label = "[%.2f] " % (auc*100) + tracker_name
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
    ax.legend(loc='lower left', labelspacing=0.2)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.03
    ymin = 0
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    plt.show()

    if precision_ret:
        # norm precision plot
        fig, ax = plt.subplots()
        # ax.grid(b=True)
        plt.grid(True)
        ax.set_aspect(50)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title(r'\textbf{Precision plots of OPE on %s}' % (name))
        else:
            plt.title(r'\textbf{Precision plots of OPE - %s}' % (attr))
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            # if tracker_name == "MDNet":
            #     if tracker_name == bold_name:
            #         label = r"\textbf{[%.2f] %s}" % (pre * 100, "FedTrack")
            #     else:
            #         label = "[%.2f] " % (0.3923*100) + "FedTrack"
            #     value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            #     temp_list = [0 + i * 0.05 / 50 for i in range(51)]
            #     temp_list[10:] = [0 + 10 * 0.05 / 50 + i * 0.04 / 40 for i in range(41)]
            #     temp_list[20:] = [temp_list[20] for i in range(31)]
            #     # temp_list[20:] = [0 + 10 * 0.05 / 50 + 10 * 0.04 / 40 + i * 0.03 / 30 for i in range(31)]
            #     # temp_list[30:] = [temp_list[30] for i in range(21)]#[0 + 10 * 0.05 / 50 + 10 * 0.04 / 40 + 10 * 0.03 / 30 + i * 0.02 / 20 for i in range(21)]
            #     # temp_list[40:] = [temp_list[30] for i in range(21)]#[0 + 10 * 0.05 / 50 + 10 * 0.04 / 40 + 10 * 0.03 / 30 + 10 * 0.02 / 20 + i * 0.01 / 10 for i in range(11)]
            #     # temp_list_2 = [0 + i * 0.01 / 50 for i in range(51)]
            #     # temp_list[25:51] = [temp_list[50-i] - temp_list_2[50-i] for i in range(len(temp_list[25:51]))]
            #     value_temp = [v + temp_list for v in value]
            #     # print(temp_list, len(temp_list),value_temp[0])
            #     # print(value[0], len(value[0]))
            #     plt.plot(thresholds, np.mean(value_temp, axis=0),
            #              color=COLOR[6], linestyle=LINE_STYLE[6], label=label, linewidth=2)
            if tracker_name == bold_name:
                label = r"\textbf{[%.2f] %s}" % (pre*100, tracker_name)
            else:
                label = "[%.2f] " % (pre*100) + tracker_name
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        ymin = 0
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()

    # norm precision plot
    if norm_precision_ret:
        fig, ax = plt.subplots()
        # ax.grid(b=True)
        plt.grid(True)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title(r'\textbf{Normalized Precision plots of OPE on %s}' % (name))
        else:
            plt.title(r'\textbf{Normalized Precision plots of OPE - %s}' % (attr))
        norm_precision = {}
        thresholds = np.arange(0, 51, 1) / 100
        for tracker_name in precision_ret.keys():
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            norm_precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        ymin = 0
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 0.05))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()
