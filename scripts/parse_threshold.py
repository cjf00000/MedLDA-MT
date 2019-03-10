import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from utils import *

if __name__ == '__main__':
    all_data=[('20news', '20NewsGroups'),
              ('rcv1_small', 'RCV1(small)'),
              ('wiki_small', 'Wiki(small)')]
    thresholds = [0.0001, 0.0002, 0.0003, 0.0006, 0.001, 0.002, 0.003, 0.006, 0.01, 0.02, 0.03, 0.06, 0.1]

    for data, data_name in all_data:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        fig2, ax2 = plt.subplots(figsize=(4.4, 4.4))
        fig3, ax3 = plt.subplots(figsize=(4.4, 4.4))
        sps = []
        rejs = []
        tims = []
        thrs = []
        for threshold in thresholds:
            log_file = '../logs/threshold/{}_{}.log'.format(data, threshold)
            log = read_log(log_file)

            sparsity = []
            reject_rate = []
            times = []
            for iter in range(0, 100):
                reject_rate.append(log[iter]['reject_rate'])
                sparsity.append(log[iter]['sparse_rate'])
                times.append(log[iter]['ldaTime'])

            sps.append(np.mean(sparsity))
            rejs.append(np.mean(reject_rate))
            tims.append(np.mean(times))
            thrs.append(threshold)

        ax.plot(thrs, rejs, 'k:', label='Rejection rate')
        ax.plot(thrs, sps, 'k-', label='Sparsity rate')
        ax.set_xlabel('Threshold')
        ax.set_title(data_name)
        ax.set_ylim([0.0, 1.0])
        ax.set_xscale('log')
        ax.legend()
        fig.savefig('threshold_thr_rej_sps_{}.pdf'.format(data))

        ax3.plot(thrs, tims, 'k-')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Time (s)')
        ax3.set_title(data_name)
        ax3.set_xscale('log')
        fig3.savefig('threshold_thr_time_{}.pdf'.format(data))
