import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from utils import *

if __name__ == '__main__':
    all_data=[('20news', '20NewsGroups', [0.0, 4.0]),
              ('rcv1_small', 'RCV1(small)', [0.0, 10.0]),
              ('wiki_small', 'Wiki(small)', [0.0, 5.0])]
    algs = [('naive', 'Baseline', 'k:'),
            ('fast', 'Fast', 'k-')]
    topics = range(100, 1001, 100)

    for data, data_name, lim in all_data:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        for alg, alg_name, style in algs:
            times = []
            for topic in topics:
                log_file = '../logs/topics/{}_{}_{}.log'.format(data, alg, topic)
                log = read_log(log_file)

                total_time = 0
                for iter in range(0, 20):
                    total_time += log[iter]['classTime'] + log[iter]['ldaTime'] + log[iter]['svmTime']

                times.append(total_time / 20)

            ax.plot(topics, times, style, label=alg_name)

        ax.set_xlabel('K')
        ax.set_ylabel('Time (s)')
        ax.set_ylim(lim)
        ax.set_title(data_name)
        ax.legend()
        fig.savefig('topics_{}.pdf'.format(data))