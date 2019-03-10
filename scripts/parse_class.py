import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from utils import *
import sys

if __name__ == '__main__':
    all_data=[('20news', '20NewsGroups', False, 225380),
              ('rcv1_small', 'RCV1(small)', True, 2060000),
              ('wiki_small', 'Wiki(small)', True, 400000)]

    algs = [('none', 'None', 'k:'),
            ('fast', 'Fast', 'k-')]

    for data, data_name, multi_label, DL in all_data:
        logs = []
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        for alg, name, style in algs:
            svs = []
            log_file = '../logs/class/{}_{}.log'.format(data, alg)
            log = read_log(log_file)

            class_time = 0
            nSVs = 0
            for iter in range(0, 100):
                class_time += log[iter]['classTime']
                nSVs += log[iter]['nSV']
                svs.append(log[iter]['nSV'])

            logs.append(class_time / 100)
            ax.plot(svs, style, label=name)

        logs.append(nSVs / 100)
        logs.append(DL)
        sys.stdout.write(data_name)
        for l in logs:
            sys.stdout.write(' & {}'.format(l))
        sys.stdout.write('\\\\\n')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('# support vectors')
        ax.set_title(data_name)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        fig.savefig('class_{}.pdf'.format(data))

