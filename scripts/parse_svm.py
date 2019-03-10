import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from utils import *

if __name__ == '__main__':
    all_data=[('20news', '20NewsGroups', [0.76, 0.83], False),
          ('rcv1_small', 'RCV1(small)', [0.1, 0.6], True),
          ('wiki_small', 'Wiki(small)', [0, 0.4], True)]
    algs = [('1', '1', 'k:'),
            ('2', '2', 'k:x'),
            ('3', '3', 'k--.'),
            ('4', '4', 'k--^'),
            ('infty', 'max', 'k-')]

    for data, data_name, data_lim, multi_label in all_data:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        fig2, ax2 = plt.subplots(figsize=(4.4, 4.4))
        fig3, ax3 = plt.subplots(figsize=(4.4, 4.4))
        svmIters = []
        for alg, name, style in algs:
            log_file = '../logs/svm/{}_{}.log'.format(data, alg)
            log = read_log(log_file)

            total_time = 0
            times = []
            accs = []
            iters = []
            total_svm_iter = 0
            for iter in range(0, 100):
                total_time += log[iter]['svmTime'] + log[iter]['ldaTime'] + log[iter]['classTime']
                total_svm_iter += log[iter]['nSVMIteres']
                if iter % 10 == 0:
                    if multi_label:
                        acc = log[iter]['testing_macro_f1']
                    else:
                        acc = log[iter]['testing_accuracy']
                    times.append(total_time)
                    accs.append(acc)
                    iters.append(iter+1)

            ax.plot(times, accs, style, label=name)
            ax2.plot(iters, accs, style, label=name)
            svmIters.append(total_svm_iter / 100)

        ax.set_xlabel('Time (s)')
        if multi_label:
            ax.set_ylabel('Testing F1')
        else:
            ax.set_ylabel('Testing accuracy')
        ax.set_title(data_name)
        ax.set_ylim(data_lim)
        ax.legend()
        fig.savefig('svm_{}.pdf'.format(data))

        ax2.set_xlabel('Iteration')
        if multi_label:
            ax2.set_ylabel('Testing F1')
        else:
            ax2.set_ylabel('Testing accuracy')
        ax2.set_title(data_name)
        ax2.set_ylim(data_lim)
        ax2.legend()
        fig2.savefig('svm_{}_iter.pdf'.format(data))

        N = len(algs)
        ind = np.arange(N)
        width = 0.5

        ax3.bar(ind, svmIters, width, fill=False)
        ax3.set_ylabel('SVM Iterations')
        ax3.set_title(data_name)
        plt.xticks(ind, (alg[1] for alg in algs))
        fig3.savefig('svm_{}_niter.pdf'.format(data))
