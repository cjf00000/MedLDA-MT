import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from utils import *

if __name__ == '__main__':
    all_data=[#('20news', '20NewsGroups', False),
              # ('rcv1', 'RCV1', True),
              # ('wiki', 'Wiki', True),
              ('rcv1_small', 'RCV1(Small)', True),
              ('wiki_small', 'Wiki(Small)', True),]
    algs = [('none', 'None', 'k:'),
            ('fs', '+FS', 'k:x'),
            ('fsfs', '+FSFS', 'k--.'),
            ('fsfsfc', 'This', 'k-')]

    for data, data_name, multi_label in all_data:
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        for alg, name, style in algs:
            log_file = '../logs/comparison/{}_{}.log'.format(data, alg)
            log = read_log(log_file)

            total_time = 0
            times = []
            accs = []
            for iter in range(0, 20):
                total_time += log[iter]['svmTime'] + log[iter]['ldaTime'] + log[iter]['classTime']
                if iter % 1 == 0:
                    if multi_label:
                        acc = log[iter]['testing_macro_f1']
                    else:
                        acc = log[iter]['testing_accuracy']
                    times.append(total_time)
                    accs.append(acc)

            ax.plot(times, accs, style, label=name)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Testing accuracy')
        ax.set_title(data_name)
        plt.legend()
        fig.savefig('comparison_{}.pdf'.format(data))


        fig, ax = plt.subplots(figsize=(4.2, 4.2))

        class_times = []
        svm_times = []
        lda_times = []
        for alg, name, style in algs:
            log_file = '../logs/comparison/{}_{}.log'.format(data, alg)
            log = read_log(log_file)

            class_time = 0
            svm_time = 0
            lda_time = 0

            for iter in range(0, 20):
                class_time += log[iter]['classTime']
                svm_time += log[iter]['svmTime']
                lda_time += log[iter]['ldaTime']

            class_times.append(class_time / 20)
            svm_times.append(svm_time / 20)
            lda_times.append(lda_time / 20)
            print(data, alg, class_time, svm_time, lda_time)

        class_times = np.array(class_times)
        svm_times = np.array(svm_times)
        lda_times = np.array(lda_times)
        N = len(algs)
        ind = np.arange(N)
        width = 0.5

        ax.bar(ind, lda_times, width, label='lda', fill=False, hatch='//')
        ax.bar(ind, svm_times, width, bottom=lda_times, label='svm', fill=False)
        ax.bar(ind, class_times, width, bottom=lda_times+svm_times, label='class', fill=False, hatch='\\')

        ax.set_ylabel('Time (s)')
        ax.set_title(data_name)
        # plt.ylim([0.0, 1.0])
        plt.xticks(ind, (alg[1] for alg in algs))
        plt.legend()
        fig.savefig('comparison_{}_time.pdf'.format(data))
