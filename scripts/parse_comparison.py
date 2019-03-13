import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from utils import *
import sys

if __name__ == '__main__':
    all_data=[('20news', '20NewsGroups', False, 20, 11269),
              ('rcv1', 'RCV1', True, 103, 790000),
              ('wiki', 'Wiki', True, 20, 1100000),
              ('rcv1_small', 'RCV1(Small)', True, 103, 20000),
              ('wiki_small', 'Wiki(Small)', True, 20, 20000),]
    algs = [('none', 'Baseline', 'k:'),
            ('fs', 'Reject', 'k:x'),
            ('fsfs', 'Reject, iSVM=3', 'k--.'),
            ('fsfsfc', 'Fast', 'k-')]

    for data, data_name, multi_label, L, D in all_data:
        sys.stdout.write(data_name)
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        for alg, name, style in algs:
            log_file = '../logs/comparison/{}_{}.log'.format(data, alg)
            log = read_log(log_file)

            total_time = 0
            times = []
            accs = []
            total_isvm = 0
            total_lsv = 0
            total_kd = 0
            total_klambda = 0
            for iter in range(0, 20):
                total_time += log[iter]['svmTime'] + log[iter]['ldaTime'] + log[iter]['classTime']
                total_isvm += float(log[iter]["nSVMIteres"]) / L
                total_lsv += float(log[iter]["nSV"]) / D
                total_kd += float(log[iter]["cdk_nnz"]) / D
                total_klambda += float(log[iter]["doc_prob_nnz"]) / D
                if iter % 1 == 0:
                    if multi_label:
                        acc = log[iter]['testing_macro_f1']
                    else:
                        acc = log[iter]['testing_accuracy']
                    times.append(total_time)
                    accs.append(acc)

            if alg == 'none':
                sys.stdout.write(" & {:.0f}".format(total_isvm / 20))
            elif alg == 'fsfsfc':
                sys.stdout.write(" & {:.0f} & {:.0f} & {:.0f} \\\\\n".format(total_kd / 20, total_klambda / 20, total_lsv / 20))
            ax.plot(times, accs, style, label=name)

        ax.set_xlabel('Time (s)')
        if multi_label:
            ax.set_ylabel('Testing macro F1')
        else:
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
            # print(data, alg, class_time, svm_time, lda_time)

        class_times = np.array(class_times)
        svm_times = np.array(svm_times)
        lda_times = np.array(lda_times)
        N = len(algs)
        ind = np.arange(N)
        width = 0.5

        ax.bar(ind, lda_times, width, label='Sampling', fill=False, hatch='//')
        ax.bar(ind, svm_times, width, bottom=lda_times, label='Classifier', fill=False)
        ax.bar(ind, class_times, width, bottom=lda_times+svm_times, label='Precomputing', fill=False, hatch='\\')

        ax.set_ylabel('Time (s)')
        ax.set_title(data_name)
        # plt.ylim([0.0, 1.0])
        plt.xticks(ind, (alg[1] for alg in algs))
        plt.legend()
        fig.savefig('comparison_{}_time.pdf'.format(data))
