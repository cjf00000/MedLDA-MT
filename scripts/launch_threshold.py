from multiprocessing import Pool
import os

def run(args):
    params, id = args
    cmd = 'MedLDA/medlda {} > ../logs/threshold/{}.log'.format(params, id)
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':
    tasks = []
    common_params = [('20news', '--data ../data/20news --K 100 --C 0.1 --ell 3 --fast_sampling --fast_precompute --max_svm_iters=3 '),
                     ('rcv1_small', '--data ../data/rcv1_small --K 800 --C 1 --ell 0.1 --multi_label --fast_sampling --fast_precompute --max_svm_iters=3 '),
                     ('wiki_small', '--data ../data/wiki_small --K 400 --C 2.56 --ell 128 --multi_label --fast_sampling --fast_precompute --max_svm_iters=3 ')]
    for data, data_params in common_params:
        for threshold in [0.0001, 0.0002, 0.0003, 0.0006, 0.001, 0.002, 0.003, 0.006, 0.01, 0.02, 0.03, 0.06, 0.1]:
            params = '{} --epsilon {}'.format(data_params, threshold)
            id = '{}_{}'.format(data, threshold)
            tasks.append((params, id))

    p = Pool(10)
    p.map(run, tasks)
