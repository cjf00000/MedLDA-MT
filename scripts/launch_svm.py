from multiprocessing import Pool
import os

def run(args):
    params, id = args
    cmd = 'MedLDA/medlda {} > ../logs/svm/{}.log'.format(params, id)
    # cmd = '../light_medlda/med {} > ../logs/param_search/{}.log 2>&1'.format(params, id)
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':
    tasks = []
    common_params = [('20news', '--data ../data/20news --K 100 --C 0.1 --ell 3 '),
                     ('rcv1_small', '--data ../data/rcv1_small --K 800 --C 1 --ell 0.1 --multi_label '),
                     ('wiki_small', '--data ../data/wiki_small --K 400 --C 2.56 --ell 128 --multi_label ')]
    param_space = [('infty', '--fast_sampling --fast_precompute'),
                   ('1', '--fast_sampling --fast_precompute --max_svm_iters=1'),
                   ('2', '--fast_sampling --fast_precompute --max_svm_iters=2'),
                   ('3', '--fast_sampling --fast_precompute --max_svm_iters=3'),
                   ('4', '--fast_sampling --fast_precompute --max_svm_iters=4')]
    for pid, param in common_params:
        for id, params in param_space:
            tasks.append((param + params, pid + '_' + id))

    p = Pool(10)
    p.map(run, tasks)
