from multiprocessing import Pool
import os

def run(args):
    params, id = args
    cmd = 'MedLDA/medlda {} > ../logs/param_search/{}.log'.format(params, id)
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':
    tasks = []
    common_params = '--data ../data/rcv1_small --fast_precompute --fast_sampling --max_svm_iters=3 --multi_label'
    for K in [100, 200, 400, 600, 800, 1000]:
        for C in [0.1, 0.3, 1, 3, 10]:
            for ell in [0.1, 0.3, 1, 3, 10]:
                params = '{} --K {} --C {} --ell {}'.format(common_params, K, C, ell)
                id = 'rcv1_small_fast_K{}_C{}_ell{}'.format(K, C, ell)
                tasks.append((params, id))

    p = Pool(20)
    p.map(run, tasks)
