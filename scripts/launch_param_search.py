from multiprocessing import Pool
import os

def run(args):
    params, id = args
    cmd = 'MedLDA/medlda {} > ../logs/param_search/{}.log'.format(params, id)
    # cmd = '../light_medlda/med {} > ../logs/param_search/{}.log 2>&1'.format(params, id)
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

    # common_params = ' --train_file ../data/20news.train --test_file ../data/20news.test --eval_every 10 --num_iter 100 --alpha_sum 50'
    # for K in [20, 40, 60, 80, 100]:
    #     for C in [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]:
    #         for ell in [1, 2, 4, 8, 16, 32, 64, 128]:
    #             params = '{} --num_topic {} --cost {} --ell {}'.format(common_params, K, C, ell)
    #             id = '20news_light_K{}_C{}_ell{}'.format(K, C, ell)
    #             tasks.append((params, id))

    # common_params = '--data ../data/wiki_small --multi_label --fast_precompute --fast_sampling --max_svm_iters=3'
    # for K in [20, 40, 60, 80, 100, 200, 400]:
    #     for C in [3, 10, 30]:
    #         for ell in [3, 10, 30, 100, 300]:
    #             params = '{} --K {} --C {} --ell {}'.format(common_params, K, C, ell)
    #             id = 'wiki_small_K{}_C{}_ell{}'.format(K, C, ell)
    #             tasks.append((params, id))

    p = Pool(20)
    p.map(run, tasks)
