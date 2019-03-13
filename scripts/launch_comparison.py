from multiprocessing import Pool
import os

def run(args):
    params, id = args
    cmd = 'stdbuf -o 0 MedLDA/medlda {} > ../logs/comparison/{}.log'.format(params, id)
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':
    tasks = []
    common_params = [('20news', '--data ../data/20news --K 100 --C 0.1 --ell 3 --test_every 1 --epsilon 0.02 --alpha_sum 10 --num_iters 20 '),
                     ('rcv1', '--data ../data/rcv1 --K 800 --C 1 --ell 0.1 --multi_label --num_iters 20 --test_every 1 --epsilon 0.0025 --alpha_sum 10 '),
                     ('wiki', '--data ../data/20_label_wiki --K 400 --C 2.56 --ell 128 --multi_label --num_iters 20 --test_every 1 --epsilon 0.005 --alpha_sum 10 '),
                     ('rcv1_small', '--data ../data/rcv1_small --K 800 --C 1 --ell 0.1 --multi_label --num_iters 20 --test_every 1 --epsilon 0.0025 --alpha_sum 10 '),
                     ('wiki_small', '--data ../data/wiki_small --K 400 --C 2.56 --ell 128 --multi_label --num_iters 20 --test_every 1 --epsilon 0.005 --alpha_sum 10 ')]
    param_space = [('none', ''),
                   ('fs', '--fast_sampling'),
                   ('fsfs', '--fast_sampling --max_svm_iters=3'),
                   ('fsfsfc', '--fast_sampling --fast_precompute --max_svm_iters=3')]
    for pid, param in common_params:
        for id, params in param_space:
            tasks.append((param + params, pid + '_' + id))

    p = Pool(2)
    p.map(run, tasks)
