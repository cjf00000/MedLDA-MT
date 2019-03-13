from multiprocessing import Pool
import os

def run(args):
    params, id = args
    cmd = 'MedLDA/medlda {} > ../logs/topics/{}.log'.format(params, id)
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':
    tasks = []
    common_params = [('20news_naive', '--data ../data/20news --C 0.1 --ell 3 --test_every 1 --alpha_sum 10 --num_iters 20 '),
                     ('20news_fast', '--data ../data/20news --C 0.1 --ell 3 --test_every 1 --alpha_sum 10 --num_iters 20 --fast_sampling --fast_precompute --max_svm_iters=3 '),
                     ('rcv1_small_naive', '--data ../data/rcv1_small --C 1 --ell 0.1 --multi_label --num_iters 20 --test_every 1 --alpha_sum 10 '),
                     ('rcv1_small_fast', '--data ../data/rcv1_small --C 1 --ell 0.1 --multi_label --num_iters 20 --test_every 1 --alpha_sum 10 --fast_sampling --fast_precompute --max_svm_iters=3 '),
                     ('wiki_small_naive', '--data ../data/wiki_small --C 2.56 --ell 128 --multi_label --num_iters 20 --test_every 1 --alpha_sum 10 '),
                     ('wiki_small_fast', '--data ../data/wiki_small --C 2.56 --ell 128 --multi_label --num_iters 20 --test_every 1 --alpha_sum 10 --fast_sampling --fast_precompute --max_svm_iters=3 ')]

    for data, data_params in common_params:
        for topics in range(100, 1001, 100):
            params = '{} --K {} --epsilon {}'.format(data_params, topics, 2.0/topics)
            id = '{}_{}'.format(data, topics)
            tasks.append((params, id))

    p = Pool(10)
    p.map(run, tasks)
