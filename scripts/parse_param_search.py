from utils import *

if __name__ == '__main__':
    tasks = []
    best_acc = 0
    best_params = []
    # 20news
    for K in [20, 40, 60, 80, 100]:
        for C in [0.1, 0.3, 1, 3, 10]:
            for ell in [0.1, 0.3, 1, 3, 10]:
                id = '20news_K{}_C{}_ell{}'.format(K, C, ell)
                log = read_log('../logs/param_search/{}.log'.format(id))
                acc = log[-10]["testing_accuracy"]
                print(K, C, ell, acc)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (K, C, ell)

    print(best_params, best_acc)

    # rcv1
    best_acc = 0
    best_params = []
    for K in [100, 200, 400, 600, 800, 1000]:
        for C in [0.1, 0.3, 1, 3, 10]:
            for ell in [0.1, 0.3, 1, 3, 10]:
                id = 'rcv1_small_fast_K{}_C{}_ell{}'.format(K, C, ell)
                log = read_log('../logs/param_search/{}.log'.format(id))
                acc = log[-10]["testing_macro_f1"]
                print(K, C, ell, acc)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (K, C, ell)
    print(best_params, best_acc)

    best_acc = 0
    best_params = []
    for K in [20, 40, 60, 80, 100]:
        for C in [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]:
            for ell in [1, 2, 4, 8, 16, 32, 64, 128]:
                id = '20news_light_K{}_C{}_ell{}'.format(K, C, ell)
                acc = read_light_log('../logs/param_search/{}.log'.format(id))
                if len(acc) > 0:
                    acc = acc[-1]
                else:
                    acc = 0
                print(K, C, ell, acc)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (K, C, ell)
    print(best_params, best_acc)