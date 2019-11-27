# -*- coding: utf-8 -*-
"""
Created on 2019/11/24 20:21 
@file: DataMgr.py
@author: Matt
"""
import numpy as np
from six.moves import cPickle as pickle


def load_cifar10(root):
    # train_set
    X_train = []
    y_train = []
    for batch in range(1, 5 + 1):
        file = root + '/data_batch_{}'.format(batch)
        with open(file, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
            X_train.append(d['data'])
            y_train.append(d['labels'])
    X_train = np.concatenate(X_train).reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32')
    y_train = np.concatenate(y_train)
    # test_set
    file = root + '/test_batch'
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        X_test = d['data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32')
        y_test = np.array(d['labels'])
    X_train /= 255.0
    X_test /= 255.0
    return X_train, y_train, X_test, y_test


def write_performance(perf, filename):
    with open(filename, 'a+') as f:
        text = 'iter\tbest_pop\tbest_loss\tbest_acc\tavg_fitness\n'
        for e in perf:
            text += ('\t'.join(list(map(str, [e['iter'], e['best_fit']['pop'], e['best_fit']['train_loss'],
                                              e['best_fit']['train_acc'], e['avg_fitness']]))) + '\n')
        f.write(text)
