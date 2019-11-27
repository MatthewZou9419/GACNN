# -*- coding: utf-8 -*-
"""
Created on 2019/11/20 23:13 
@file: GA.py
@author: Matt
"""
from abc import abstractmethod

import numpy as np
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.models import Sequential


class GA:
    def __init__(self, _X_train, _y_train, _X_test, _y_test, _pop_size, _r_mutation, _p_crossover, _p_mutation,
                 _max_iter, _min_fitness, _elite_num, _mating_pool_size, _batch_size=32, _dataset='cifar10'):
        # input params
        self.X_train = _X_train
        self.y_train = _y_train
        self.X_test = _X_test
        self.y_test = _y_test
        self.pop_size = _pop_size
        self.r_mutation = _r_mutation
        self.p_crossover = _p_crossover  # for steady-state
        self.p_mutation = _p_mutation  # for generational
        self.max_iter = _max_iter
        self.min_fitness = _min_fitness
        self.elite_num = _elite_num  # for elitism
        self.mating_pool_size = _mating_pool_size  # for elitism
        self.batch_size = _batch_size
        self.dataset = _dataset
        # other params
        self.chroms = []
        self.evaluation_history = []
        self.stddev = 0.5
        self.loss_func = 'categorical_crossentropy'
        self.metrics = ['accuracy']

    @property
    def cur_iter(self):
        return len(self.evaluation_history)

    def shuffle_batch(self):
        series = list(range(len(self.X_train)))
        np.random.shuffle(series)
        return series

    def initialization(self):
        for i in range(self.pop_size):
            model = Sequential()
            if self.dataset == 'cifar10':
                model.add(Conv2D(6, (1, 1), activation='relu', use_bias=False, input_shape=(32, 32, 3)))
                model.add(AveragePooling2D((2, 2)))
                model.add(Conv2D(16, (1, 1), activation='relu', use_bias=False))
                model.add(AveragePooling2D((2, 2)))
                model.add(Conv2D(120, (1, 1), activation='relu', use_bias=False))
                model.add(AveragePooling2D((2, 2)))
                model.add(Flatten())
                model.add(Dense(120, activation='relu', use_bias=False))
                model.add(Dense(84, activation='relu', use_bias=False))
                model.add(Dense(10, activation='softmax', use_bias=False))
            elif self.dataset == 'mnist':
                model.add(Conv2D(40, (1, 1), activation='relu', use_bias=False, input_shape=(28, 28, 1)))
                model.add(AveragePooling2D((2, 2)))
                model.add(Conv2D(40, (1, 1), activation='relu', use_bias=False))
                model.add(AveragePooling2D((2, 2)))
                model.add(Conv2D(5, (1, 1), activation='relu', use_bias=False))
                model.add(AveragePooling2D((2, 2)))
                model.add(Conv2D(1, (1, 1), activation='relu', use_bias=False))
                model.add(Flatten())
                model.add(Dense(40, activation='relu', use_bias=False))
                model.add(Dense(10, activation='softmax', use_bias=False))
            else:
                raise Exception('Unknown dataset.')
            self.chroms.append(model)
        print('{} network initialization({}) finished.'.format(self.dataset, self.pop_size))

    def evaluation(self, _X, _y, _is_batch=True):
        cur_evaluation = []
        for i in range(self.pop_size):
            model = self.chroms[i]
            model.compile(loss=self.loss_func, metrics=self.metrics, optimizer='adam')
            train_loss, train_acc = model.evaluate(_X, _y, verbose=0)
            if not _is_batch:
                test_loss, test_acc = model.evaluate(self.X_test, self.y_test, verbose=0)
                cur_evaluation.append({
                    'pop': i,
                    'train_loss': round(train_loss, 4),
                    'train_acc': round(train_acc, 4),
                    'test_loss': round(test_loss, 4),
                    'test_acc': round(test_acc, 4),
                })
            else:
                cur_evaluation.append({
                    'pop': i,
                    'train_loss': round(train_loss, 4),
                    'train_acc': round(train_acc, 4),
                })
        best_fit = sorted(cur_evaluation, key=lambda x: x['train_acc'])[-1]
        self.evaluation_history.append({
            'iter': self.cur_iter + 1,
            'best_fit': best_fit,
            'avg_fitness': np.mean([e['train_acc'] for e in cur_evaluation]).round(4),
            'evaluation': cur_evaluation,
        })
        print('\nIter: {}'.format(self.evaluation_history[-1]['iter']))
        print('Best_fit: {}, avg_fitness: {:.4f}'.format(self.evaluation_history[-1]['best_fit'],
                                                         self.evaluation_history[-1]['avg_fitness']))

    def roulette_wheel_selection(self):
        sorted_evaluation = sorted(self.evaluation_history[-1]['evaluation'], key=lambda x: x['train_acc'])
        cum_acc = np.array([e['train_acc'] for e in sorted_evaluation]).cumsum()
        extra_evaluation = [{'pop': e['pop'], 'train_acc': e['train_acc'], 'cum_acc': acc}
                            for e, acc in zip(sorted_evaluation, cum_acc)]
        rand = np.random.rand() * cum_acc[-1]
        for e in extra_evaluation:
            if rand < e['cum_acc']:
                return e['pop']
        return extra_evaluation[-1]['pop']

    @abstractmethod
    def run(self):
        raise NotImplementedError('Run not implemented.')

    @abstractmethod
    def selection(self):
        raise NotImplementedError('Selection not implemented.')

    @abstractmethod
    def crossover(self, _selected_pop):
        raise NotImplementedError('Crossover not implemented.')

    @abstractmethod
    def mutation(self, _selected_pop):
        raise NotImplementedError('Mutation not implemented.')

    @abstractmethod
    def replacement(self, _child):
        raise NotImplementedError('Replacement not implemented.')
