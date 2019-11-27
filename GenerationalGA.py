# -*- coding: utf-8 -*-
"""
Created on 2019/11/21 14:37 
@file: GenerationalGA.py
@author: Matt
"""
import numpy as np
from GA import GA
from keras.layers import Conv2D, Dense
from keras.models import Sequential, clone_model


class GenerationalGA(GA):
    def run(self):
        print('Generational GA is running...')
        self.initialization()
        while 1:
            series = self.shuffle_batch()
            for i in range(self.batch_size, len(series) + 1, self.batch_size):
                idx = series[i - self.batch_size:i]
                X_batch = self.X_train[idx]
                y_batch = self.y_train[idx]
                if i + self.batch_size > len(series):
                    self.evaluation(X_batch, y_batch, False)
                else:
                    self.evaluation(X_batch, y_batch)
                self.selection()
                if self.cur_iter >= self.max_iter:
                    print('Maximum iterations({}) reached.'.format(self.max_iter))
                    return
                if self.evaluation_history[-1]['best_fit']['train_acc'] >= self.min_fitness:
                    print('Minimum fitness({}) reached.'.format(self.min_fitness))
                    return

    def selection(self):
        mating_pool = np.array([self.roulette_wheel_selection() for _ in range(self.pop_size)])
        np.random.shuffle(mating_pool)
        pairs = mating_pool.reshape(int(self.pop_size / 2), 2)
        print('Pairs: {}'.format(list(map(list, pairs))))
        children = []
        for pair in pairs:
            children += self.crossover(pair)
        print('Cross over finished.')
        self.replacement(children)
        for i in range(self.pop_size):
            if np.random.rand() < self.p_mutation:
                mutated_child = self.mutation(i)
                del self.chroms[i]
                self.chroms.insert(i, mutated_child)

    def crossover(self, _selected_pop):
        # identical pops
        if _selected_pop[0] == _selected_pop[1]:
            return [clone_model(self.chroms[_selected_pop[0]]), clone_model(self.chroms[_selected_pop[1]])]
        child1 = Sequential()
        child2 = Sequential()
        chrom1 = clone_model(self.chroms[_selected_pop[0]])
        chrom2 = clone_model(self.chroms[_selected_pop[1]])
        chrom1_layers = chrom1.layers
        chrom2_layers = chrom2.layers
        for i in range(len(chrom1_layers)):
            layer1 = chrom1_layers[i]
            layer2 = chrom2_layers[i]
            if type(layer1) is Conv2D:
                if np.random.rand() < 0.5:
                    child1.add(layer1)
                    child2.add(layer2)
                else:
                    child1.add(layer2)
                    child2.add(layer1)
            elif type(layer1) is Dense:
                weights1 = layer1.get_weights()[0]  # only the kernel weights
                weights2 = layer2.get_weights()[0]
                rand1 = np.random.randint(0, 2, weights1.shape[1])  # cols
                rand2 = 1 - rand1
                layer1.set_weights([weights1 * rand1 + weights2 * rand2])
                layer2.set_weights([weights1 * rand2 + weights2 * rand1])
                child1.add(layer1)
                child2.add(layer2)
            else:
                child1.add(layer1)
                child2.add(layer2)
        del chrom1
        del chrom2
        return [child1, child2]

    def mutation(self, _selected_pop):
        child = Sequential()
        chrom = clone_model(self.chroms[_selected_pop])
        chrom_layers = chrom.layers
        for layer in chrom_layers:
            if type(layer) is Conv2D:
                if np.random.rand() < self.r_mutation:
                    weights = layer.get_weights()[0]
                    layer.set_weights([weights + np.random.normal(0, self.stddev, weights.shape)])
                child.add(layer)
            elif type(layer) is Dense:
                weights = layer.get_weights()[0]
                rand = np.where(np.random.rand(weights.shape[1]) < self.r_mutation, 1, 0)
                layer.set_weights([weights + rand * np.random.normal(0, self.stddev, weights.shape)])
                child.add(layer)
            else:
                child.add(layer)
        del chrom
        print('Mutation({}) finished.'.format(_selected_pop))
        return child

    def replacement(self, _child):
        self.chroms[:] = _child
        print('Replacement finished.')
