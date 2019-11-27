# -*- coding: utf-8 -*-
"""
Created on 2019/11/12 17:59 
@file: SteadyStateGA.py
@author: Matt
"""
import numpy as np
from GA import GA
from keras.layers import Conv2D, Dense
from keras.models import Sequential, clone_model


class SteadyStateGA(GA):
    def run(self):
        print('Steady-state GA is running...')
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
                child = self.selection()
                self.replacement(child)
                if self.cur_iter >= self.max_iter:
                    print('Maximum iterations({}) reached.'.format(self.max_iter))
                    return
                if self.evaluation_history[-1]['best_fit']['train_acc'] >= self.min_fitness:
                    print('Minimum fitness({}) reached.'.format(self.min_fitness))
                    return

    def selection(self):
        if np.random.rand() < self.p_crossover:
            selected_pop = []
            while len(selected_pop) < 2:
                pop = self.roulette_wheel_selection()
                if pop not in selected_pop:
                    selected_pop.append(pop)
            print('Selected pop: {}'.format(selected_pop))
            return self.crossover(selected_pop)
        else:
            selected_pop = self.roulette_wheel_selection()
            print('Selected pop: {}'.format(selected_pop))
            return self.mutation(selected_pop)

    def crossover(self, _selected_pop):
        child = Sequential()
        chrom1 = clone_model(self.chroms[_selected_pop[0]])
        chrom2 = clone_model(self.chroms[_selected_pop[1]])
        chrom1_layers = chrom1.layers
        chrom2_layers = chrom2.layers
        for i in range(len(chrom1_layers)):
            layer1 = chrom1_layers[i]
            layer2 = chrom2_layers[i]
            if type(layer1) is Conv2D:
                child.add(layer1 if np.random.rand() < 0.5 else layer2)
            elif type(layer1) is Dense:
                weights1 = layer1.get_weights()[0]  # only the kernel weights
                weights2 = layer2.get_weights()[0]
                rand1 = np.random.randint(0, 2, weights1.shape[1])  # cols
                rand2 = 1 - rand1
                layer1.set_weights([weights1 * rand1 + weights2 * rand2])
                child.add(layer1)
            else:
                child.add(layer1)
        del chrom1
        del chrom2
        print('Crossover finished.')
        return child

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
        print('Mutation finished.')
        return child

    def replacement(self, _child):
        sorted_evaluation = sorted(self.evaluation_history[-1]['evaluation'], key=lambda x: x['train_acc'])
        least_fit_pop = sorted_evaluation[0]['pop']
        del self.chroms[least_fit_pop]
        self.chroms.insert(least_fit_pop, _child)
        print('Replacement({}, fitness: {:.4f}) finished.'
              .format(least_fit_pop, sorted_evaluation[0]['train_acc']))
