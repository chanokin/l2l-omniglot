from __future__ import (print_function,
                        unicode_literals,
                        division)
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from pprint import pprint
from multiprocessing import Process, Queue
import time
import sys
import os
import logging

from l2l.optimizees.functions.optimizee import Optimizee
from six import iterkeys, iteritems

def randnum(vmin, vmax, div=None, rng=None):
    if isinstance(vmin, int):
        return randint_float(vmin, vmax, div, rng)
    v = rng.uniform(vmin, vmax)
    # print("RANDNUM: uniform(%s, %s) = %s"%(vmin, vmax, v))
    return v


def bound(val, num_range):
    if len(num_range) == 1:
        v = num_range[0]
    else:
        v = np.clip(val, num_range[0], num_range[1])

    # print("BOUND: (%s, %s, %s) -> %s"%(num_range[0], num_range[1], val, v))
    if np.issubdtype(type(num_range[0]), np.integer):
        v = np.round(v)
        # print("INT-BOUND %s"%v)

    return v


def randint_float(vmin, vmax, div=None, rng=None):
    rand_func = np.random if rng is None else rng
    if div is None:
        return np.float(rng.randint(vmin, vmax))
    else:
        return np.float(np.floor(np.floor(rng.randint(vmin, vmax) / float(div)) * div))



class config(object):
    ATTR_RANGES = {
        "{:05d}".format(k): (-1.0, 1.0) for k in range(50)
    }

EPSILON = 1e-12
class FitOptimizee(Optimizee):
    def __init__(self, traj, seed):
        super().__init__(traj)
        self.seed = np.uint32(seed)
        self.rng = np.random.RandomState(seed=self.seed)
        self.ind_param_ranges = config.ATTR_RANGES.copy()
        self.sim_params = traj.simulation.params.copy()

    def create_individual(self):
        ipr = self.ind_param_ranges
        return {k: randnum(ipr[k][0], ipr[k][1], rng=self.rng) \
                if len(ipr[k]) == 2 else ipr[k][0] for k in ipr}

    def bounding_func(self, individual):
        ipr = self.ind_param_ranges
        for k in ipr:
            range = ipr[k]
            val = individual[k]
            individual[k] = bound(val, range)

        return individual

    def simulate(self, traj, queue=None):
        ind_idx = traj.individual.ind_idx
        generation = traj.individual.generation

        for i in range(1):
            # print("generation, ind_idx = (%s, %s)\tt = %02d"%(generation, ind_idx, i))
            time.sleep(0.01)

        ipr = self.ind_param_ranges
        ind_params = {k: getattr(traj.individual, k) for k in ipr}
        ps = [ind_params[k] for k in sorted(list(ind_params.keys()))]
        n_points = len(self.sim_params['target'])
        xs = np.arange(n_points) / float(n_points)
        ys = np.polynomial.chebyshev.chebval(xs, ps)
        dy = self.sim_params['target'] - ys
        mse = np.mean(dy ** 2)
        mse = EPSILON if mse <= EPSILON else mse
        fitness = 1.0/mse

        name = 'gen{:05d}_ind{:05d}'.format(generation, ind_idx)

        p = '/home/gp283/l2l-omniglot'
        os.makedirs(os.path.join(p, 'temp_fit'), exist_ok=True)
        with open(os.path.join(p, 'temp_fit', 'fitnesses.txt'), 'a+') as f:
            f.write('{}, {}, {}\n'.format(generation, ind_idx, fitness))
            f.close()

        # os.makedirs(os.path.join(p, 'temp_figs'), exist_ok=True)
        #
        # os.makedirs(os.path.join(p, 'temp_figs', 'gen_{:06d}'.format(generation)), exist_ok=True)

        # fig = plt.figure()
        # ax =  plt.subplot(1,1,1)
        # plt.plot(xs, ys, 'r', label='Individiual')
        # plt.plot(xs, self.sim_params['target'], 'b', label='Target')
        # plt.legend()
        # fname = os.path.join(p, 'temp_figs',
        #                      'gen_{:06d}'.format(generation),
        #                      'simulation_{}.pdf'.format(name))
        # plt.savefig(fname)
        # plt.close('all')

        if queue is not None:
            queue.put([fitness])
            return

        return [fitness]
