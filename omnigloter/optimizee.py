from __future__ import (print_function,
                        unicode_literals,
                        division)
import numpy as np
from scipy.special import comb
from pprint import pprint
from multiprocessing import Process, Queue
import time
import sys
import os
import logging
from omnigloter import config, utils, analyse_network as analysis
from omnigloter.snn_decoder import Decoder

logger = logging.getLogger("optimizee.mushroom_body")

# L2L imports
from l2l.optimizees.functions.optimizee import Optimizee
from six import iterkeys, iteritems

SOFT_ZERO_PUNISH = bool(1)

class OmniglotOptimizee(Optimizee):
    def __init__(self, traj, seed, gradient_desc=bool(0)):

        super().__init__(traj)
        self.seed = np.uint32(seed)
        self.rng = np.random.RandomState(seed=self.seed)
        self.grad_desc = gradient_desc
        self.ind_param_ranges = config.ATTR_RANGES.copy()
        self.sim_params = traj.simulation.params.copy()

        print("In OmniglotOptimizee:")
        print(self.ind_param_ranges)
        print(self.sim_params)

    def create_individual(self):
        ipr = self.ind_param_ranges
        return {k: utils.randnum(ipr[k][0], ipr[k][1], rng=self.rng) \
                    if len(ipr[k]) == 2 else ipr[k][0] for k in ipr}

    def bounding_func(self, individual):
        ipr = self.ind_param_ranges
        for k in ipr:
            range = ipr[k]
            val = individual[k]
            individual[k] = utils.bound(val, range)

        return individual


    def snn_multiproc(self, name, params):
        # trying to avoid huge memory consumption after many individuals


        # queue = Queue()
        # p = Process(target=self.create_and_run, args=(name, params, queue))
        # p.start()
        # print("launched process, waiting ...")
        # p.join()  # this blocks until the process terminates
        # print("done running!")
        # data = queue.get()
        # return data
        queue = None
        return self.create_and_run(name, params, queue)

    def create_and_run(self, name, params, queue=None):
        print("Creating and running SNN")
        snn = Decoder(name, params)
        data = snn.run_pynn()
        print("Done running simulation")
        if queue is None:
            return data
        queue.put(data)

    def simulate(self, traj, queue=None):
        work_path = traj._parameters.JUBE_params.params['work_path']
        results_path = os.path.join(work_path, 'run_results')
        os.makedirs(results_path, exist_ok=True)

        bench_start_t = time.time()
        ipr = self.ind_param_ranges
        for k in traj.par:
            try:
                print("{}:".format(k))
                print("\t{}\n".format(traj.par[k]))
            except:
                print("\tNOT FOUND!\n")

        n_out = self.sim_params['output_size']
        n_test = self.sim_params['test_per_class']
        n_class = self.sim_params['num_classes']
        n_dots = comb(n_class, 2)
        same_class_count = 0

        # ind_idx = np.random.randint(0, 1000)
        ind_idx = traj.individual.ind_idx
        generation = traj.individual.generation
        name = 'gen{:05d}_ind{:05d}'.format(generation, ind_idx)
        ind_params = {k: getattr(traj.individual, k) for k in ipr}
        print("ind_params:")
        print(ind_params)
        params = {
            'ind': ind_params,
            'sim': self.sim_params,
            'gen': {'gen': generation, 'ind': ind_idx},
        }

        snn = Decoder(name, params)
        data = snn.run_pynn()
        
        if data['died']:
            print(data['recs'])

        diff_class_dots = []
        min_v = -1.0 if data['died'] else 0.0
        apc, ipc = None, None
        if not data['died']:
            ### Analyze results
            dt = self.sim_params['sample_dt']
            out_spikes = data['recs']['output'][0]['spikes']
            labels = data['input']['labels']
            end_t = self.sim_params['duration']
            start_t = end_t - n_class * n_test * dt
            apc, ipc = analysis.spiking_per_class(labels, out_spikes, start_t, end_t, dt)

            print("\n\n\napc")
            print(apc)
            
            print("\nipc")
            print(ipc)

            diff_class_vectors = [np.zeros(n_out) for _ in apc]
            for c in apc:
                if len(apc[c]):
                    kv = np.array(list(apc[c].keys()), dtype='int')
                    diff_class_vectors[c - 1][kv] += 1

            # punish inactivity on output cells,
            # every test sample should produce at least one spike in
            # the output population
            any_zero = False
            all_zero = True
            n_out_class = 0
            for v in diff_class_vectors:
                if np.sum(v) > 0:
                    all_zero = False
                    n_out_class += 1
                    continue
                any_zero = True
        else:
            any_zero = True
            all_zero = True


        if (not all_zero) and (SOFT_ZERO_PUNISH or not any_zero):
            a = []

            for ix, x in enumerate(diff_class_vectors):
                for iy, y in enumerate(diff_class_vectors):
                    if iy > ix:
                        xnorm = np.sqrt(np.sum(x ** 2))
                        ynorm = np.sqrt(np.sum(y ** 2))
                        if xnorm <= 1e-9 or ynorm <= 1e-9:
                            dot = 0.0
                        else:
                            xnorm = x / xnorm
                            ynorm = y / ynorm

                            dot = np.sqrt(np.sum( (xnorm - ynorm) ** 2 )) / np.sqrt(2)
                        a.append(dot)

            diff_class_distances = np.asarray(a)
            diff_dist = np.mean(diff_class_distances)

            overlap = np.zeros_like(diff_class_vectors[0])
            for _class, v in enumerate(diff_class_vectors):
                overlap += v

            overlap_len = np.sum(overlap > 0)
            overlap[:] = overlap > 1
            if overlap_len < n_class:
                diff_class_overlap = 0.0
                diff_class_repr = 0
            else:
                diff_class_overlap = 1.0 - (np.sum(overlap)/overlap_len)
                # diff_class_overlap = 1.0 - (np.sum(overlap)/overlap_len)
                diff_class_repr = float(n_out_class) / float(n_class)
                # diff_class_overlap = overlap_len - np.sum(overlap)

            diff_class_norms = np.linalg.norm(diff_class_vectors, axis=1)
            print("{}\tdiff vectors - norms".format(name))
            print(diff_class_norms)

            a = []
            for ix, x in enumerate(diff_class_vectors):
                for iy, y in enumerate(diff_class_vectors):
                    if iy > ix:
                        dot = np.dot(x, y) / (diff_class_norms[ix] * diff_class_norms[iy])
                        a.append(dot)

            diff_class_dots = np.asarray(a)

            print("{}\tdiff dots".format(name))
            print(diff_class_dots)

            same_class_vectors = {c-1: [np.zeros(n_out) for _ in ipc[c]] for c in ipc}
            for c in ipc:
                for i, x in enumerate(ipc[c]):
                    for nid in ipc[c][x]:
                        same_class_vectors[c - 1][i][nid] = 1

            # punish inactivity on output cells,
            # every test sample should produce at least one spike in
            # the output population
            for c in same_class_vectors:
                for i, v in enumerate(same_class_vectors[c]):
                    if np.sum(v) > 0:
                        continue
                    any_zero = True
                    break

            if SOFT_ZERO_PUNISH or not any_zero:
                same_class_norms = {c: np.linalg.norm(same_class_vectors[c], axis=1) \
                                                            for c in same_class_vectors}

                print("{}\tsame vectors - norms".format(name))
                print(same_class_norms)
                same_class_dots = {}
                same_class_distances = {}
                same_class_count = 0.0
                for c in same_class_vectors:
                    a = []
                    b = []
                    for ix, x in enumerate(same_class_vectors[c]):
                        for iy, y in enumerate(same_class_vectors[c]):
                            if iy > ix:
                                dot = np.dot(x, y) / (same_class_norms[c][ix] * same_class_norms[c][iy])
                                a.append(dot)
                                xnorm = x / np.sqrt(np.sum(x ** 2))
                                ynorm = y / np.sqrt(np.sum(y ** 2))
                                dist = np.sqrt(np.sum((xnorm - ynorm) ** 2)) / np.sqrt(2)
                                b.append(dist)
                                same_class_count += 1.0

                    same_class_dots[c] = np.asarray(a)
                    same_class_distances[c] = np.asarray(b)


                print("{}\tsame dots".format(name))
                print(same_class_dots)

            else:
                diff_class_dots = []
        else:
            diff_class_dots = []
        # except:
        #     print("Error in simulation, setting fitness to 0")
        #     diff_class_dots = []

        print("\n\nExperiment took {} seconds\n".format(time.time() - bench_start_t))

        vmin = -1.0 if all_zero else 0.0
        
        if len(diff_class_dots) == 0:# or any_zero:
            print("dots == 0, fitness = ", 0)
            same_class_vectors = []
            diff_class_vectors = [] ###already defined

            diff_class_norms = []
            diff_class_dots = []

            same_class_norms = []
            same_class_dots = []

            diff_class_fitness = min_v#n_dots
            same_class_fitness = min_v

            diff_class_distances = []
            same_class_distances = []
            diff_dist = min_v

            diff_class_overlap = min_v
            diff_class_repr = min_v

            apc = []

            n_dots = 0
            ipc = []
            same_class_count = 0

        else:
            if np.any(diff_class_norms == 0.):
                print("At least one of the norms was 0")
                whr = np.where(np.isnan(diff_class_dots))[0]
                if len(whr):
                    diff_class_dots[whr] = 1.0 # 1 means same vector == bad

                whr = np.where(np.isinf(diff_class_dots))[0]
                if len(whr):
                    diff_class_dots[whr] = 1.0 # 1 means same vector == bad

            # invert (1 - x) so that 0 == bad and 1 == good
            # diff_class_fitness = 1.0 - np.mean(diff_class_dots)
            diff_class_fitness = np.mean(1.0 - diff_class_dots)
            # diff_class_fitness = 1.0 - np.sum(diff_class_dots)
            # diff_class_fitness /= float(len(diff_class_dots))
            # print("diff_fitness %s - %s = %s"%(1, np.sum(diff_class_dots)/n_dots, diff_class_fitness))

            same_fitnesses = np.asarray([
                np.sum(same_class_dots[c]) if len(same_class_dots[c]) else 0.0 \
                                                for c in sorted(same_class_dots.keys())
            ])

            # 0 means orthogonal vector == bad for same class activity
            same_fitnesses[np.where(np.isnan(same_fitnesses))] = 0.0
            same_fitnesses[np.where(np.isinf(same_fitnesses))] = 0.0
            same_class_fitness = np.sum(same_fitnesses)
            same_class_fitness /= same_class_count

            print("same fitness ", same_class_fitness)



        data['analysis'] = {
            'aggregate_per_class': {
                'spikes': apc,
                'vectors': diff_class_vectors,
                'norms': diff_class_norms,
                'dots': diff_class_dots,
                'cos_dist': diff_class_fitness,
                'distances': diff_class_distances,
                'euc_dist': diff_dist,
                'fitness': diff_class_overlap,
                'num_dots': n_dots,
                'overlap_dist': diff_class_overlap,
                'class_dist': diff_class_repr,
            },
            'individual_per_class': {
                'spikes': ipc,
                'vectors': same_class_vectors,
                'norms': same_class_norms,
                'dots': same_class_dots,
                'fitness': same_class_fitness,
                'cos_dist': same_class_fitness,
                'distances': same_class_distances,
                'num_dots': same_class_count,
            },

        }

        fit0 = 0.35 * data['analysis']['aggregate_per_class']['overlap_dist'] + \
               0.35 * data['analysis']['aggregate_per_class']['class_dist'] + \
               0.2 * data['analysis']['aggregate_per_class']['euc_dist'] + \
               0.1 * data['analysis']['individual_per_class']['cos_dist']

        data['fitness'] = fit0
        ### Save results for this individual

        fname = 'data_{}.npz'.format(name)
        np.savez_compressed(os.path.join(results_path, fname), **data)
        time.sleep(0.1)


        # fit1 = data['analysis']['individual_per_class']['cos_dist']

        ### Clear big objects
        import gc

        data.clear()
        params.clear()

        del data
        del params

        gc.collect()
        print("Done running simulation")


        if queue is not None:
            queue.put([fit0])#, fit1,])
            return

        return [fit0]#, fit1,]
