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
import copy

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
#         print(self.sim_params)

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


    def simulate(self, traj):
        # TODO: trying to run many (N) simulations per GPU
        #  in Juwels.
        #  We need to:
        #    - make N copies of the trajectory and
        #      slightly alter them
        #    - obtain the results and grade them accordingly
        #    - replace the current trajectory with the best one
        #    - return the winning fitness
        #  This will (temporarily) increase the population size
        #  and help explore the parameter space faster.
        n_sims = self.sim_params['num_sims']
        if n_sims == 1:
            return self.simulate_one(traj)

        n_params = len(traj.individual.keys)
        p_change = 1.0/n_params
        original_ind_idx = copy.copy(traj.individual.ind_idx)
        ipr = self.ind_param_ranges
        q = Queue()
        trajs = [traj.copy() for _ in range(n_sims)]
        n_inds = len(traj.individuals[0])
        for tid, t in enumerate(trajs):
            new_ind_idx = t.individual.ind_idx
            new_ind_idx *= n_sims
            new_ind_idx += tid
            t.individual.ind_idx = new_ind_idx

            if tid == 0:
                continue

            for k in t.individual.keys:
                if np.random.uniform(0., 1.) <= p_change:
                    print(tid, k)
                    dv = np.random.normal(0., 1.0)
                    k = k.split('.')[1]
                    v = utils.bound(
                            getattr(t.individual, k) + dv,
                            ipr[k])

                    setattr(t.individual, k, v)


        procs = [
            Process(target=self.simulate_one, args=(trajs[i], q))
            for i in range(n_sims)]

        for p in procs:
            p.start()


        for p in procs:
            p.join()

        res = []
        for _ in procs:
            res.append(q.get())

        wfits = []
        for r in res:
            wfits.append(np.sum(r))

        win_idx = int(np.argmax(wfits))
        for k in traj.individual.keys:
            k = k.split('.')[1]
            v = getattr(trajs[win_idx].individual, k)
            setattr(traj.individual, k, v)

        traj.individual.ind_idx = original_ind_idx

        return res[win_idx]






    def simulate_one(self, traj, queue=None):
        work_path = traj._parameters.JUBE_params.params['work_path']
        results_path = os.path.join(work_path, 'run_results')
        os.makedirs(results_path, exist_ok=True)

        bench_start_t = time.time()
        ipr = self.ind_param_ranges
#        for k in traj.par:
#            try:
#                print("{}:".format(k))
#                print("\t{}\n".format(traj.par[k]))
#            except:
#                print("\tNOT FOUND!\n")

        n_out = self.sim_params['output_size']
        n_test = self.sim_params['test_per_class']
        n_class = self.sim_params['num_classes']
        n_dots = comb(n_class, 2)
        same_class_count = 0

        # ind_idx = np.random.randint(0, 1000)
        ind_idx = traj.individual.ind_idx
        generation = traj.individual.generation
        name = 'gen{:010d}_ind{:010d}'.format(generation, ind_idx)
        ind_params = {k: getattr(traj.individual, k) for k in ipr}
#        print("ind_params:")
#        print(ind_params)
        params = {
            'ind': ind_params,
            'sim': self.sim_params,
            'gen': {'gen': generation, 'ind': ind_idx},
        }

        snn = Decoder(name, params)
        data = snn.run_pynn()
        data['binned'] = {}
        data['correlations'] = {}

        print("\n\nExperiment took {} seconds\n".format(time.time() - bench_start_t))

#         if data['died']:
#             print(data['recs'])
        overactive = 0.
        vmin = -1.0 if data['died'] else -0.5

        sum_dists = 0 
        n_empty = n_test * n_class
        n_sharing = n_out 
        n_spikes = n_out * n_empty

        diff_class_vectors = []
        diff_class_distances = []
        diff_class_fitness = vmin

        same_class_vectors = []
        same_class_distances = []

        same_class_fitness = vmin

        diff_class_overlap = vmin
        diff_class_repr = vmin
        apc, ipc = [], []

        any_zero, all_zero = False, False
        dt = self.sim_params['sample_dt']
        n_out = config.OUTPUT_SIZE
        labels = data['input']['labels']
        end_t = self.sim_params['duration']
        # start_t = end_t - n_class * n_test * dt
        start_t = analysis.get_test_start_t(data) 
        avg_class_sample_distance = -(n_test * n_class) ** 2
        avg_activity_error = (config.OUTPUT_SIZE - config.TARGET_ACTIVITY_PER_SAMPLE) ** 3
        avg_sharing_class_error = n_class ** 3
        avg_freq_error = (n_test * n_class - config.TARGET_FREQUENCY_PER_OUTPUT_NEURON) ** 3
        act_per_samp = []
        if not data['died']:
            ### Analyze results
            #from pprint import pprint
            #pprint(data['recs']['output'][0])
            out_spikes = data['recs']['output'][0]['spikes']
            apc, ipc = analysis.spiking_per_class(labels, out_spikes, start_t, end_t, dt)

            # print("\n\n\napc")
            # print(apc)

            # print("\nipc")
            # print(ipc)

            diff_class_vectors = analysis.diff_class_vectors(apc, n_out)
            # punish inactivity on output cells,
            # every test sample should produce at least one spike in
            # the output population
            any_zero, all_zero = analysis.any_all_zero(apc, ipc)
            # balance the push to activity with punishment for
            # over-active 'classes'
            above_thresh = analysis.vectors_above_threshold(
                    diff_class_vectors, config.ACTIVITY_THRESHOLD)

        print("any_zero, all_zero = {}, {}".format(any_zero, all_zero))
        if not all_zero:
            diff_class_distances = analysis.diff_class_dists(diff_class_vectors)
            max_dist = config.MAX_VECTOR_DIST
            diff_class_distances[:] = np.clip(diff_class_distances, 0., max_dist)/max_dist
            diff_class_overlap = analysis.overlap_score(apc, n_out)
            diff_class_repr = analysis.individual_score(ipc, n_test, n_class)

            same_class_vectors = analysis.same_class_vectors(ipc, n_out)

            same_class_distances = \
                        analysis.same_class_distances(same_class_vectors)

            # invert (1 - x) so that 0 == bad and 1 == good
            # diff_class_fitness = 1.0 - np.mean(diff_class_distances) 
            # punish overactive classes
            overactive = len(above_thresh) * config.ABOVE_THRESH_W
            diff_class_fitness = np.mean(diff_class_distances)

            same_fitnesses = np.asarray([ np.mean(same_class_distances[c])
                                    for c in sorted(same_class_distances.keys()) ])

            # 0 means orthogonal vector == bad for same class activity
            # same_class_fitness = np.sum(same_fitnesses) / (n_class * n_test)
            same_class_fitness = np.mean(same_fitnesses)
            print("same fitness ", same_class_fitness)

        if 'output' in data['recs']:
            start_t = analysis.get_test_start_t(data)
            dt = data['params']['sim']['sample_dt']  
            _spikes, _labels = analysis.get_test_spikes_and_labels(data) 
            n_empty = analysis.get_num_inactive(_labels, _spikes, start_t, dt) 
            n_sharing = analysis.num_neurons_sharing_class(_labels, _spikes, start_t, dt)
            n_spikes = analysis.get_num_spikes(_spikes)
            sum_dists = np.sum(diff_class_distances) 

            op = np.mean if bool(1) else np.sum

            act_per_samp = analysis.activity_per_sample(
                _labels, _spikes, start_t, dt)

            avg_class_sample_distance = op( analysis.per_sample_class_distance(
                act_per_samp, _labels, n_out) )
        
            
            avg_activity_error = op( analysis.error_sample_target_activity(
                config.TARGET_ACTIVITY_PER_SAMPLE, act_per_samp, power=1) )

            avg_sharing_class_error = op( analysis.neurons_sharing_class(
                _labels, _spikes, start_t, dt, power=2) )

            avg_freq_error = op( analysis.target_frequency_error(
                config.TARGET_FREQUENCY_PER_OUTPUT_NEURON, _spikes, power=1) )
            
            duration = data['params']['sim']['duration']
            output_binned = analysis.bin_spikes(data['recs']['output'][0]['spikes'], dt, 0, duration)
            data['binned']['output'] = output_binned
            labels = data['input']['labels']
            n_total_tests = n_test * n_class
            data['correlations']['output'] = analysis.spikes_correlations(
                                                analysis.bin_to_dict(
													output_binned[-n_total_tests:], 
													labels[-n_total_tests:]))

        if 'mushroom' in data['recs']:
            start_t = analysis.get_test_start_t(data)
            dt = data['params']['sim']['sample_dt']
            duration = data['params']['sim']['duration']
            mushroom_binned = analysis.bin_spikes(data['recs']['mushroom'][0]['spikes'], dt, 0, duration)
            data['binned']['mushroom'] = mushroom_binned
            labels = data['input']['labels']
            data['correlations']['mushroom'] = analysis.spikes_correlations(
                                                analysis.bin_to_dict(mushroom_binned, labels))
            

            
        data['analysis'] = {
            'aggregate_per_class': {
                'spikes': apc,
                'vectors': diff_class_vectors,
                'distances': diff_class_distances,
                'fitness': diff_class_fitness,
                'num_dots': n_dots,
                'overlap_dist': diff_class_overlap,
                'class_dist': diff_class_repr,
                'n_empty': n_empty,
                'n_sharing': n_sharing,
                'n_spikes': n_spikes,
                'sum_dists': sum_dists,
                'avg_sharing_class_error': avg_sharing_class_error,
                'avg_freq_error': avg_freq_error,
                'weights': {
                    #overlapping activity is present
                    'overlap_dist': config.OVERLAP_WEIGHT,
                    #how many classes are represented
                    'class_dist': config.REPRESENTATION_WEIGHT,
                    # different class distance
                    'fitness': config.DIFFERENT_CLASS_DISTANCE_WEIGHT,
                },
            },
#fit0 = avg_class_sample_distance - avg_activity_error - avg_sharing_class_error - avg_freq_error

            'individual_per_class': {
                'spikes': ipc,
                'vectors': same_class_vectors,
                'distances': same_class_distances,
                'fitness': same_class_fitness,
                'act_per_sample': act_per_samp,
                'avg_class_samp_distance': avg_class_sample_distance,
                'avg_activity_error': avg_activity_error,
                'avg_sharing_class_error': avg_sharing_class_error,
                'avg_freq_error': avg_freq_error,
                'weights': {
                    # same class distance
                    'fitness': config.SAME_CLASS_DISTANCE_WEIGHT,
                },
            },

        }


        # overlap of output vectors, ideally should be 0, so we inverted the average
        woverlap = data['analysis']['aggregate_per_class']['weights']['overlap_dist']
        # spikes active per test presentation, ideally should be 1 per presentation, average
        wclass = data['analysis']['aggregate_per_class']['weights']['class_dist']
        # MANHATTAN distance between output vectors, 1 is bad so we inverted the average
        wdiff = data['analysis']['aggregate_per_class']['weights']['fitness']
        # cosine distance between output vectors per class, 1 is bad so we inverted the average
        # data['analysis']['individual_per_class']['weights']['fitness'] = 1.0 - woverlap - wclass - wdiff
        wsame = data['analysis']['individual_per_class']['weights']['fitness']

        fit00 = woverlap * data['analysis']['aggregate_per_class']['overlap_dist'] + \
               wclass * data['analysis']['aggregate_per_class']['class_dist'] + \
               wdiff * data['analysis']['aggregate_per_class']['fitness'] + \
               wsame * data['analysis']['individual_per_class']['fitness'] - \
               overactive

        data['fitness0'] = fit00

        fit01 = sum_dists / (n_empty + n_sharing + n_spikes)
        data['fitness1'] = fit01

        fit0 = avg_class_sample_distance - avg_activity_error - avg_sharing_class_error - avg_freq_error
        #fit0 = - avg_activity_error - avg_sharing_class_error - avg_freq_error
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
        print("Fitness {}".format(fit0))
        print("Done running simulation\n\n\n")

        if queue is not None:
            queue.put([fit0])
            return



        return [fit0]#, fit1,]
