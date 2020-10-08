from __future__ import (print_function,
                        unicode_literals,
                        division)
import random as pyrand
from glob import glob
import numpy as np
import os

import sys
import time
import datetime
from omnigloter import stdp_mech as __stdp__
from omnigloter import neuron_model as __neuron__
from omnigloter import config
from omnigloter import utils
from omnigloter.MaxDistanceFixedProbabilityConnector import \
    MaxDistanceFixedProbabilityConnector
from pynn_genn.random import NativeRNG, NumpyRNG, RandomDistribution

if config.SIM_NAME == config.SPINNAKER:
    import pyNN.spiNNaker as sim
elif config.SIM_NAME == config.GENN:
    import pynn_genn as sim

from pyNN.space import Grid3D

# import matplotlib.pyplot as plt




if config.DEBUG:
    class Logging:
        def __init__(self):
            pass
        def info(self, txt):
            sys.stdout.write(str(txt)+'\n')
            sys.stdout.flush()

    logging = Logging()
else:
    import logging

class Decoder(object):
    def __init__(self, name, params):
        self._network = None
        self.inputs = None
        self.in_shapes = None
        self.in_labels = None
        self.name = name
        self.params = params
        self.decode(params)
        logging.info("In Decoder init, %s"%name)

        self.np_rng = None # NumpyRNG(seed=config.SEED)
        self.rng = None # NativeRNG(self.np_rng, seed=config.SEED)
        
        self.initial_weights = None

        print('\n\n')
        env = os.environ
        for k in env:
            klow = k.lower()
            if 'gpu' in klow or 'cuda' in klow or 'cpu' in klow or \
               'proc' in klow or 'nodelist' in klow:
                print( '{} = {}'.format(k, env[k]) )


        if "GPU_DEVICE_ORDINAL" in os.environ:
            print("\n\n\nGPU_DEVICE_ORDINAL = {}\n\n".format(os.environ["GPU_DEVICE_ORDINAL"]))
        else:
            print("\n\n\nGPU_DEVICE_ORDINAL WAS NOT FOUND!\n\n")


        if "CUDA_VISIBLE_DEVICES" in os.environ:
            print("\n\n\nCUDA_VISIBLE_DEVICES = {}\n\n".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        else:
            print("\n\n\nCUDA_VISIBLE_DEVICES NOT FOUND!!!\n\n")

#        print("\n\n\n{}\n\n\n".format(os.environ))
        # pprint(params)

    def decode(self, params):
        self.np_rng = NumpyRNG(seed=config.SEED)
        self.rng = NativeRNG(self.np_rng, seed=config.SEED)

        self._network = {}
        self._network['timestep'] = params['sim'].get('timestep', config.TIMESTEP)
        self._network['min_delay'] = params['sim'].get('min_delay', config.TIMESTEP)
        self._network['run_time'] = params['sim']['duration']

        logging.info("\n\nCurrent time is: {}\n".format(datetime.datetime.now()))

        logging.info("Setting up simulator")

        setup_args = {
            'timestep': self._network['timestep'],
            'min_delay': self._network['min_delay'],
        }

        if config.SIM_NAME == config.GENN:
            name = self.name
            name = name[name.find('ind'):]
            setup_args['model_name'] = name
            # setup_args['model_name'] = self.name
            setup_args['backend'] = config.BACKEND

            if params['sim']['on_juwels']:
                ind_idx = self.params['gen']['ind']
                GPU_ID = os.environ["CUDA_VISIBLE_DEVICES"]
                try:
                    GPU_ID = int(GPU_ID)
                except:
                    from omnigloter import cuda_utils
#                    np.random.seed(None)
#                    GPU_ID = np.random.randint(0, 4)
                    time.sleep(config.RNG.randint(1, 5))
                    GPU_ID = cuda_utils.pick_gpu_lowest_memory()

                print("\n{}\n\nchosen gpu id = {}\n".format(
                    os.environ["CUDA_VISIBLE_DEVICES"], GPU_ID))

                setup_args['selected_gpu_id'] = GPU_ID
            else:
                setup_args['selected_gpu_id'] = config.GPU_ID

        sim.setup(**setup_args)

        if config.SIM_NAME == config.SPINNAKER:
            sim.set_number_of_neurons_per_core(__neuron__.IF_curr_exp_i, 150)
            sim.set_number_of_neurons_per_core(sim.SpikeSourceArray, 150)

        logging.info("\tGenerating spikes")
        self.in_labels, self.in_shapes, self.inputs = self.get_in_spikes(params)

        pops = {}
        logging.info("\tPopulations: Input")
        pops['input'] = self.input_populations()

        if params['sim']['use_gabor']:
            logging.info("\tPopulations: Gabor")
            self.gabor_shapes, pops['gabor'] = self.gabor_populations(params)

        logging.info("\tPopulations: Mushroom")
        self.mushroom_size(params)
        pops['mushroom'] = self.mushroom_population(params)

        logging.info("\tPopulations: Mushroom Inhibitory")
        pops['inh_mushroom'] = self.inh_mushroom_population(params)
        #pops['noise_mushroom'] = self.noise_mushroom_population(params)
        
        if config.GAIN_CONTROL:
            logging.info("\tPopulations: Gain Control")
            pops['gain_control'] = self.gain_control_population(params)
        
        if not config.TEST_MUSHROOM:
            logging.info("\tPopulations: Output")
            pops['output'] = self.output_population(params)

            logging.info("\tPopulations: Output Inhibitory")
            pops['inh_output'] = self.inh_output_population(params)

        self._network['populations'] = pops


        projs = {}
        if params['sim']['use_gabor']:
            logging.info("\tProjections: Input to Gabor")
            projs['input to gabor'] = self.in_to_gabor(params)

            logging.info("\tProjections: Gabor to Mushroom")
            projs['gabor to mushroom'] = self.gabor_to_mushroom(params)

            logging.info("\tProjections: Gabor sWTA")
            projs['wta_mushroom'] = self.wta_gabor(params)
        else:
            logging.info("\tProjections: Input to Mushroom")
            projs['input to mushroom'] = self.input_to_mushroom(params)

        if config.GAIN_CONTROL:
            logging.info("\tProjections: In to GainControl")
            projs['in to gain'] = self.input_to_gain(params)
            projs['gain to mushroom'] = self.gain_to_mushroom(params)

        logging.info("\tProjections: Mushroom sWTA")
        projs['wta_mushroom'] = self.wta_mushroom(params)
        #projs['noise to mushroom'] = self.noise_to_mushroom(params)
        
        if not config.TEST_MUSHROOM:
            logging.info("\tProjections: Mushroom to Output")
            projs['mushroom to output'] = self.mushroom_to_output(params)
        
            logging.info("\tProjections: Output sWTA")
            projs['wta_output'] = self.wta_output(params)


        self._network['projections'] = projs


    def get_in_spikes(self, params):
        db = pyrand.choice(params['sim']['database'])
        print(db)
        self._db_name = db
        path = os.path.join(params['sim']['spikes_path'], db)
        nclass = params['sim']['num_classes']
        nsamp = params['sim']['samples_per_class']
        prob_noise = params['sim']['prob_noise']
        ntest = params['sim']['test_per_class']
        nepochs = params['sim']['num_epochs']
        # nlayers = params['sim']['input_layers']
        in_shape = params['sim']['input_shape']
        in_divs = params['sim']['input_divs']
        total_fs = nclass*nsamp*nepochs + nclass*ntest
        in_path = params['sim']['noisy_spikes_path']
        fname = "input_spikes_%s__width_%s_div_%s__nclass_%02d__nepoch_%04d__totalsamples_%010d.npz"%\
                (db, in_shape[0], in_divs[0], nclass, nepochs, total_fs)
        fname = os.path.join(in_path, fname)
#        print(fname)
        duration = params['sim']['duration']
        steps = params['sim']['steps']
        if os.path.isfile(fname):
            t_creation_start = time.time()

            with np.load(fname, allow_pickle=True) as data:
                labels=data['labels']
                shapes=data['shapes'].item()
                spikes = utils.split_ssa(
                    data['spikes'].item(), steps, duration, config.SIM_NAME==config.SPINNAKER)

                total_t_creation = time.time() - t_creation_start
                hours = total_t_creation // 3600
                minutes = (total_t_creation - hours * 3600) // 60
                seconds = total_t_creation - hours * 3600 - minutes * 60
                print("\tIt took %d:%d:%05.2f" % (hours, minutes, seconds))
                print(shapes)
#                 print(labels)
#                 print(len(spikes))
                return labels, shapes, spikes

        else:
            print("FILE NOT FOUND!!!!!!")

        fnames = []
        class_dirs = sorted(os.listdir(path))[:nclass]

        from random import shuffle

        e_fnames = []
        for e in range(nepochs):
            e_fnames[:] = []
            for cidx in class_dirs:
                cpath = os.path.join(path, cidx)
                files = sorted(glob(os.path.join(cpath, '*.npz')))
                for f in files[:nsamp]:
                    e_fnames.append(f)

            shuffle(e_fnames)

            fnames += e_fnames


        test_fnames = []
        for cidx in class_dirs:
            cpath = os.path.join(path, cidx)
            files = sorted(glob(os.path.join(cpath, '*.npz')))
            for f in files[nsamp:]:
                test_fnames.append(f)

        f = test_fnames[0]
        nlayers = 0
        with np.load(f, allow_pickle=True) as spk:
            nlayers = len(spk['kernels'].item())

        t_creation_start = time.time()
        tmp = {}
        labels = []
        spikes = {i: [] for i in range(nlayers)}
        shapes = {i: None for i in range(nlayers)}

        dt = params['sim']['sample_dt']
        dt_idx = 0
        total_fs = float(len(fnames) + len(test_fnames))
        for i, f in enumerate(fnames):

            with np.load(f, allow_pickle=True) as spk:
                all_divs = spk['keepers'].item()
                in_shape = spk['scaled_image'].shape
                labels.append(spk['label'].item())
                tmp.clear()
                tmp = utils.split_spikes(
                        #utils.compress_spikes_array(
                            spk['spike_source_array'],
                        #    start_t=0, end_t=10.,
                        #    randomize=False,
                        #    period=dt//3,
                        #    decimals=0),
                        nlayers)

                for tidx in range(nlayers):
                    # divs = (1, 1) if tidx < 2 else params['sim']['input_divs']
                    divs = all_divs[tidx]
                    # print(len([1 for ts in tmp[tidx] if ts.size > 0]))
                    shape, tmp[tidx][:] = utils.reduce_spike_place(tmp[tidx], in_shape, divs)
                    # print(len([1 for ts in tmp[tidx] if ts.size > 0]))
                    if shapes[tidx] is None:
                        shapes[tidx] = shape

                    # print(len([1 for ts in tmp[tidx] if ts.size > 0]))
                    tmp[tidx][:] = utils.randomize_ssa(tmp[tidx], config.SAMPLE_OFFSET, 
                                                    config.SAMPLE_MAX_T, 
                                                    decimals=1)
                    # print(len([1 for ts in tmp[tidx] if ts.size > 0]))
                    tmp[tidx][:] = utils.add_noise(prob_noise, tmp[tidx], 0., dt*0.5)
                    # print(len([1 for ts in tmp[tidx] if ts.size > 0]))
                    if len(spikes[tidx]) == 0:
                        # print('first spikes set')
                        # print(tmp[tidx])
                        spikes[tidx][:] = tmp[tidx]
                        # print(np.mean([len(ts) for ts in spikes[tidx][:]]))
                    else:
                        # print(np.mean([len(ts) for ts in spikes[tidx][:] ]))
                        spikes[tidx][:] = utils.append_spikes(spikes[tidx], tmp[tidx], dt_idx*dt)
                        # print(np.mean([len(ts) for ts in spikes[tidx][:] ]))

                        no_small = 0
                        for ts in spikes[tidx]:
                            whr = np.where(np.asarray(ts) < dt)[0]
                            if len(whr) == 0:
                                no_small += 1
                            else:
                                break
                        # print("no small ", no_small == len(spikes[tidx]))
                        if no_small == len(spikes[tidx]):
                            print(i, dt_idx)
                            sys.exit()

            dt_idx += 1
            sys.stdout.write("\r\t\tTrain %s -> %s %06.2f%%"%(
                dt_idx*dt, dt_idx * dt + dt * 0.5, 100.0 * dt_idx / total_fs))
            sys.stdout.flush()


        # plt.close('all')
        # plt.figure()
        # plt.hist(labels[-nclass*nsamp:], bins=nclass)
        # plt.savefig("label_histogram_last.pdf")

        # plt.show()

        shuffle(test_fnames)
        shuffle(test_fnames)
        for f in test_fnames:
            with np.load(f, allow_pickle=True) as spk:
                all_divs = spk['keepers'].item()
                in_shape = spk['scaled_image'].shape
                labels.append(spk['label'].item())
                tmp.clear()
                tmp = utils.split_spikes(
        #                utils.compress_spikes_array(
                            spk['spike_source_array'],
        #                    start_t=0, end_t=500.,
        #                    randomize=False,
        #                    period=dt//3,
        #                    decimals=0),
                        nlayers)

                for tidx in range(nlayers):
                    # divs = (1, 1) if tidx < 2 else params['sim']['input_divs']
                    divs = all_divs[tidx]
                    shape, tmp[tidx][:] = utils.reduce_spike_place(tmp[tidx], in_shape, divs)
                    #if shapes[tidx] is None:
                    #    shapes[tidx] = shape

                    tmp[tidx][:] = utils.randomize_ssa(tmp[tidx], 
                                                    config.SAMPLE_OFFSET,
                                                    config.SAMPLE_MAX_T,
                                                    decimals=1)
                    tmp[tidx][:] = utils.add_noise(prob_noise, tmp[tidx], 0., dt*0.5)
                    #if spikes[tidx] is None:
                    #    spikes[tidx] = tmp[tidx]
                    #else:
                    spikes[tidx][:] = utils.append_spikes(spikes[tidx], tmp[tidx], dt_idx*dt)

            dt_idx += 1
            sys.stdout.write("\r\t\tTest %06.2f%%"%(100.0 * dt_idx / total_fs))
            sys.stdout.flush()

        total_t_creation = time.time() - t_creation_start
        hours = total_t_creation // 3600
        minutes = (total_t_creation - hours * 3600) // 60
        seconds = total_t_creation - hours * 3600 - minutes * 60
        print("\tIt took %d:%d:%05.2f" % (hours, minutes, seconds))

        np.savez_compressed(fname, labels=labels, shapes=shapes, spikes=spikes)

        return labels, shapes, utils.split_ssa(spikes, steps, duration, config.SIM_NAME==config.SPINNAKER)


    ### ----------------------------------------------------------------------
    ### -----------------          populations           ---------------------
    ### ----------------------------------------------------------------------

    def input_populations(self):
        if self.inputs is None:
            raise Exception("Input spike arrays are not defined")

        if 'populations' in self._network and 'input' in self._network['populations']:
            return self._network['populations']['input']

        ins = {}
        for i in self.inputs[0]:
            shp = tuple( [int(v) for v in list(self.in_shapes[i])] )
            s = len(self.inputs[0][i])
            p = sim.Population(shp, sim.SpikeSourceArray,
                               {'spike_times': self.inputs[0][i]},
                               label='input layer %s'%i)

            if 'input' in config.RECORD_SPIKES:
                p.record('spikes')

            ins[i] = p
        return ins

    def gabor_populations(self, params=None):
        if 'populations' in self._network and 'gabor' in self._network['populations']:
            return self.gabor_shapes, self._network['populations']['gabor']

        gs = {}
        stride = (params['ind']['stride'], params['ind']['stride'])
        pad = (params['sim']['kernel_pad'], params['sim']['kernel_pad'])
        k_shape = (params['sim']['kernel_width'], params['sim']['kernel_width'])
        ndivs = int(params['ind']['n_pi_divs'])
        _shapes = {
            i: utils.compute_region_shape(self.in_shapes[i], stride, pad, k_shape) \
                                                            for i in self.in_shapes
        }
        neuron_type = getattr(sim, config.GABOR_CLASS)
        for lyr in _shapes:
            lrd = gs.get(lyr, {})
            for row in np.arange(_shapes[lyr][config.HEIGHT]).astype('int'):
                lrc = lrd.get(row, {})
                for col in np.arange(_shapes[lyr][config.WIDTH]).astype('int'):
                    lrc[col] = sim.Population(ndivs, neuron_type, config.GABOR_CLASS,
                                label='gabor - {} ({}, {})'.format(lyr, row, col))
                    if 'gabor' in config.RECORD_SPIKES:
                        lrc[col].record('spikes')

                lrd[row] = lrc
            gs[lyr] = lrd

        return _shapes, gs


    def gain_control_population(self, params=None):
        k = 'gain_control'
        if 'populations' in self._network and k in self._network['populations']:
            return self._network['populations'][k]

        size = config.GAIN_CONTROL_SIZE
        params = config.GAIN_CONTROL_PARAMS
        neuron_type = sim.IF_curr_exp
        pop = {}
        for i in range(1):
            p = sim.Population(size, neuron_type, params, 
                               label='gain_control_{}'.format(i))
            if k in config.RECORD_SPIKES:
                p.record('spikes')
            
            if k in config.RECORD_VOLTAGES:
                p.record('v')

            pop[i] = p
        
        return pop

    def mushroom_population(self, params=None):
        if 'populations' in self._network and 'mushroom' in self._network['populations']:
            return self._network['populations']['mushroom']

        count = self.mushroom_size(params)
        radius = np.copy(params['ind']['conn_dist'])
        shapes = self.in_shapes
        divs = params['sim']['input_divs']
        nz = self.num_zones_mushroom(shapes, radius, divs)

        sys.stdout.write("\tMushroom size: {}\n".format(count))
        sys.stdout.flush()
        neuron_type = getattr(__neuron__, config.MUSHROOM_CLASS)
        _Y, _X = 0, 1
        n_y, n_x = int(nz[0][_Y]), int(nz[0][_X])
        n_z = int( count // (n_y * n_x) )
        shape_post = (n_x, n_y, n_z)
        count = int(np.prod(shape_post))
        self._mushroom_size = count

        ratioXY = float(shape_post[0]) / shape_post[1]
        ratioXZ = float(shape_post[0]) / shape_post[2]
        dx = shapes[0][_X] // float(n_x)
        dy = shapes[0][_Y] // float(n_y)
        structure = Grid3D(ratioXY, ratioXZ,
                           dx=dx, dy=dy, dz=0.001 * (1. / n_z))

        p = sim.Population(count, neuron_type, config.MUSHROOM_PARAMS,
                           structure=structure, label='mushroom')

        if 'mushroom' in config.RECORD_SPIKES:
            p.record('spikes')

        return p

    def mushroom_size(self, params=None):
        if hasattr(self, '_mushroom_size'):
            return self._mushroom_size

        count = 0
        if params['sim']['use_gabor']:
            gshapes = self.gabor_shapes
            ndivs = int(params['ind']['n_pi_divs'])
            for l in gshapes:
                count += int(gshapes[l][0]*gshapes[l][1]*ndivs)
        else:
            for i in self.inputs[0]:
                count += len(self.inputs[0][i])

        expand = params['ind']['expand']
        sys.stdout.write("\tCount: {}\tExpand: {}\n".format(count, expand))
        count = int(np.ceil(count * expand))
        self._mushroom_size = count
        return count


    def num_zones_mushroom(self, in_shapes, radius, divs):
        if hasattr(self, 'n_zones'):
            return self.n_zones

        total = 0
        nz = {}
        for k in in_shapes:
            d = (1, 1) if k < 2 else divs
            nz[k] = []
            for i in range(len(d)):
                r = np.floor((2.0 * radius + 1.) / d[i])
                r = max(1.0, r * (2./3.))
                nz[k].append( max(1.0, np.ceil(in_shapes[k][i]/r)) )

            total += np.prod(nz[k])
            # max_div = max(d[0], d[1])
            # r = max(1.0, np.round((2.0 * radius) / max_div))
            # nz[k] = [max(1.0, v//r) for v in in_shapes[k]]
            # total += np.prod(nz[k])

        nz['total'] = total

        self.n_zones = nz

        return nz

    def noise_mushroom_population(self, params=None):
        if ('populations' in self._network and
                'noise_mushroom' in self._network['populations']):
            return self._network['populations']['noise_mushroom']

        count = config.NOISE_MUSHROOM_SIZE

        rate = config.NOISE_MUSHROOM_RATE

        p = sim.Population(count,
                sim.SpikeSourcePoisson,
                {'rate': rate},
                label='noise mushroom')

        #if 'mushroom' in config.RECORD_SPIKES:
        #    p.record('spikes')

        return p


    def inh_mushroom_population(self, params=None):
        if 'populations' in self._network and 'inh_mushroom' in self._network['populations']:
            return self._network['populations']['inh_mushroom']

        # count = 0
        # if params['sim']['use_gabor']:
        #     gshapes = self.gabor_shapes
        #     ndivs = int(params['ind']['n_pi_divs'])
        #     for l in gshapes:
        #         count += int(gshapes[l][0]*gshapes[l][1]*ndivs)
        # else:
        #     for i in self.inputs:
        #         count += len(self.inputs[i])
        #
        # expand = params['ind']['expand']
        # count = int(count * expand * 0.25)
        radius = params['ind']['conn_dist']
        shapes = self.in_shapes
        divs = params['sim']['input_divs']
        nz = self.num_zones_mushroom(shapes, radius, divs)
        count = (int(nz['total']) ) * config.N_INH_PER_ZONE + 1
        print('MUSHROOM INH TOTAL NEURONS = {}'.format(count))

        neuron_type = getattr(sim, config.INH_MUSHROOM_CLASS)
        p = sim.Population(count, neuron_type, config.INH_MUSHROOM_PARAMS,
                           label='inh_mushroom')

        if 'inh_mushroom' in config.RECORD_SPIKES:
            p.record('spikes')

        return p


    def output_population(self, params=None):
        if 'populations' in self._network and 'output' in self._network['populations']:
            return self._network['populations']['output']

        try:
            neuron_type = getattr(sim, config.OUTPUT_CLASS)
        except:
            neuron_type = getattr(__neuron__, config.OUTPUT_CLASS)

        n_out = params['sim']['output_size']
        vr = config.OUTPUT_PARAMS['v_rest'] 
        vrm = config.OUTPUT_PARAMS['v_reset']
        dist_params = {'low': vrm, 'high': vr}
        print(dist_params)
        dist = 'uniform'
        rdist = RandomDistribution(dist, rng=config.NP_RNG, **dist_params)
         
        p = sim.Population(n_out, neuron_type, config.OUTPUT_PARAMS,
                           label='output')

        p.initialize(v=rdist)

        if 'output' in config.RECORD_SPIKES:
            p.record('spikes')

        if 'output' in config.RECORD_VOLTAGES:
            p.record('v')

        return p

    def inh_output_population(self, params=None):
        if 'populations' in self._network and 'inh_output' in self._network['populations']:
            return self._network['populations']['inh_output']

        neuron_type = getattr(sim, config.INH_OUTPUT_CLASS)
        n_out = config.N_INH_OUTPUT # params['sim']['output_size']
        p = sim.Population(n_out, neuron_type, config.INH_OUTPUT_PARAMS,
                           label='inh_output')

        if 'inh_output' in config.RECORD_SPIKES:
            p.record('spikes')

        return p

    def supervisor(self, params=None):
        if 'populations' in self._network and 'supervisor' in self._network['populations']:
            return self._network['populations']['supervisor']

        ntype = sim.StepCurrentSource
        nsources = config.N_CLASSES
        nsamp = config.N_SAMPLES * config.N_EPOCHS
        ntrain = nsources * nsamp

        labels = self.in_labels
        sample_dt = config.SAMPLE_DT
        sup_dt = config.SUP_DURATION
        on = config.SUP_CORRECT_AMPLITUDE
        off = config.SUP_WRONG_AMPLITUDE
        # start current input right before spikes start 
        start_t = config.SAMPLE_OFFSET - sup_dt - config.TIMESTEP 
        times = np.zeros((nsources, (nsamp * 2) + 1)) # we need to say when it goes on and off u_u
        amps = np.zeros((nsources, (nsamp * 2) + 1))
        for i, lbl in enumerate(labels):
            idx = i * 2
            # start of supervision (set high) 
            times[lbl][idx] = start_t + sample_dt * i
            amps[lbl][idx] = on
            
            # end of supervision (set low)
            times[lbl][idx + 1] = start_t +  sample_dt * i + sup_dt
            amps[lbl][idx + 1] = off


        return times, amps, sim.StepCurrentSource(times=times, amplitudes=amps)
                
        
    ### ----------------------------------------------------------------------
    ### -----------------          projections           ---------------------
    ### ----------------------------------------------------------------------

    def in_to_gabor(self, params=None):
        if 'projections' in self._network and 'input to gabor' in self._network['projections']:
            return self._network['projections']['input to gabor']

        gabor_weight = [params['ind'][s] for s in sorted(params['ind'].keys()) \
                                                    if s.startswith('gabor_weight')]
        stride = (int(params['ind']['stride']), int(params['ind']['stride']))
        pad = (int(params['sim']['kernel_pad']), int(params['sim']['kernel_pad']))
        k_shape = (int(params['sim']['kernel_width']), int(params['sim']['kernel_width']))
        ndivs = params['ind']['n_pi_divs']
        # 0 about the same as 180?
        adiv = (np.pi / ndivs)

        gabor_params = {
            'omega': [params['ind']['omega']],
            'theta': (np.arange(ndivs) * adiv).tolist(),
            'shape': k_shape,
        }
        pre_shapes = self.in_shapes
        pres = self.input_populations()
        post_shapes, posts = self.gabor_populations()

        projs = {}
        for i in self.in_shapes:
            lyrdict = projs.get(i, {})
            pre_shape = pre_shapes[i]
            pre_indices = utils.pre_indices_per_region(pre_shape, pad, stride, k_shape)
            pre = pres[i]
            for r in posts[i]:
                rowdict = lyrdict.get(r, {})
                for c in posts[i][r]:
                    k, conns = utils.gabor_connect_list(pre_indices[r][c], gabor_params, delay=1.0,
                                                  w_mult=gabor_weight[i])
                    ilist, elist = utils.split_to_inh_exc(conns)

                    if len(elist) == 0:
                        continue

                    post = posts[i][r][c]
                    econ = sim.FromListConnector(elist)
                    elabel = 'exc - in {} to gabor {}-{}'.format(i, r, c)
                    rowdict[c] = {
                        'exc': sim.Projection(pre, post, econ, sim.StaticSynapse(),
                                              label=elabel, receptor_type='excitatory')
                    }

                    if len(ilist) > 0:
                        icon = sim.FromListConnector(ilist)
                        ilabel = 'inh - in {} to gabor {}-{}'.format(i, r, c)
                        rowdict[c]['inh'] = sim.Projection(pre, post, icon, sim.StaticSynapse(),
                                                           label=ilabel, receptor_type='inhibitory')

                lyrdict[r] = rowdict
            projs[i] = lyrdict

        return projs

    def gabor_to_mushroom(self, params=None):
        if 'projections' in self._network and 'gabor to mushroom' in self._network['projections']:
            return self._network['projections']['gabor to mushroom']

        post = self.mushroom_population()
        prob = params['ind']['exp_prob']
        mushroom_weight = params['ind']['mushroom_weight']
        projs = {}
        pre_shapes, pres = self.gabor_populations()
        for lyr in pres:
            lyrdict = projs.get(lyr, {})
            for r in pres[lyr]:
                rdict = lyrdict.get(r, {})
                for c in pres[lyr][r]:
                    pre = pres[lyr][r][c]
                    rdict[c] = sim.Projection(pre, post,
                                sim.FixedProbabilityConnector(prob),
                                sim.StaticSynapse(weight=mushroom_weight),
                               label='gabor {}{}{} to mushroom'.format(lyr, r, c),
                                receptor_type='excitatory')
                lyrdict[r] = rdict
            projs[lyr] = lyrdict

        return projs

    def input_to_mushroom(self, params=None):
        if 'projections' in self._network and 'input to mushroom' in self._network['projections']:
            return self._network['projections']['input to mushroom']

        post = self.mushroom_population()
        prob = params['ind']['exp_prob']  # this now means how many input connections a post has
        weight = params['ind']['mushroom_weight']
        delay = 3.
        radius = float(np.copy(params['ind']['conn_dist']))
        shapes = self.in_shapes
        divs = params['sim']['input_divs']
        nz = self.num_zones_mushroom(shapes, radius, divs)

        conns = []
        iconns = []
        if config.ONE_TO_ONE_EXCEPTION == True:
            conns = utils.o2o_conn_list(shapes, nz, post.size, radius, prob, weight, delay)
        else:
            conns, iconns = utils.dist_conn_list(shapes, nz, post.size, radius, prob, weight, delay)

        syn = sim.StaticSynapse(weight=weight, delay=delay)
        self._in_to_mush_conns = conns
        self._in_to_mush_iconns = iconns
        projs = {}

        for k, pre in self.input_populations().items():
            if len(conns[k]):
                projs[k] = sim.Projection(pre, post,
        #                    MaxDistanceFixedProbabilityConnector(max_dist=radius,
        #                                                         probability=prob,
        #                                                         rng=self.rng),
                            sim.FromListConnector(conns[k]),
                            synapse_type=syn,
                            label='input to mushroom - {}'.format(k),
                            receptor_type='excitatory',
                            use_procedural=config.USE_PROCEDURAL,
                            num_threads_per_spike=16
                           )
         #       print(k)
                if config.INH_INPUT:
                    sim.Projection(pre, post,
        #                    MaxDistanceFixedProbabilityConnector(max_dist=radius,
        #                                                         probability=prob,
        #                                                         rng=self.rng),
                            sim.FromListConnector(iconns[k]),
                            synapse_type=syn,
                            label='input to mushroom INH - {}'.format(k),
                            receptor_type='inhX',
        #                    use_procedural=config.USE_PROCEDURAL,
        #                    num_threads_per_spike=16
                           )

            else:
                projs[k] = None

        return projs


    def noise_to_mushroom(self, params=None):
        if ('projections' in self._network and
            'noise to mushroom' in self._network['projections']):
            return self._network['projections']['noise to mushroom']

        pre = self.noise_mushroom_population()
        post = self.mushroom_population()
        w = config.NOISE_MUSHROOM_WEIGHT
        prob = config.NOISE_MUSHROOM_PROB

        p = sim.Projection(pre, post,
                sim.FixedProbabilityConnector(prob),
                sim.StaticSynapse(weight=w),
                label='noise to mushroom',
                receptor_type='excitatory',
                use_procedural=config.USE_PROCEDURAL,
                num_threads_per_spike=16)

        return p


    def wta_gabor(self, params=None):
        if 'projections' in self._network and 'wta_gabor' in self._network['projections']:
            return self._network['projections']['wta_gabor']

        iw = config.INHIBITORY_WEIGHT['gabor']
        projs = {}
        pre_shapes, pres = self.gabor_populations()
        for lyr in pres:
            lyrdict = projs.get(lyr, {})
            for r in pres[lyr]:
                rdict = lyrdict.get(r, {})
                for c in pres[lyr][r]:
                    pre = pres[lyr][r][c]
                    post = pres[lyr][r][c]

                    rdict[c] = sim.Projection(pre, post,
                                sim.AllToAllConnector(),
                                sim.StaticSynapse(weight=iw, delay=config.TIMESTEP),
                               label='wta gabor {}{}{}'.format(lyr, r, c),
                                receptor_type='inhibitory',
                                use_procedural=config.USE_PROCEDURAL,
                                num_threads_per_spike=16)
                lyrdict[r] = rdict
            projs[lyr] = lyrdict

        return projs


    def wta_mushroom(self, params=None):
        if 'projections' in self._network and 'wta_mushroom' in self._network['projections']:
            return self._network['projections']['wta_mushroom']

        prjs = {}
        exc = self.mushroom_population()
        inh = self.inh_mushroom_population()
        print('inh size ', inh.size)
        exp_size = params['ind']['expand']
        ew = config.EXCITATORY_WEIGHT['mushroom']# / exp_size
        iw = config.INHIBITORY_WEIGHT['mushroom']
        delay = config.TIMESTEP
        radius = params['ind']['conn_dist']
        shapes = self.in_shapes
        divs = params['sim']['input_divs']
        nz = self.num_zones_mushroom(shapes, radius, divs)
        n_post = int( exc.size // nz['total'] )
        d = config.TIMESTEP
#        pe = []
#        pi = []
#        for i, p_start in enumerate(range(0, exc.size, n_post)):
#            p_end = min(exc.size, p_start + n_post)
#            print(p_start, p_end, i)
#            p = sim.Projection(exc[p_start:p_end], inh[i:i+1],
#                        sim.AllToAllConnector(),
#                        sim.StaticSynapse(weight=ew, delay=d),
#                        label='mushroom to mushroom exc',
#                        receptor_type='excitatory')
#            pe.append(p)
#
#            p = sim.Projection(inh[i:i+1], exc[p_start:p_end],
#                        sim.AllToAllConnector(),
#                        sim.StaticSynapse(weight=ew, delay=d),
#                        label='mushroom to mushroom inh',
#                        receptor_type='inhibitory')
#            pi.append(p)
#
#        prjs['e to i'] = pe
#        prjs['i to e'] = pi

        # e2e_inh_conn = utils.wta_mush_conn_list_a2a(shapes, nz, exc.size, iw, config.TIMESTEP)
        #prjs['e to i'] = sim.Projection(exc, exc,
        #                     MaxDistanceFixedProbabilityConnector(max_dist=0.5,
        #                                                          probability=1.,
        #                                                          rng=self.rng),
        #                     sim.StaticSynapse(weight=iw,
        #                                       delay=config.TIMESTEP),
        #                     sim.FromListConnector(e2e_inh_conn),
        #                     label='mushroom to mushroom inh',
        #                     receptor_type='inhibitory',
        #                     use_procedural=config.USE_PROCEDURAL,
        #                     num_threads_per_spike=16
        #                    )


        icon, econ = utils.wta_mush_conn_list(shapes, config.N_INH_PER_ZONE, nz, exc.size, 
                                              iw, ew, config.TIMESTEP)
        prjs['e to i'] = sim.Projection(exc, inh,
                            sim.FromListConnector(econ),
                            label='mushroom to inh_mushroom',
                            receptor_type='excitatory')

        prjs['i to e'] = sim.Projection(inh, exc,
                            sim.FromListConnector(icon),
                            label='inh_mushroom to mushroom',
                            #receptor_type='inhibitory'
                            receptor_type='inhShunt',
                            )

        # prjs['e to i'] = sim.Projection(exc, inh,
        #                     sim.AllToAllConnector(),
        #                     sim.StaticSynapse(weight=ew, delay=config.TIMESTEP),
        #                     label='mushroom to inh_mushroom',
        #                     receptor_type='excitatory')
        #
        # prjs['i to e'] = sim.Projection(inh, exc,
        #                     sim.AllToAllConnector(),
        #                     sim.StaticSynapse(weight=config.INHIBITORY_WEIGHT['mushroom'],
        #                                       delay=config.TIMESTEP),
        #                     label='inh_mushroom to mushroom',
        #                     receptor_type='inhibitory')

        # prjs['e to e'] = sim.Projection(exc, exc,
        #                     sim.FixedProbabilityConnector(MUSH_SELF_PROB),
        #                     sim.StaticSynapse(weight=config.INHIBITORY_WEIGHT['mushroom'],
        #                                       delay=config.TIMESTEP),
        #                     label='inh_mushroom to mushroom',
        #                     receptor_type='inhibitory')

        return prjs

    def input_to_gain(self, params=None):
        if 'projections' in self._network and 'in to gain' in self._network['projections']:
            return self._network['projections']['in to gain']

        pres = self.input_populations()
        posts = self.gain_control_population()
        w_min = config.GAIN_CONTROL_MIN_W
        w_max = config.GAIN_CONTROL_MAX_W
        cutoff = config.GAIN_CONTROL_CUTOFF
        ps = {}
        conns = {}
        conn_list = []
        for lyr in pres:
            pre = pres[lyr]
            projs = {}
            for p in posts:
                post = posts[p]
                conn_list = utils.gain_control_list(pre.size, post.size, w_max, cutoff)
                # post_idx, w_max, n_cutoff
                #w = utils.gain_control_w(p, w_max, cutoff)
                #ps[p] = w
                pj = sim.Projection(pre, post,
                #            sim.AllToAllConnector(),
                #            sim.StaticSynapse(weight=w, delay=1.),
                            sim.FromListConnector(conn_list),
                            sim.StaticSynapse(),
                            label='in {} to gain {}'.format(lyr, p),
                            receptor_type='excitatory',
                #            use_procedural=config.USE_PROCEDURAL,
                #            num_threads_per_spike=16
                )
                projs[p] = pj
            conns[lyr] = projs
        np.savez_compressed('in_to_gain.npz', conns=ps, conn_list=conn_list)
        return conns


    def gain_to_mushroom(self, params=None):
        if 'projections' in self._network and 'gain to mushroom' in self._network['projections']:
            return self._network['projections']['gain to mushroom']

        wi = config.GAIN_CONTROL_INH_W
        pres = self.gain_control_population()
        post = self.mushroom_population()

        prj = {}
        for p in pres:
            pre = pres[p]
            pj = sim.Projection(pre, post,
                    sim.AllToAllConnector(),
                    sim.StaticSynapse(weight=wi, delay=1.),
                    receptor_type='inhibitory',
                    label='gain to mush inh',
                    use_procedural=config.USE_PROCEDURAL,
                    num_threads_per_spike=16
            )
            prj[p] = pj

        return prj

    def mushroom_to_output(self, params=None):
        if 'projections' in self._network and 'mushroom to output' in self._network['projections']:
            return self._network['projections']['mushroom to output']

        ind_par = params['ind']
        pre = self.mushroom_population()
        post = self.output_population()
        prob = ind_par['out_prob']
        exp_size = params['ind']['expand']
        max_w = ind_par['out_weight'] #/ exp_size
        min_w = ind_par['w_min_mult']
        # min_w = 0.#-ind_par['out_weight']
        conn_list = utils.output_connection_list(pre.size, post.size, prob,
                                                 max_w, 0.1, max_pre=config.MAX_PRE_OUTPUT# ,
                                                 #seed=config.SEED,
                                                )
        #make sure we initialize everything as excitatory weights
        conn_list[:, 2] = np.clip(conn_list[:, 2], 0., max_w*ind_par['w_max_mult'])

        if config.SAVE_INITIAL_WEIGHTS:
            self.initial_weights = conn_list

        tdeps = {k: ind_par[k] if k in ind_par else config.TIME_DEP_VARS[k] \
                                                for k in config.TIME_DEP_VARS}
#         print("time deps ", tdeps)
        tdep = getattr(__stdp__, config.TIME_DEP)(**tdeps)
        wdep = getattr(__stdp__, config.WEIGHT_DEP)(min_w, ind_par['w_max_mult']*max_w)
        stdp = getattr(__stdp__, config.STDP_MECH)(timing_dependence=tdep, weight_dependence=wdep)

        p = sim.Projection(pre, post, sim.FromListConnector(conn_list), stdp,
                           label='mushroom to output', receptor_type='excitatory')

#         p = sim.Projection(pre, post, sim.FixedProbabilityConnector(prob), stdp,
#                            label='mushroom to output', receptor_type='excitatory')

        return p


    def wta_output(self, params=None):
        if 'projections' in self._network and 'wta_output' in self._network['projections']:
            return self._network['projections']['wta_output']

        prjs = {}
        exc = self.output_population()
        inh = self.inh_output_population()
        ew = config.EXCITATORY_WEIGHT['output']
        iw = config.INHIBITORY_WEIGHT['output']

        prjs['e to i'] = sim.Projection(exc, inh,
                            sim.AllToAllConnector(),
                            sim.StaticSynapse(weight=ew, delay=config.TIMESTEP),
                            label='output to inh_output',
                            receptor_type='excitatory',
                            use_procedural=config.USE_PROCEDURAL,
                            num_threads_per_spike=16)

        prjs['i to e'] = sim.Projection(inh, exc,
                            sim.AllToAllConnector(),
                            sim.StaticSynapse(weight=iw, delay=config.TIMESTEP),
                            label='inh_output to output',
                            receptor_type='inhShunt',
                            #receptor_type='inhibitory',
                            use_procedural=config.USE_PROCEDURAL,
                            num_threads_per_spike=16)

        #prjs['i to e'] = sim.Projection(exc, exc,
        #                     sim.AllToAllConnector(),
        #                     sim.StaticSynapse(weight=iw, delay=config.TIMESTEP),
        #                     label='wta - output to output',
        #                     receptor_type='inhibitory',
        #                     #receptor_type='inhShunt',
        #                     use_procedural=config.USE_PROCEDURAL,
        #                     num_threads_per_spike=16)

        return prjs


    def _get_recorded(self, layer):
        data = {}
        if layer == 'input':
            pops = self.input_populations()
            for i in pops:
                data[i] = grab_data(pops[i])

        elif layer == 'gain_control':
            pops = self.gain_control_population()
            for i in pops:
                data[i] = grab_data(pops[i])

        elif layer == 'gabor':
            _, pops = self.gabor_populations()
            for i in pops:
                idict = data.get(i, {})
                for r in pops[i]:
                    rdict = idict.get(r, {})
                    for c in pops[i][r]:
                        rdict[c] = grab_data(pops[i][r][c])
                    idict[r] = rdict
                data[i] = idict

        else:
            data[0] = grab_data(self._network['populations'][layer])

        return data

    def run_pynn(self):
        net = self._network
        steps = self.params['sim']['steps']
        duration = self.params['sim']['duration']//steps
        # pprint(net)

        logging.info("\tRunning experiment for {} milliseconds".format(net['run_time']))

        #sys.stdout.write("\n\n\tRunning step {} out of {}\n\n".format(1, steps))
        #sys.stdout.flush()

        __died__ = False
        records = {}
        weights = {}


        for step, st in enumerate(sorted(self.inputs.keys())):
            ssa = self.inputs[st]
            pops = self.input_populations()
            if config.SIM_NAME == config.SPINNAKER:
                for layer in ssa:
                    pops[layer].set(spike_times=ssa[layer])

            sys.stdout.write("\n\n\tRunning step {} out of {}\t".format(step + 1, steps))
            sys.stdout.write("From {} to {} \n\n\n".format(st, st + duration))
            sys.stdout.flush()

            sim.run(duration)
            #try:
            #    sim.run(duration)
            #except Exception as inst:
            #    sys.stdout.write("\n\n\tExperiment died!!!\n\n")
            #    sys.stdout.write("Exception is {}\n\n\n".format(inst))
            #    sys.stdout.flush()
            #    __died__ = True
            #    break


        for pop in net['populations']:
            if pop in config.RECORD_SPIKES:
                try:
                    records[pop] = self._get_recorded(pop)
                except:
                    sys.stdout.write("\n\n\n\tUnable to get data from {}\n\n".format(pop))
                    sys.stdout.flush()
                    __died__ = True

        if config.SAVE_INITIAL_WEIGHTS:
            weights['initial'] = self.initial_weights

        if not __died__:
            for proj in net['projections']:
                if proj in config.RECORD_WEIGHTS:
                    if proj == 'input to mushroom':
                        weights[proj] = self._in_to_mush_conns
                    else:
                        weights[proj] = grab_weights(net['projections'][proj])

        try:
            sim.end()
        except Exception as inst:
            sys.stdout.write("Exception is {}\n\n\n".format(inst))
            sys.stdout.flush()

        ### todo: change start and end for labels and runtimes
        # dt = self.params['sim']['sample_dt']
        # nclass = self.params['sim']['num_classes']
        # ntrain = self.params['sim']['samples_per_class']
        # start_t = dt * nclass * ntrain
        # cls_labels = self.in_labels[int(start_t//dt):]
        # spk_p_class = spiking_per_class(cls_labels,
        #                                 records['output'][0]['spikes'],
        #                                 start_t, net['run_time'], dt),
        radius = self.params['ind']['conn_dist']
        divs = self.params['sim']['input_divs']
        shapes = self.in_shapes
        nz = self.num_zones_mushroom(shapes, radius, divs)

        data = {
            'recs': records,
            'weights': weights,
            'input': {
                'labels': self.in_labels,
                'spikes': self.inputs,
                'shapes': self.in_shapes,
                'n_zones': nz,
            },
            'params': self.params,
            'db_name': self._db_name,
            'died': __died__,
            # 'analysis':{
            #     'per_class': spk_p_class
            # }
        }

        if self.params['sim']['use_gabor']:
            data['gabor'] = {
                'shapes': self.gabor_shapes,
            }

        for p in net['populations']:
            del p

        for p in net['projections']:
            del p

        import gc
        self._network.clear()
        del self._network
        gc.collect()

        return data


def grab_data(pop):
    data = pop.get_data()
    try:
        spikes = spikes_from_data(data)
    except:
        spikes = []
    try:
        voltage = voltage_from_data(data)
    except:
        voltage = []
    return {'spikes': spikes, 'voltage': voltage}


def spikes_from_data(data):

    spikes = [[] for _ in range(len(data.segments[0].spiketrains))]
    for train in data.segments[0].spiketrains:
        spikes[int(train.annotations['source_index'])][:] = \
            [float(t) for t in train]
    return spikes


def voltage_from_data(data):
    #print(data.segments[0])
    #print(data.segments[0].filter(name='v')[0])
    volts = data.segments[0].filter(name='v')[0]
    volts = np.vstack(
                [np.asarray(row) for row in volts])
    #print(volts)
    #print(volts.shape)
    return volts
#    return [[[float(a), float(b), float(c)] for a, b, c in volts]]

def safe_get_weights(p):
    # try:
        return p.getWeights(format='array')
    # except:
    #     return []

def grab_weights(proj):
    if isinstance(proj, dict): #gabor connections are a lot! :O
        w = {}
        for k in proj:
            wk = {}
            if isinstance(proj[k], dict):
                for r in proj[k]:
                    wr = {}
                    if isinstance(proj[k][r], dict):
                        for c in proj[k][r]:
                            wc = {}
                            if isinstance(proj[k][r][c], dict):
                                for x in proj[k][r][c]:
                                    # print(k,r,c,x, proj[k][r][c][x])
                                    wc[x] = safe_get_weights(proj[k][r][c][x])
                            else:
                                # print(k, r, c, proj[k][r][c])
                                wc[-1] = safe_get_weights(proj[k][r][c])
                            wr[c] = wc
                    else:
                        # print(k, r, proj[k][r])
                        wr[-1] = safe_get_weights(proj[k][r])
                    wk[r] = wr
            else:
                # print(k, proj[k])
                wk[-1] = safe_get_weights(proj[k])

            w[k] = wk

        return w
    else:
        try:
            return safe_get_weights(proj)
        except:
            return []


