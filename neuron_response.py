import matplotlib.pyplot as plt
import sys
import argparse
from pprint import pprint
import pynn_genn as sim
# import stdp_mech as __stdp__
from omnigloter import neuron_model as __neuron__

def plot_spiketrains(segment):
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '|')
        plt.ylabel(segment.name)
        plt.setp(plt.gca().get_xticklabels(), visible=False)

backend = 'genn'
neuron_class = __neuron__.IF_curr_exp_i
# heidelberg's brainscales seems to like these params
e_rev = 92 #mV
# e_rev = 500.0 #mV
# i_offsets = np.arange(0.0, 1.0, 0.05)
i_offsets = np.array([0.1])
n_neurons = i_offsets.size
### Neuron configuration
VTHRESH = -55.0
BASE_PARAMS = {
    'cm': 0.09,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'tau_m': 10.,  # ms
    'tau_refrac': 1.,  # ms
    'tau_syn_E': 3., # ms
    'tau_syn_I': 5., # ms
}

tau_thresh = 30.0
# tau_thresh = 50.0
mult_thresh = 1.8

base_params = BASE_PARAMS.copy()
base_params['v_threshold'] = VTHRESH  # mV
# OUTPUT_PARAMS['v_thresh_adapt'] = OUTPUT_PARAMS['v_threshold']
base_params['tau_threshold'] = tau_thresh
base_params['w_threshold'] = mult_thresh
base_params['tau_syn_I'] = 5.
base_params['i_offset'] = i_offsets
# base_params = {
#     'cm': 0.09,  # nF
#     'v_reset': -70.,  # mV
#     'v_rest': -65.,  # mV
#     'v_thresh': -55.,  # mV
#     'v_thresh_adapt': -55.0,
#     # 'e_rev_I': -e_rev, #mV
#     # 'e_rev_E': 0.,#e_rev, #mV
#     'tau_m': 10.,  # ms
#     'tau_refrac': 0.1,  # ms
#     'tau_syn_E': 1.0,  # ms
#     'tau_syn_I': 5.0,  # ms
#     'tau_thresh': 50.0,
#     'mult_thresh': 1.8,
#     'i_offset': i_offsets,
# }

sim_time = 250.0
timestep = 1.0

sim.setup(timestep=timestep, min_delay=timestep,
          backend='SingleThreadedCPU')



post = sim.Population(n_neurons, neuron_class(**base_params))
post.record('spikes')
post.record('v_thresh_adapt')


sim.run(sim_time)

data = post.get_data()

sim.end()

vthresh = data.segments[0].filter(name='v_thresh_adapt')

np.savez_compressed("genn_spiking_behaviour.npz", spikes=data.segments[0])

plt.figure()
ax = plt.subplot()
plt.plot(vthresh[0])
# plot_spiketrains(data.segments[0])
plt.grid()
plt.show()


