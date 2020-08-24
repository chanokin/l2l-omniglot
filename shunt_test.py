import pynn_genn as sim
from omnigloter.neuron_model_genn import IF_curr_exp_i
import matplotlib.pyplot as plt
import numpy as np


def plot_data(segExc, segInh, segment, figsize=None, title=None):
    fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax0.set_title(title)

    for t in segInh.spiketrains[0]:
        plt.axvline(t, color='red', linewidth=1.)

    for t in segExc.spiketrains[0]:
        plt.axvline(t, color='green', linewidth=1.)

    max_v = np.max( segment.filter(name='v')[0] )
    ax0.plot(segment.spiketrains[0],
             0.99 * max_v * np.ones_like(segment.spiketrains[0]), '.b')

    ax0.plot(segment.filter(name='v')[0], color='blue')


timestep = 1.
runtime = 1000.
sim.setup(timestep)

n_neurons = 1
preExc = sim.Population(n_neurons,
                        sim.SpikeSourcePoisson(rate=100),
                        label='input'
                        )
preExc.record('spikes')

preInh = sim.Population(n_neurons,
                        sim.SpikeSourcePoisson(rate=100),
                        label='input'
                        )
preInh.record('spikes')

parameters = {
    'v_rest': -65.0,  # Resting membrane potential in mV.
    'cm': 1.0,  # Capacity of the membrane in nF
    'tau_m': 20.0,  # Membrane time constant in ms.
    'tau_refrac': 0.1,  # Duration of refractory period in ms.
    'tau_syn_E': 3.0,  # Decay time of excitatory synaptic current in ms.
    'tau_syn_I': 5.0,  # Decay time of inhibitory synaptic current in ms.
    'tau_syn_S': 5.0,
    'i_offset': 0.0,  # Offset current in nA
    'v_reset': -65.0,  # Reset potential after a spike in mV.
    'v_threshold': -50.0,  # Spike threshold in mV. STATIC, MIN
    'i': 0.0,  # nA total input current

    ### https://www.frontiersin.org/articles/10.3389/fncom.2018.00074/full
    # 'tau_thresh': 80.0,
    # 'mult_thresh': 1.8,
    ### https://www.frontiersin.org/articles/10.3389/fncom.2018.00074/full
    'tau_threshold': 1.0,
    'w_threshold': 1.0000000000,
    'v_thresh_adapt': -50.0,  # Spike threshold in mV.

}
postStd = sim.Population(n_neurons,
                      IF_curr_exp_i(**parameters),
                      label='output standard',
                      )
postStd.record(['v', 'spikes'])
postShunt = sim.Population(n_neurons,
                      IF_curr_exp_i(**parameters),
                      label='output shunt',
                      )
postShunt.record(['v', 'spikes'])

eProjStd = sim.Projection(preExc, postStd,
                       sim.OneToOneConnector(),
                       sim.StaticSynapse(weight=5.),
                       receptor_type='excitatory',
                       label='exc_proj_std',
                       )
eProjShunt = sim.Projection(preExc, postShunt,
                       sim.OneToOneConnector(),
                       sim.StaticSynapse(weight=5.),
                       receptor_type='excitatory',
                       label='exc_proj_shunt',
                       )

iProj = sim.Projection(preInh, postStd,
                       sim.OneToOneConnector(),
                       sim.StaticSynapse(weight=-5.0),
                       receptor_type='inhibitory',
                       label='inh_proj',
                       )
sProj = sim.Projection(preInh, postShunt,
                       sim.OneToOneConnector(),
                       sim.StaticSynapse(weight=-5.0),
                       receptor_type='inhShunt',
                       label='shunt_proj',
                       )

sim.run(runtime)
data_exc = preExc.get_data().segments[0]
data_inh = preInh.get_data().segments[0]
data_std = postStd.get_data().segments[0]
data_shunt = postShunt.get_data().segments[0]
sim.end()

figsize = (15, 5)

plot_data(data_exc, data_inh, data_shunt, title='shunt', figsize=figsize)
plt.tight_layout()

plot_data(data_exc, data_inh, data_std, title='standard', figsize=figsize)
plt.tight_layout()

plt.show()