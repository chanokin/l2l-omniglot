import pynn_genn as sim
from omnigloter.neuron_model_genn import IF_curr_exp_i
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['r', 'g', 'b', 'o', 'k', 'm', 'c']

def plot_data(segExc, segment, figsize=None, title=None, ax=None):
    if ax is None:
        fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    else:
        ax0 = ax

    ax0.set_title(title)

    for t in segExc.spiketrains[0]:
        ax0.axvline(t, color='green', linestyle='--', linewidth=1.)

    max_v = np.max( segment.filter(name='v')[0] )
    for nid, times in enumerate(segment.spiketrains):
        ax0.plot(times, (float(0.9 * max_v) + nid*2) * np.ones_like(times), '.',
                 color=COLORS[nid])

    volts = segment.filter(name='v')[0]
    for nid, vs in enumerate(volts.T):
        # print(volts.shape)
        ax0.plot(vs, color=COLORS[nid])


timestep = 1.
runtime = 200.
sim.setup(timestep)

n_neurons = 3
preExc = sim.Population(n_neurons,
                        sim.SpikeSourcePoisson(rate=200),
                        label='input'
                        )
preExc.record('spikes')


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

postBoth = sim.Population(n_neurons,
                      IF_curr_exp_i(**parameters),
                      label='output both',
                      )
postBoth.record(['v', 'spikes'])


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
eProjBoth = sim.Projection(preExc, postBoth,
                       sim.OneToOneConnector(),
                       sim.StaticSynapse(weight=5.),
                       receptor_type='excitatory',
                       label='exc_proj_both',
                       )

iProj = sim.Projection(postStd, postStd,
                       sim.AllToAllConnector(),
                       sim.StaticSynapse(weight=-5.0),
                       receptor_type='inhibitory',
                       label='inh_proj',
                       )

sProj = sim.Projection(postShunt, postShunt,
                       sim.AllToAllConnector(),
                       sim.StaticSynapse(weight=-5.0),
                       receptor_type='inhShunt',
                       label='shunt_proj',
                       )


sProjB = sim.Projection(postBoth, postBoth,
                       sim.AllToAllConnector(),
                       sim.StaticSynapse(weight=-5.0),
                       receptor_type='inhShunt',
                       label='shunt_proj',
                       )

xProjB = sim.Projection(postBoth, postBoth,
                       sim.AllToAllConnector(),
                       sim.StaticSynapse(weight=-3.0),
                       receptor_type='inhX',
                       label='x_proj',
                       )


sim.run(runtime)
data_exc = preExc.get_data().segments[0]
data_std = postStd.get_data().segments[0]
data_shunt = postShunt.get_data().segments[0]
data_both = postBoth.get_data().segments[0]
sim.end()

figsize = (15, 15)
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True, sharey=True)

plot_data(data_exc, data_std, title='standard', figsize=figsize, ax=ax0)

# plt.tight_layout()
plot_data(data_exc, data_shunt, title='shunt', figsize=figsize, ax=ax1)

plot_data(data_exc, data_both, title='both', figsize=figsize, ax=ax2)
# plt.tight_layout()


# plt.tight_layout()

plt.show()