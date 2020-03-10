import numpy as np
import matplotlib.pyplot as plt
import pynn_genn as sim


def spikes_from_data(data):
    spikes = [[] for _ in range(len(data.segments[0].spiketrains))]
    for train in data.segments[0].spiketrains:
        spikes[int(train.annotations['source_index'])][:] = \
            [float(t) for t in train]
    return spikes


np.random.seed(1)

sim.setup(timestep=1.0, min_delay=1.0)

max_t = 50 #ms
spikes_per_neuron = int(max_t * 0.1)
n_ssa = 5
n_neurons = [np.random.randint(100, 1000) for _ in range(n_ssa)]


sources = {}

for ssa_id in range(n_ssa):
    sources[ssa_id] = {}

    spike_times = []
    for nid in range(n_neurons[ssa_id]):
        n_spikes = int(np.random.choice(spikes_per_neuron))
        spikes = np.round(np.random.choice(max_t, size=n_spikes, replace=False)).tolist()
        spikes[:] = sorted(spikes)
        spike_times.append(sorted(spikes))

    sources[ssa_id]['times'] = spike_times

    ssa = sim.Population(n_neurons[ssa_id],
             sim.SpikeSourceArray(spike_times=spike_times), label='ssa %d'%ssa_id)
    ssa.record(['spikes'])

    sources[ssa_id]['pop'] = ssa



sim.run(int(1.5 * max_t))

for ssa_id in range(n_ssa):
    data = sources[ssa_id]['pop'].get_data()
    sources[ssa_id]['spikes'] = spikes_from_data(data)
    sources[ssa_id]['data'] = data

sim.end()

for ssa_id in range(n_ssa):

    plt.figure()
    for nid in range(n_neurons[ssa_id]):
        in_spikes = sources[ssa_id]['times']
        out_spikes = sources[ssa_id]['spikes']
        plt.plot(in_spikes[nid], nid * np.ones_like(in_spikes[nid]),
                 'xg', markersize=5, markeredgewidth=1)
        plt.plot(out_spikes[nid], nid * np.ones_like(out_spikes[nid]),
                 '+m', markersize=5, markeredgewidth=1)

plt.show()
