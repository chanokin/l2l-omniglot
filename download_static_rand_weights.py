import pynn_genn as sim
backend = 'genn'
neuron_class = sim.IF_curr_exp

base_params = {
    'cm': 0.09,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'v_thresh': -55.,  # mV
    'tau_m': 10.,  # ms
    'tau_refrac': 2.0,  # ms
    'tau_syn_E': 1.0,  # ms
    'tau_syn_I': 5.0,  # ms
}

timestep = 0.1
max_w = 0.01
start_w = max_w / 2.0
min_delay = 0.1

sim.setup(timestep, min_delay=min_delay,
          backend='CUDA'
          # backend='SingleThreadedCPU'
          )

pre = sim.Population(10, sim.SpikeSourcePoisson(**{'rate': 10}))
post = sim.Population(100, neuron_class(**base_params))

# w = 1.0
w = sim.RandomDistribution('normal', [5.0, 1.0])
synapse = sim.StaticSynapse(weight=w)

prj = sim.Projection(pre, post, sim.FixedProbabilityConnector(0.25),
# prj = sim.Projection(pre, post, sim.AllToAllConnector(),
# prj = sim.Projection(pre, post, sim.OneToOneConnector(),
                     synapse_type=synapse, receptor_type='excitatory')
sim.run(100)
weights = prj.get('weight', format='array')

sim.end()

print(weights)
