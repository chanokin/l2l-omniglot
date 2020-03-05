import os
import sys
import glob
from omnigloter import traj_utils as tutils

# path = '/home/gp283/l2l-omniglot/L2L-OMNIGLOT/run-num-test/per_gen_trajectories'
path = '/home/gp283/l2l-omniglot/FIT-TEMPS/run-num-test/per_gen_trajectories'

trajs = tutils.open_all_trajectories(path)
gens = sorted(trajs.keys())

last_gen = gens[-1]
last_traj = trajs[last_gen]

params = tutils.get_params(last_traj)
print(params)

fitnesses = tutils.get_fitnesses(last_traj)
print(fitnesses)