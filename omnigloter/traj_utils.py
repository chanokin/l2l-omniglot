import pickle
import os
import sys
import glob

def open_traj(path):
    with open(path, 'rb') as t:
        return pickle.load(t)

def name2gen(txt):
    return int( (txt.split('.')[0]).split('_')[-1] )

def open_all_trajectories(path, pattern='Trajectory_final_*.bin'):
    fnames = sorted(glob.glob(os.path.join(path, pattern)))
    ts = {}
    for f in fnames:
        gen = name2gen(f)
        ts[gen] = open_traj(f)

    return ts

def get_params(traj):
    inds = traj.individuals
    all_params = {}
    for gen in inds:
        params = {}
        for i, p in enumerate(inds[gen]):
            i_params = {}
            for k in sorted(p.keys):
                sk = k.split('.')[-1]
                v = getattr(p, sk)
                i_params[sk] = v
            params[i] = i_params
        all_params[gen] = params
    return all_params

def get_fitnesses(traj):
    fits = traj.results._data['all_results']._data
    return {g: {v[0]: v[1] for v in fits[g]} for g in fits}

def combine_params_fitnesses(params, fitnesses):
    c = {}
    for k in params:
        c[k] = {
            'params': params[k],
            'fitness': fitnesses[k]
        }
    return c