import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def bin_spikes(spikes, dt, end_t):
    bs = [[[] for _ in range(len(spikes))] 
              for _ in np.arange(0, end_t, dt)]
    
    for i, st in enumerate(np.arange(0, end_t, dt)):
        et = st + dt
        for j, times in enumerate(spikes):
            ts = np.asarray(times)
            whr = np.where(np.logical_and(st <= ts, ts < et))[0]
            if len(whr):
                bs[i][j] += ts[whr].tolist()
    return bs

def norm_rate(binned_spikes, total=False):
    w = 1./len(spikes)
    rate = [0. for _ in binned_spikes]
    for i, bs in enumerate(binned_spikes):
        rate[i] = np.sum([1 if (not total) and len(times) else len(times)
                            for times in bs]) * w
    return rate
    
def unique_times(bin_spikes, dt):
    rt = [set() for _ in bin_spikes]
    for i, curr_bin in enumerate(bin_spikes):
        for ts in curr_bin:
            ts = np.asarray(ts)
            rt[i].update(set(ts))
        
    return [np.asarray(list(s)) for s in rt]

def collapse_times(bin_spikes, dt):
    ct = [[] for _ in bin_spikes]
    for i, curr_bin in enumerate(bin_spikes):
        for ts in curr_bin:
            ct[i] += ts
        
    return ct

def relative_collapse_times(bin_spikes, dt):
    ct = []
    for i, curr_bin in enumerate(bin_spikes):
        for ts in curr_bin:
            ct += (np.asarray(ts) - dt * i).tolist()
        
    return ct



dirs = sorted(glob.glob('./L2L-OMNIGLOT/mushroom_test/*'))
# print(dirs)
spikes = []
bs = []
rates = []
uts = []
for d in dirs[:]:
    dd = d.split('/')[-1]
    for f in sorted(glob.glob(os.path.join(d, '*.npz')))[:]:
        print(f)
        ff = os.path.basename(f)[:-4]
        with np.load(f, allow_pickle=True) as npz:
#             print(sorted(npz['recs'].item()['mushroom'][0]['spikes'].keys()))
            sim_ps = npz['params'].item()['sim']
            dt = sim_ps['sample_dt']
            start_t = 0
            end_t = 1400#0
            spikes[:] = npz['recs'].item()['mushroom'][0]['spikes']

            bs[:] = bin_spikes(spikes, dt, end_t)
            rates[:] = norm_rate(bs)
            uts[:] = relative_collapse_times(bs, dt)

            
            plt.figure(figsize=(16, 5))
            ax = plt.subplot(1, 2, 1)
            plt.suptitle(f"{dd}\n{ff}")
            plt.axhline(0.0, linestyle='--', color='gray')
            plt.plot(rates)
            ax.set_xlabel('sample')
            ax.set_ylabel('norm rate (total spikes/num neurons)')

            ax = plt.subplot(1, 2, 2)
            plt.hist(rates, bins=50)
            ax.set_xlabel('norm rate (total spikes/num neurons)')
            plt.savefig(f"raw_and_histogram_rates_{dd}_{ff}.png", dpi=150)
            
        
            plt.figure(figsize=(5, 5))
            ax = plt.subplot(1, 1, 1)
            plt.suptitle(f"{dd}\n{ff}")

            plt.hist(uts, bins=50)
            plt.savefig(f"histogram_spike_times_{dd}_{ff}.png", dpi=150)
            # plt.show()


