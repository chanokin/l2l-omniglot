import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys

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


def get_delta_t_per_sample(bin_spikes, dt):
    dts = []
    mins = []
    maxs = []
    for i, bs in enumerate(bin_spikes):
        mint = np.min([np.min(ts) for ts in bs if len(ts)])
        maxt = np.max([np.max(ts) for ts in bs if len(ts)])
        mins.append(mint - i*dt)
        maxs.append(maxt - i*dt)
        dts.append( maxt - mint )
    return dts, mins, maxs


def get_first_n_per_sample(bin_spikes, dt, n=5):
    firsts = [[] for _ in range(n)]
    for i, bs in enumerate(bin_spikes):
        all_ts = set()
        for ts in bs:
            all_ts |= set(ts)
        sorted_ts = np.asarray(sorted(list(all_ts))) - dt * i
        # print(sorted_ts)
        for i, t in enumerate(sorted_ts):
            if i == n:
                break
            firsts[i].append(t)
            
    return firsts





dirs = sorted(glob.glob('./L2L-OMNIGLOT/mushroom_test/*'))
# print(dirs)
spikes = []
bs = []
rates = []
uts = []
for d in dirs[:]:
    dd = d.split('/')[-1]
    sdd = dd.split('_')
    dd = '_'.join([sdd[0], sdd[-1], sdd[-2]])
    for f in sorted(glob.glob(os.path.join(d, '*.npz')))[:]:
        print(f)
        ff = os.path.basename(f)[:-4]
        with np.load(f, allow_pickle=True) as npz:
            
            plt.close('all')

#             print(sorted(npz['recs'].item()['mushroom'][0]['spikes'].keys()))
            sim_ps = npz['params'].item()['sim']
            dt = sim_ps['sample_dt']
            start_t = 0
            end_t = 14000
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

            plt.tight_layout()
            plt.savefig(f"raw_and_histogram_rates_{dd}_{ff}.png", dpi=150)
# ----------------------------            

        
            plt.figure(figsize=(5, 5))
            ax = plt.subplot(1, 1, 1)
            plt.suptitle(f"{dd}\n{ff}")

            plt.hist(uts, bins=50)

            plt.tight_layout()
            plt.savefig(f"histogram_spike_times_{dd}_{ff}.png", dpi=150)
# ----------------------------            

            
            dts, mins, maxs = get_delta_t_per_sample(bs, dt)
        
            plt.figure(figsize=(17, 5))
            ax = plt.subplot(1, 3, 1)
            ax.set_title(f"min spike times")
            plt.hist(mins, bins=50)
        
            ax = plt.subplot(1, 3, 2)
            ax.set_title(f"max spike times")
            plt.hist(maxs, bins=50)
        
            ax = plt.subplot(1, 3, 3)
            ax.set_title(f"(max - min) spike times")
            plt.hist(dts, bins=50)
            plt.tight_layout()
            plt.savefig(f"hist_min_max_times_{dd}_{ff}.png", dpi=150)
# ----------------------------            

            
            n = 5
            firsts = get_first_n_per_sample(bs, dt, n)
        
            plt.figure(figsize=(17, 5))
            for i in range(n):
                ax = plt.subplot(1, n, i + 1)
                if i == 0:
                    place = "st"
                elif i == 1:
                    place = "nd"
                else:
                    place = "th"

                ax.set_title(f"{i+1}{place} spike times")
                plt.hist(firsts[i], bins=50)
            
            plt.tight_layout()        
            plt.savefig(f"hist_spike_places_{dd}_{ff}.png", dpi=150)
# ----------------------------            

            # plt.show()


