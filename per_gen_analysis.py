import matplotlib
matplotlib.use('Agg')

import numpy as np

import matplotlib.pyplot as plt

import os
from pprint import pprint
from matplotlib.lines import Line2D
from glob import glob
import sys
from datetime import datetime
import time
from scipy.special import comb



# PREFIX = 'GA'
PREFIX = 'GD'
# PREFIX = 'ES'

TIME_SUFFIX = datetime.now().strftime("%d-%m-%Y-%H-%M")
print("generating plot on {}".format(TIME_SUFFIX))

def plot_input_spikes(in_spikes, start_t, total_t, dt=1.0, img_shape=(28, 28), in_divs=(5, 3)):
    for lyr in sorted(in_spikes.keys()):
        s1 = in_spikes[lyr]
        w, h = img_shape if lyr < 2 else ((img_shape[0] // in_divs[0]) + 1, (img_shape[1] // in_divs[1]) + 1)
        num_images = int(np.ceil(total_t / float(dt)))
        imgs = [np.zeros((h, w)) for _ in range(num_images)]
        for idx, times in enumerate(s1):
            for t in sorted(times):
                r, c = int(idx // w), int(idx % w)
                if t < start_t or t >= start_t + total_t:
                    continue
                img_idx = int((t - start_t) // dt)
                if img_idx > num_images:
                    continue
                imgs[img_idx][r, c] += 1.0
        fw = 2
        plt.figure(figsize=(total_t * fw, fw))
        for idx, img in enumerate(imgs):
            ax = plt.subplot(1, total_t, idx + 1)
            plt.imshow(img)
        plt.show()

if len(sys.argv) == 1:
    input_path = os.path.abspath('./L2L-OMNIGLOT/run_results')

else:
    input_path = os.path.abspath(sys.argv[1])
    # base_dir = os.path.abspath('.')

base_dir = input_path

result_files = sorted(glob(os.path.join(input_path, 'data_*.npz')))

total_different = 1.0 #comb(14, 2)
total_same = 0.1 # 4 * 14 * 0.1
total = total_different + total_same

data = {}
with np.load(result_files[0], allow_pickle=True) as tmp:
    for k in tmp:
        try:
            data[k] = tmp[k].item()
        except:
            data[k] = tmp[k]


# print(data.keys())
pkeys = [k for k in sorted(data['params']['ind'].keys()) \
                            if not (k == 'w_max_mult')]


fit_fname_pat = "*_fitness_and_params_data_per_generation_*.npz"
fit_fnames = sorted(glob(os.path.join(base_dir, fit_fname_pat)))
last_fit_fname = fit_fnames[-1] if len(fit_fnames) else None
pre_comp = {}
last_gen = -1
try:
    with np.load(last_fit_fname, allow_pickle=True) as tmp:
        for k in tmp:
            try:
                pre_comp[k] = tmp[k].item()
            except:
                pre_comp[k] = tmp[k]

    last_gen = sorted(pre_comp['fitnesses'].keys())[-1]
    all_params = {k: pre_comp['params'][k]
                    for k in sorted(pre_comp['params'].keys())[:-1]}
    fitnesses = {k: pre_comp['fitnesses'][k]
                    for k in sorted(pre_comp['fitnesses'].keys())[:-1]}
    all_scores = []
    for k in sorted(pre_comp['fitnesses'].keys())[:-1]:
        all_scores += pre_comp['fitnesses'][k]

except:
    all_params = {}
    fitnesses = {}
    all_scores = []

data = {}
for rf in result_files[:]:
    sys.stdout.write("\r{}".format(rf))
    sys.stdout.flush()
    fn = os.path.basename(rf)
    fns = (fn.split('.')[0]).split('_')
    gen = int(fns[1].split('gen')[-1])

    if gen < last_gen:
        time.sleep(0.01)
        continue

    ind = int(fns[2].split('ind')[-1])

    data.clear()
    try:
        with np.load(rf, allow_pickle=True) as tmp:
            for k in tmp:
                try:
                    data[k] = tmp[k].item()
                except:
                    data[k] = tmp[k]
    except:
        continue
    ag = data['analysis']['aggregate_per_class']['fitness']
    ig = data['analysis']['individual_per_class']['fitness']
    # fit0 = 0.3 * data['analysis']['aggregate_per_class']['overlap_dist'] + \
    #        0.3 * data['analysis']['aggregate_per_class']['euc_dist'] + \
    #        0.3 * data['analysis']['aggregate_per_class']['class_dist']
    # fit1 = data['analysis']['individual_per_class']['cos_dist']
    # _fit = (fit0 + 0.1*fit1)#/2.0
    _fit = data['fitness']
    # _fit = ag + ig
    all_scores.append(_fit)
    l = fitnesses.get(gen, [])
    l.append(_fit)

    fitnesses[gen] = l

    ap = {k: data['params']['ind'][k] for k in pkeys}
    apl = all_params.get(gen, [])
    apl.append(ap)
    all_params[gen] = apl




fit_fname = "{}_fitness_and_params_data_per_generation_{}.npz".format(PREFIX, TIME_SUFFIX)
np.savez_compressed(os.path.join(base_dir, fit_fname),
                    fitnesses=fitnesses, params=all_params)

print()
n_bins = int(np.ceil(total / 5.0) + 1)
minimum = []
maximum = []
average = []
for g in fitnesses:
    minimum.append(np.min(np.clip(fitnesses[g], 0, np.inf)))
    maximum.append(np.max(np.clip(fitnesses[g], 0, np.inf)))
    average.append(np.mean(np.clip(fitnesses[g], 0, np.inf)))

#####################################################################
#####################################################################
#####################################################################

fw = 8
fig = plt.figure(figsize=(fw*np.sqrt(2), fw))
ax = plt.subplot(1, 1, 1)

for g in fitnesses:
    plt.plot(g * np.ones_like(fitnesses[g]), np.clip(fitnesses[g],0, np.inf), '.b', alpha=0.1)

plt.plot(np.asarray(maximum), '^', linestyle=':', label='max')
plt.plot(np.asarray(average), 'o', linestyle='-', label='avg')
plt.plot(np.asarray(minimum), 'v', linestyle='-.', label='min')

# plt.axhline(total, linestyle='--', color='magenta', linewidth=1)
# plt.axhline(total_different, linestyle='--', color='magenta', linewidth=0.5)
plt.axhline(0, linestyle='--', color='gray', linewidth=1)
ax.set_xlabel('generation')
ax.set_ylabel('fitness')
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.025))
ax.margins(0.1)
plt.tight_layout()
fname = "{}_fitness_per_generation_{}.pdf".format(PREFIX, TIME_SUFFIX)
plt.savefig(os.path.join(base_dir, fname))


#####################################################################
#####################################################################
#####################################################################


fw = 8
fig = plt.figure(figsize=(fw*np.sqrt(2), fw))
ax = plt.subplot(1, 1, 1)

plt.plot(np.asarray(maximum), '^', linestyle=':', label='max')

# plt.axhline(total, linestyle='--', color='magenta', linewidth=1)
# plt.axhline(total_different, linestyle='--', color='magenta', linewidth=0.5)
ax.set_xlabel('generation')
ax.set_ylabel('fitness')
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.025))
ax.margins(0.1)
plt.tight_layout()
fname = "{}_max_fitness_per_generation_{}.pdf".format(PREFIX, TIME_SUFFIX)
plt.savefig(os.path.join(base_dir, fname))

#####################################################################
#####################################################################
#####################################################################


n_ind = len(fitnesses[0])
epochs = len(fitnesses)
ncols = 3
nrows =  epochs//ncols + int(epochs % ncols > 0)
fw = 5
fig = plt.figure(figsize=(fw*ncols, fw*nrows))
plt.suptitle("Fitness histogram per generation\n")
for g in fitnesses:
#     if len(fitnesses[g]) < n_ind:
#         continue
    ax = plt.subplot(nrows, ncols, g+1)
    ax.set_title("Gen %d   n_ind %d"%(g+1, len(fitnesses[g])))
    plt.hist(fitnesses[g], range=(0.0, 1.0), bins=10)
#     ax.set_xticks(np.arange(0, total+11, 10))
ax.margins(0.1)
plt.tight_layout()
fname = "{}_histogram_per_gen_{}.pdf".format(PREFIX, TIME_SUFFIX )
plt.savefig(os.path.join(base_dir, fname))

#####################################################################
#####################################################################
#####################################################################

scores = np.asarray(all_scores)
argsort = np.argsort(scores)

n_params = len(pkeys)
n_figs = comb(n_params, 2)
n_cols = 3
n_rows = n_figs // n_cols + int(n_figs % n_cols > 0)
fw = 5.0
fig = plt.figure(figsize=(fw * n_cols * 1.25, fw * n_rows))
plt_idx = 1
accum_params = {k: [] for k in pkeys}
for g in all_params:
    for ind in all_params[g]:
        for k in ind:
            accum_params[k].append(ind[k])
alpha = np.clip(scores, 0, np.inf)
alpha = alpha / (1.0 + alpha)
for i in range(n_params):
    for j in range(i + 1, n_params):
        i_params = np.asarray(accum_params[pkeys[i]])
        j_params = np.asarray(accum_params[pkeys[j]])
        ax = plt.subplot(n_rows, n_cols, plt_idx)
        im = plt.scatter(i_params[argsort], j_params[argsort],
                c=scores[argsort],
                s= alpha[argsort] * 100.0,
#                 s=(100.0 - scores)+ 5.0,
#                 s=scores + 5.0,
                vmin=0.0, vmax=1.0,
                cmap='Spectral',
#                 alpha=0.7,
                linewidths=0,
                edgecolors='none',
        )
        plt.colorbar(im)

        ax.set_xlabel(pkeys[i])
        ax.set_ylabel(pkeys[j])

        plt_idx += 1

ax.margins(0.1)
plt.tight_layout()
fname = '{}_parameter_pairs_{}.pdf'.format(PREFIX, TIME_SUFFIX)
plt.savefig(os.path.join(base_dir, fname))
