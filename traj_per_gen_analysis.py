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
from omnigloter import traj_utils as tutils


# PREFIX = 'GA'
PREFIX = 'GD'
# PREFIX = 'ES'

TIME_SUFFIX = datetime.now().strftime("%d-%m-%Y-%H-%M")
print("generating plot on {}".format(TIME_SUFFIX))


if len(sys.argv) == 1:
    input_path = os.path.abspath('./L2L-OMNIGLOT/run-num-test/per_gen_trajectories')

else:
    input_path = os.path.abspath(sys.argv[1])
    # base_dir = os.path.abspath('.')

base_dir = input_path

result_files = sorted(glob(os.path.join(input_path, 'Trajectory_*.bin')))

total_different = 1.0 #comb(14, 2)
total_same = 0.1 # 4 * 14 * 0.1
total = total_different + total_same


traj = tutils.open_traj(result_files[-1])
params = tutils.get_params(traj)
d_fitnesses = tutils.get_fitnesses(traj)
gkeys = sorted( list(d_fitnesses.keys()) )
pkeys = [k for k in sorted(params[gkeys[0]][0].keys()) \
                            if not (k == 'w_max_mult')]


all_params = {}
all_scores = []
fitnesses = {}
for g in gkeys:
    sys.stdout.write("\rGeneration {}".format(g))
    sys.stdout.flush()

    lfit = []    
    apl = all_params.get(g, [])
    for ind_idx in sorted(params[g]):
        ap = {k: params[g][ind_idx][k] for k in pkeys}

        score = np.sum(d_fitnesses[g][ind_idx])
        lfit.append(score)
    
        all_scores.append(score)
        apl.append(ap)

    fitnesses[g] = lfit


    all_params[g] = apl



print()
n_bins = int(np.ceil(total / 5.0) + 1)
minimum = []
maximum = []
average = []

for g in gkeys:
    # minimum.append(np.min(np.clip(fitnesses[g],  0, np.inf)))
    # maximum.append(np.max(np.clip(fitnesses[g],  0, np.inf)))
    # average.append(np.mean(np.clip(fitnesses[g], 0, np.inf)))
    minimum.append(np.min(  fitnesses[g] ))
    maximum.append(np.max(  fitnesses[g] ))
    average.append(np.mean( fitnesses[g] ))



#####################################################################
#####################################################################
#####################################################################
print('plotting max fitness per generation')

fw = 8
fig = plt.figure(figsize=(fw*np.sqrt(2), fw))
ax = plt.subplot(1, 1, 1)

for g in gkeys:
    plt.plot(g * np.ones_like(fitnesses[g]), fitnesses[g], '.b', alpha=0.3)
#    plt.plot(g * np.ones_like(fitnesses[g]), np.clip(fitnesses[g],0, np.inf), '.b', alpha=0.3)

plt.plot(gkeys, np.asarray(maximum), linestyle=':', label='max')
plt.plot(gkeys, np.asarray(average), linestyle='-', label='avg')
# plt.plot(gkeys, np.asarray(minimum), 'v', linestyle='-.', label='min')

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

print('plotting max fitness per generation')
fw = 8
fig = plt.figure(figsize=(fw*np.sqrt(2), fw))
ax = plt.subplot(1, 1, 1)

plt.plot(gkeys, np.asarray(maximum), linestyle=':', label='max')

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

print('plotting histograms')
kmin = int(np.min( list(fitnesses.keys()) ))
n_ind = len(fitnesses[kmin])
epochs = len(fitnesses)
ncols = 3
nrows =  epochs//ncols + int(epochs % ncols > 0)
fw = 5
fig = plt.figure(figsize=(fw*ncols, fw*nrows))
plt.suptitle("Fitness histogram per generation\n")
for g in gkeys:
#     if len(fitnesses[g]) < n_ind:
#         continue
    ax = plt.subplot(nrows, ncols, g+1)
    ax.set_title("Gen %d   n_ind %d"%(g+1, len(fitnesses[g])))
    plt.hist(fitnesses[g], bins=20)
#     ax.set_xticks(np.arange(0, total+11, 10))
ax.margins(0.1)
plt.tight_layout()
fname = "{}_histogram_per_gen_{}.pdf".format(PREFIX, TIME_SUFFIX )
plt.savefig(os.path.join(base_dir, fname))

#####################################################################
#####################################################################
#####################################################################
print('plotting parameter pairs')
print("len(all_scores) = {}".format(len(all_scores)))
scores = np.clip(np.asarray(all_scores), -196., np.inf)
argsort = np.argsort(scores)

n_params = len(pkeys)
n_figs = comb(n_params, 2)
n_cols = 3
n_rows = n_figs // n_cols + int(n_figs % n_cols > 0)
fw = 5.0
fig = plt.figure(figsize=(fw * n_cols * 1.25, fw * n_rows))
plt_idx = 1
accum_params = {k: [] for k in pkeys}
for g in gkeys:
    for ind in all_params[g]:
        for k in pkeys:
            accum_params[k].append(ind[k])
alpha = np.clip(scores, 0, np.inf)
alpha = alpha / (1.0 + alpha)
for i in range(n_params):
    for j in range(i + 1, n_params):
        i_params = np.asarray(accum_params[pkeys[i]])
        j_params = np.asarray(accum_params[pkeys[j]])
        # print("len({}_params) = {}".format(i, len(i_params)))
        # print("len({}_params) = {}".format(j, len(j_params)))
        
        ax = plt.subplot(n_rows, n_cols, plt_idx)
        im = plt.scatter(i_params[argsort], j_params[argsort],
                c=scores[argsort],
#                 s= alpha[argsort] * 100.0,
#                 s=(100.0 - scores)+ 5.0,
#                 s=scores + 5.0,
#                 vmin=0.0, vmax=1.0,
                cmap='jet',
#                 alpha=0.7,
                linewidths=1,
                edgecolors='black',
        )
        plt.colorbar(im)

        ax.set_xlabel(pkeys[i])
        ax.set_ylabel(pkeys[j])

        plt_idx += 1

ax.margins(0.1)
plt.tight_layout()
fname = '{}_parameter_pairs_{}.pdf'.format(PREFIX, TIME_SUFFIX)
plt.savefig(os.path.join(base_dir, fname))
