import matplotlib
matplotlib.use('Agg')

import os
import sys
import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("usage:\n\t"
              "python plot_fitness_per_gen.py path/to/input/file")
        sys.exit(1)

    fname = sys.argv[1]

    data = np.load(fname, allow_pickle=True)

    fitnesses = data['fitnesses'].item()

    fw = 4
    fig = plt.figure(figsize=(fw*1.5 , fw))
    ax = plt.subplot(1, 1, 1)

    average = []
    maximum = []
    for g in fitnesses:
        plt.plot(g * np.ones_like(fitnesses[g]), fitnesses[g], '.', color='orange', alpha=0.2)
        average.append(np.mean(fitnesses[g]))
        maximum.append(np.max(fitnesses[g]))
    plt.plot(np.asarray(maximum), linestyle='-', linewidth=2.0, label='Top')
    plt.plot(np.asarray(average), linestyle='-', linewidth=2.0, label='Average')
    # plt.plot(np.asarray(minimum), 'v', linestyle='-.', label='min')

    plt.axhline(1.0, linestyle='--', color='magenta', linewidth=1, label='Maximum')
    # plt.axhline(total_different, linestyle='--', color='magenta', linewidth=0.5)
    plt.axhline(0, linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel('generation')
    ax.set_ylabel('fitness')
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 0.4), ncol=3)
    ax.margins(0.05)
    plt.tight_layout()
    basedir = os.path.dirname(fname)
    fname = os.path.basename(fname)
    fname = "plot_of_{}.pdf".format(fname[:-4])
    plt.savefig(os.path.join(basedir, fname))



if __name__ == '__main__':
    main()
    sys.exit(0)