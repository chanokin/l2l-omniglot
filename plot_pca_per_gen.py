import matplotlib
matplotlib.use('Agg')

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

def param_matrix(par):
    # print(par)
    mtx = np.zeros( (len(par), len(par[0])) )
    for row, p in enumerate(par):
        for col, k in enumerate( sorted(p.keys()) ):
            mtx[row, col] = p[k]
    return mtx

def main():
    if len(sys.argv) != 2:
        print("usage:\n\t"
              "python plot_pca_per_gen.py path/to/input/file")
        sys.exit(1)

    fname = sys.argv[1]

    data = np.load(fname, allow_pickle=True)

    params = data['params'].item()
    basedir = os.path.dirname(fname)
    fname = os.path.basename(fname)

    for g in params:
        fw = 8
        fig = plt.figure(figsize=(fw * np.sqrt(2), fw))
        ax = plt.subplot(1, 1, 1)

        mtx = param_matrix(params[g])
        pca.fit(mtx)
        comps = pca.transform(mtx)

        plt.plot(comps[:, 0], comps[:, 1], '.')

        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
        ax.margins(0.1)
        plt.tight_layout()
        ffname = "plot_pca_of_{}_{:010d}.pdf".format(fname[:-4], g)
        plt.savefig(os.path.join(basedir, ffname))

        plt.close(fig)

if __name__ == '__main__':
    main()
    sys.exit(0)