import os
import sys
import glob
import re
import shutil

def get_results_dirs(in_path):
    dirs = sorted( glob.glob(os.path.join(in_path, 'res*')) )
    return [d for d in dirs if d[-1].isdigit()]

def get_trajectories_dirs(in_path):
    dirs = sorted( glob.glob(os.path.join(in_path, 'traj*')) )
    return [d for d in dirs if d[-1].isdigit()]

def get_gen_ind(fname):
    '''/path/to/files/data_genXYZ_indABC.npz'''
    t = ( fname.split('.')[0] ).split('_')
    g = int( t[-2][3:] )
    i = int( t[-1][3:] )
    return g, i

def get_run(path):
    # from https://stackoverflow.com/questions/14471177/python-check-if-the-last-characters-in-a-string-are-numbers
    m = re.search(r'\d+$', path)
    # if the string ends in digits m will be a Match object, or None otherwise.
    if m is not None:
        # print(m.group(0))
        return int(m.group(0))
    else:
        raise Exception('No number found in run path')

def get_max_gen_in_run(run_dir):
    files = glob.glob(os.path.join(run_dir, 'data_gen*_ind*.npz'))
    max_gen = -1
    for f in files:
        igen, _ = get_gen_ind(f)
        if igen > max_gen:
            max_gen = igen

    return max_gen

def get_max_gen_first_run(results_dirs):
    min_idx, min_run = 1e10, 1e10
    for i, d in enumerate(results_dirs):
        r = get_run(d)
        if r < min_run:
            min_run = r
            min_idx = i

    min_dir = results_dirs[min_idx]
    max_gen = get_max_gen_in_run(min_dir)

    return max_gen, min_run

def merge_results(results_dirs, output_dir):
    res_out = os.path.abspath(os.path.join(output_dir, 'final_results'))
    os.makedirs(res_out, exist_ok=True)

    max_gen, min_run = get_max_gen_first_run(results_dirs)
    base_gen = 0
    for rid, rdir in enumerate( sorted(results_dirs) ):
        run = get_run(rdir)
        print(min_run, run, rdir)

        if min_run != run:
            min_run = run
            base_gen += max_gen
            max_gen = get_max_gen_in_run(rdir)

        files = sorted(glob.glob(os.path.join(rdir, 'data_gen*_ind*.npz')))
        for fid, fname in enumerate(files):
            gen, ind = get_gen_ind(fname)
            gen += base_gen

            out_fname = 'data_gen{:06d}_ind{:06d}.npz'.format(gen, ind)
            out_fpath = os.path.join(res_out, out_fname)

            sys.stdout.write('copying {} to \n\t{}\n'.format(fname, out_fname))
            sys.stdout.flush()
            shutil.copy2(fname, out_fpath)



def main():
    in_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    if os.path.isdir(in_dir):
        res_dirs = get_results_dirs(in_dir)

        os.makedirs(out_dir, exist_ok=True)

        merge_results(results_dirs=res_dirs, output_dir=out_dir)

    else:
        raise Exception('Cannot find input directory')


if __name__ == '__main__':
    main()




