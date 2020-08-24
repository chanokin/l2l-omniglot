import numpy as np
import os
import sys
import glob
import copy
from . import config

HEIGHT, WIDTH = range(2)
ROWS, COLS = HEIGHT, WIDTH
PRE, POST, WEIGHT, DELAY = range(4)


def to_post(val, pad, stride):
    return ((val - pad) // stride)


def post_shape(val, stride, kernel_width):
    return (((val - kernel_width) // stride) + 1)


def randnum(vmin, vmax, div=None, rng=config.NATIVE_RNG):
    if isinstance(vmin, int):
        return randint_float(vmin, vmax, div, rng)
    v = rng.uniform(vmin, vmax)
    # print("RANDNUM: uniform(%s, %s) = %s"%(vmin, vmax, v))
    return v


def bound(val, num_range):
    if len(num_range) == 1:
        v = num_range[0]
    else:
        v = np.clip(val, num_range[0], num_range[1])
    # print("BOUND: (%s, %s, %s) -> %s"%(num_range[0], num_range[1], val, v))
    if np.issubdtype(type(num_range[0]), np.integer):
        v = np.round(v)
        # print("INT-BOUND %s"%v)

    return v


def randint_float(vmin, vmax, div=None, rng=config.RNG):
    rand_func = np.random if rng is None else rng
    if div is None:
        return np.float(rng.randint(vmin, vmax))
    else:
        return np.float(np.floor(np.floor(rng.randint(vmin, vmax) / float(div)) * div))


def compute_num_regions(shape, stride, padding, kernel_shape):
    ins = np.array(shape)
    s = np.array(stride)
    ks = np.array(kernel_shape)
    ps = post_shape(ins, s, ks)
    ps[WIDTH] = max(1, ps[WIDTH])
    ps[HEIGHT] = max(1, ps[HEIGHT])
    return int(ps[HEIGHT] * ps[WIDTH])


def compute_region_shape(shape, stride, padding, kernel_shape):
    ins = np.array(shape)
    s = np.array(stride)
    ks = np.array(kernel_shape)
    ps = post_shape(ins, s, ks).astype('int')
    ps[WIDTH] = max(1, ps[WIDTH])
    ps[HEIGHT] = max(1, ps[HEIGHT])

    return [ps[HEIGHT], ps[WIDTH]]


def n_neurons_per_region(num_in_layers, num_pi_divs):
    return num_in_layers * num_pi_divs


def n_in_gabor(shape, stride, padding, kernel_shape, num_in_layers, num_pi_divs):
    return compute_num_regions(shape, stride, padding, kernel_shape) * \
           n_neurons_per_region(num_in_layers, num_pi_divs)


def generate_input_vectors(num_vectors, dimension, on_probability, seed=None):
    n_active = int(on_probability * dimension)
    # np.random.seed(seed)
    # vecs = (np.random.uniform(0., 1., (num_vectors, dimension)) <= on_probability).astype('int')
    vecs = np.zeros((num_vectors, dimension))
    for i in range(num_vectors):
        indices = config.NATIVE_RNG.choice(
                    np.arange(dimension, dtype='int'), 
                    size=n_active, replace=False)
        vecs[i, indices] = 1.0
    #np.random.seed()
    return vecs


def generate_samples(input_vectors, num_samples, prob_noise, seed=None, method=None):
    """method='all' means randomly choose indices where we flip 1s and 0s with probability = prob_noise"""
    #np.random.seed(seed)

    samples = None

    for i in range(input_vectors.shape[0]):
        samp = np.tile(input_vectors[i, :], (num_samples, 1)).astype('int')
        if method == 'all':
            dice = config.NATIVE_RNG.uniform(0., 1., samp.shape)
            whr = np.where(dice < prob_noise)
            samp[whr] = 1 - samp[whr]
        elif method == 'exact':
            n_flips = int(np.mean(input_vectors.sum(axis=1)) * prob_noise)
            for j in range(num_samples):
                # flip zeros to ones
                indices = config.NATIVE_RNG.choice(
                            np.where(samp[j] == 0)[0], size=n_flips, replace=False)
                samp[j, indices] = 1

                # flip ones to zeros
                indices = config.NATIVE_RNG.choice(
                            np.where(samp[j] == 1)[0], size=n_flips, replace=False)
                samp[j, indices] = 0
        else:
            n_flips = int(np.mean(input_vectors.sum(axis=1)) * prob_noise) * 2
            for j in range(num_samples):
                indices = config.NATIVE_RNG.choice(
                            np.arange(input_vectors.shape[1]), size=n_flips, replace=False)
                samp[j, indices] = 1 - samp[j, indices]

        if samples is None:
            samples = samp
        else:
            samples = np.append(samples, samp, axis=0)

    np.random.seed()
    return samples


def samples_to_spike_times(samples, sample_dt, start_dt, max_rand_dt, seed=None,
                           randomize_samples=False):
    #np.random.seed(seed)
    t = 0
    spike_times = [[] for _ in range(samples.shape[-1])]
    if randomize_samples:
        indices = config.NATIVE_RNG.choice(
                    np.arange(samples.shape[0]), 
                    size=samples.shape[0],
                    replace=False)
    else:
        indices = np.arange(samples.shape[0])

    for idx in indices:
        samp = samples[idx]
        active = np.where(samp == 1.)[0]
        ts = t + start_dt + config.NATIVE_RNG.randint(
                                -max_rand_dt, max_rand_dt + 1, size=active.size)
        for time_id, neuron_id in enumerate(active):
            if ts[time_id] not in spike_times[neuron_id]:
                spike_times[neuron_id].append(ts[time_id])

        t += sample_dt
    #np.random.seed()
    return indices, spike_times

def gain_control_w(post_idx, w_max, n_cutoff=15):
    return w_max / (n_cutoff + 1.0 + post_idx)

def gain_control_list(input_size, horn_size, max_w, cutoff=0.75):
    n_cutoff = 15  # int(cutoff*horn_size)
    matrix = np.ones((input_size * horn_size, 4))
    matrix[:, 0] = np.repeat(np.arange(input_size), horn_size)
    matrix[:, 1] = np.tile(np.arange(horn_size), input_size)

    matrix[:, 2] = np.tile(gain_control_w(np.arange(horn_size), max_w, n_cutoff), input_size)

    return matrix

def o2o_conn_list(in_shapes, num_zones, out_size, radius, prob, weight, delay):
    print("in ONE TO ONE con_list")
    print(" pre shapes {}".format(in_shapes))
    print(" num zones {}".format(num_zones))
    print("out size {}".format(out_size))
    print("radius {}".format(radius))
    print("prob {}".format(prob))

    conns = [[] for _ in in_shapes]
    start_post = 0
    for pre_pop in in_shapes:
        height, width = in_shapes[pre_pop][0], in_shapes[pre_pop][1]
        max_pre = width * height
        for pre in range(max_pre):
            post = start_post + pre
            conns[pre_pop].append([pre, post, weight, delay])
            pc = (100.0 * (float(post+1.0) / out_size))
            sys.stdout.write("\r\tIn to Mushroom\t%6.2f%%" % pc)
            sys.stdout.flush()

        start_post += max_pre

    sys.stdout.write("\n")
    sys.stdout.flush()
    return conns

def dist_conn_list(in_shapes, num_zones, out_size, radius, prob, weight, delay):
    print("\tin dist_con_list")
    print("\t\tpre shapes {}".format(in_shapes))
    print("\t\tnum zones {}".format(num_zones))
    print("\t\tout size {}".format(out_size))
    print("\t\tradius {}".format(radius))
    print("\t\tprob {}".format(prob))
    # I changed the prob parameter to number of pre

    n_in = len(in_shapes)
    if n_in > 1:
        div = max(in_shapes[0][0]//in_shapes[2][0],
                  in_shapes[0][1]//in_shapes[2][1])
    else:
        div = 1.0

    conns = [[] for _ in range(n_in)]
    n_per_zone = int(out_size // num_zones['total'])
    zone_idx = 0
    for pre_pop in in_shapes:
        height, width = in_shapes[pre_pop][0], in_shapes[pre_pop][1]
        max_pre = width * height
        # how many rows and columns resulted from dividing in_shape / (2 * radius)
        nrows, ncols = int(num_zones[pre_pop][0]), int(num_zones[pre_pop][1])
        # select minimum distance (adjust for different in_shapes)
        _radius = np.round( np.round(radius) if pre_pop < 2 else int(np.round(radius)//div) )
        
        _max_n = (2. * _radius) ** 2
        _n_pre = min(_max_n, prob)

        for zr in range(nrows):
            # centre row in terms of in_shape
            pre_r = int( min(_radius + zr * 2 * _radius, height - 1) )
            # low and high limits for rows
            row_l, row_h = int(max(0, pre_r - _radius)), int(min(height, pre_r + _radius + 1))

            for zc in range(ncols):
                # centre column in terms of in_shape
                pre_c = int( min(_radius + zc * 2 * _radius, width - 1) )
                # low and high limits for columns
                col_l, col_h = int(max(0, pre_c - _radius)), int(min(width, pre_c + _radius + 1))

                # square grid of coordinates
                cols, rows = np.meshgrid(np.arange(col_l, col_h,),
                                         np.arange(row_l, row_h))

                if len(cols) == 0 or len(rows) == 0:
                    continue

                # how many indices to select at random
                # I changed the meaning of prob, now its how many pre neurons
                # n_idx = int(np.round(rows.size * prob))
                n_idx = int( min(_n_pre, cols.size) )
 
                # post population partition start and end
                start_post = int(zone_idx * n_per_zone)
                end_post = min(int(start_post + n_per_zone), out_size)

                # choose pre coords sets for each post neuron
                for post in range(start_post, end_post):
                    rand_indices = config.NP_RNG.choice(rows.size, size=n_idx, replace=False).astype('int')
                    # randomly selected coordinates
                    rand_r = rand_indices // rows.shape[1]
                    rand_c = rand_indices % rows.shape[1]

                    # randomly selected coordinates converted to indices
                    pre_indices = rows[rand_r, rand_c] * width + cols[rand_r, rand_c]
                    for pre_i in pre_indices:
                        if pre_i < max_pre:
                            w = np.clip(config.RNG.normal(weight, weight * 0.1), 0., np.inf)
                            d = 1# np.round(np.random.uniform(delay, delay + 10.))
                            conns[pre_pop].append((pre_i, post, w, d))
                        else:
                            print("pre is larger than max ({} >= {})".format(pre_i, max_pre))
                            print("pre_r, row_l, row_h = {} {} {}".format(pre_r, row_l, row_h))
                            print("pre_c, col_l, col_h = {} {} {}".format(pre_c, col_l, col_h))

                zone_idx += 1
                pc = (100.0 * (zone_idx) / num_zones['total'])
                sys.stdout.write("\r\tIn to Mushroom\t%6.2f%%" % pc)
                sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return conns

def get_pre_indices_dist(in_shape, post_row, post_col, radius):
    print("\tin get_pre_indices_dist")
    print("\t\tpre shape {}".format(in_shape))
    print("\t\tpost_row {}".format(post_row))
    print("\t\tpost_col {}".format(post_col))
    print("\t\tradius {}".format(radius))

    zr = post_row
    zc = post_col
    pad = radius
    height, width = in_shape[HEIGHT], in_shape[WIDTH]

    # centre row in terms of in_shape

    pre_r = int( min(pad + zr * 2 * radius, height - 1) )
    # low and high limits for rows
    row_l, row_h = int(max(0, pre_r - radius)), int(min(height, pre_r + radius))

    # centre column in terms of in_shape
    pre_c = int( min(pad + zc * 2 * radius, width - 1) )
    # low and high limits for columns
    col_l, col_h = int(max(0, pre_c - radius)), int(min(width, pre_c + radius))

    # square grid of coordinates
    cols, rows = np.meshgrid(np.arange(col_l, col_h,),
                             np.arange(row_l, row_h))


    return (rows * width + cols).flatten()


def wta_mush_conn_list(in_shapes, num_zones, out_size, iweight, eweight, delay):
    econns = []
    iconns = []
    n_per_zone = out_size // num_zones['total']
    zone_idx = 0
    for pre_pop in in_shapes:
        nrows, ncols = int(num_zones[pre_pop][0]), int(num_zones[pre_pop][1])
        for zr in range(nrows):
            for zc in range(ncols):
                start_post = int(zone_idx * n_per_zone)
                end_post = int(start_post + n_per_zone)
                for post in range(start_post, end_post):
                    econns.append((post, zone_idx, eweight, delay))
                    iconns.append((zone_idx, post, iweight, delay))

                zone_idx += 1
                pc = (100.0 * (zone_idx) / num_zones['total'])
                sys.stdout.write("\r\tWTA to Mushroom\t%6.2f%%" % pc)
                sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return iconns, econns

def wta_mush_conn_list_a2a(in_shapes, num_zones, out_size, iweight, delay):
    iconns = []
    n_per_zone = out_size // num_zones['total']
    zone_idx = 0
    for pre_pop in in_shapes:
        nrows, ncols = int(num_zones[pre_pop][0]), int(num_zones[pre_pop][1])
        for zr in range(nrows):
            for zc in range(ncols):
                start_post = int(zone_idx * n_per_zone)
                end_post = int(start_post + n_per_zone)
                for post in range(start_post, end_post):
                    for ipost in range(start_post, end_post):
                        iconns.append((post, ipost, iweight, delay))

                zone_idx += 1
                pc = (100.0 * (zone_idx) / num_zones['total'])
                sys.stdout.write("\r\tWTA to Mushroom\t%6.2f%%" % pc)
                sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return iconns

    
    
def output_connection_list(kenyon_size, decision_size, prob_active, active_weight,
                           inactive_scaling, need=None, max_pre=config.MAX_PRE_OUTPUT):
    n_pre = min(max_pre, kenyon_size)
    n_conns = n_pre * decision_size
    print("output_connection_list: n_conns = {}".format(n_conns))
    matrix = np.ones((n_conns, 4))
    if kenyon_size < max_pre:
        matrix[:, 0] = np.repeat(np.arange(kenyon_size), decision_size)
        matrix[:, 1] = np.tile(np.arange(decision_size), n_pre)
    else:
        for i in range(0, n_conns, n_pre):
            matrix[i:i + n_pre, 0] = config.NATIVE_RNG.randint(0, kenyon_size, size=n_pre)
            matrix[i:i + n_pre, 1] = i // n_pre

    #np.random.seed(seed)

    inactive_weight = active_weight * inactive_scaling
    scale = max(active_weight * 0.1, 0.00001)
    matrix[:, 2] = config.NATIVE_RNG.normal(loc=inactive_weight, 
                                            scale=scale,
                                            size=n_conns)

    dice = config.NATIVE_RNG.uniform(0., 1., size=(n_conns))
    active = np.where(dice <= prob_active)
    print(prob_active, len(active[0]), n_conns)
    scale *= 0.1
    matrix[active, 2] = config.NATIVE_RNG.normal(loc=active_weight, 
                                                # scale=active_weight * 0.05,
                                                 scale=scale,
                                                 size=active[0].shape)

    #np.random.seed()

    return matrix

DEFAULT_SPIKE_DIR = "/home/gp283/brainscales-recognition/codebase/images_to_spikes"
def load_mnist_spike_file(
        dataset, digit, index, base_dir=DEFAULT_SPIKE_DIR):
    if dataset not in ['train', 't10k']:
        dataset = 'train'

    return sorted(glob.glob(
        os.path.join(base_dir, "mnist-db/spikes", dataset, str(digit), '*.npz')))[index]


def load_omniglot_spike_file(dataset, character, index,
                             base_dir="/home/gp283/brainscales-recognition/codebase/images_to_spikes"):
    datasets = ['Alphabet_of_the_Magi', 'Cyrillic', 'Gujarati', 'Japanese_-katakana-',
                'Sanskrit', 'Japanese_-hiragana-', 'Korean', 'Malay_-Jawi_-_Arabic-', 'Balinese',
                'Latin', 'Mkhedruli_-Georgian-', 'Blackfoot_-Canadian_Aboriginal_Syllabics-', 'Grantha',
                'Asomtavruli_-Georgian-', 'Burmese_-Myanmar-', 'Armenian', 'Bengali', 'Anglo-Saxon_Futhorc',
                'Tifinagh', 'Ojibwe_-Canadian_Aboriginal_Syllabics-', 'Braille', 'Greek', 'Tagalog',
                'N_Ko', 'Early_Aramaic', 'Arcadian', 'Inuktitut_-Canadian_Aboriginal_Syllabics-', 'Futurama',
                'Hebrew', 'Syriac_-Estrangelo-']
    if dataset not in datasets:
        raise Exception('Dataset not found!')
    char = "character%02d" % character
    return sorted(glob.glob(
        os.path.join(base_dir, "omniglot/spikes", dataset, char, '*.npz')))[index]


def pre_indices_per_region(pre_shape, pad, stride, kernel_shape):
    ps = compute_region_shape(pre_shape, stride, pad, kernel_shape)
    hk = np.array(kernel_shape) // 2
    pres = {}
    for _r in range(pad[HEIGHT], pre_shape[HEIGHT], stride[HEIGHT]):
        post_r = int(to_post(_r, pad[HEIGHT], stride[HEIGHT]))
        if post_r < 0 or post_r >= ps[HEIGHT]:
            continue
        rdict = pres.get(post_r, {})
        for _c in range(pad[WIDTH], pre_shape[WIDTH], stride[WIDTH]):
            post_c = int(to_post(_c, pad[WIDTH], stride[WIDTH]))
            if post_c < 0 or post_c >= ps[WIDTH]:
                continue
            clist = rdict.get(post_c, [])

            for k_r in range(-hk[HEIGHT], hk[HEIGHT] + 1, 1):
                for k_c in range(-hk[WIDTH], hk[WIDTH] + 1, 1):
                    pre_r, pre_c = _r + k_r, _c + k_c
                    outbound = pre_r < 0 or pre_c < 0 or \
                               pre_r >= pre_shape[HEIGHT] or \
                               pre_c >= pre_shape[WIDTH]

                    pre = None if outbound else (pre_r * pre_shape[WIDTH] + pre_c)
                    clist.append(pre)
            rdict[post_c] = clist
        pres[post_r] = rdict

    return pres


def prob_conn_from_list(pre_post_pairs, n_per_post, probability, weight, delay, weight_off_mult=None):
    posts = np.unique(pre_post_pairs[:, 1])
    conns = []
    for post_base in posts:
        pres = pre_post_pairs[np.where(pre_post_pairs[:, 1] == post_base)]
        for i in range(n_per_post):
            for pre in pres:
                if config.RNG.uniform <= probability:
                    post = post_base * n_per_post + i
                    conns.append([pre, post, weight, delay])
                else:
                    if weight_off_mult is None:
                        continue

                    post = post_base * n_per_post + i
                    conns.append([pre, post, weight * weight_off_mult, delay])

    return np.array(conns)


def gabor_kernel(params):
    # adapted from
    # http://vision.psych.umn.edu/users/kersten/kersten-lab/courses/Psy5036W2017/Lectures/17_PythonForVision/Demos/html/2b.Gabor.html
    shape = np.array(params['shape'])
    omega = params['omega']  # amplitude1 (~inverse)
    theta = params['theta']  # rotation angle
    k = params.get('k', np.pi / 2.0)  # amplitude2
    sinusoid = params.get('sinusoid func', np.cos)
    normalize = params.get('normalize', True)

    r = np.floor(shape / 2.0).astype('int')

    # create coordinates
    [x, y] = np.meshgrid(range(-r[0], r[0] + 1), range(-r[1], r[1] + 1))
    # rotate coords
    ct, st = np.cos(theta), np.sin(theta)
    x1 = x * ct + y * st
    y1 = x * (-st) + y * ct

    gauss = (omega ** 2 / (4.0 * np.pi * k ** 2)) * np.exp(
        (-omega ** 2 * (4.0 * x1 ** 2 + y1 ** 2)) * (1.0 / (8.0 * k ** 2)))
    sinus = sinusoid(omega * x1) * np.exp(k ** 2 / 2.0)
    k = gauss * sinus

    if normalize:
        k -= k.mean()
        k /= np.sqrt(np.sum(k ** 2))

    return k


def gabor_connect_list(pre_indices, gabor_params, delay=1.0, w_mult=1.0):
    omegas = gabor_params['omega']
    omegas = omegas if isinstance(omegas, list) else [omegas]

    thetas = gabor_params['theta']
    thetas = thetas if isinstance(thetas, list) else [thetas]

    shape = gabor_params['shape']

    kernels = [gabor_kernel({'shape': shape, 'omega': o, 'theta': t})
               for o in omegas for t in thetas]

    conns = []
    for ki, k in enumerate(kernels):
        for pre_i, pre in enumerate(pre_indices):
            if pre is None:
                continue

            r = pre_i // shape[WIDTH]
            c = pre_i % shape[WIDTH]
            conns.append([pre, ki, k[r, c] * w_mult, delay])

    return kernels, conns


def split_to_inh_exc(conn_list, epsilon=float(1e-6)):
    e = []
    i = []
    for v in conn_list:
        if v[WEIGHT] > epsilon:
            e.append(v)
        elif v[WEIGHT] < epsilon:
            i.append(v)

    return i, e


def split_spikes(spikes, n_types):
    spikes_out = {k: [] for k in range(n_types)}
    n_per_type = spikes.shape[0] // n_types
    for type_idx in range(n_types):
        for nidx in range(n_per_type):
            arr = np.array(spikes[type_idx * n_per_type + nidx])
            # if arr.size > 0:
            #    print(arr)
            spikes_out[type_idx].append(arr)

        #print(len([1 for ts in spikes_out[type_idx] if ts.size > 0]))

    return spikes_out


def div_index(orig_index, orig_shape, divs):
    w = orig_shape[1] // divs[1] + int(orig_shape[1] % divs[1] > 0)
    r = (orig_index // orig_shape[1])
    r = int(r / float(divs[0]))
    c = (orig_index % orig_shape[1])
    c = int(c / float(divs[1]))
    return r * w + c


def reduce_spike_place(spikes, shape, divs):
    fshape = [shape[0] // divs[0] + int(shape[0] % divs[0] > 0),
              shape[1] // divs[1] + int(shape[1] % divs[1] > 0)]
    fspikes = [None for _ in range(fshape[0] * fshape[1])]
    for pre, times in enumerate(spikes):
        fpre = div_index(pre, shape, divs)
        if fspikes[fpre] is None:
            fspikes[fpre] = times
        else:
            fspikes[fpre] = np.append(fspikes[fpre], times)
            fspikes[fpre][:] = np.unique(sorted(fspikes[fpre]))

    return fshape, fspikes


def scaled_pre_templates(pre_shape, pad, stride, kernel_shape, divs):
    pre_indices = []
    for scale_divs in divs:
        _indices = pre_indices_per_region(pre_shape, pad, stride, kernel_shape)
        if scale_divs[0] == 1 and scale_divs[1] == 1:
            pre_indices.append(_indices)
        else:
            d = {}
            for r in _indices:
                dr = d.get(r, {})
                for c in _indices[r]:
                    _scaled = set(dr.get(c, list()))
                    for pre in _indices[r][c]:
                        _scaled.add(div_index(pre, pre_shape, scale_divs))
                    dr[c] = list(_scaled)
                d[r] = dr

            pre_indices.append(d)

    return pre_indices


def append_spikes(source, added, dt):
    for i, times in enumerate(added):
        if len(times) == 0:
            continue
        #print(type(source[i]), type(times)) 
        source[i] = np.append(source[i], np.sort(times + dt))
    return source


def add_noise(prob, spikes, start_t, end_t):

    n_toggle = int(len(spikes) * prob)
    #n_toggle = int(len(on_neurons) * prob) if n_toggle >= len(on_neurons) else n_toggle
    on_neurons = np.arange(len(spikes))#[i for i in range(len(spikes)) if len(spikes[i]) > 0]
    on_to_toggle = config.NATIVE_RNG.choice(on_neurons, size=n_toggle, replace=False)
    for tog in on_to_toggle:
        # this is awful
        spikes[tog] = np.array([])


    off_neurons = np.arange(len(spikes))#[i for i in range(len(spikes)) if len(spikes[i]) == 0]
    off_to_toggle = config.NATIVE_RNG.choice(off_neurons, size=n_toggle, replace=False)
    for tog in off_to_toggle:
        spikes[tog] = config.NATIVE_RNG.randint(start_t, end_t, size=(1,))


    return spikes

def split_ssa(ssa, n_steps, duration, round_times):
    if n_steps == 1:
        return {0.0: ssa}

    dt = duration // n_steps
    s = {}
    for loop, st in enumerate(np.arange(0, duration, dt)):
        sys.stdout.write("\rsplitting spikes to steps {:6.2f}%".format(100.0*float(loop+1)/float(n_steps)))
        sys.stdout.flush()
        et = st + dt
        s[st] = {}
        for layer in ssa:
            s[st][layer] = [] 
            for times in ssa[layer]:
                ts = np.asarray(times)
                whr = np.where(np.logical_and(st <= ts, ts < et))
                arr = np.round(ts[whr]) if round_times else ts[whr]
                s[st][layer].append(arr.tolist())

    sys.stdout.write("\n\n")
    sys.stdout.flush()

    return s

def randomize_ssa(ssa, start_t, end_t, decimals):
    return [np.around(
                config.NATIVE_RNG.uniform(start_t, end_t, size=ts.shape), 
                decimals=decimals) if ts.size > 0 else ts
                    for ts in ssa]
    
def compress_spikes_list(spikes_list, start_t, end_t, randomize=True, period=10, decimals=0):
    sl = np.array(spikes_list)
    whr = np.where(np.logical_and(start_t <= sl, sl < end_t))[0]
    if len(whr) == 0:
        return []

    if randomize:
        t = np.around(
                config.NATIVE_RNG.uniform(start_t, start_t + period),
                decimals=decimals)
    else:
        t = np.around(
                period * sl[whr]/float(end_t - start_t),
                decimals=decimals)
    return [t]

def compress_spikes_array(spikes, start_t, end_t, randomize=True, period=10, decimals=0):
    return np.asarray([
        compress_spikes_list(s, start_t, end_t, randomize, period, decimals)
            for s in spikes
    ])


def load_last_trajs(path):
    import pickle
    def g(txt):
        x = os.path.basename(txt).split('.bin')[0]
        return int( x.split('_')[-1] )

    files = glob.glob(os.path.join(path, '*.bin'))

    if not files:
        return {}

    gens = np.asarray([g(f) for f in files])
    max_gen = np.max(gens)
    rows = np.where(gens == max_gen)[0]
    last_fnames = {gens[r]: files[r] for r in rows}

    trajs = {k: pickle.load(open(last_fnames[k], 'rb'))\
                                    for k in last_fnames}

    trajs['generation'] = max_gen
    for k in trajs.keys():
        if k == 'generation':
            continue

        trajs[k].par['generation'] = k

    return trajs

def trajectories_to_individuals(trajs, target_number, optimizee, generation=-1):
    gen = sorted( [k for k in trajs if k != 'generation'] )[-1]
    inds = trajs[gen].individuals[gen]
    max_id = len(inds)
    n_inds = len(inds)
    if n_inds < target_number:
        from l2l.utils.individual import Individual
        for i in range(target_number - n_inds):
            zee = optimizee.bounding_func(
                    optimizee.create_individual())
            ind_idx = max_id + i + 1
            params = [{'individual.{}'.format(k): zee[k]} for k in zee]
            inds.append(Individual(ind_idx=ind_idx, params=params))

    return {generation: inds}
