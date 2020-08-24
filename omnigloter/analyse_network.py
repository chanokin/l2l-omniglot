import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import OrderedDict
from omnigloter import config

ZERO_FLOAT = 1.0e-9


def bin_spikes(spikes, dt, start_t, end_t):
    bins = np.arange(start_t, end_t, dt)
    
    bs = [[[] for _ in range(len(spikes))] 
              for _ in np.arange(start_t, end_t, dt)]

    for i, times in enumerate(spikes):
        inds = np.digitize(times, bins)
        for j in range(len(times)):
            b = inds[j] - 1
            bs[b][i].append(times[j])
    return bs


def bin_to_dict(bin_spikes, labels):
    un_lbl = np.unique(labels)
    vs = {l: [] for l in un_lbl}
    for i, s in enumerate(bin_spikes):
        v = [1 if len(spk) else 0 for spk in s]
        lbl = labels[i]
        vs[lbl].append(v)
    return vs

def spikes_correlations(vec_dict):
    vs = vec_dict
    cls = sorted(vs.keys())
    lens = [len(vs[c]) for c in cls]
    n_samp = int(np.sum(lens))
    over = np.ones((n_samp, n_samp)) * np.nan
    acc_lens = []
    for i0, c in enumerate(cls):
        if i0 == 0:
            acc_lens.append(0)
        else:
            acc_lens.append(lens[i0-1] + acc_lens[i0-1])

    for i0, c0 in enumerate(cls):
        for i1, c1 in enumerate(cls):
            if i1 < i0:
                continue
            for j0, v0 in enumerate(vs[c0]):
                for j1, v1 in enumerate(vs[c1]):
                    if j1 < j0:
                        continue
                    v0 = np.asarray(v0)
                    v1 = np.asarray(v1)
                    w0 = np.dot(v0, v0)
                    w1 = np.dot(v1, v1)

                    norm = np.sqrt(w0 * w1)
                    v = np.correlate(v0, v1)[0]
                    v = v / norm if norm > 0 else np.nan

                    over[acc_lens[i0] + j0, acc_lens[i1] + j1] = v
                    over[acc_lens[i0] + j1, acc_lens[i1] + j0] = v
                    over[acc_lens[i1] + j0, acc_lens[i0] + j1] = v
                    over[acc_lens[i1] + j1, acc_lens[i0] + j0] = v
    return over


def target_frequency_error(target, spikes, power=1):
    err = [np.clip(len(times) - target, 0, np.inf)**power for times in spikes]
    return err

def mean_target_frequency_error(target, spikes, power=1):
    return np.mean( target_frequency_error(target, spikes, power) )
    
def inter_class_distance(_activity_per_sample, labels, n_out):
    class_samples = {}
    max_active = 0
    for idx, lbl in enumerate(labels):
        lbl_list = class_samples.get(lbl, [])
        lbl_list.append( _activity_per_sample[idx])
        class_samples[lbl] = lbl_list
        
        if len(_activity_per_sample[idx]) > max_active:
            max_active = len(_activity_per_sample[idx])

    if max_active == 0:
        return [-( len(labels)**2 )]
    
    dists = []
    classes = sorted(class_samples.keys())
    pcd = {c: np.zeros_like(n_out) for c in classes}
    for c in classes:
        for s in class_samples[c]:
            if len(s) == 0:
                continue
            pcd[c][s] = 1

    for i0, c0 in enumerate(clss[:-1]):
        for c1 in clss[i0+1]:
            v0 = pcd[c0]
            v1 = pcd[c1]
            l0 = np.sum(v0)
            l1 = np.sum(v1)

            if l0 > 0 and l1 > 0:
                w = 1./np.sqrt(l0 + l1)
                d = ( np.sum( np.abs(v0 - v1) ) ) * w
            else:
                d = -1

            dists.append(d)

    return dists

           

def mean_per_sample_class_distance(_activity_per_sample, labels, n_out):
    return np.mean(
        per_sample_class_distance(_activity_per_sample, labels, n_out))

def per_sample_class_distance(_activity_per_sample, labels, n_out):
    class_samples = {}
    max_active = 0
    for idx, lbl in enumerate(labels):
        lbl_list = class_samples.get(lbl, [])
        lbl_list.append( _activity_per_sample[idx])
        class_samples[lbl] = lbl_list
        
        if len(_activity_per_sample[idx]) > max_active:
            max_active = len(_activity_per_sample[idx])

    if max_active == 0:
        return [-2.]
    
    v0 = np.zeros(n_out)
    v1 = np.zeros(n_out)
    dists = []
    classes = sorted(class_samples.keys())
    for idx0, cls0 in enumerate(classes[:-1]):
        for cls1 in classes[idx0+1:]:
            for samp0 in class_samples[cls0]:
                v0[:] = 0
                if len(samp0):
                    v0[samp0] = 1
                    
                for samp1 in class_samples[cls1]:
                    v1[:] = 0
                    if len(samp1):
                        v1[samp1] = 1
                    
                    if len(samp0) > 0 and len(samp1) > 0:
                        w = 1./(np.sqrt(len(samp0) + len(samp1)))
                        dists.append( np.sum(np.abs(v0 - v1)) * w )
                    else:
                        dists.append( -1.  )

    return dists


def error_sample_target_activity(target, _activity_per_sample, power=1.0, div=1.0):
    act = np.asarray([len(ids) for ids in _activity_per_sample])
    return ( ( np.abs( act - target ) * (1.0/div) ) ** power )


def mean_error_sample_target_activity(target, _activity_per_sample, power=1.0, div=1.0):
    err = error_sample_target_activity(target, _activity_per_sample, power, div)
    return np.mean( err ) 


def vectors_above_threshold(vectors, threshold):
    vs = [np.sum(v) for v in vectors]
    return [i for i, s in enumerate(vs) if s >= threshold]

def get_test_label_idx(data):
    n_class = data['params']['sim']['num_classes']
    n_train_per_class = data['params']['sim']['samples_per_class']
    n_epochs = data['params']['sim']['num_epochs']
    return n_class * n_train_per_class * n_epochs

def get_test_start_t(data):
    dt = data['params']['sim']['sample_dt']
    n_train = get_test_label_idx(data)
    return n_train * dt

def get_test_spikes_and_labels(data):
    spk = data['recs']['output'][0]['spikes']
    lbl = data['input']['labels']
    dt = data['params']['sim']['sample_dt']
    start_t = get_test_start_t(data)
    start_idx = get_test_label_idx(data)
    print(start_t, dt, start_idx * dt)
    out_spk = []
    out_ids = lbl[start_idx:]
    for times in spk:
        ts = np.asarray(times)
        whr = np.where(ts >= start_t)
        out_spk.append(ts[whr])
    
    return out_spk, out_ids


def get_labels_per_neuron(labels, spikes, start_t, dt):
    lbls_per_nrn = {k: [] for k in range(len(spikes))}

    for nid, times in enumerate(spikes):
        if np.isscalar(times):
            times = [times]
            print(times)

        if len(times) == 0:
            continue
        ts = np.asarray(times).astype('int')
        ts -= int(start_t)
        ts //= int(dt)
        ids = ts[np.where(ts < len(labels))]
        for lbl in [labels[i] for i in ids]:
            lbls_per_nrn[nid].append(lbl)

    return lbls_per_nrn

def get_neurons_per_label(labels, spikes, start_t, dt):
    unique = np.unique(labels)
    nrns_per_lbl = {k: [] for k in unique}
    for nid, times in enumerate(spikes):
        ts = np.asarray(times).astype('int')
        ts -= int(start_t)
        ts //= int(dt)
        ids = ts[np.where(ts < len(labels))]
        for lbl in [labels[i] for i in ids]:
            nrns_per_lbl[lbl].append(nid)

    return nrns_per_lbl


def activity_per_sample(labels, spikes, start_t, dt):
    end_t = start_t + len(labels) * dt
    aps = [[] for _ in labels]
    for st in np.arange(start_t, end_t, dt):
        et = st + dt
        idx = int((st - start_t) // dt)
        for nid, times in enumerate(spikes):
            ts = np.asarray(times)
            whr = np.where(np.logical_and(st <= ts, ts < et))[0]
            if len(whr):
                aps[idx].append(nid)
    return aps


def get_num_inactive(labels, spikes, start_t, dt):
    aps = activity_per_sample(labels, spikes, start_t, dt)
    n_aps = [len(v) for v in aps]
    return np.sum([1 for s in n_aps if s == 0])


def get_vectors(neurons_per_label):
    vectors = {}
    for k in neurons_per_label:
        v = np.zeros(len(out_spikes))
        v[neurons_per_label[k]] = 1.
        vectors[k] = v
    
    return vectors


def get_distances(neurons_per_label):
    n_labels = len(neurons_per_label)
    vectors = get_vectors(neurons_per_label)
    dists = np.zeros((n_labels, n_labels))
    labels = sorted(neurons_per_label.keys())
    for k0 in labels[:-1]:
        for k1 in labels[k0+1:]:
            if np.sum(vectors[k0]) < 1 or \
                np.sum(vectors[k1]) < 1:
                s = 0.
            else:
                diff = np.abs(vectors[k0] - vectors[k1])
                s = np.sum(diff)
                
            dists[k0, k1] = s
            dists[k1, k0] = dists[k0, k1]

    return dists


def neurons_sharing_class(labels, spikes, start_t, dt, power=1):
    labels_per_neuron = get_labels_per_neuron(
                            labels, spikes, start_t, dt)
    n_labels_per_neuron = [len(np.unique(labels_per_neuron[k]))
                                for k in labels_per_neuron]
    hi_n_labels_per_neuron = [(n - 1)**power for n in n_labels_per_neuron if n > 0]
    if len(hi_n_labels_per_neuron) == 0:
        return [0]

    return hi_n_labels_per_neuron

def mean_neurons_sharing_class(labels, spikes, start_t, dt, power=1):
    return np.mean( neurons_sharing_class(labels, spikes, start_t, dt, power) )    


def num_neurons_sharing_class(labels, spikes, start_t, dt):
    labels_per_neuron = get_labels_per_neuron(
                            labels, spikes, start_t, dt)
    n_labels_per_neuron = [len(np.unique(labels_per_neuron[k])) 
                                for k in labels_per_neuron]
    hi_n_labels_per_neuron = [n for n in n_labels_per_neuron if n > 1]
    
    return len(hi_n_labels_per_neuron)

def empty(st, dt, spikes):
    for ts in spikes:
        ts = np.asarray(ts)
        whr = np.where(np.logical_and(st <= ts, ts < st + dt))[0]
        if len(whr):
            return False
    return True


def get_num_spikes(spikes):
    return np.sum([len(ts) for ts in spikes])



def per_neuron_rate(spikes, start_t):
    return [len(np.where(np.asarray(times) >= start_t)[0])
            for times in spikes]

def spiking_per_class(indices, spikes, start_t, end_t, dt):
    uindices = np.unique(indices)
    aggregate_per_class = {}
    individual_per_class = {}
    for st in np.arange(start_t, end_t, dt):
        et = st + dt
        class_idx = int(st // dt)
        cls = int(indices[class_idx])
        apc = aggregate_per_class.get(cls, {})
        ipc = individual_per_class.get(cls, {})
        ind = {}
        for nid, ts in enumerate(spikes):
            times = np.array(ts)
            whr = np.where(np.logical_and(st <= times, times < et))[0]
            if len(whr):
                narray = apc.get(nid, None)
                if narray is None:
                    narray = times[whr]
                else:
                    narray = np.append(narray, times[whr])

                ind[nid] = times[whr]
                apc[nid] = narray

        aggregate_per_class[cls] = apc

        ipc[class_idx] = ind
        individual_per_class[cls] = ipc

    return aggregate_per_class, individual_per_class

def overlap_score(apc, n_output):
    classes = sorted(apc.keys())
    class_overlaps_sets = [set() for _ in classes]
    zero_counts = np.array(
        [1 if len(apc[cls0].keys()) == 0 else 0 for cls0 in apc]
    )
    for cls0_id, cls0 in enumerate(classes[:-1]):
        for cls1 in classes[cls0_id + 1:]:
            for nid in np.unique(list(apc[cls0].keys())):
                nids1 = list(apc[cls1].keys())
                if nid in nids1:
                    class_overlaps_sets[cls0] |= set([cls1])
                    class_overlaps_sets[cls1] |= set([cls0])

    co = np.asarray( [float(len(s)) for s in class_overlaps_sets] )
    # print(zero_counts)
    coco = co / (len(classes) - 1)
    coco[zero_counts != 0] = 1
    # print(coco)
    # print(class_overlaps_sets)
    # print(co)
    # print( co / (len(classes) - 1) )
    # print( 1.0 - co / (len(classes) - 1) )
    # print( 1.0 - coco )
    # print(np.min( 1.0 - co / (len(classes) - 1) ))

    return np.min( 1.0 - coco)

def individual_score(ipc, n_tests, n_classes):
    events = np.zeros(n_classes)
    for cls in sorted(ipc.keys()):
        for idx in sorted(ipc[cls].keys()):
            if len(ipc[cls][idx]):
                events[cls] += 1
    max_score = n_tests * n_classes
    return np.sum(events) / max_score


def diff_class_vectors(apc, n_output):
    dcv = [np.zeros(n_output) for _ in apc]
    for c in apc:
        nids = list(apc[c].keys())
        if len(nids):
            dcv[c][nids] = 1.

    return dcv


def vec_list_diffs(vec_list, norm=2):
    norms = [np.sqrt(np.sum(x ** 2)) for x in vec_list]
    dots = []
    eucs = []
    zero_ids = []
    euc = 0.
    xn, yn = 0., 0.
    # max_idx = np.argmax(np.sum(v) for v in vec_list)
    for ix, x in enumerate(vec_list[:-1]):
        for iy, y in enumerate(vec_list[ix+1:]):
            if iy > ix:
                xn = norms[ix]
                xx = x / xn if xn > ZERO_FLOAT else x

                yn = norms[iy]
                yy = y / yn if yn > ZERO_FLOAT else y
                zid = [None, None]
                if xn < 1:
                    zid[0] = ix
                if yn < 1:
                    zid[1] = iy
                zero_ids.append( zid )

                if norm == 2:
                    # sqrt(2) == max distance
                    euc = np.sqrt(np.sum((xx - yy) ** 2)) / np.sqrt(2)
                elif norm == 1:
                    euc = np.sum(np.abs(xx - yy))
                else:
                    euc = 0.

                eucs.append(euc)

                dot = np.dot(xx, yy)
                dots.append(dot)

    return np.asarray(norms), np.asarray(dots), np.asarray(eucs), np.asarray(zero_ids)


def diff_class_dists(diff_class_vectors):
    norms, dots, eucs, zids = vec_list_diffs(diff_class_vectors, norm=1)
    for out_id, ii in enumerate(zids):
        if ii[0] is not None or ii[1] is not None:
            eucs[out_id] = 0.
    return eucs

def diff_unique_dists(dists):
    unique_d = []
    for i in range(dists.shape[0])[:-1]:
        for j in range(i+1, dists.shape[0]):
            unique_d.append(dists[i, j])
    return unique_d

def sum_unique_dists(dists):
    return np.sum(diff_unique_dists(dists))

def any_all_zero(apc, ipc):
    any_zero = False
    all_zero = True
    for cls in sorted(ipc.keys()):
        for idx in sorted(ipc[cls].keys()):
            if len(ipc[cls][idx]) == 0:
                any_zero = True
                break
        if any_zero:
            break

    for c in apc:
        nids = list(apc[c].keys())
        if len(nids) != 0:
            all_zero = False
            break

    return any_zero, all_zero

def same_class_vectors(ipc, n_out):
    smc = {c: [np.zeros(n_out) for _ in ipc[c]] for c in ipc}
    for c in sorted(ipc.keys()):
        for i, x in enumerate(sorted(ipc[c].keys())):
            for nid in ipc[c][x]:
                smc[c][i][nid] = 1
    return smc

def same_class_distances(same_class_vectors):
    scd = {}
    for c in same_class_vectors:
        norms, dots, eucs, zids = vec_list_diffs(same_class_vectors[c])
        for out_id, ii in enumerate(zids):
            if ii[0] is not None or ii[1] is not None:
                dots[out_id] = 0.

        scd[c] = np.asarray(dots)

    return scd

def spiking_per_class_split(indices, spikes, start_t, end_t, dt):
    uindices = np.unique(indices)
    aggregate_per_class = {u: {} for u in uindices}
    individual_per_class = {u: {} for u in uindices}
    for st in np.arange(start_t, end_t, dt):
        sample_idx = int(st // dt)
        cls = int(indices[sample_idx])
        ind = {}
        for nid, ts in enumerate(spikes[sample_idx]):
            times = np.asarray(ts)
            if len(times):
                narray = aggregate_per_class[cls].get(nid, None)
                if narray is None:
                    narray = times
                else:
                    narray = np.append(narray, times)

                ind[nid] = times

                aggregate_per_class[cls][nid] = narray

        individual_per_class[cls][sample_idx] = ind

    return aggregate_per_class, individual_per_class


def split_per_dt(spikes, start_t, end_t, dt):
    spikes_dt = []
    for st in np.arange(start_t, end_t, dt):
        et = st + dt
        dt_act = []
        for nid, ts in enumerate(spikes):
            times = np.array(ts)
            whr = np.where(np.logical_and(st <= times, times < et))[0]
            new_ts = times[whr] if len(whr) else []
            dt_act.append(new_ts)

        spikes_dt.append(dt_act)
    return spikes_dt


def count_active_per_dt(split_spikes):
    count = []
    for spikes in split_spikes:
        count.append(np.sum([len(ts) for ts in spikes]))
    return count

def count_non_spiking_samples(spikes, start_t, end_t, dt):
    not_spiking = 0
    for st in np.arange(start_t, end_t, dt):
        sample_idx = int(st // dt)
        sample_not_spiking = np.sum([len(ts) > 0 for ts in spikes[sample_idx]])
        not_spiking += int(sample_not_spiking == 0)
    return not_spiking

def get_test_region(spikes, start_time, labels, t_per_sample):
    spks = []
    lbls = []
    for times in spikes:
        ts = np.asarray(times)
        whr = np.where(ts >= start_time)
        if len(whr[0]):
            spks.append(ts[whr].tolist())
        else:
            spks.append([])

    start_idx = int(start_time // t_per_sample)
    for l in labels[start_idx:]:
        lbls.append(l)

    return spks, lbls
