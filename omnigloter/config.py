import numpy as np
import os

GENN = 'genn'
SPINNAKER = 'spinnaker'

DEBUG = bool(0)
ONE_TO_ONE_EXCEPTION = bool(0)
BACKEND = 'SingleThreadedCPU' if bool(0) else 'CUDA'

INF = float(10e10)

#SEED = 7
SEED = None
RNG = np.random.RandomState(seed=SEED)


USE_GABOR_LAYER = bool(0)

SIM_NAME = GENN

if SIM_NAME == GENN:
    from pynn_genn.random import NativeRNG, NumpyRNG
    NP_RNG = NumpyRNG(seed=SEED)
    NATIVE_RNG = NativeRNG(NP_RNG, seed=SEED)

else:
    NP_RNG = RNG
    NATIVE_RNG = RNG


GPU_ID = 0
USE_PROCEDURAL = bool(0)

TIMESTEP = 0.10 #ms
SAMPLE_DT = 50.0 #ms
SAMPLE_OFFSET = 10. # ms
SAMPLE_MAX_T = SAMPLE_OFFSET + 5. # ms
# iw = 28
iw = 32
# iw = 48
# iw = 56
# iw = 64
# iw = 105
INPUT_SHAPE = (iw, iw)
INPUT_DIVS = (3, 5)
# INPUT_DIVS = (3, 3)
# INPUT_DIVS = (2, 2)
# INPUT_DIVS = (1, 1)
# INPUT_DIVS = (2, 3)
N_CLASSES = 14 if DEBUG else 3
N_SAMPLES = 14 if DEBUG else 14
N_EPOCHS = 10 if DEBUG else 5
N_TEST = 6 if DEBUG else 6
TOTAL_SAMPLES = N_SAMPLES * N_EPOCHS + N_TEST
DURATION = N_CLASSES * TOTAL_SAMPLES * SAMPLE_DT
PROB_NOISE_SAMPLE = 0.0#5
STEPS = 1 if SIM_NAME == GENN else 100


TEST_MUSHROOM = bool(0)
GAIN_CONTROL = bool(1)
INH_INPUT = bool(1)
SUPERVISION = bool(1) and (not TEST_MUSHROOM) 

SUP_DELAY = 5 # ms
SUP_DURATION = 15 # ms
SUP_CORRECT_AMPLITUDE = 1. # nA ?
SUP_WRONG_AMPLITUDE = -0.2 # nA ?

KERNEL_W = 7
N_INPUT_LAYERS = 4
PAD = KERNEL_W//2
PI_DIVS_RANGE = (6, 7) if DEBUG else (2, 7)
STRIDE_RANGE = (2, 3) if DEBUG else (1, KERNEL_W//2 + 1)
OMEGA_RANGE = (0.5, 1.0)

if ONE_TO_ONE_EXCEPTION:
    EXPANSION_RANGE = (1., 1.0000000000000000000001)
else:
    # EXPANSION_RANGE = (10., 10.0001) if DEBUG else (0.25, 11.0)
    EXPANSION_RANGE = (20., 21.0) if DEBUG else (10, )#(5, 40)


EXP_PROB_RANGE = (0.5, 0.75000001) if DEBUG else (3,)# 65)#0.025, 0.25)

MUSH_MAX = 3.2 #/ float(EXP_PROB_RANGE[0])

if ONE_TO_ONE_EXCEPTION:
    MUSHROOM_WEIGHT_RANGE = (5.0, 5.0000000001)
else:
    MUSHROOM_WEIGHT_RANGE = (1.0, 5.0000001) if DEBUG else  (1., MUSH_MAX)
# MUSHROOM_WEIGHT_RANGE = (0.50, 0.500000001) if DEBUG else (0.05, 1.0)
# MUSHROOM_WEIGHT_RANGE = (0.025, 0.02500001) if DEBUG else (0.05, 1.0) ### for (64,64)

MAX_PRE_OUTPUT = 40000

OUTPUT_PROB_RANGE = (0.5, 0.750000001) if DEBUG else (0.01, )#0.5)
# OUT_WEIGHT_RANGE = (0.1, 0.100000001) if DEBUG else (1.0, 5.0)
if ONE_TO_ONE_EXCEPTION:
    OUT_WEIGHT_RANGE = (0.1, 0.1000000001)
else:
    OUT_WEIGHT_RANGE = (2.0, 5.000000001) if DEBUG else (0.1, 1.2)# (0.01, 0.5)
# OUT_WEIGHT_RANGE = (1.5, 1.500001) if DEBUG else (0.01, 0.5) ### 64x64


A_PLUS = (0.1, 5.0000000001) if DEBUG else (0.001, 1.0)
A_MINUS = (0.1, 1.000000001) if DEBUG else (0.001, 1.0)
CONN_DIST = (5, 15) if DEBUG else (1,)# 16)#(1, 15)


STD_DEV = (3.0, 3.00000001) if DEBUG else (0.5, 5.0)
DISPLACE = (0.0,)#01, 0.00100000001) if DEBUG else (0.0001, 0.1)
MAX_DT = (80.0, 80.00000001) if DEBUG else (float(SAMPLE_DT), SAMPLE_DT*2.0)
W_MIN_MULT = (0.0, 0.00000001) if DEBUG else (0.00, )#(-1, 1)
W_MAX_MULT = (1.,)# 1.200000001) if DEBUG else (0.1, 2.0


GABOR_WEIGHT_RANGE = (2.0, 5.000001) if DEBUG else (1.0, 5.0)

GAIN_CONTROL_SIZE = 20
GAIN_CONTROL_MIN_W = 0.
GAIN_CONTROL_MAX_W = 0.250000000000008#0000#1
GAIN_CONTROL_INH_W = -0.10000
GAIN_CONTROL_CUTOFF = 15


NOISE_MUSHROOM_SIZE = 20
NOISE_MUSHROOM_RATE = 50
NOISE_MUSHROOM_WEIGHT = 0.01
NOISE_MUSHROOM_PROB = 0.0

###############
# if ONE_TO_ONE_EXCEPTION:
#     EXPANSION_RANGE = (1., 1.0000000000000000000001)
# else:
#     # EXPANSION_RANGE = (10., 10.0001) if DEBUG else (0.25, 11.0)
#     EXPANSION_RANGE = (0.25, 0.25) if DEBUG else (0.25, 11.0)
#
# EXP_PROB_RANGE = (0.15, 0.15000001) if DEBUG else (0.05, 0.3)
# OUTPUT_PROB_RANGE = (0.15, 0.150000001) if DEBUG else (0.05, 0.3)
# A_PLUS = (2.0, 2.0000000001) if DEBUG else (0.01, 5.0)
# A_MINUS = (1.0, 1.000000001) if DEBUG else (0.001, 1.0)
# STD_DEV = (3.0, 3.00000001) if DEBUG else (0.5, 5.0)
# DISPLACE = (0.0,)#01, 0.00100000001) if DEBUG else (0.0001, 0.1)
# MAX_DT = (80.0, 80.00000001) if DEBUG else (float(SAMPLE_DT), SAMPLE_DT*2.0)
# W_MIN_MULT = (0.0, 0.00000001) if DEBUG else (-2.0, 0.0)
# W_MAX_MULT = (1.2,)# 1.200000001) if DEBUG else (0.1, 2.0)
# CONN_DIST = (10, 11) if DEBUG else (3, 25)
#
#
# GABOR_WEIGHT_RANGE = (2.0, 2.000001) if DEBUG else (1.0, 5.0)
#
# # OUT_WEIGHT_RANGE = (0.1, 0.100000001) if DEBUG else (1.0, 5.0)
# if ONE_TO_ONE_EXCEPTION:
#     OUT_WEIGHT_RANGE = (0.1, 0.1000000001)
# else:
#     OUT_WEIGHT_RANGE = (2.0, 2.000000001) if DEBUG else (0.5, 5.0)
# # OUT_WEIGHT_RANGE = (1.5, 1.500001) if DEBUG else (0.01, 0.5) ### 64x64
#
# if ONE_TO_ONE_EXCEPTION:
#     MUSHROOM_WEIGHT_RANGE = (5.0, 5.0000000001)
# else:
#     MUSHROOM_WEIGHT_RANGE = (1.0, 1.0000001) if DEBUG else (1.0, 5.0)
# # MUSHROOM_WEIGHT_RANGE = (0.50, 0.500000001) if DEBUG else (0.05, 1.0)
# # MUSHROOM_WEIGHT_RANGE = (0.025, 0.02500001) if DEBUG else (0.05, 1.0) ### for (64,64)
###############


#################################################################
# WEIGHTS FOR FITNESS #
#################################################################

if SUPERVISION:
    N_PER_CLASS = 1
else:
    N_PER_CLASS = 50

OUTPUT_SIZE = N_CLASSES * N_PER_CLASS
MAX_ACTIVE_PER_CLASS = int(OUTPUT_SIZE / 0.5)
ACTIVITY_THRESHOLD = 0.5 * OUTPUT_SIZE
MAX_VECTOR_DIST = 100.0
ABOVE_THRESH_W = 1.0 / N_CLASSES

TARGET_ACTIVITY_PER_SAMPLE = np.round(OUTPUT_SIZE * 0.05)
TARGET_FREQUENCY_PER_OUTPUT_NEURON = np.round(N_TEST * 1.5)

OVERLAP_WEIGHT = 0.3
REPRESENTATION_WEIGHT = 0.4
DIFFERENT_CLASS_DISTANCE_WEIGHT = 0.2
SAME_CLASS_DISTANCE_WEIGHT = 0.


# CONN_DIST = 3
# CONN_DIST = 9
# CONN_DIST = 15
# CONN_ANGS = 9
# CONN_RADII = [3, ]

### static weights
# gabor_weight = [1.0, 1.0, 2.0, 2.0]
# mushroom_weight = 0.25
INHIBITORY_WEIGHT = {
    'gabor': -5.0,
    'mushroom': -(0.5 if USE_PROCEDURAL else 0.5),
    'output': -0.01,
}

N_INH_PER_ZONE = 3

N_INH_OUTPUT = 5 # int( 0.25 * N_PER_CLASS * N_CLASSES )

EXCITATORY_WEIGHT = {
    'gabor': 3.0,
    'mushroom': 5.0,
    'output': 5.0,
}
MUSH_SELF_PROB = 0.0075

ATTRS = [
    'out_weight',
    # 'n_pi_divs', 'stride', 'omega',
     'expand', 'exp_prob', 'out_prob',
    'mushroom_weight'
]
# ATTRS += ['gabor_weight-%d'%i for i in range(N_INPUT_LAYERS)]

N_ATTRS = len(ATTRS)

ATTR2IDX = {attr: i for i, attr in enumerate(ATTRS)}

ATTR_RANGES = {
    'out_weight': OUT_WEIGHT_RANGE,
    'mushroom_weight': MUSHROOM_WEIGHT_RANGE,
    'expand': EXPANSION_RANGE,
    'exp_prob': EXP_PROB_RANGE,
    'out_prob': OUTPUT_PROB_RANGE,
    'conn_dist': CONN_DIST,

    'A_plus': A_PLUS,
    'A_minus': A_MINUS,
    # 'std': STD_DEV,
    # 'displace': DISPLACE,
    # 'maxDt': MAX_DT,
    'w_max_mult': W_MAX_MULT,
    'w_min_mult': W_MIN_MULT,

}
ATTR_STEPS_DEVS = {
    'out_weight': 1.0,
    'mushroom_weight': 1.0,
    'expand': 1.0,
    'exp_prob': 1.0,
    'out_prob': 1.0,
    'A_plus': 1.0,
    'A_minus': 1.0,
    'std': 1.0,
    'displace': 1.0,
    'maxDt': 1.0,
    'w_max_mult': 1.0,
    'w_min_mult': 1.0,
    'conn_dist': 1.0,
}
# ATTR_STEPS_BASE = {
#     'out_weight': 1.0,
#     'mushroom_weight': 1.0,
#     'expand': 5.0,
#     'exp_prob': 0.05,
#     'out_prob': 0.05,
#     'A_plus': 0.1,
#     'A_minus': 0.1,
#     'std': 0.5,
#     'displace': 0.01,
#     'maxDt': 10.0,
#     'w_max_mult': 0.05,
#     'w_min_mult': 0.05,
#     'conn_dist': 5.0,
# }
# cheap attempt to scale the variance for normal-distributed mutation
ATTR_STEPS_BASE = {
    k: ATTR_STEPS_DEVS[k] * ((ATTR_RANGES[k][1] - ATTR_RANGES[k][0]) / 3.14159)
      if len(ATTR_RANGES[k]) > 1 else
       ATTR_STEPS_DEVS[k] * ((ATTR_RANGES[k][0]) / 3.14159)
        for k in ATTR_RANGES
}

ATTR_STEPS = {k: ATTR_STEPS_BASE[k] for k in ATTR_STEPS_BASE}

# for s in ATTRS:
#     if s.startswith('gabor_weight'):
#         ATTR_RANGES[s] = GABOR_WEIGHT_RANGE


### Neuron types
NEURON_CLASS = 'IF_curr_exp'
GABOR_CLASS = 'IF_curr_exp'
MUSHROOM_CLASS = 'IF_curr_exp_i' # i
INH_MUSHROOM_CLASS = 'IF_curr_exp'
OUTPUT_CLASS = 'IF_curr_exp_i'
INH_OUTPUT_CLASS = 'IF_curr_exp'

### Neuron configuration
VTHRESH = -55.0
BASE_PARAMS = {
    'cm': 0.1,  # nF
    'v_reset': -70.,  # mV
    'v_rest': -65.,  # mV
    'tau_m': 10.,  # ms
    'tau_refrac': 5.,  # ms
    'tau_syn_E': 2., # ms
    'tau_syn_I': 5., # ms
    'i_offset': 0.
}

INH_PARAMS = BASE_PARAMS.copy()
INH_PARAMS['v_thresh'] = -55.0
INH_PARAMS['tau_m'] = 16.0

tau_thresh = 30.0
#tau_thresh = 50.0
mult_thresh = 1.8
# mult_thresh = 0.00000000001
mult_thresh = 1.000000000000000001

GABOR_PARAMS = BASE_PARAMS.copy()
MUSHROOM_PARAMS = BASE_PARAMS.copy()
MUSHROOM_PARAMS['v_threshold'] = VTHRESH  # mV
MUSHROOM_PARAMS['v_thresh_adapt'] = MUSHROOM_PARAMS['v_threshold']
MUSHROOM_PARAMS['tau_threshold'] = tau_thresh
MUSHROOM_PARAMS['w_threshold'] = mult_thresh
MUSHROOM_PARAMS['tau_syn_E'] = 5.
MUSHROOM_PARAMS['tau_syn_I'] = 5.
MUSHROOM_PARAMS['cm'] = 1.0
MUSHROOM_PARAMS['tau_m'] = 20.0

INH_MUSHROOM_PARAMS = INH_PARAMS.copy()
INH_OUTPUT_PARAMS = INH_PARAMS.copy()
GAIN_CONTROL_PARAMS = BASE_PARAMS.copy()

tau_thresh = 50.0
mult_thresh = 1.8
mult_thresh = 1.00000000000000000000001

OUTPUT_PARAMS = BASE_PARAMS.copy()
OUTPUT_PARAMS['v_threshold'] = VTHRESH  # mV
OUTPUT_PARAMS['v_thresh_adapt'] = OUTPUT_PARAMS['v_threshold']
OUTPUT_PARAMS['tau_threshold'] = tau_thresh
OUTPUT_PARAMS['w_threshold'] = mult_thresh
OUTPUT_PARAMS['tau_syn_E'] = 5.
OUTPUT_PARAMS['tau_syn_I'] = 5.
OUTPUT_PARAMS['cm'] = 1.0
OUTPUT_PARAMS['tau_m'] = 20.0
OUTPUT_PARAMS['tau_syn_S'] = 5.




RECORD_SPIKES = [
    # 'input',
    # 'gabor',
#    'gain_control',
#    'mushroom',
    # 'inh_mushroom',
    'output',
    # 'inh_output',
]
if TEST_MUSHROOM and 'mushroom' not in RECORD_SPIKES:
   RECORD_SPIKES.append('mushroom')

RECORD_WEIGHTS = [
    # 'input to gabor',
    # 'gabor to mushroom',
    # 'input to mushroom',
#    'mushroom to output'
]

RECORD_VOLTAGES = [
    'output',
#    'gain_control'
]

SAVE_INITIAL_WEIGHTS = bool(1)

# STDP_MECH = 'STDPMechanism'
#
# time_dep = 'SpikePairRule'
# time_dep_vars = dict(
#     tau_plus = 20.0,
#     tau_minus = 20.0,
#     A_plus = 0.01,
#     A_minus = 0.01,
# )
#
# weight_dep = 'AdditiveWeightDependence'
# weight_dep_vars = dict(
# )
# w_min_mult = 0.0
# w_max_mult = 1.2

STDP_MECH = 'MySTDPMechanism'

TIME_DEP = 'MyTemporalDependence'
TIME_DEP_VARS = {
    "A_plus": 0.10,
    "A_minus": 0.01,
    "tau_plus": 5.0,
    "tau_plus1": 5.0,
    "tau_minus": 80.0,
    "max_learn_t": N_CLASSES * N_SAMPLES * SAMPLE_DT * N_EPOCHS + 1.0,
}

WEIGHT_DEP = 'MyWeightDependence'
WEIGHT_DEP_VARS = dict(
)
W_MIN_MULT = 0#-2.0
W_MAX_MULT = 1.2

