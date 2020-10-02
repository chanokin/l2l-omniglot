from copy import deepcopy
from functools import partial
import numpy as np
import lazyarray as la
from pyNN.standardmodels import cells, build_translations
from pynn_genn.standardmodels.cells import tau_to_decay, tau_to_init, \
    genn_postsyn_defs
from pynn_genn.simulator import state
import logging
from pynn_genn.model import GeNNStandardCellType, GeNNDefinitions
from pygenn.genn_model import create_custom_neuron_class


def inv_val(val_name, **kwargs):
    return 1.0 / kwargs[val_name]


def inv_tau_to_decay(val_name, **kwargs):
    return 1.0 / la.exp(-state.dt / kwargs[val_name])


ADD_DVDT = bool(0)

_genn_neuron_defs = {}
_genn_postsyn_defs = {}


_genn_postsyn_defs["ExpCurrShunt"] = GeNNDefinitions(
    definitions={
        "decay_code": {
            "inhShunt": "$(inSyn) *= $(expDecay);",
            "inhX": "$(inSyn) *= $(expDecay);",
            "inh": "$(inSyn) *= $(expDecay);",
            "exc": "$(inSyn) *= $(expDecay);",
        },

        "apply_input_code": {
            "inhShunt": "$(IsynShunt) += $(init) * $(inSyn);",
            "inhX": "$(IsynX) += $(init) * $(inSyn);",
            "inh": "$(Isyn) += $(init) * $(inSyn);",
            "exc": "$(Isyn) += $(init) * $(inSyn);",
        },

        "var_name_types": [],
        "param_name_types": {
            "expDecay": "scalar",
            "init": "scalar",
        }
    },
    translations=(
        ("tau_syn_E", "exc_expDecay", partial(tau_to_decay, "tau_syn_E"), None),
        ("tau_syn_I", "inh_expDecay", partial(tau_to_decay, "tau_syn_I"), None),
        ("tau_syn_S", "inhShunt_expDecay", partial(tau_to_decay, "tau_syn_S"), None),
        ("tau_syn_X", "inhX_expDecay", partial(tau_to_decay, "tau_syn_X"), None),
    ),
    extra_param_values={
        "exc_init": partial(tau_to_init, "tau_syn_E"),
        "inh_init": partial(tau_to_init, "tau_syn_I"),
        "inhShunt_init": partial(tau_to_init, "tau_syn_S"),
        "inhX_init": partial(tau_to_init, "tau_syn_X"),

    })

_genn_neuron_defs['IFAdapt'] = GeNNDefinitions(
    definitions={
        "sim_code": """
            $(I) = $(Isyn);
            if ($(RefracTime) <= 0.0) {
                scalar alpha = (($(Isyn) * exp( $(IsynShunt) ) + $(IsynX) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
                $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
                $(VThreshAdapt) = $(Vthresh) + ($(VThreshAdapt) - $(Vthresh))* $(DownThresh);
            }
            else {
                $(RefracTime) -= DT;
            }
        """,

        "threshold_condition_code": "$(RefracTime) <= 0.0 && $(V) >= $(VThreshAdapt)",

        "reset_code": """
            $(V) = $(Vreset);
            $(RefracTime) = $(TauRefrac);
            $(VThreshAdapt) += $(UpThresh)*($(Vthresh) - $(Vrest)); 
        """,

        "var_name_types": [
            ("V", "scalar"),
            ("I", "scalar"),
            ("RefracTime", "scalar"),
            ("VThreshAdapt", "scalar"),
        ],

        "param_name_types": {
            "Rmembrane": "scalar",  # Membrane resistance
            "ExpTC": "scalar",  # Membrane time constant [ms]
            "Vrest": "scalar",  # Resting membrane potential [mV]
            "Vreset": "scalar",  # Reset voltage [mV]
            "Vthresh": "scalar",  # Spiking threshold [mV]
            "Ioffset": "scalar",  # Offset current
            "TauRefrac": "scalar",
            "UpThresh": "scalar",
            "DownThresh": "scalar",
        },
        "additional_input_vars": [
            ("IsynShunt", "scalar", 0.0),
            ("IsynX", "scalar", 0.0),
        ],
    },
    translations=(
        ("v_rest", "Vrest"),
        ("v_reset", "Vreset"),
        ("cm", "Rmembrane", "tau_m / cm", ""),
        ("tau_m", "ExpTC", partial(tau_to_decay, "tau_m"), None),
        ("tau_refrac", "TauRefrac"),
        ("v_threshold", "Vthresh"),
        ("i_offset", "Ioffset"),
        ("v", "V"),
        ("i", "I"),
        ("w_threshold", "UpThresh"),
        ("tau_threshold", "DownThresh", partial(tau_to_decay, "tau_threshold"),
         None),
        ("v_thresh_adapt", "VThreshAdapt"),
    ),
    extra_param_values={
        "RefracTime": 0.0,
    })


class IF_curr_exp_i(cells.IF_curr_exp, GeNNStandardCellType):
    __doc__ = cells.IF_curr_exp.__doc__

    default_parameters = {
        'v_rest': -65.0,  # Resting membrane potential in mV.
        'cm': 1.0,  # Capacity of the membrane in nF
        'tau_m': 20.0,  # Membrane time constant in ms.
        'tau_refrac': 0.1,  # Duration of refractory period in ms.
        'tau_syn_E': 5.0,  # Decay time of excitatory synaptic current in ms.
        'tau_syn_I': 5.0,  # Decay time of inhibitory synaptic current in ms.
        'tau_syn_S': 5.0,
        'tau_syn_X': 5.0,
        'i_offset': 0.0,  # Offset current in nA
        'v_reset': -65.0,  # Reset potential after a spike in mV.
        'v_threshold': -50.0,  # Spike threshold in mV. STATIC, MIN
        'i': 0.0,  # nA total input current

        ### https://www.frontiersin.org/articles/10.3389/fncom.2018.00074/full
        # 'tau_thresh': 80.0,
        # 'mult_thresh': 1.8,
        ### https://www.frontiersin.org/articles/10.3389/fncom.2018.00074/full
        'tau_threshold': 120.0,
        'w_threshold': 1.8,
        'v_thresh_adapt': -50.0,  # Spike threshold in mV.

    }

    recordable = ['spikes', 'v', 'i', 'v_thresh_adapt']

    default_initial_values = {
        'v': -65.0,  # 'v_rest',
        'isyn_exc': 0.0,
        'isyn_inh': 0.0,
        'isyn_inh_s': 0.0,
        'isyn_inh_x': 0.0,
        'i': 0.0,
    }

    units = {
        'v': 'mV',
        'isyn_exc': 'nA',
        'isyn_inh': 'nA',
        'isyn_inh_s': 'nA',
        'isyn_inh_x': 'nA',
        'v_rest': 'mV',
        'cm': 'nF',
        'tau_m': 'ms',
        'tau_refrac': 'ms',
        'tau_syn_E': 'ms',
        'tau_syn_I': 'ms',
        'tau_syn_S': 'ms',
        'tau_syn_X': 'ms',
        'i_offset': 'nA',
        'v_reset': 'mV',
        'v_threshold': 'mV',
        'i': 'nA',
        'tau_threshold': 'ms',
        'w_threshold': '',
        'v_thresh_adapt': 'mV',
    }

    receptor_types = (
        'excitatory', 'inhibitory', 'inhShunt', 'inhX',
    )

    genn_neuron_name = "IF_i"
    genn_postsyn_name = "ExpCurrShunt"
    neuron_defs = _genn_neuron_defs['IFAdapt']
    postsyn_defs = _genn_postsyn_defs[genn_postsyn_name]
