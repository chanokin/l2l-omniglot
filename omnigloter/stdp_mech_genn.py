from copy import deepcopy
from pyNN.standardmodels import synapses, StandardModelType, build_translations
from pynn_genn.simulator import state
import numpy as np
import logging
from pygenn.genn_wrapper.WeightUpdateModels import StaticPulse
from pynn_genn.model import GeNNStandardSynapseType, GeNNDefinitions
from pynn_genn.standardmodels.synapses import WeightDependence, delayMsToSteps, delayStepsToMs, DDTemplate


class MySTDPMechanism(synapses.STDPMechanism, GeNNStandardSynapseType):
    __doc__ = synapses.STDPMechanism.__doc__

    mutable_vars = set(["g"])

    base_translations = build_translations(
        ("weight", "g"),
        ("delay", "delaySteps", delayMsToSteps, delayStepsToMs),
        ("dendritic_delay_fraction", "_dendritic_delay_fraction"))

    base_defs = {
        "vars": {"g": "scalar"},
        "pre_var_name_types": [],
        "post_var_name_types": [],

        "sim_code": DDTemplate("""
            $(addToInSyn, $(g));
            scalar dt = $(t) - $(sT_post);
            $${TD_CODE}
        """),
        "learn_post_code": DDTemplate("""
            scalar dt = $(t) - $(sT_pre);
            $${TD_CODE}
        """),

        "is_pre_spike_time_required": True,
        "is_post_spike_time_required": True,
    }

    def _get_minimum_delay(self):
        if state._min_delay == "auto":
            return state.dt
        else:
            return state._min_delay

    def __init__(self, timing_dependence=None, weight_dependence=None,
                 voltage_dependence=None, dendritic_delay_fraction=1.0,
                 weight=0.0, delay=None):
        super(MySTDPMechanism, self).__init__(
            timing_dependence, weight_dependence, voltage_dependence,
            dendritic_delay_fraction, weight, delay)

        # Create a copy of the standard STDP defs
        self.wum_defs = deepcopy(self.base_defs)

        # Adds variables from timing and weight dependence to definitions
        self.wum_defs["vars"].update(self.timing_dependence.vars)
        self.wum_defs["vars"].update(self.weight_dependence.vars)

        # Add pre and postsynaptic variables from timing dependence to definition
        if hasattr(self.timing_dependence, "pre_var_name_types"):
            self.wum_defs["pre_var_name_types"].extend(
                self.timing_dependence.pre_var_name_types)

        if hasattr(self.timing_dependence, "post_var_name_types"):
            self.wum_defs["post_var_name_types"].extend(
                self.timing_dependence.post_var_name_types)

        # Apply substitutions to sim code
        td_sim_code = self.timing_dependence.sim_code.substitute(
            WD_CODE=self.weight_dependence.depression_update_code)
        self.wum_defs["sim_code"] = \
            self.wum_defs["sim_code"].substitute(TD_CODE=td_sim_code)

        # Apply substitutions to post learn code
        td_post_learn_code = self.timing_dependence.learn_post_code.substitute(
            WD_CODE=self.weight_dependence.potentiation_update_code)
        self.wum_defs["learn_post_code"] = \
            self.wum_defs["learn_post_code"].substitute(TD_CODE=td_post_learn_code)

        # Use pre and postsynaptic spike code from timing dependence
        if hasattr(self.timing_dependence, "pre_spike_code"):
            self.wum_defs["pre_spike_code"] = self.timing_dependence.pre_spike_code

        if hasattr(self.timing_dependence, "post_spike_code"):
            self.wum_defs["post_spike_code"] = self.timing_dependence.post_spike_code


#class MyWeightDependence(synapses.MultiplicativeWeightDependence, WeightDependence):
#    __doc__ = synapses.MultiplicativeWeightDependence.__doc__
#
#    depression_update_code = """
#        if(update <= 0){
#            $(g) += ($(g) - $(Wmin)) * update;
#        }else{
#            $(g) += ($(Wmax) - $(g)) * update;
#        }
#        $(g) = fmin( fmax($(Wmin), $(g)) , $(Wmax));
#    """
#
#    potentiation_update_code = depression_update_code
#
#    translations = build_translations(*WeightDependence.wd_translations)


class MyWeightDependence(synapses.AdditiveWeightDependence, WeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__
    depression_update_code = "$(g) = fmin($(Wmax), fmax($(Wmin), $(g) + (($(Wmax) - $(Wmin)) * update)));\n"
    potentiation_update_code = depression_update_code

    translations = build_translations(*WeightDependence.wd_translations)


# class MyWeightDependence(synapses.AdditiveWeightDependence, WeightDependence):
#     __doc__ = synapses.AdditiveWeightDependence.__doc__
#
#     depression_update_code = "$(g) = fmin($(Wmax), fmax($(Wmin), $(g) + (($(Wmax) - $(Wmin)) * update)));\n"
#
#     potentiation_update_code = "$(g) = fmin($(Wmax), fmax($(Wmin), $(g) + (($(Wmax) - $(Wmin)) * update)));\n"
#
#     translations = build_translations(*WeightDependence.wd_translations)

class MyTemporalDependence(synapses.STDPTimingDependence):
    # __doc__ = synapses.SpikePairRule.__doc__
    ### will try to do a displaced gaussian
    """tau_plus: scalar: how wide is the Gaussian
       tau_minus: scalar: temporal range, how far in future/past to see spikes
       Aminus: scalar: if total input is inhibitory, how much should we decrease weights
       Aplus: scalar: rate, how high the Gaussian is
    """
    vars = {
        # "tauPlus": "scalar",  # 0 - Potentiation time constant (ms)
        # "tauMinus": "scalar", # 1 - Depression time constant (ms)
        # "Aplus": "scalar",    # 2 - Rate of potentiation
        # "Aminus": "scalar",   # 3 - Rate of depression
        "Aplus": "scalar",
        "Aminus": "scalar",
        "tau_plus": "scalar",
        "tau_plus1": "scalar",
        "tau_minus": "scalar",
        "max_t": "scalar",
        # "": "scalar",
    }

    pre_var_name_types = [("preTrace", "scalar")]
    post_var_name_types = [("postTrace", "scalar")]

    # post then pre (dt = pre - post)
    # using {.brc} for left{ or right} so that .format() does not freak out
    sim_code = DDTemplate("""
        // std::cout << "pre(" << dt << ")" << std::endl;
        // std::cout << "t [" << $(t) << "]" << std::endl;
        if( $(max_t) < 0.0 || $(t) <= $(max_t) ){
            // std::cout << "t [" << $(t) << "] <= max_t [" << $(max_t) << "]" << std::endl;
            if (dt > 0){
                scalar update = 0.0;
                if (dt <= $(tau_plus1)){
                    update = $(Aplus);                
                }
                else
                if ( dt <= $(tau_minus) ){
                    update = -$(Aminus);
                }
                
                $${WD_CODE}
            }
        }
        """)

    # pre then post (dt = post - pre)
    learn_post_code = DDTemplate("""
        // std::cout << "post(" << dt << ")" << std::endl;
        // std::cout << "t [" << $(t) << "]" << std::endl;
        if( $(max_t) < 0.0 || $(t) <= $(max_t) ){
            // std::cout << "t [" << $(t) << "] <= max_t [" << $(max_t) << "]" << std::endl;
            if (dt >= 0){
                scalar update = 0.0;
                if (dt <= $(tau_plus)){
                    update = $(Aplus);
                    
                }
                else if ( dt <= $(tau_minus) ){
                    update = -$(Aminus);
                }
    
                $${WD_CODE}
            }
        }
        """)

    pre_spike_code = """\
        const scalar dt = $(t) - $(sT_pre);
        $(preTrace) = $(preTrace) * exp(-dt) + 1.0;
        """

    post_spike_code = """\
        const scalar dt = $(t) - $(sT_post);
        $(postTrace) = $(postTrace) * exp(-dt) + 1.0;
        """
    translations = build_translations(
        ("A_plus", "Aplus"),
        ("A_minus", "Aminus"),
        ("tau_plus", "tau_plus"),
        ("tau_plus1", "tau_plus1"),
        ("tau_minus", "tau_minus"),
        ("max_learn_t", "max_t"),
    )

    default_parameters = {
        'tau_plus': 1.0,
        'tau_plus1': -1.0,
        'tau_minus': 20.0,
        'A_plus': 0.01,
        'A_minus': 0.0,
        'max_learn_t': -1.0,
    }

    def __init__(self, A_plus=0.01, A_minus=0.01, tau_plus=1.0, tau_plus1=-1.0, tau_minus=20.0, max_learn_t=-1.0):
        """
        Create a new specification for the timing-dependence of an STDP rule.
        """
        parameters = dict(locals())
        parameters.pop('self')
        if np.isinf(max_learn_t):
            parameters['max_learn_t'] = -1.0

        synapses.STDPTimingDependence.__init__(self, **parameters)
