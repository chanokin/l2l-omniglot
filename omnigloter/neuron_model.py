from omnigloter import config

if config.SIM_NAME == config.SPINNAKER:
    from python_models8.neuron.builds import IF_curr_exp_i
elif config.SIM_NAME == config.GENN:
    from neuron_model_genn import IF_curr_exp_i