import os, sys
sys.path.append(os.path.abspath(os.path.join('../../BayesFlow_dev/BayesFlow'))) # path Lasse
import bayesflow as bf
import numpy as np
sys.path.append("..")
import prior_simulator_functions

### Parameter estimation




### Model comparison

# Helper function to pass variable N as shared context for model comparison
def variable_n_obs():
    n_obs = prior_simulator_functions.prior_N()
    return {'n_obs' : n_obs}

# Custom configurator to handle variable n_obs in model comparison
default_config = bf.configuration.DefaultModelComparisonConfigurator(num_models=2)

def variable_n_obs_configurator(forward_dict):
    config = default_config(forward_dict)
    batch_size = config['model_indices'].shape[0]
    config['direct_conditions'] = np.zeros((batch_size, 1), dtype=np.float32) + np.sqrt(forward_dict['n_obs'])
    return config

