import sys, os
import numpy as np
from SpikingBGRL import Experiment

if __name__ == '__main__':
    # Build experiment
    exp = Experiment()

    # Make any tweaks here. For example:
    #exp.brain.vta.DA_pars['A_plus'] = .15 * exp.brain.vta.DA_pars['weight']

    # Run normal conditioning
    success_history = exp.train_brain(n_trials=150, save_dir='run1')
    result = np.sum(success_history[-100:])
    # Run reversal learning
    success_history = exp.train_brain(n_trials=150, rev_learn=True, save_dir='run1')
    result += np.sum(success_history[-100:])
    
    print('Successful trials:' result)
