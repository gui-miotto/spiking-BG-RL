# Spiking-BG-RL

An example of how to use [NEST](https://www.nest-simulator.org/) to create a spiking neural network that can learn associations between cues and actions. It uses a plasticity rule modulated by dopamine, i.e. three-factor STDP. If you are interested in knowing more about this model and the neuroscience supporting it, you can check out the [report](https://github.com/gui-miotto/spiking-BG-RL/blob/master/Report.pdf) available at the root directory.

# The model

The decision-making setup modeled here was based on a common experiment conducted with animals where they are presented with one of two possible cues and have to decide between two different actions. At each trial, just one action is rewarded, and the rewarded action can be predicted based on the cue. After some training, the animals learn to consider the cue to make the decisions.

The neural network is based on the interaction between cortex and striatum; the main input nucleus of the basal ganglia. It is made of 1451 neurons: 1000 excitatory, 450 inhibitory and 1 dopaminergic. Cues are sensed by the cortical network and actions are originated at the striatal network.

![alt text](https://github.com/gui-miotto/spiking-BG-RL/blob/master/prot_net.png "protocol and network")

# Example

An example of how to run an experiment simulation is provided in the file main.py:

```python
import sys, os
import numpy as np
from SpikingBGRL import Experiment

if __name__ == '__main__':
    # Build experiment
    exp = Experiment()

    # Run normal conditioning
    success_history = exp.train_brain(n_trials=150, save_dir='run1')
    result = np.sum(success_history[-100:])
    # Run reversal learning
    success_history = exp.train_brain(n_trials=150, rev_learn=True, save_dir='run1')
    result += np.sum(success_history[-100:])
    
    print('Successful trials:' result)
```
Functions to plot results and the progress of training are available in the files make_plots_0.py and make_plots_1.py


# Prerequisites:

* [NEST](https://www.nest-simulator.org/): I recommend [installing using conda](https://nest-simulator.readthedocs.io/en/latest/installation/conda_install.html). It is required to activate support to MPI, so don't forget the flag `nest-simulator=*=mpi_openmpi*`
* [mpi4py](https://pypi.org/project/mpi4py/)
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)

