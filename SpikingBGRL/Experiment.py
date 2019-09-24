import nest
import numpy as np
from mpi4py import MPI
from time import time
from datetime import timedelta
from .BrainStructures import Brain
from .DataIO import ExperimentResults, ExperimentMethods, NetworkSnapshot

class Experiment():
    """Class representing the instrumental conditioning of a brain. A experiment is sequence of 
    trials. At each trial, a cue is presented to the brain and an action is taken by the brain. 
    Class members whose names are followed by a trailing _ (e.g. self.success_) are updated at every
    trial, the others are constant throughout the whole experiment.
    """
    def __init__(self, seed=42, debug_mode=False):
        """Constructor
        
        Parameters
        ----------
        seed : int, optional
            Master seed for EVERYTHING. Runs with the same seed and number of virtual processes
            should yeld the same results. By default 42
        """
        self.debug = debug_mode
        
        # Experiment parameters
        self.trial_duration = 1100. if self.debug else 6000.  # Trial duration
        self.eval_time_window = 20. # Time window to check response via spike count
        self.tail_of_trial = self.trial_duration - self.eval_time_window
        self.min_DA_wait_time = 100. # Minimum waiting time to reward
        self.max_DA_wait_time = 1000. # Maximum waiting time to reward
        self.warmup_magnitude = 1. if self.debug else 25. # The duration of the warmup period is        
                                                          # given by warmup_magnitude * vta.tau_n

        # A random number generator (used to determine the sequence of cues)
        self.rng = np.random.RandomState(seed)

        # The brain to be trained
        scale = .2 if self.debug else 1.
        self.brain = Brain(master_seed=seed, scale=scale)
        self.brain_initiated = False

        #MPI rank (here used basically just to avoid multiple printing)
        self.mpi_rank = MPI.COMM_WORLD.Get_rank()
        self.rank0 = self.mpi_rank == 0

    
    def train_brain(self, n_trials=400, syn_scaling=True, aversion=True, 
        rev_learn=False, baseline_only=False, full_io=True, save_dir='/tmp/learner'):
        """ Creates a brain and trains it for a specific number of trials.
        
        Parameters
        ----------
        n_trials : int, optional
            Number of trials to perform, by default 400
        syn_scaling : bool, optional
            If True, a homeostatic plasticity rule (synaptic scaling like) will be applied at the
            end of every trial, by default True
        aversion : bool, optional
            If True, taking wrong actions makes dopamine sink bellow the baseline. If False, taking 
            wrong actions will keep dopamine concentrarion at baseline levels. By default True.
        rev_learn : bool, optional
            If True the stimuli/action association that results in reward is reversed, by default 
            False
        baseline_only : bool, optional
            If True dopamine is kept at baseline levels regardless of the action taken, by default 
            False
        full_io : bool, optional
            If False, there are no IOs to files and not essential MPI messages are not sent. Setting
            this variable to False is useful for tests and automated optimization processes that 
            depend only on the success rate By default True.
        save_dir : str, optional
            Directory where the outputs will be saved (if full_io=True). Existing files will be 
            overwritten. By default '/tmp/learner'
        
        Returns
        -------
        list[bool]
            A list with the success history
        """
        # Some handy variables
        color = {'red' : '\033[91m', 'green' : '\033[92m', 'none' : '\033[0m'}

        # Create brain and simulate a warmup
        if not self.brain_initiated:
            self._initiate_brain(full_io, save_dir)

        # Simulate trials
        trials_wall_clock_time = list()
        for trial in range(1, n_trials +1):
            self.trial_begin_ = nest.GetKernelStatus('time')
            if self.rank0:
                print(f'Simulating trial {trial} of {n_trials}:')

            # Adjust the amplitude of the dopamine bursts/dips
            self.brain.vta.adjust_salience_size(self.success_)
            
            # Simulate one trial and measure time taken to do it
            trial_start = time()
            self._simulate_one_trial(
                aversion=aversion, rev_learn=rev_learn, baseline_only=baseline_only)
            wall_clock_time = time() - trial_start
            trials_wall_clock_time.append(wall_clock_time)

            # Synaptic scaling
            if syn_scaling:
                self.brain.homeostatic_scaling(log_syn_change_factor=full_io)
            
            # Store experiment results on file(s):
            if full_io:
                self.brain.read_spike_detectors()
                self.brain.read_synaptic_weights()
                ExperimentResults(self).write(save_dir)
            self.brain.reset_spike_detectors()

            # Print some useful monitoring information
            n_suc = np.sum(self.success_)
            if self.rank0:
                print(f'Trial simulation concluded in {wall_clock_time:.1f} seconds')
                print(f'End-of-trial weight change: {self.brain.syn_change_factor_:.5f}')
                if self.success_[-1]:
                    print(f'{color["green"]}Correct action{color["none"]}')
                else:
                    print(f'{color["red"]}Wrong action{color["none"]}')
                print(f'{n_suc} correct actions so far ({n_suc * 100. / len(self.success_):.2f}%)')
                mean_wct = np.mean(trials_wall_clock_time)
                print(f'Average elapsed time per trial: {mean_wct:.1f} seconds')
                remaining_wct = round(mean_wct * (n_trials - trial))
                print(f'Expected remaining time: {timedelta(seconds=remaining_wct)}\n')
        
        if full_io:
            self.brain.store_network_snapshot()
            NetworkSnapshot(self).write(save_dir)
        
        return self.success_

    def _initiate_brain(self, full_io, save_dir):
        # Create the whole neural network
        if self.rank0:
            print('\nBuilding network')
        build_start = time()
        n_nodes = self.brain.build_local_network()
        build_elapsed_time = time() - build_start

        # Write to file the experiment properties which are trial-independent
        if full_io:
            ExperimentMethods(self).write(save_dir)

        # Print build information
        warmup_duration = self.warmup_magnitude * self.brain.vta.tau_n
        if self.rank0:
            print(f'Building completed in {build_elapsed_time:.1f} seconds')
            print('Number of nodes:', n_nodes)
            print(f'Initial total plastic weight: {self.brain.initial_total_weight:,}')
            print(f'Simulating warmup for {warmup_duration} ms')
        
        # Simulate warmup
        warmup_start = time()
        syn_change = self.simulate_rest_state(
            duration=warmup_duration, reset_weights=True, return_change_factor=full_io)
        warmup_elapsed_time = time() - warmup_start
        
        # Print warmup statistics
        if self.rank0:
            print(f'Warmup simulated in {warmup_elapsed_time:.1f} seconds')
            print(f'Synaptic change during warmup: {syn_change:.5f}\n')
        
        # Some variable initiation
        self.success_ = list()
        self.brain_initiated = True


    def _simulate_one_trial(self, aversion, rev_learn, baseline_only):
        # Decide randomly what will be the next cue and do the corresponding stimulation
        self.cue_ = ['low', 'high'][self.rng.randint(2)]
        self.brain.cortex.stimulate_subpopulation(spop=self.cue_, delay=self.brain.dt)
        
        # Simulate evaluation window and count the resulting decision spikes
        self.brain.vta.set_drive(length=self.eval_time_window, drive_type='baseline')
        nest.Simulate(self.eval_time_window)
        decision_spikes = self.brain.striatum.count_decision_spikes()

        # Check if the action the correct one
        self.lminusr_ = decision_spikes['left'] - decision_spikes['right']
        if self.lminusr_ == 0:
            success = False
        else:
            success = (self.cue_ == 'low' and self.lminusr_ > 0) or \
                      (self.cue_ == 'high' and self.lminusr_ < 0)
            success = not success if rev_learn else success
        self.success_.append(success)
        
        # According to the action outcome, deliver the appropriate DA response
        if self.lminusr_ == 0 or baseline_only:  # just keep the baseline  
            self.brain.vta.set_drive(length=self.tail_of_trial, drive_type='baseline')
        else:
            wait_time = self.max_DA_wait_time - (abs(self.lminusr_) - 1) * 100.  #TODO: calibrate this
            wait_time = round(np.clip(wait_time, self.min_DA_wait_time, self.max_DA_wait_time))
            drive_type = 'rewarding' if success else 'aversive' if aversion else 'baseline'
            self.brain.vta.set_drive(
                length=self.tail_of_trial, drive_type=drive_type, delay=wait_time)

        # Simulate the rest of the trial
        nest.Simulate(self.tail_of_trial)

    
    def simulate_rest_state(self, duration=100., reset_weights=True, return_change_factor=True):
        """Simulates the network in its resting state, i.e.: no stimulus and under dopamine baseline
        levels. This function is used to simulate the warmup period and is a great debuging tool.
        
        Parameters
        ----------
        duration : float, optional
            Simulation duration, by default 100.
        reset_weights : bool, optional
            If true corticostriatal synapses will be set to it initial value after the simulation, 
            by default True
        return_change_factor : bool, optional
            If True returns the synaptic change factor that happened during the simulation. 
            by default True

        Returns
        -------
        [type]
            Synaptic change factor (i.e. the original total plastic weight divide by the total 
            weight after simulation). Ideally should be as close to 1. as possible.
        """        
        self.brain.vta.set_drive(length=duration, drive_type='baseline')
        nest.Simulate(duration)
        syn_change_factor = self.brain.get_total_weight_change() if return_change_factor else -1.
        self.brain.reset_spike_detectors()
        if reset_weights:
            self.brain.reset_corticostriatal_synapses()

        return syn_change_factor

