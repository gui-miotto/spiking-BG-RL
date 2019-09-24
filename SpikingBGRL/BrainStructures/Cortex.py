import math, nest
from .BaseBrainStructure import BaseBrainStructure


class Cortex(BaseBrainStructure):
    def __init__(self, neu_params, J_E, **args):
        super().__init__(**args)

        # Number of neurons
        self.N['I'] = int(250 * self.scale)  # number of inhibitory neurons
        self.N['E'] = 4 * self.N['I']  # number of excitatory neurons
        self.N['E_rec'] = self.N['I_rec'] = int(100 * self.scale)  # num of neurons to record from
        self.N['low'] = self.N['high'] = int(100 * self.scale)  # subpops associated to stimuli
        self.N['E_no_S'] = self.N['E'] - self.N['low'] - self.N['high'] 
        self.N['ALL'] = self.N['I'] + self.N['E']

        # Connectivity
        epsilon = .1  # connection probability
        self.C = {pop : int(epsilon * n) for pop, n in self.N.items()} # num synapses per neuron

        # Synapse parameters
        g = 8.  # ratio inhibitory weight/excitatory weight
        self.J = {'E' : J_E} # amplitude of excitatory postsynaptic current
        self.J['I'] = -g * self.J['E']  # amplitude of inhibitory postsynaptic current

        # Background firing rate
        eta = .88  # external rate relative to threshold rate
        nu_th = neu_params['V_th'] * neu_params['C_m']
        nu_th /= self.J['E'] * self.C['E'] * math.e * neu_params['tau_m'] * neu_params['tau_syn_ex']
        nu_ex = eta * nu_th
        self.bg_rate = 1000.0 * nu_ex * self.C['E']

        # Stimulation protocol
        self.stim_duration = 10. #3.  #ms
        self.stim_intensity = 300.  #pA


    def build_local_network(self):
        # Create neurons
        for pop in ['E', 'I']:
            self.neurons[pop] = nest.Create('default_neuron', self.N[pop])
        
        # Sample subpopulations
        cut = 0
        for pop in ['low', 'high', 'E_rec']:
            self.neurons[pop] = self.neurons['E'][cut:cut+self.N[pop]]
            cut += self.N[pop]  
        self.neurons['I_rec'] = self.neurons['I'][:self.N['I_rec']]
        self.neurons['E_no_S'] = tuple(
            set(self.neurons['E']) - set(self.neurons['low']) - set(self.neurons['high']))
        self.neurons['ALL'] = self.neurons['E'] + self.neurons['I']

        # Connect subpopulations to spike detectors
        for pop in ['low', 'high', 'E_rec', 'I_rec']:
            self.spkdets[pop] = nest.Create('spike_detector')
            nest.Connect(self.neurons[pop], self.spkdets[pop])

        # Connect neurons with each other
        for pop in ['E', 'I']:
            syn_model_name = f'cortex_{pop}_synapse'
            nest.CopyModel('default_synapse', syn_model_name, {"weight": self.J[pop]})
            conn_params = {'rule': 'fixed_indegree', 'indegree': self.C[pop]}
            nest.Connect(self.neurons[pop], self.neurons['ALL'], conn_params, syn_model_name)

        # Create and connect background activity
        background_activity = nest.Create('poisson_generator', params={"rate": self.bg_rate})
        nest.Connect(background_activity, self.neurons['ALL'], syn_spec='cortex_E_synapse')

        # initiate membrane potentials
        self.initiate_membrane_potentials_randomly()

        # Create and connect sensory stimulus
        self.stimulus = dict()
        for pop in ['low', 'high']:
            self.stimulus[pop] = nest.Create('step_current_generator')
            nest.Connect(self.stimulus[pop], self.neurons[pop])

    def stimulate_subpopulation(self, spop, delay=0.):
        stim_onset = nest.GetKernelStatus()['time'] + delay
        nest.SetStatus(self.stimulus[spop], params={
            'amplitude_times' : [stim_onset, stim_onset + self.stim_duration],
            'amplitude_values' : [self.stim_intensity, 0.],
        })

