import nest
import numpy as np

class BaseBrainStructure(object):
    # static numpy random number generators
    _py_rngs = None
    @property
    def py_rngs(self):
        return type(self)._py_rngs
    

    def __init__(self, scale=1):
        self.scale = scale  #TODO: make it static?
        self.N = dict()  # Number of neurons in each subpopulation
        self.neurons = dict()  # Neuron handles for each subpopulation
        self.spkdets = dict()  # Spike detectors
        self.events_ = dict()  # Events registered by the spike detectors
        self.grouped_synapses = list()  # A list of lists of plastic synapses grouped by target
        self.plastic_weight_setpoint = None # Total plastic weight per target neuron - will be used
                                            # as homeostatic setpoint for each neuron


    def build_local_network(self):
        raise NotImplementedError('All brain scructures must implement build_local_network()')


    def initiate_membrane_potentials_randomly(self, v_min=None, v_max=None, pops=['ALL']):
        if v_min == None and v_max == None:
            neu_pars = nest.GetDefaults('default_neuron')
            v_min, v_max = neu_pars['V_reset'], neu_pars['V_th']

        for pop in pops:
            node_info = nest.GetStatus(self.neurons[pop])
            local_nodes = [(ni['global_id'], ni['vp']) for ni in node_info if ni['local']]
            for gid, proc in local_nodes:
                nest.SetStatus([gid], {'V_m': self.py_rngs[proc].uniform(v_min, v_max)})
    

    def read_spike_detectors(self):
        for pop, spkdet in self.spkdets.items():
            self.events_[pop] = nest.GetStatus(spkdet, 'events')[0]
    
    
    def reset_spike_detectors(self):
        for spkdet in self.spkdets.values():
            nest.SetStatus(spkdet, {'n_events' : 0 })


    def group_synapses_per_target(self, sources, targets, syn_model):
        local_gids = [ni['global_id'] for ni in nest.GetStatus(targets) if ni['local']]
        self.grouped_synapses = [nest.GetConnections(sources, [gid], syn_model) for gid in local_gids]

    
    def homeostatic_scaling(self):
        # TODO: this loop is naturally parallelized if using mpi. Maybe it could be interesting to
        # paralelize this loop also for multithreading
        for syns in self.grouped_synapses:
            current_weights = np.array(nest.GetStatus(syns, 'weight'))
            scaling_factor = self.plastic_weight_setpoint / np.sum(current_weights)
            new_weights = scaling_factor * current_weights
            nest.SetStatus(syns, params='weight', val=new_weights)



