import nest
import numpy as np
from itertools import product
from mpi4py import MPI
from .BaseBrainStructure import BaseBrainStructure

class Striatum(BaseBrainStructure):
    """ Abstraction of a striatum. Contains just inhibitiony neurons mutually connected randomly 
    with constant indegree. Can be divided into two subpopulations. Connections within a 
    subpopulation have greater (i.e. less negative) weights than those across subpopulations. Class 
    members whose names are followed by a trailing _ (e.g. self.firing_rates_) are updated at every
    trial, the others are constant throughout the whole experiment.
    """
    def __init__(self, C_E, J_I, **args):
        super().__init__(**args)
    
        # Number of neurons
        #n = int(1.25 * C_E)  # neurons per subpopulation
        n = int(100 * self.scale)
        self.N['left'] = self.N['right'] = n
        self.N['ALL'] = self.N['left'] + self.N['right']

        # Connectivity
        epsilon = .1  # connection probability
        self.conn_params = {'rule': 'fixed_indegree', 'indegree': int(epsilon * n)} 

        # synapse parameters
        self.w = 0. # deviation between strength of inter and intra-subpopulation synapses
        self.J_inter = J_I * (1. + self.w)  # weight between neurons of distinct sub populations
        self.J_intra = J_I * (1. - self.w)  # weight between neurons of the same sub populations;

        # Background activity
        self.bg_rate = 7950.

        # MPI communication
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_procs = self.mpi_comm.Get_size()
        
    def build_local_network(self):
        # Create neurons and connect them to spike detectors
        for pop in ['left', 'right']:
            self.neurons[pop] = nest.Create('default_neuron', self.N[pop])
            self.spkdets[pop] = nest.Create('spike_detector')
            nest.Connect(self.neurons[pop], self.spkdets[pop])
        self.neurons['ALL'] = self.neurons['left'] + self.neurons['right']

        # Connect neurons to each other
        nest.CopyModel('default_synapse', 'striatum_intra_syn', {"weight": self.J_intra})
        nest.CopyModel('default_synapse', 'striatum_inter_syn', {"weight": self.J_inter})
        for origin, target in product(['left', 'right'], ['left', 'right']):
            syn_model = 'striatum_intra_syn' if origin == target else 'striatum_inter_syn'
            nest.Connect(self.neurons[origin], self.neurons[target], self.conn_params, syn_model)

        # Create and connect background activity
        background_activity = nest.Create('poisson_generator', params={"rate": self.bg_rate})
        nest.Connect(background_activity, self.neurons['ALL'], syn_spec='cortex_E_synapse')

        # initiate membrane potentials
        self.initiate_membrane_potentials_randomly()
    
    def count_decision_spikes(self):
        dec_spk = [nest.GetStatus(self.spkdets[pop], 'n_events')[0] for pop in ['left', 'right']]
        dec_spk = np.array(dec_spk, dtype='i')
        recvbuf = np.empty([self.mpi_procs, 2], dtype='i') if self.mpi_rank == 0 else None
        self.mpi_comm.Gather(dec_spk, recvbuf, root=0)
        if self.mpi_rank == 0:
            recvbuf = np.sum(recvbuf, axis=0)
            decision_spikes = {pop : recvbuf[it] for it, pop in enumerate(['left', 'right'])}
        else:
            decision_spikes = dict()
        decision_spikes = self.mpi_comm.bcast(decision_spikes, root=0)
        return decision_spikes
    