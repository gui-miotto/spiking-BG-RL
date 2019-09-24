import pickle, time, os.path

class ExperimentMethods():

    def __init__(self, exp):
        self.rank = exp.mpi_rank
        self.striatum_N = exp.brain.striatum.N
        self.weights_count = exp.brain.weights_count
        self.trial_duration = exp.trial_duration
        self.eval_time_window = exp.eval_time_window
        self.wmax = exp.brain.vta.DA_pars['Wmax']
        self.neurons = {
            'I' : exp.brain.cortex.neurons['I'],
            'E' : exp.brain.cortex.neurons['E'],
            'E_rec' : exp.brain.cortex.neurons['E_rec'],
            'low' : exp.brain.cortex.neurons['low'],
            'high' : exp.brain.cortex.neurons['high'],
            'left' : exp.brain.striatum.neurons['left'],
            'right' : exp.brain.striatum.neurons['right'],
        }


    def write(self, save_dir):
        while not os.path.exists(save_dir):
            if self.rank == 0:
                os.mkdir(save_dir)
            else:
                time.sleep(.1)
        file_path = os.path.join(save_dir, 'methods-rank-'+str(self.rank).rjust(3, '0')+'.data')
        pickle.dump(self, open(file_path, 'wb'))
    
    