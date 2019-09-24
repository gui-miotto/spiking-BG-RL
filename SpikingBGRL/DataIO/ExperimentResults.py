import pickle, time, os.path

class ExperimentResults(object):

    def __init__(self, exp):
        self.rank = exp.mpi_rank
        self.trial = len(exp.success_)
        self.trial_begin = exp.trial_begin_
        self.cue = exp.cue_
        self.reward_size = exp.brain.vta.reward_size_
        self.aversion_size = exp.brain.vta.aversion_size_
        self.lminusr = exp.lminusr_
        self.success = exp.success_[-1]
        self.events = exp.brain.events_
        self.weights_mean = exp.brain.weights_mean_
        self.weights_hist = exp.brain.weights_hist_
        self.syn_change_factor = exp.brain.syn_change_factor_


    def write(self, save_dir):
        save_dir = os.path.join(save_dir, 'trial-'+str(self.trial).rjust(3, '0'))
        while not os.path.exists(save_dir):
            if self.rank == 0:
                os.mkdir(save_dir)
            else:
                time.sleep(.1)
        file_path = os.path.join(save_dir, 'rank-'+str(self.rank).rjust(3, '0')+'.data')
        pickle.dump(self, open(file_path, 'wb'))