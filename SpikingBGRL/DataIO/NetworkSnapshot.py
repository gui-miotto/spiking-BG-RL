import pickle, time, os.path

class NetworkSnapshot(object):

    def __init__(self, exp):
        self.rank = exp.mpi_rank
        self.snapshot = exp.brain.snapshot_


    def write(self, save_dir):
        while not os.path.exists(save_dir):
            if self.rank == 0:
                os.mkdir(save_dir)
            else:
                time.sleep(.1)
        file_path = os.path.join(save_dir, 'snapshot-rank-'+str(self.rank).rjust(3, '0')+'.data')
        pickle.dump(self, open(file_path, 'wb'))