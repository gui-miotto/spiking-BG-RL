import os
from itertools import product
from collections import defaultdict
import numpy as np
from scipy.signal import butter, lfilter
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from SpikingBGRL.DataIO.Reader import Reader

def get_raster_data(events, gids=None, shift_senders=False, tmin=None, tmax=None):
    senders = events['senders']
    times = events['times']
    if gids is not None:
        matches = np.isin(senders, gids)
        senders = senders[matches]
        times = times[matches]
    if tmin is not None:
        matches = np.where(times >= tmin)
        senders = senders[matches]
        times = times[matches]
    if tmax is not None:
        matches = np.where(times <= tmax)
        senders = senders[matches]
        times = times[matches]
    senders = senders - np.min(senders) if shift_senders and len(senders) > 0 else senders
    return senders, times


def build_trial_plots(figs_dir, data, run_parallel=True):
    # If debuging it may be a good idea to NOT run it in parallel
    if run_parallel:
        Parallel(n_jobs=-1)(delayed(
            build_one_trial_plot)(figs_dir, data, trial) for trial in range(data.num_of_trials))
    else:
        for trial in range(data.num_of_trials):
            build_one_trial_plot(figs_dir, data, trial)


def build_one_trial_plot(figs_dir, data, trial):
    print('Plotting trial', trial+1)
    
    # Create figure
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15., 9.))
    plt.subplots_adjust(wspace = 0.25, hspace = 0.4, left=0.1, right=0.95, bottom=.07)
    trial_num_str = str(trial + 1).rjust(3, '0')
    lminusr = data.lminusr[trial]
    cue = 'high tone' if data.cue[trial]=='high' else 'low tone'
    sel_act = 'went left' if lminusr > 0 else 'went right' if lminusr < 0 else 'no action'
    outcome = 'success' if data.success[trial] else 'fail'
    suptitle = f'Trial {trial_num_str}\n{cue} + {sel_act} = {outcome}'
    plt.suptitle(suptitle, size=16., weight='normal')
    
    # Control what gets plotted, where and how big
    n_rows, n_cols = 3, 4
    plot_grid = {
        'raster_decision' : 1,
        'raster_full' : (2, 3),
        'w_ctx_str' : 4,
        'w_pop_to_left' : 8,
        'w_pop_to_right' : 12,
        'cnx_mat' : 10,
    }

    # Format raster plot data
    raster_data, nsamp = dict(), 50 
    for i, (pop, gids) in enumerate(data.neurons.items()):
        if pop in ['low', 'high', 'left', 'right', 'E_rec']:
            senders, times = get_raster_data(
                data.events[trial][pop], 
                gids=gids[:nsamp], 
                shift_senders=True)
            times -= data.trial_begin[trial]
            raster_data[pop] = {'senders' : senders + i * nsamp, 'times' : times}
    
    # Raster plot: decision period
    if 'raster_decision' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['raster_decision'])
        plt.title('Decision period raster plot')
        for pop, events in raster_data.items():
            dec_events = get_raster_data(events, tmax=data.eval_time_window)
            plt.scatter(dec_events[1], dec_events[0], marker='o', s=5., label=pop)
        t_min = -.1 * data.eval_time_window
        t_max = 1.1 * data.eval_time_window
        plt.xlim(t_min, t_max)
        plt.xlabel('time (ms)')

    # Raster plot: full trial
    if 'raster_full' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['raster_full'])
        plt.title('Full trial raster plot')
        for pop, events in raster_data.items():
            plt.scatter(events['times'] / 1000., events['senders'], marker='.', s=5., label=pop)
        plt.legend(loc='upper right')
        plt.xlabel('time (s)')

    # Histogram: Mean weight between cortex and striatum 
    if 'w_ctx_str' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['w_ctx_str'])
        plt.title(f'Cortex to striatum weights')
        left_h = data.weights_hist[trial].loc['E', 'left']
        right_h = data.weights_hist[trial].loc['E', 'right']
        n_bars = len(left_h)
        x = np.linspace(0, data.wmax, n_bars)
        width = data.wmax / n_bars / 2.
        plt.bar(x - width/2, left_h, width, label='left', log=True, color='C2')
        plt.bar(x + width/2, right_h, width, label='right', log=True, color='C3')
        #plt.ylim(1., EM.N['A'] * EM.C['S'] * 10)
        plt.xlabel('weight')
        plt.legend(loc='upper center')

    # Histogram: Mean weight between cortex and striatum subpopulations
    def pop_to_pop_histogram(target, position):
        plt.subplot(n_rows, n_cols, position)
        plt.title(f'Cortex to {target} striatum weights')
        low_h = data.weights_hist[trial].loc['low', target]
        high_h = data.weights_hist[trial].loc['high', target]
        #Erec_h = data.weights_hist[trial].loc['E_rec', target]
        n_bars = len(low_h)
        x = np.linspace(0, data.wmax, n_bars)
        width = data.wmax / n_bars / 2.
        plt.bar(x - width/2, low_h, width, label='low', log=True)
        plt.bar(x + width/2, high_h, width, label='high', log=True)
        #plt.ylim(1., EM.N['A'] * EM.C['S'] * 10)
        plt.xlabel('weight')
        plt.legend(loc='upper center')
    if 'w_pop_to_left' in plot_grid.keys():
        pop_to_pop_histogram('left', plot_grid['w_pop_to_left'])
    if 'w_pop_to_right' in plot_grid.keys():
        pop_to_pop_histogram('right', plot_grid['w_pop_to_right'])

    # Connectivity matrix
    if 'cnx_mat' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['cnx_mat'])
        plt.title('Connectivity matrix')
        source = ['low', 'high', 'E_rec']
        target = ['left', 'right']
        cnn_matrix = data.weights_mean[trial].loc[source, target].to_numpy(dtype ='float32').T
        plt.imshow(cnn_matrix, origin='lower', vmin=0., vmax=data.wmax) 
        for (j,i),val in np.ndenumerate(cnn_matrix):
            plt.gca().text(i, j, f'{val:.2f}', ha='center', va='center')
        padx, pady = 10, '\n\n'
        plt.xticks(
            [.5, 1.5, 2.5], 
            ('low'.ljust(padx), 'high'.ljust(padx), 'E_rec'.ljust(padx)), 
            ha='right')
        plt.yticks([.5, 1.5], (pady + 'left', pady + 'right'), va='top')
        plt.xlabel('cortical sources')
        plt.ylabel('striatal targets')
    
    # Save figure
    fig_file = 'trial_' + trial_num_str + '.png'
    #pdf.savefig()
    plt.savefig(os.path.join(figs_dir, fig_file), transparent=False)
    plt.close(fig)

def build_experiment_plot(figs_dir, data):
    print('Plotting experiment overview')

    # Create figure
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15., 9.))
    plt.subplots_adjust(wspace = 0.25, hspace = 0.4, left=0.1, right=0.95, bottom=.07)
    plt.suptitle('Experiment overview', size=16., weight='normal')
    trials = range(1, data.num_of_trials+1)

    # Control what gets plotted, where and how big
    n_rows, n_cols = 3, 4
    plot_grid = {
        'dec_spikes' : (1, 2),
        #'syn_scaling' : (5, 6),
        'salience' : (5, 6),
        #'str_fr' : (9, 10),
        #'w_pop_to_left' : (3, 4),
        #'w_pop_to_right' : (7, 8),
        'w_low' : (3, 4),
        'w_high' : (7, 8),
        'w_E_rec' : (11, 12),
        #'act_selec' : (11, 12),
        'act_selec2' : (9, 10),
    }

    # Difference on decision spikes counts
    if 'dec_spikes' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['dec_spikes'])
        plt.title('Decision spikes difference')
        plt.plot(trials, np.abs(data.lminusr))
        plt.xlabel('trials')

    # Plot: Synaptic scaling factors
    if 'syn_scaling' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['syn_scaling'])
        plt.title('Synaptic scaling factor')
        plt.plot(trials, data.syn_rescal_factor)
        reg_line = np.poly1d(np.polyfit(trials, data.syn_rescal_factor, 1))
        plt.plot(trials, reg_line(trials), label='lin reg')
        plt.legend()
        #plt.ylim(.9, 1.1)
        plt.xlabel('trials')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot: Salience size
    if 'salience' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['salience'])
        plt.title('Salience size')
        plt.plot(trials, data.reward_size, label='rewarding')
        plt.plot(trials, data.aversion_size, label='aversive')
        plt.legend()
        #plt.ylim(bottom=-.1)
        plt.xlabel('trials')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot: Striatal firing rates
    if 'str_fr' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['str_fr'])
        plt.title('Striatum firing rates')
        n_events = defaultdict(list)
        #filter_numer, filter_denom = butter(5, .8)
        for pop in ['left', 'right']:
            for events in data.events:
                n_events[pop].append(len(events[pop]['times']))
            frs = np.array(n_events[pop]) / data.striatum_N[pop] / data.trial_duration * 1000.
            #frs = lfilter(filter_numer, filter_denom, frs)
            plt.plot(trials, frs, label=pop)
        n_events_all = (np.array(n_events['left']) + np.array(n_events['right']))
        frs_all = n_events_all / data.striatum_N['ALL'] / data.trial_duration * 1000.
        plt.plot(trials, frs_all, label='all')
        plt.legend()
        plt.xlabel('trials')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Mean weight between stimulus and actions
    def pop_to_pop_weight_plot(sources, targets, position):
        plt.subplot(n_rows, n_cols, position)
        #plt.title(f'Mean synaptic weight arriving at the {target} striatal subnetwork')
        for source, target in product(sources, targets):
            weights_mean = [wm.loc[source, target] for wm in data.weights_mean]
            #plt.plot(trials, weights_mean, label=source)
            plt.plot(trials, weights_mean, label=target)
        plt.xlabel('trials')
        plt.legend()
    if 'w_pop_to_left' in plot_grid.keys():
        pop_to_pop_weight_plot(['low', 'high', 'E_rec'], ['left'], plot_grid['w_pop_to_left'])
    if 'w_pop_to_right' in plot_grid.keys():
        pop_to_pop_weight_plot(['low', 'high', 'E_rec'], ['right'], plot_grid['w_pop_to_right'])
    if 'w_low' in plot_grid.keys():
        pop_to_pop_weight_plot(['low'], ['left', 'right'], plot_grid['w_low'])
    if 'w_high' in plot_grid.keys():
        pop_to_pop_weight_plot(['high'], ['left', 'right'], plot_grid['w_high'])
    if 'w_E_rec' in plot_grid.keys():
        pop_to_pop_weight_plot(['E_rec'], ['left', 'right'], plot_grid['w_E_rec'])


    # Probability of action selection (original version)
    if 'act_selec' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['act_selec'])
        plt.title('Probability of correct action selection')
        prob_sucess, bin_size = [], 25
        trials_coarse = range(0, data.num_of_trials, bin_size)
        for begin in trials_coarse:
            cut = data.success[begin:begin+bin_size]
            prob_sucess.append(np.sum(cut) / len(cut))
        plt.plot(trials_coarse, prob_sucess)
        plt.ylim(-.05, 1.05)
        plt.xlabel('trial')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Probability of action selection (new version)
    if 'act_selec2' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['act_selec2'])
        plt.title('Probability of correct action selection')
        prob_sucess, bin_size = [], 30
        for i in range(len(data.success)):
            suc_slice = data.success[:i+1][-bin_size:]
            prob_sucess.append(np.sum(suc_slice) / len(suc_slice))
        plt.plot(trials, prob_sucess)
        plt.ylim(-.05, 1.05)
        plt.xlabel('trial')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save figure
    fig_file = 'experiment_overview.png'
    #pdf.savefig()
    plt.savefig(os.path.join(figs_dir, fig_file), transparent=False)
    plt.close(fig)


if __name__ == '__main__':
    data_dir = '/tmp/learner'
    figs_dir = os.path.join(data_dir, 'plots')
    if not os.path.exists(figs_dir):
        os.mkdir(figs_dir)
    data = Reader().read(data_dir)

    build_trial_plots(figs_dir, data, run_parallel=True)
    build_experiment_plot(figs_dir, data)

    






