import os
from itertools import product
from collections import defaultdict
import numpy as np
from scipy.signal import butter, lfilter
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from SpikingBGRL.DataIO.Reader import Reader


light_green = (.807843137 , .921568627 , .921568627)  # (206, 235, 235)
dark_green = (.0 , 0.635294118, .701960784)  #(0, 162, 179)
dark_blue = (.047058824, .180392157, .509803922)  # (12, 46, 130)
lilac = 'xkcd:bright lilac'  # (201,94,251)
light_lilac = 'xkcd:light lilac'  # (237,200,255)
uni_red = (.764705882, .101960784, .211764706)  # (195,26,54)
mango = 'xkcd:mango'  # (255,166,43)
color_cycle = [dark_green, dark_blue, lilac, uni_red, mango]  
rev_color_cycle = color_cycle[::-1]

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

def build_methods_figure(figs_dir, data):
    print('Plotting methods figure')

    trial = 145  # just get a random one

    # Create figure
    plt.style.use('seaborn')
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
    mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{helvet}',    
       r'\usepackage{sansmath}',  
       r'\sansmath'               
    ] 
    fig = plt.figure(figsize=(12., 9.))
    plt.subplots_adjust(wspace=.25, hspace=.4, left=.06, right=.99, top=.95, bottom=.05)

    # Control what gets plotted, where and how big
    n_rows, n_cols = 3, 3
    plot_grid = {
        'raster_full' : (4, 5),
        'raster_decision' : 6,
        'DA_delay' : 7,
        'salience_mult' : 8,
    }

   # Format raster plot data
    raster_data, nsamp, = dict(), 50
    pops = ['E_rec', 'low', 'high', 'left', 'right']
    for ind, pop in enumerate(pops):
        gids = data.neurons[pop]
        senders, times = get_raster_data(
            data.events[trial][pop], 
            gids=gids[:nsamp], 
            shift_senders=True)
        times -= data.trial_begin[trial]
        raster_data[pop] = {'senders' : senders + (len(pops) - 1 - ind) * nsamp, 'times' : times}
    
    # Raster plot: decision period
    if 'raster_decision' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['raster_decision'])
        plt.title(r'\textbf{E)} Decision window raster plot')
        for ind, (pop, events) in enumerate(raster_data.items()):
            dec_events = get_raster_data(events, tmax=data.eval_time_window)
            plt.scatter(dec_events[1], dec_events[0], marker='o', s=5., label=pop, color=rev_color_cycle[ind])
        t_min = -.1 * data.eval_time_window
        t_max = 1.1 * data.eval_time_window
        plt.xlim(t_min, t_max)
        plt.ylim(-10, 260)
        plt.xlabel('time (ms)')

    # Raster plot: full trial
    if 'raster_full' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['raster_full'])
        plt.title(r'\textbf{D)} Full trial raster plot')
        for ind, (pop, events) in enumerate(raster_data.items()):
            pop_lbl = 'control' if pop == 'E_rec' else pop
            plt.scatter(events['times'] / 1000., events['senders'], marker='.', s=5., label=pop_lbl, color=rev_color_cycle[ind])
        plt.legend(loc='upper right', frameon=True, framealpha=1., markerscale=2.)
        plt.ylim(-10, 260)
        plt.xlabel('time (s)')

    # Plot: dopamine response delay
    if 'DA_delay' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['DA_delay'])
        plt.title(r'\textbf{F)} Dopamine response delay')
        plt.plot([1, 10, 13], [1000., 100., 100.], color=dark_green)
        plt.plot([13, 16], [100., 100.], color=dark_green, linestyle='--')
        plt.yticks(range(0, 1001, 100))
        plt.xticks(range(1, 17, 3))
        plt.xlabel('decision spikes difference')
        plt.ylabel('delay (ms)')

    # Plot: salience multiplier
    if 'salience_mult' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['salience_mult'])
        plt.title(r'\textbf{G)} Salience multiplier')
        sh31 = np.arange(0., 1.01, .001)
        reward_mult, aversi_mult = [], []
        for sh in sh31:
            reward_mult.append((2 - 2 * sh) ** .4625462336213272 if sh > .5 else 1.)
            aversi_mult.append((2 * sh - 1) ** .4625462336213272 if sh > .5 else 0.)
        plt.plot(sh31, reward_mult, label='rewarding', color=dark_green)
        plt.plot(sh31, aversi_mult, label='aversive', color=dark_blue)
        plt.legend()
        plt.xticks([0., .25, .5, .75, 1.])
        plt.xlabel('recent success rate')
        plt.ylabel('salience multiplier')
    
    # Save figure
    pdf = PdfPages(os.path.join(figs_dir, 'fig_methods_base.pdf'))
    pdf.savefig()
    pdf.close()
    plt.close(fig)


def build_results_figure(figs_dir, data):
    print('Plotting results figure')

    # Create figure
    plt.style.use('seaborn')
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
    mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{helvet}',    
       r'\usepackage{sansmath}',  
       r'\sansmath'               
    ] 

    fig = plt.figure(figsize=(15., 9.))
    plt.subplots_adjust(wspace=.25, hspace=.4, left=.03, right=.99, top=.95, bottom=.05)
    trials = range(1, data.num_of_trials+1)
    trial_ticks = range(0, data.num_of_trials+1, 50)
    reversal_trial = 151  # TODO: read from reader

    # Control what gets plotted, where and how big
    n_rows, n_cols = 3, 4
    plot_grid = {
        'w_low' : (1, 2),
        'w_high' : (5, 6),
        'w_E_rec' : (9, 10),
        'syn_scaling' : 3,
        'dec_spikes' : 4,
        'salience' : (7, 8),
        'act_selec' : (11, 12),
    }

    # Mean weight between stimulus and actions
    def pop_to_pop_weight_plot(source, position, letter):
        plt.subplot(n_rows, n_cols, position)
        source_lbl = source
        if source == 'low':
            letter = r'\textbf{A)}'
        elif source == 'high':
            letter = r'\textbf{B)}'
        elif source == 'E_rec':
            letter = r'\textbf{C)}'
            source_lbl = 'control'
        plt.title(letter + r' Mean weight of corticostriatal neurons from the ' + source_lbl + r' subnetwork')
        for tid, target in enumerate(['left', 'right']):
            weights_mean = [wm.loc[source, target] for wm in data.weights_mean]
            plt.plot(trials, weights_mean, label=target, color=color_cycle[tid])
        plt.axvline(x=reversal_trial, color=lilac, linestyle='--')
        plt.xticks(trial_ticks)
        plt.ylim(13., 50.)
        plt.xlabel('trial')
        plt.legend(loc='upper left')
    if 'w_low' in plot_grid.keys():
        pop_to_pop_weight_plot('low', plot_grid['w_low'], 'A')
    if 'w_high' in plot_grid.keys():
        pop_to_pop_weight_plot('high', plot_grid['w_high'], 'B')
    if 'w_E_rec' in plot_grid.keys():
        pop_to_pop_weight_plot('E_rec', plot_grid['w_E_rec'], 'C')


    # Plot: Synaptic scaling factors
    if 'syn_scaling' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['syn_scaling'])
        plt.title(r'\textbf{D)} Global synaptic changing factor')
        plt.plot(trials, data.syn_rescal_factor, color=dark_green)
        #reg_line = np.poly1d(np.polyfit(trials, data.syn_rescal_factor, 1))
        #plt.plot(trials, reg_line(trials), label='regression', color=dark_blue)
        #filter_numer, filter_denom = butter(1, .3)
        #synscal_filtered = lfilter(filter_numer, filter_denom, data.syn_rescal_factor)
        #plt.plot(trials, synscal_filtered, label='low-pass filtered', color=dark_blue)
        plt.axvline(x=reversal_trial, color=lilac, linestyle='--')
        #plt.legend(loc='upper left')
        plt.xticks(trial_ticks)
        plt.ylim(.9, 1.1)
        plt.xlabel('trial')
        #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Difference on decision spikes counts
    if 'dec_spikes' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['dec_spikes'])
        plt.title(r'\textbf{E)} Decision spikes difference')
        lminusr_abs = np.abs(data.lminusr)
        plt.plot(trials, lminusr_abs, color=dark_green)
        filter_numer, filter_denom = butter(10, .3)
        lminusr_abs_filtered = lfilter(filter_numer, filter_denom, lminusr_abs)
        plt.plot(trials, lminusr_abs_filtered, label='low-pass filtered', color=dark_blue)
        plt.axvline(x=reversal_trial, color=lilac, linestyle='--')
        plt.xticks(trial_ticks)
        plt.legend()
        plt.xlabel('trial')

    # Plot: Salience size
    if 'salience' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['salience'])
        plt.title(r'\textbf{F)} Salience size')
        plt.plot(trials, data.reward_size, label='rewarding', color=dark_green)
        plt.plot(trials, data.aversion_size, label='aversive', color=dark_blue)
        plt.axvline(x=reversal_trial, color=lilac, linestyle='--')
        plt.xticks(trial_ticks)
        plt.legend()
        plt.xlabel('trial')
        #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Probability of action selection (new version)
    if 'act_selec' in plot_grid.keys():
        plt.subplot(n_rows, n_cols, plot_grid['act_selec'])
        plt.title(r'\textbf{G)} Probability of correct action selection')
        prob_sucess, bin_size = [], 31
        for i in range(len(data.success)):
            suc_slice = data.success[:i+1][-bin_size:]
            prob_sucess.append(np.sum(suc_slice) / len(suc_slice))
        plt.plot(trials[10:], prob_sucess[10:], color=dark_blue, label='last 31 trials')
        #plt.scatter(trials, data.success, color=dark_blue, marker="$|$", label='trial-by-trial')
        plt.scatter(trials, data.success, color=dark_green, marker='|', label='trial-by-trial')
        plt.axvline(x=reversal_trial, color=lilac, linestyle='--')
        plt.ylim(-.05, 1.05)
        plt.xticks(trial_ticks)
        plt.legend(loc='center right')
        plt.xlabel('trial')
        #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save figure
    pdf = PdfPages(os.path.join(figs_dir, 'fig_results.pdf'))
    pdf.savefig()
    pdf.close()
    plt.close(fig)


if __name__ == '__main__':
    data_dir = '../../results/run1'
    figs_dir = os.path.join(data_dir, 'plots')
    if not os.path.exists(figs_dir):
        os.mkdir(figs_dir)
    data = Reader().read(data_dir)

    build_results_figure(figs_dir, data)
    build_methods_figure(figs_dir, data)

    







