import numpy as np
import logging as log
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors

# SVOM SPECIFIC imports
from svom.utils import En2Ch
from svom.utils import Ch2En


###############################################################
# GENERAL PLOT FUNCTIONS
###############################################################


# BASIC ENERGY SPECTRUM (LINER AND LOG SPACE)
def plot_spec_en(energy, cts):
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.suptitle("Input energy spectrum (linear and log scales)")

    ax[0].set_xlabel('energy (keV)')
    ax[0].set_ylabel('Nb counts')
    ax[0].plot(energy, cts)
    ax[0].set_xlim(1, 110)
    
    ax[1].set_xlabel('(energy (keV)')
    ax[1].set_ylabel('Nb counts')
    ax[1].loglog(energy, cts)
    ax[1].set_xlim(1, 110)
    
    return fig


# BASIC CHANNEL SPECTRUM (LINER AND LOG SPACE)
def plot_spec_ch(chan, cts):
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.suptitle("Input channel spectrum (linear and log scales)")
    
    ax[0].set_xlabel('channels')
    ax[0].set_ylabel('Nb counts')
    ax[0].plot(chan, cts)
    
    ax[1].set_xlabel('channels')
    ax[1].set_ylabel('Nb counts')
    ax[1].loglog(chan, cts)

    return fig


# BASIC GAIN OFFSET MATRICES
def plot_matrix(gains, offsets):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
    # fig.tight_layout()
    
    # ax[0].set_xlabel('')
    # ax[0].set_ylabel('')
    ax[0].set_title('GAINS (in keV/channel)')
    gain_mat = ax[0].matshow(gains, interpolation='none')
    fig.colorbar(gain_mat, ax=ax[0], fraction=0.046, pad=0.04)

    # ax[1].set_xlabel('')
    # ax[1].set_ylabel('')
    ax[1].set_title('OFFSETS (in keV)')
    offset_mat = ax[1].matshow(offsets, interpolation='none')
    fig.colorbar(offset_mat, ax=ax[1], fraction=0.046, pad=0.04)

    return fig


###############################################################
# DIAGNOSTIC PLOTS PER PIXEL (to plot with args option --plotall)
###############################################################

# DEVELOPMENT ONLY
#    INPUT SPECTRUM WITHOUT REDISTRIBUTION
def plot_raw(spec_raw, centroids_ini):
    plt.plot(spec_raw[:, 0], spec_raw[:, 1], label='rawspec')
    for (i, cent) in enumerate(centroids_ini):
        plt.axvline(x=cent, color='r', linestyle='--', linewidth=1, label='Init. centr.' if i == 0 else '')
    return 


# Relation channel-energy fits
def plot_relation(ini,fin,err=None,fit_rel=None,bad_cent=None,
                  stats=None, tolerance=4,
                  originals=None):
    fig = plt.figure(figsize=(12, 8))
    fig.tight_layout()

    # Plotting true centroids (ini) and fitted centroids (fin)
    ax1 = plt.subplot(211)
    ax1.errorbar(ini,fin,yerr=err,fmt='.', label='centroids')
    ax1.set_ylabel('channel')
    
    if fit_rel is not None:
        # Plot best-fit relation (line fit_rel)
        ax1.plot(ini,fit_rel, color='orange', label='Fit relation')
        ax1.set_xlim(0,95)
        ax1.set_ylim(0,600)

        # Plot residuals of best-fit
        ax2 = plt.subplot(212, sharex=ax1)
        ax2.errorbar(ini,fin-fit_rel,yerr=err,fmt='.', label='centroids')
        ax2.axhline(y=0, color='orange')                # That's just a horizontal line!!!
        ax2.set_ylabel('Difference (in channels)')
        ax2.set_xlabel('keV')
        ax2.set_ylim(-4.,4.)
    else:
        ax1.set_xlabel('keV')
        ax1.set_xlim(0,95)

    # Show bad centroids in RED
    if np.any(bad_cent):
        ax1.errorbar(ini[bad_cent],fin[bad_cent],yerr=err[bad_cent],fmt='o', color='red',elinewidth=3, label='excluded')
        ax2.errorbar(ini[bad_cent],fin[bad_cent]-fit_rel[bad_cent],yerr=err[bad_cent],fmt='o', color='red',elinewidth=3, label='excluded')

    # Show 3-sigma error in GREEN
    if stats is not None and fit_rel is not None:
        #ax2.axhspan(stats[0]-3.0*stats[1],stats[0]+3.0*stats[1], alpha=0.5, color='green')
        low_bound = ini*(stats[0].nominal_value-tolerance*stats[0].std_dev) + (stats[1].nominal_value-tolerance*stats[1].std_dev)
        high_bound = ini*(stats[0].nominal_value+tolerance*stats[0].std_dev) + (stats[1].nominal_value+tolerance*stats[1].std_dev)
        #ax1.fill_between(ini,low_bound, high_bound, color='green')
        ax2.fill_between(ini,low_bound-fit_rel, high_bound-fit_rel, color='orange', alpha=0.2, label='Fit +/- {}-sigma'.format(tolerance))
        #ax2.plot(ini, lowB-fit_rel, color='r')
        #ax2.plot(ini, highB-fit_rel, color='r')

    if originals is not None:
        orig_relation = (ini-originals[1])/originals[0]
        ax2.plot(ini, orig_relation-fit_rel, color='r', linestyle='--', label='Original relation')

    ax1.legend()
    ax2.legend()
    return fig


# Plot block of spectral lines with fits and residuals 
def plot_block(spec_block,spec_cont,
               cont_func,fit_func,guess_func,
               centroids_ini,centroids_fit,centroids_err, tolerance=4,
               only_residuals=False):

    # Plot a spectral block
    if only_residuals==False:
        plt.errorbar(spec_block[:,0],spec_block[:,1],yerr=np.sqrt(spec_block[:,1]),label='spectrum')    # plots the data (with error bars)
        plt.plot(spec_block[:,0],cont_func,color='orange',zorder=5,label='continuum')                   # plots the continuum
        plt.plot(spec_block[:,0],guess_func,'-.',color='black',zorder=5,label='guess')                  # plots the initial guess function 
        plt.plot(spec_cont[:,0],spec_cont[:,1], 'o', color='black',zorder=5,label='spec. cont.')        # plots the points used for continuum guesses
        plt.plot(spec_block[:,0],fit_func,color='red', zorder=10, label='best fit')                     # plots the best-fit function

        # Plots vertical lines for the centroids 
        for (i,cent) in enumerate(centroids_ini):
            # Initial centroid values
            plt.axvline(x=cent, color='r', linestyle='--',linewidth=1, label='Init. centr.' if i==0 else '')
            plt.axvline(x=centroids_fit[i], alpha=0.6, color='g', linestyle='-', label='Fit centr.' if i == 0 else '')
            if centroids_err[i] is not None:
                # Best-fit centroids WITH 3-sigma uncertainties
                plt.axvspan(centroids_fit[i]-tolerance*centroids_err[i], centroids_fit[i]+tolerance*centroids_err[i], alpha=0.2, color='g',linestyle='-', label='Fit centr.' if i==0 else '')

    # Plot the residual of a spectral block
    else:
        # Plots the residuals of the data
        plt.errorbar(spec_block[:,0],(spec_block[:,1]-fit_func)/np.sqrt(spec_block[:,1]),yerr=1.0,label='residuals', fmt='.')
        for (i,cent) in enumerate(centroids_ini):
            # Initial centroid values
            plt.axvline(x=cent, color='r', linestyle='--',linewidth=1, label='Init. centr.' if i==0 else '')
            plt.axvline(x=centroids_fit[i], alpha=0.6, color='g', linestyle='-', label='Fit centr.' if i == 0 else '')
            if centroids_err[i] is not None:
                # Best-fit centroids WITH 3-sigma uncertainties
                plt.axvspan(centroids_fit[i]-tolerance*centroids_err[i], centroids_fit[i]+tolerance*centroids_err[i], alpha=0.2, color='g',linestyle='-', label='Fit centr.' if i==0 else '')

    return 


###############################################################
# OUTPUT RESULTS PLOTS (to plot with args option --plots)
###############################################################

# Histogram of differences between fitted and true/input values
def plot_difference(diff_gains, diff_offset, bins=30):

    fig = plt.figure(figsize=(12, 9))
    
    # Plots the difference in gain
    ax0 = plt.subplot(211)
    ax0.hist(diff_gains, bins=bins)
    ax0.axvline(x=0.0, color='red')
    ax0.set_xlabel("Gain difference (% difference)")

    # Plots the difference in offset
    ax1 = plt.subplot(212)
    ax1.hist(diff_offset, bins=bins)
    ax1.axvline(x=0.0, color='red')
    ax1.set_xlabel("Offset difference (% difference)")
    
    return fig


# Comparison of best fit gain/offset for all pixels
def plot_comparison(indices,                                        # Index for the pixels
                    final_gains,  initial_gains,  final_gains_err,  # Gains  (best fit, initial, and best fit uncertainties)
                    final_offset, initial_offset, final_offset_err, # Offset (best fit, initial, and best fit uncertainties)
                    original_gain = None, original_offset=None,     # True values of Gain and Offset (used to make input spectra) -- For development only
                    pix_over_specs=None):                           # Flagged pixels with reconstruction difference over specs    -- For development only

    # fig = plt.figure(figsize = (12,9))
    fig, axes = plt.subplots(2, 2,  gridspec_kw={'width_ratios': [4, 1]}, figsize=(12,8))
    plt.subplots_adjust(wspace=0.00)

    # Plots for gain
    # Best-fit gains
    axes[0, 0].errorbar(indices, final_gains, fmt='x', color='green', yerr=final_gains_err, label='Fitted value')
    if pix_over_specs is not None:
        if len(pix_over_specs)>0:
            axes[0, 0].errorbar(indices[pix_over_specs], final_gains[pix_over_specs], fmt='x', color='red', yerr=final_gains_err[pix_over_specs], label='Fitted value (over specs)')
    # Initial values of the gains
    axes[0, 0].plot(indices, initial_gains, color='orange', linewidth=0.5, label='Previous value (from input matrix)')
    # Horizontal line for gain=0.145 keV/Channel
    axes[0, 0].axhline(y=0.145, color='red', linewidth=1)
    # Grid and ticks
    axes[0, 0].grid(which='major', axis='x', linewidth=0.7, alpha=1.0)
    axes[0, 0].minorticks_on()
    axes[0, 0].grid(which='minor', axis='x', linewidth=0.5, alpha=0.5)
    axes[0, 0].set_ylabel("Gain (keV/channel)")
    # Dev. only option
    if original_gain is not None:
        axes[0, 0].plot(indices, original_gain, 'o', color='blue', linewidth=2, label='Value used of input spectra')

    axes[0, 1].hist(final_gains, bins=50, orientation='horizontal')
    axes[0, 1].tick_params(bottom=True, top=False, left=True, right=False)
    axes[0, 1].tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    axes[0, 1].minorticks_on()
    axes[0, 1].set_ylim(axes[0, 0].get_ylim())

    # Plots for offset
    # Best-fit offset
    axes[1, 0].errorbar(indices, final_offset, fmt='x', color='green', yerr=final_offset_err, label='Fitted value')
    if pix_over_specs is not None:
        if len(pix_over_specs)>0:
            axes[1, 0].errorbar(indices[pix_over_specs], final_offset[pix_over_specs], fmt='x', color='red', yerr=final_offset_err[pix_over_specs], label='Fitted value (over specs)')
    # Initial values of the offset
    axes[1, 0].plot(indices, initial_offset, color='orange', linewidth=0.5, label='Previous value (from input matrix)')
    # Horizontal line for offset=3.0 keV
    axes[1, 0].axhline(y=3.000, color='red', linewidth=1)
    # Grid and ticks
    axes[1, 0].grid(which='major', axis='x', linewidth=0.7, alpha=1.0)
    axes[1, 0].minorticks_on()
    axes[1, 0].grid(which='minor', axis='x', linewidth=0.5, alpha=0.5)
    axes[1, 0].set_xlabel("Pixel number")
    axes[1, 0].set_ylabel("Offset (keV)")
    # Dev. only option
    if original_offset is not None:
        axes[1, 0].plot(indices, original_offset, 'o', color='blue', linewidth=2, label='Value used to generate spectra')

    axes[1, 1].hist(final_offset, bins=50, orientation='horizontal')
    axes[1, 1].tick_params(bottom=True, top=False, left=True, right=False)
    axes[1, 1].tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    axes[1, 1].minorticks_on()
    axes[1, 1].set_ylim(axes[1, 0].get_ylim())

    #handles, labels = axes[0, 0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right')
    axes[0, 0].legend(loc='best')

    return fig


# Reconstruction uncertainty, requirements, and pass fraction
def plot_reconstruction(gain,offset,
                        gain_err,offset_err):

    yaxis_max = 0.5
    lowband_thres=0.3
    highband_thres=0.5
    channels = np.arange(1,1024,1)

    pix_over_specs = []

    # Energies at which to calculate the pass fraction and make five histogram subplots
    SelectE = np.array([4,15,50,80,150])
    
    fig = plt.figure(figsize=(12, 8))
    
    ax0 = plt.subplot2grid((2, len(SelectE)), (0, 0), rowspan=1, colspan=len(SelectE))
    ax0.set_ylim([0,1.1*yaxis_max])

    # Table to store the reconstruction errors at the given Selected energies (for histogram subplots)
    reconst_err = np.zeros(shape=(len(gain),len(SelectE)))
    
    # Calculate and plot reconstruct error/uncertainty for each pixel
    for (i,g) in enumerate(gain):
        # Get energy and uncertainty from gain, offset and their uncertainties
        energies, energies_err = Ch2En(channels,g,offset[i], gain_err[i], offset_err[i])

        line_color = 'black'
        line_alpha = 0.1

        # Interpolate to calculate the reconstruction errors at the given Selected energies
        for (j,E) in enumerate(SelectE):
            reconst_err[i,j]  = np.interp(E,energies,energies_err)
            if (E<=80 and reconst_err[i,j]>lowband_thres):
                line_color = 'orange'
                line_alpha = 0.5
                pix_over_specs.append(i)
            if (E>80 and reconst_err[i,j]>highband_thres):
                line_color = 'orange'
                line_alpha = 0.5
                pix_over_specs.append(i)

        ax0.plot(energies, energies_err, color=line_color, alpha=line_alpha)

        # Adjust the y-axis if needed 
        if energies_err[0]>yaxis_max:
            yaxis_max = energies_err[0]
            ax0.set_ylim([0,1.1*yaxis_max])

    # Show requirements at < 80 keV and > 80 keV     
    ax0.plot([4,80],[lowband_thres,lowband_thres],color='red')
    ax0.plot([80,150],[highband_thres,highband_thres],color='red')
    ax0.set_ylabel('Reconstruction uncertainty (keV)')
    ax0.set_xlabel('keV')

    # Makes the histogram subplots at each fo the given Selected energies
    for (j,E) in enumerate(SelectE):
        ax = plt.subplot2grid((2, len(SelectE)), (1, j), rowspan=1, colspan=1, yticklabels=[])
        ax.hist(reconst_err[:,j], bins=30)
        ax.set_xlabel('Reconst. error at {:0.1f} keV'.format(E))
        
        low_y, high_y = ax.get_ylim()
        low_x, high_x = ax.get_xlim()

        # Calculate the "Pass fraction" and draw line for requirement at < 80 keV and > 80 keV
        if E<=80:
            NbPass = len(reconst_err[(reconst_err[:,j]<=lowband_thres),j])
            if high_x > lowband_thres:
                ax.axvline(x=lowband_thres, color='r')
        else:
            NbPass = len(reconst_err[(reconst_err[:,j]<=highband_thres),j])
            if high_x > highband_thres:
                ax.axvline(x=highband_thres, color='r')

        # Writes "Pass fraction" on the histrogram subplots
        txt = ax.text(0.1*(high_x-low_x), 0.9*high_y,"{:0.2f}% pass".format(100*NbPass/len(gain)),fontsize=12)
        txt.set_bbox(dict(facecolor='white', alpha=0.7))

    pix_over_specs = np.unique(pix_over_specs)

    return fig, pix_over_specs


# Comparison of best fit gain/offset for all pixels
def plot_fit_stats(indices,                                        # Index for the pixels
                   redchi2,
                   LargeErr,
                   cent):

    # Color for False and True
    binary_cmap = matplotlib.colors.ListedColormap(['white', 'red'])

    # fig = plt.figure(figsize = (12,9))
    fig, axes = plt.subplots(2, 2,  gridspec_kw={'width_ratios': [4, 1]})
    plt.subplots_adjust(wspace=0.00)

    # Plots for gain
    # Reduced Chi Sq. of Line fits
    axes[0, 0].scatter(indices, redchi2, marker='.', color='green', label='Reduced Chi Square')
    # Grid and ticks
    axes[0, 0].grid(which='major', axis='x', linewidth=0.7, alpha=1.0)
    axes[0, 0].minorticks_on()
    axes[0, 0].grid(which='minor', axis='x', linewidth=0.5, alpha=0.5)
    axes[0, 0].set_ylabel("Reduced Chi Square")
    axes[0, 0].set_xlim(0, axes[0, 0].get_xlim()[1])
    axes[0, 0].set_ylim(0.6,1.6)

    axes[0, 1].hist(redchi2, bins=50, orientation='horizontal')
    axes[0, 1].tick_params(bottom=True, top=False, left=True, right=False)
    axes[0, 1].tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False)
    axes[0, 1].minorticks_on()
    axes[0, 1].set_ylim(axes[0, 0].get_ylim())

    ## TODO: Ticks Labels are hardcoded for 14 centroids!!! Fix this!

    # Plots for offset
    # Best-fit offset
    #axes[1, 0].scatter(indices, LargeErr, marker='.', color='green', label='Reduced Chi Square')
    #axes[1, 0].imshow(LargeErr.T, interpolation='none', aspect='auto', cmap=binary_cmap)
    LargeErrIdx = np.argwhere(LargeErr==1)
    axes[1, 0].scatter(LargeErrIdx[:,0],LargeErrIdx[:,1], s=1.0, marker='s', color='red')
    axes[1, 0].tick_params(bottom=True, top=False, left=True, right=False)
    axes[1, 0].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    # Grid and ticks
    axes[1, 0].minorticks_on()
    axes[1, 0].set_xlabel("Pixel number")
    axes[1, 0].set_ylabel("Centroids w/ Large % error")
    #axes[1, 0].set_yticks(np.arange(0,15,2), np.arange(1,16,2) )  ## MATPLOTLIB 3.5
    axes[1, 0].set_yticks(np.arange(0,15,2))                       ## MATPLOTLIB 3.4
    axes[1, 0].set_yticklabels(np.arange(1,16,2) )                 ## MATPLOTLIB 3.4
    axes[1, 0].grid(which='major', axis='x', linewidth=0.7, alpha=1.0)
    axes[1, 0].grid(which='minor', axis='x', linewidth=0.5, alpha=0.5)
    axes[1, 0].grid(which='major', axis='y', linewidth=0.7, alpha=1.0)
    axes[1, 0].set_xlim(axes[0, 0].get_xlim())
    axes[1, 0].set_ylim(-0.5,14)
    #axes[1, 0].invert_yaxis()

    # Input Centroid energies
    labels_centroids = ["{:.2f} keV".format(e) for e in np.concatenate(cent).ravel()]

    centroids = np.arange(0,len(LargeErr[0]),1)
    a = np.sum(LargeErr, axis=0)
    axes[1, 1].barh(centroids, width=a)
    axes[1, 1].tick_params(bottom=True, top=False, left=True, right=False)
    axes[1, 1].tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=True)
    #axes[1, 1].minorticks_off()
    #axes[1, 1].set_yticks(np.arange(0,15,1), centroids)     ## MATPLOTLIB 3.5
    axes[1, 1].set_yticks(np.arange(0,14,1))                 ## MATPLOTLIB 3.4
    axes[1, 1].set_yticklabels(labels_centroids)             ## MATPLOTLIB 3.4
    axes[1, 1].set_ylim(axes[1, 0].get_ylim())
    axes[1, 1].yaxis.set_tick_params(labelsize=7)
    axes[1, 1].grid(which='major', axis='y', linewidth=0.7, alpha=1.0)
    #handles, labels = axes[0, 0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right')
    axes[0, 0].legend(loc='best')

    return fig