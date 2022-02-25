import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ASTROPY imports
# from astropy.io import fits
from astropy import log
from astropy.stats import sigma_clipped_stats

# SCIPY imports
from scipy.optimize import curve_fit

# LMFIT
# from lmfit import Model
# import lmfit.models as models
from lmfit import Parameters, minimize,  fit_report
# from lmfir import report_fit

# SVOM SPECIFIC imports
# from svom.io_utils import read_spec
# from svom.io_utils import read_rel
# from svom.io_utils import read_lines
# from svom.utils import Ch2En
from svom.utils import En2Ch
import svom.plot_utils as plot_utils

# Warning suppress
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def LinearRelation(x, *params):
    """Straight line."""
    return (params[0] * x) + params[1]


def gauss(x, amp, cen, sigma):
    """Single Gaussian lineshape."""
    return amp * np.exp(-(x-cen)**2 / (2.*sigma**2))


def BlockModel(params, x, i, nb):
    """Calculate model of given block i from parameters (with nb Gaussians) for data set x."""
    
    # Start with the continuum (linear relation)
    model = params['a_{:d}'.format(i)]*x + params['b_{:d}'.format(i)]

    # Now add the gaussians
    for j in np.arange(nb):
        amp = params['amp_{:d}_{:d}'.format(i, j)]
        cen = params['cen_{:d}_{:d}'.format(i, j)]
        sig = params['sig_{:d}_{:d}'.format(i, j)]
        model = model + gauss(x, amp, cen, sig)  
    return model


def ModelResiduals(params, xdata, ydata, NbGaussPerBlock, weights=False):
    """Calculate total residual for fits of Gaussians to several data sets."""
    
    ndata = len(NbGaussPerBlock)
    resid = 0.0*ydata[:]
          
    # Calculate residual per data block, with or without uncertainties weights, i.e., sqrt(ydata)
    for i in range(ndata):
        if not weights:
            resid[i] = ydata[i] - BlockModel(params, xdata[i], i, NbGaussPerBlock[i])
        else:
            resid[i] = (ydata[i] - BlockModel(params, xdata[i], i, NbGaussPerBlock[i]))/(np.sqrt(ydata[i]))
            
    # Flatten the residuals for each data block this into 1D array (as required by minimize() function)
    return np.hstack(resid)


class FittingEngine(object):

    def __init__(self, all_spectra, intervals, centroids, width, rootname, exposure=0, rawspec=None, plots=False):
        """Initialize function """            

        # ALL SPECTRA (ARRAY OF SPECTRAL ARRAY, ONE FOR EACH PIXEL)
        self.all_spectra = all_spectra
        
        # RAW SPECTRUM (FOR DEV. ONLY)
        if rawspec is not None:
            self.rawspec = np.column_stack((rawspec['energy'], rawspec['pix']))
        else:
            self.rawspec = None
            
        # PLOTS - save plots
        self.plots = plots

        # FILENAMES
        self.rootname = rootname
        
        # Intervals, centroids and line width 
        self.width = width                                          # Width in keV
        self.intervals = intervals                                  # lower/upper bounds for each blocks (in keV)
        self.centroids = centroids                                  # Centroids in each of the blocks (in keV)
        self.all_ini_centroids = np.concatenate(centroids).ravel()  # All centroids (flattened array), in keV
        self.nb_blocks = len(centroids)                             # Number of spectral blocks
        self.NbGaussians = []                                       # Number of Gaussians (i.e. centroids) in each block
        for i in self.centroids:
            self.NbGaussians.append(len(i))
        self.NbGaussians = np.array(self.NbGaussians)

        # OTHERS
        self.exposure = exposure

    def __call__(self, initial_relation):
        """FittingEngine call, either from Pool() or from simple 'for' loop"""
        
        idx0    = int(initial_relation[0])                      # Pixel number
        gain0   = initial_relation[1]                           # Previously known value of gain in keV/chan
        offset0 = initial_relation[2]                           # Previously known value of offset in keV
        # log.info("Pixel {:4.0f}".format(idx0))

        # Sanity check!
        if idx0 != self.all_spectra['pixel'][idx0]:
            log.error('Problem with pixel number. Exiting...')
            exit()

        # Background spectrum for pixel idx0
        #     DEV:  GEANT4 BACKGROUND made into a matrix of spectra (columns: "channels", "spectrum")
        spectrum = np.column_stack((np.arange(self.all_spectra['channels'][idx0]), self.all_spectra['spectrum'][idx0]))
        
        # Some things needed for LMFIT
        MaxAmplitude = 1.5 * np.max(spectrum[:, 1])              # Use as bounds for line amplitudes in fits
        ApproxSig = (self.width/gain0)/2.35                     # from line width in keV to sigma in channel
        fit_params = Parameters()                               # Initialize the Parameters object for LMFIT
        xdata = []                                              
        ydata = []

        #  Loop on all blocks defined by the boundaries in interval.
        for (i, interval) in enumerate(self.intervals):
            # log.info("    Block {:.0f} in {:2.0f}-{:2.0f} keV".format((i+1),interval[0],interval[1]))

            # Convert centroids and intervals into channel, assuming input gain and offset
            centroid = self.centroids[i]                          # get numpy array of centroids for current block
            interval_ch, _  = En2Ch(interval, gain0, offset0)     # convert interval  in energy to channels
            centroid_ch, _  = En2Ch(centroid, gain0, offset0)     # convert centroids in energy to channels
        
            # Sanity check!
            if len(interval) > 2:                                 # checking size of the interval array
                log.error("Why is 'interval' more than 2 values?... Exiting!")
                exit()

            # Use masks to select the channel range of current block
            mask_block = (spectrum[:, 0] >= interval_ch[0]) & (spectrum[:, 0] <= interval_ch[1])
            spec_block = spectrum[mask_block]
            x = spec_block[:, 0]
            y = spec_block[:, 1]

            # Guess the continuun (lines excluded) with a line
            #   and define the line parameters (a,b) for current block
            cont_param = np.polyfit(spec_block[[0, -1], 0], spec_block[[0, -1], 1], 1)  # 1-order polynome from 1st and last points
            cont_func = np.poly1d(cont_param)
            fit_params.add('a_{:d}'.format(i), value=cont_param[0])   # TRY TO set min and max 
            fit_params.add('b_{:d}'.format(i), value=cont_param[1])   # TRY TO set min and max
            
            # Loop on all centroids of current block to define gaussian parameters
            for (j, cen) in enumerate(centroid_ch):

                # Guess the amplitude by interpolating number of counts over the continuun at centroid channel
                guess_amp = np.abs(np.interp(cen, spectrum[:, 0], spectrum[:, 1]) - np.interp(cen, x, cont_func(x)))

                # Define the gaussian parameters from the guessed amplitude, centroid channel, and approx line width
                fit_params.add('amp_{:d}_{:d}'.format(i, j), value=guess_amp, min=0.0, max=MaxAmplitude)
                fit_params.add('cen_{:d}_{:d}'.format(i, j), value=cen, min=interval_ch[0], max=interval_ch[1])
                fit_params.add('sig_{:d}_{:d}'.format(i, j), value=ApproxSig, min=0.5*ApproxSig, max=1.5*ApproxSig)

                # Link centroid and sigma to other the values for the reference line.
                if j > 0:                    
                    fit_params['cen_{:d}_{:d}'.format(i, j)].expr = 'cen_{:d}_{:d} + ({})'.format(i, 0, cen-centroid_ch[0])    # with Delta_Channel
                    fit_params['sig_{:d}_{:d}'.format(i, j)].expr = 'sig_{:d}_{:d}'.format(i, 0)                               # with fixed width
                    # fit_params['sig_{:d}_{:d}'.format(i,j)].expr = 'sig_{:d}_{:d}*sqrt(cen_{:d}_{:d}/cen_{:d}_{:d})'.format(i,0,i,j,i,0)  # with width related to Reference width
                        
            # Put the X-Y array (for current block) into a single one to fit all blocks at once
            xdata.append(x)
            ydata.append(y)

        # Adding 0.5 channel to xdata (as float) to represent the centers of the channels.
        # TODO: Raises a VisibleDeprecationWarning - this could be fixed
        xdata = np.array(xdata) + 0.5
        ydata = np.array(ydata)

        # DO THE FIT WITH LMFIT.MINIMIZE
        #   This minimizes the ModelResiduals given the xdata, ydata (of the blocks),
        #   the fit_params and NbGaussians in each block.  Other options include:
        #       - True (for residuals with weights)
        #       - "least_squares" minimization
        #       - ignoring NaN values (e.g., when zero counts in a)
        # fit_params.pretty_print()
        FitResult = minimize(ModelResiduals, fit_params,
                             args=(xdata, ydata, self.NbGaussians, True),
                             calc_covar=True, method='least_squares', nan_policy='omit')

        # TODO: Check convergence
        # TODO: Check why errors are large sometimes!
        # Full report of the fit (lots of text)
        # print(fit_report(FitResult))

        # Get the best-fit centroids and their uncertainties from FitResult
        fit_centroids = np.array([FitResult.params[key].value for key in FitResult.params if key.startswith("cen")])
        err_centroids = np.array([FitResult.params[key].stderr for key in FitResult.params if key.startswith("cen")])
        #  fit_sigma = np.array([FitResult.params[key].value for key in FitResult.params if key.startswith("sig")])

        if not FitResult.success:
            print('FitError - Write a FitError Exception?')
            print('           Status = {}, RedhiSq = {}'.format(FitResult.success, FitResult.redchi))


        #  Loop on all intervals to plot the blocks    
        if self.plots:
            withrawspec = int(0)                                    # used as index for GridPlot below
            
            # Loop on each blocks
            for (i, interval) in enumerate(self.intervals):
    
                # Convert centroids and intervals into channel, assuming input gain and offset
                centroid = self.centroids[i]                        # get numpy array of centroids for that line block
                interval_ch, _  = En2Ch(interval, gain0, offset0)     # convert interval  in energy to channels
                centroid_ch, _  = En2Ch(centroid, gain0, offset0)     # convert centroids in energy to channels

                # Use masks to select the channel range of current block  
                mask_block = (spectrum[:, 0] >= interval_ch[0]) & (spectrum[:, 0] <= interval_ch[1])
                spec_block = spectrum[mask_block]
                x = spec_block[:, 0]

                # Defines the initial guess model (continuum+Gaussian) for current block i
                initguess = BlockModel(fit_params, x, i, self.NbGaussians[i])

                # Defines the best-fit continuum for current block i
                continuum = BlockModel(FitResult.params, x, i, 0)                  # last argument '0' means no Gaussians

                # Defines the best-fit model (continuum+Gaussian) for current block i
                bestfit = BlockModel(FitResult.params, x, i, self.NbGaussians[i])

                # Get the best-fit centroids and their uncertainties from FitResult for the current block i
                block_centroids = [FitResult.params[key].value for key in FitResult.params if key.startswith("cen_{}".format(i))]
                block_err_centroids = [FitResult.params[key].stderr for key in FitResult.params if key.startswith("cen_{}".format(i))]

                # This is to define the GridPlots (only once, when i==0)
                if i == 0:
                    # The size of the grid depends on the rawspec option (True/False) and number of blocks.
                    if self.rawspec is not None:
                        figure1 = plt.figure(1,figsize=(4.0*self.nb_blocks, 12.0), facecolor='white')
                        fig_grid = gridspec.GridSpec(3, self.nb_blocks, width_ratios=np.ones(self.nb_blocks))
                    else:
                        figure1 = plt.figure(1,figsize=(4.0*self.nb_blocks, 8.0), facecolor='white')
                        fig_grid = gridspec.GridSpec(2, self.nb_blocks, width_ratios=np.ones(self.nb_blocks))
                        
                # Add the RawSpectrum (without redistribution) if options was set (FOR DEV. ONLY)
                if self.rawspec is not None:
                    # Use masks to select rawspec for current block (in keV)  
                    rawmask_block = (self.rawspec[:, 0] >= interval[0]) & (self.rawspec[:, 0] <= interval[1])
                    rawspec_block = self.rawspec[rawmask_block]
                    
                    # Convert to channel with initial gain and offset
                    rawspec_block[:, 0], _ = En2Ch(rawspec_block[:, 0], gain0, offset0)

                    # Plot rawspec on first line of grid
                    ax = plt.subplot(fig_grid[0, i])
                    withrawspec = int(1)
                    plot_utils.plot_raw(rawspec_block, np.array(centroid_ch))
            
                # Plot data, best-fit continuum and model, initial guess and centroids, 
                #    and best-fit centroids with uncertainties, for the current block 
                ax1 = plt.subplot(fig_grid[withrawspec, i])
                plot_utils.plot_block(spec_block, spec_block[[0, -1]],
                                      continuum, bestfit, initguess,
                                      np.array(centroid_ch), block_centroids, block_err_centroids)
            
                # Plot residuals, initial centroids, and best-fit centroids with uncertainties, for the current block 
                ax2 = plt.subplot(fig_grid[withrawspec+1, i])
                plot_utils.plot_block(spec_block, spec_block[[0, -1]],
                                      continuum, bestfit, initguess,
                                      np.array(centroid_ch), block_centroids, block_err_centroids, only_residuals=True)
                
                # Define labels...
                if i == 0:
                    ax2.set_ylabel('residuals')
                    if self.rawspec is not None:
                        ax.set_ylabel('counts (not redistributed)')

                ax1.set_ylabel('counts')
                ax2.set_xlabel('channel')

            # Save entire figure with legend 
            handles, labels = ax1.get_legend_handles_labels()
            figure1.legend(handles, labels, loc='upper right')
            plt.tight_layout(pad=1.0, w_pad=0.3, h_pad=1.0)
            figure1.savefig("{}/spec_blocks_PIX{:0>4}.png".format(self.rootname, int(idx0)))
            plt.close('all')
            
        # Checking uncertainties values of centroids 
        if np.any(err_centroids == 0.0):
            log.warning(" Pixel {:4.0f} -- Gaussians with zero uncertainties...".format(idx0))
            err_centroids[err_centroids == 0.0] = np.max(err_centroids)                           # Work around if centroids uncertainty is zero
        if np.any(err_centroids == None):
            err_centroids = None                                                                  # If one of them is None, they are all None.

        # LargeErrors = np.sum((err_centroids/fit_centroids)>0.05)
        LargeErrors = (err_centroids / fit_centroids) > 0.05 ## 0.05
        #if LargeErrors.any():
        #    print(LargeErrors)

        # REMOVE CENTROIDS WITH LARGE ERRORS
        #fit_ini_centroids = self.all_ini_centroids
        fit_ini_centroids = self.all_ini_centroids[~LargeErrors]
        fit_centroids = fit_centroids[~LargeErrors]
        err_centroids = err_centroids[~LargeErrors]

        # Now fitting the linear relation between fitted centroids (in channels)
        #    and true energies of spectral lines


        # Initial guess are the input gain and offset
        guesses = np.array([gain0, offset0])

        # Using CurveFit:
        #   x: True lines in keV, y: fitted centroids in channel, yerr: centroids errors (in channel)
        #   Output:  gain in chan/keV, offset in channels            
        LinRelPar, LinRelCov = curve_fit(LinearRelation, fit_ini_centroids, fit_centroids,
                                         p0=guesses, sigma=err_centroids, absolute_sigma=False)
        # LinRelPar, LinRelCov = curve_fit(LinearRelation, fit_ini_centroids, fit_centroids,
        #                                  p0=guesses, sigma=None, absolute_sigma=False)
        #  Uncertainties are the square of the diagonal indices of the covariance matrix
        LinRelErrors = np.sqrt(np.diag(LinRelCov))

        # Invert the best-fit gain and offset (from LinRelPar) 
        gain_fit = 1/LinRelPar[0]                               # GAIN:   keV/chan -> chan/keV
        gain_err = gain_fit * (LinRelErrors[0]/LinRelPar[0])    # propagate the uncertainties of gain in keV/chan into chan/keV
        offs_fit = -(LinRelPar[1]/LinRelPar[0])                 # OFFSET:  channel ->  keV
        offs_err = np.abs(offs_fit) * np.sqrt(np.power((LinRelErrors[1]/LinRelPar[1]), 2)+np.power((LinRelErrors[0]/LinRelPar[0]), 2))  # propagate the uncertainties of gain in channel into keV

        # Check for large deviations, and exclude bad centroids.
        FitRel = np.array(LinearRelation(fit_ini_centroids, *LinRelPar))                           # Calculate the best-fit energy-channel relation
        cent_mean, cent_median, cent_stddev = sigma_clipped_stats((fit_centroids-FitRel),
                                                                  maxiters=3, sigma_lower=3, sigma_upper=3)  # Calculate the statistics of centroids values
        bad_idx = (np.abs(fit_centroids-FitRel)) > (cent_mean+4.0*cent_stddev)                          # Get the indices of those that deviate by >4 sigma
        if np.any(bad_idx):
            log.warning(" Pixel {:4.0f} -- Some centroids are outliers (will be shown in red on plot)".format(idx0))
            log.warning("                  Re-run for pixel {:4.0f} with option --plotall to see outliers".format(idx0))
        else:
            bad_idx = None

        # Plot channel energy relation from best fit centroids and true values
        if self.plots:
            rel_figure = plot_utils.plot_relation(fit_ini_centroids, fit_centroids,
                                                  err=err_centroids, fit_rel=FitRel, bad_cent=bad_idx, stats=[cent_mean, cent_stddev])
            rel_figure.suptitle("Channel-Energy Fit - pixel {} - Exposure: {} ks".format(int(idx0), self.exposure))
            rel_figure.savefig("{}/ch_en_relation_PIX{:0>4}.png".format(self.rootname, int(idx0)))

        # Replotting the fit without the centroids with large deviations
        if np.any(bad_idx):
            log.warning(" Pixel {:4.0f} -- Re-fitting relation without outliers...".format(idx0))
            good_idx = np.invert(bad_idx)

            # Initial guess and linear fit with curve fit, like above, but only for the good_idx
            guesses = LinRelPar
            LinRelPar, LinRelCov = curve_fit(LinearRelation, fit_ini_centroids[good_idx], fit_centroids[good_idx],
                                             p0=guesses, sigma=err_centroids[good_idx], absolute_sigma=False)
            LinRelErrors = np.sqrt(np.diag(LinRelCov))

            # Invert the best-fit gain and offset (from LinRelPar), like above
            gain_fit = 1/LinRelPar[0]                               # GAIN:   keV/chan -> chan/keV
            gain_err = gain_fit * (LinRelErrors[0]/LinRelPar[0])    # propagate the uncertainties of gain in keV/chan into chan/keV
            offs_fit = -(LinRelPar[1]/LinRelPar[0])                 # OFFSET:  channel ->  keV
            offs_err = np.abs(offs_fit) * np.sqrt(np.power((LinRelErrors[1]/LinRelPar[1]), 2)+np.power((LinRelErrors[0]/LinRelPar[0]), 2))  # propagate the uncertainties of gain in channel into keV

            # Plot channel energy relation from best fit centroids and true values, WITHOUT the bad pixels
            if self.plots:
                FitRel = np.array(LinearRelation(fit_ini_centroids[good_idx], *LinRelPar))
                rel_figure = plot_utils.plot_relation(fit_ini_centroids[good_idx], fit_centroids[good_idx],
                                                      err=err_centroids[good_idx], fit_rel=FitRel, bad_cent=None, stats=[cent_mean, cent_stddev])
                rel_figure.suptitle("Channel-Energy Fit - pixel {} - Exposure: {} ks".format(int(idx0), self.exposure))
                rel_figure.savefig("{}/ch_en_relation_PIX{:0>4}_REFITTED.png".format(self.rootname, int(idx0)))

        # Return pixel index, best fit gain and offset and their errors
        return idx0, gain_fit, offs_fit, gain_err, offs_err, FitResult.redchi, LargeErrors
        
