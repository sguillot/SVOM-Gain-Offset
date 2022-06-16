import os
import sys
import time
import argparse, configparser
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import logging as log
from os import path
from datetime import date

# MultiProc
from multiprocessing import Pool

# ASTROPY imports
# from astropy.io import fits
# from astropy import log   # DEPRECATING IN PROGRESS
from astropy.stats import sigma_clipped_stats

# SVOM SPECIFIC imports
from svom.io_utils import read_spec
from svom.io_utils import read_rel
from svom.io_utils import read_lines
from svom.io_utils import write_rel
# from svom.utils import En2Ch
# from svom.utils import Ch2En
import svom.plot_utils as plot_utils
from svom.fit_utils import FittingEngine
from svom.file_reader import FileReader as ParamReader


desc = """
Fit of fluorescence lines in background spectra 
obtained with DPIX to determine the gain and offset
of the channel-energy relation for the 6400 pixels\n

Current version still includes code for development
tests. They are marked explicitely to be removed.

Inputs include:
 * fits file with the spectra from the 6400 pixels
 * fits file for the gain-offset matrix
 * ascii file of line centroids to fit (to be replace by fits file?)

Output include:
 * fits file of the new gain-offset matrix
 * output plots with option --plots
 * diagnostic plots with option --plotall
 * ??? anything more ???

"""

def range_limited_int(arg):
    """ Type function for argparse - an int within some predefined bounds """
    min_val = 1
    max_val = 6400
    if arg is not None:
        try:
            f = int(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be an int")
        if f < min_val or f > max_val:
            raise argparse.ArgumentTypeError("Argument must be > {} and < {}".format(min_val, max_val))
        return f

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("-c", "--config_file", type=str, help='Config file')
# TODO:  THESE ARGUMENTS SHOULD PROBABLY BE MADE MANDATORY
parser.add_argument("--spectra", help="Input spectra (fits file matrix of 6400 spectra)", type=str, default=None)
parser.add_argument("--matrix", help="Input Gain-Offset matrix (fits file)", type=str, default=None)
parser.add_argument("--lines", help="Input spectral lines to fit in spectra (ascii file...for now)", type=str, default="lines_keV_4blocks.txt")
parser.add_argument("--rootname", help="Outputs rootname", type=str, default="Default")
# OPTIONAL ARGUMENTS
parser.add_argument("--tolerance", help="Filtering tolerance (default=4)", type=int, default=4)
parser.add_argument("--proc", help="Number of processors (default=1)", type=int, default=1)
parser.add_argument("--nbpix", help="Number of pixels to run (default=None, for tests only)", type=range_limited_int, default=None)
parser.add_argument("--pixels", help="One or more specific pixels to run ([1,6400], default=None, ignored if --nbpix is set)", nargs='+', type=range_limited_int, default=None)
parser.add_argument("--plots", help="Makes initial and final plots (default=False)", default=False, action='store_true')
parser.add_argument("--plotall", help="Makes all plot along the way, one per pixel (default=False)", default=False, action='store_true')
parser.add_argument("--showrawspec", help="Shows the raw spectrum (no Energy redistribution)", default=False, action='store_true')
parser.add_argument("--width", help="width (in keV) for energy redistribution", type=float, default=1.1)
parser.add_argument("--logfile", help="Name of logfile (default=rootname.log)", type=str, default=None)
parser.add_argument("--loglevel", help="Log Level (DEBUG, INFO, WARNING, ERROR, CRITICAL), Default=WARNING", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default="WARNING")

args = parser.parse_args()

if args.config_file:
    config = configparser.ConfigParser()
    config.read(args.config_file)
    defaults = {}
    defaults.update(dict(config.items("Defaults")))
    for arg in vars(args):
        if type(getattr(args, arg)) is bool:
            if arg in defaults:
                defaults[arg] = config.getboolean('Defaults', arg)
    parser.set_defaults(**defaults)
    args = parser.parse_args()        # Overwrite arguments from the parser

# DEPRECATED - plotall now works in Pool
# if (args.plotall is True) and (args.proc > 1):
#     args.proc = 1
#     log.warning("Plotting and multiprocessing has issues. Setting Nb of Proc to 1. Sorry!")



# MAIN RUN CALL
def mainrun(input_relation, output_relation,
            input_spec, outdir, outname,
            exp=None, inputrel=None):

    # OUTFILE ROOTNAME (CREATING DIRECTORY IF NEEDED)
    rootname = path.join(outdir, outname)
    if not os.path.exists(rootname):
        os.makedirs(rootname)

    # READ INPUT GAIN/OFFSET VALUES
    # NEEDED FOR PERFORMANCE TESTS RETRIEVE TRUE INPUT VALUES)
    # FOR DEVELOPMENT ONLY
    if inputrel is not None:
        original_gain, original_offset = np.loadtxt(inputrel)
    else:
        original_gain, original_offset = None, None

    # try:
    #     _ = len(original_gain)
    # except TypeError:
    #     print("not an array")
    #     original_gain = np.array([original_gain])
    #     original_offset = np.array([original_offset])

    # READ INPUT GAIN/OFFSET MATRIX
    # Option 'randomize' permits adding small errors to input gain and offset
    relations0 = read_rel(input_relation, rootname, randomize=False, plotmatrix=args.plots)

    # READ INPUT LINES INFO (fit intervals, line centroids)
    #      Currently an ASCII file - could be replaced by FITS file with spectral line info.
    #      Current file format:  low_limit, centroids, centroids, centroids, upper_limit
    #      Returns:   intervals = array of pairs (lower and upper limit)
    #                 centroids = list of arrays (centroids within a line block)
    intervals, centroids = read_lines(filelines)

    # READ INPUT SPECTRUM AND RETURN ARRAY OF SPECTRUM ARRAYS AND THE EXPOSURE
    #     DEV:  GEANT4 BACKGROUND (REPROCESSED TO CONTAIN 1 SPECTRUM FOR EACH PIXEL)
    spectra, exposure = read_spec(input_spec, rootname, plotspec=args.plots)
    if exposure is not None:
        exp = exposure

    # SPECTRUM WITHOUT REDISTRIBUTION -- FOR DEVELOPMENT ONLY
    #    Option useful to visualize relative line strength (to use with --plotall)
    if args.showrawspec:
        from svom.io_utils import read_rawspec
        rawspec_file = path.join(workdir, "BACKGROUNDS/spectrum_NoRedistribution.fits")
        rawspec = read_rawspec(rawspec_file, rootname, plotspec=args.plots)
    else:
        rawspec = None

    # If not optional number of pixel to process has been defined,
    #  setting the number of pixels to Nb of pixels in spectra matrix file.
    if args.nbpix is not None:
        pix = np.arange(0, args.nbpix)
        if args.pixels is not None:
            args.pixels = None
            log.warning("Ignoring --pixels (conflicting with --nbpix)")
    else:
        pix = np.arange(0, len(spectra['pixel']))
        if args.pixels is not None:
            pix = [p-1 for p in args.pixels]


    ################################################################################
    # Calls the main FittingEngine with either Multiprocessing Pool or simple 'for' loop

    t0 = time.time()
    if args.proc > 1:
        log.info("Running for {} pixels on {} processors...".format(len(pix), args.proc))
        log.info("...")
        try:
            # Initiates the Multiprocessing Pool
            pool = Pool(args.proc)

            # Initiates the FittingEngine Object
            fit_engine = FittingEngine(spectra, intervals, centroids, args.width, rootname, exposure=exp, tolerance=args.tolerance, rawspec=rawspec, plots=args.plotall)

            # Maps the Multiprocessing Pool with the FittingEngine
            # for the selected pixels (default, or from options --nbpix or --pixels)
            idx, final_gains, final_offset, final_gains_err, final_offset_err, LMFit_RedChi2, LMFit_LargeErr = zip(*pool.map(fit_engine, relations0[pix]))
            # FIT ENGINE RETURNS  idx0, gain_fit, offs_fit, gain_err, offs_err, FitResult.redchi, LargeErrors

        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        idx = np.asarray(idx)
        final_gains = np.asarray(final_gains)
        final_offset = np.asarray(final_offset)
        final_gains_err = np.asarray(final_gains_err)
        final_offset_err = np.asarray(final_offset_err)
        LMFit_RedChi2 = np.asarray(LMFit_RedChi2)
        LMFit_LargeErr = np.asarray(LMFit_LargeErr)

    else:

        idx = np.zeros(len(pix))
        final_gains = np.zeros(len(pix))
        final_offset = np.zeros(len(pix))
        final_gains_err = np.zeros(len(pix))
        final_offset_err = np.zeros(len(pix))
        LMFit_RedChi2 = np.zeros(len(pix))
        LMFit_LargeErr = np.zeros(( len(pix),len(np.concatenate(centroids).ravel())) )

        log.info("Running for {} pixels on {} processor (using a 'for' loop)...".format(len(pix), args.proc))
        log.info("...")

        # Initiates the FittingEngine Object
        fit_engine = FittingEngine(spectra, intervals, centroids, args.width, rootname, exposure=exp, tolerance=args.tolerance, rawspec=rawspec, plots=args.plotall)

        # Calls the FittingEngine object for the selected pixels (default, or from options --nbpix or --pixels)
        for i, idet in enumerate(tqdm.tqdm(relations0[pix], total=len(pix), position=0, leave=True)):
            tmp_idx, tmp_final_gains, tmp_final_offset, tmp_final_gains_err, tmp_final_offset_err, tmp_LMFit_RedChi2, tmp_LMFit_LargeErr = fit_engine(idet)
            idx[i] = tmp_idx
            final_gains[i] = tmp_final_gains
            final_offset[i] = tmp_final_offset
            final_gains_err[i] = tmp_final_gains_err
            final_offset_err[i] = tmp_final_offset_err
            LMFit_RedChi2[i] = tmp_LMFit_RedChi2
            LMFit_LargeErr[i,:] = tmp_LMFit_LargeErr
            #data_output.append(np.array(output))

    log.info("Time necessary for 6400 pixels: {:0.1f} sec - for {} processors".format((6400/len(pix))*(time.time()-t0), args.proc))
    ##
    #####################################################################################

    # Data exploration of outputs
    # Part of the code below contains 'development only' parts

    # Makes the output data into a numpy array
    #data = np.array(data_output)

    # Writes output gain-offset matrix to file ()
    write_rel(idx,final_gains,final_offset,
              input_relation, output_relation, rootname,
              clobber=True, plotmatrix=args.plots)

    # Organizes the initial gains and offset from input matrix
    indices        = relations0[pix, 0]
    initial_gains  = relations0[pix, 1]
    initial_offset = relations0[pix, 2]

    # Organizes the output gains and offset from FittingEngine output (data)
    #final_gains      = data[:, 1]
    #final_gains_err  = data[:, 3]
    #final_offset     = data[:, 2]
    #final_offset_err = data[:, 4]

    # RedChiSq
    #LMFit_RedChi2  = data[:, 5]
    #LMFit_LargeErr = data[:, 6]


    # Monitoring the changes in gain and offset from the initial values
    #  Setting to NaN the gain or offset that did not change by more than 3 sigma
    devsig = 2.5  # in units of sigma (calculated with astropy.sigma_clipped_stats)

    change_gain =  final_gains - initial_gains
    gain_mean, gain_med, gain_stddev = sigma_clipped_stats(change_gain, maxiters=4, sigma=3)
    mask_gain = np.abs(change_gain-gain_mean) < (devsig*gain_stddev)
    change_gain[mask_gain] = np.nan
    output_change_gain = np.column_stack((indices[np.invert(mask_gain)], final_gains[np.invert(mask_gain)]))

    change_offset = final_offset - initial_offset
    offset_mean, offset_med, offset_stddev = sigma_clipped_stats(change_offset, maxiters=4, sigma=3)
    mask_offset = np.abs(change_offset-offset_mean) < (devsig*offset_stddev)
    change_offset[mask_offset] = np.nan
    output_change_offset = np.column_stack((indices[np.invert(mask_offset)], final_offset[np.invert(mask_offset)]))

    np.savetxt("{}/change_gain_pixels.txt".format(rootname), output_change_gain, fmt='%4d %1.5f', delimiter='\t')
    np.savetxt("{}/change_offset_pixels.txt".format(rootname), output_change_offset, fmt='%4d %1.5f', delimiter='\t')

    # Summary plots if option --plots is set
    #
    if args.plots:

        # Comparison of best fit gain/offset for all pixels
        comp_fig = plot_utils.plot_comparison(indices,
                                              final_gains, initial_gains, final_gains_err,
                                              final_offset, initial_offset, final_offset_err)
        comp_fig.suptitle("Best fit gain and offset - {} pixels - Exposure: {} ks".format(len(pix), exp))
        comp_fig.savefig("{}/gain_offset_comparisons.png".format(rootname))

        # Fit Statistics figure
        fitstat_fig = plot_utils.plot_fit_stats(indices, LMFit_RedChi2, LMFit_LargeErr, centroids)
        fitstat_fig.suptitle("Fit Statistics - {} pixels - Exposure: {} ks".format(len(pix), exp))
        fitstat_fig.savefig("{}/fit_statistics.png".format(rootname))

        # Reconstruction uncertainty, requirements, and pass fraction
        rec_fig, _ = plot_utils.plot_reconstruction(final_gains, final_offset,
                                                 final_gains_err, final_offset_err)
        rec_fig.suptitle("Reconstruction uncertainty - {} pixels - Exposure: {} ks".format(len(pix), exp))
        rec_fig.savefig("{}/reconstruction_uncertainties.png".format(rootname))

        # Matrix of output differences
        # DEVELOPEMENT -- TO PERMIT LESS THAN 6400 PIXEL MATRICES
        side_size = int(np.floor(np.sqrt(len(pix))))
        tot_size = side_size*side_size
        if (len(pix)-tot_size) != 0:
            log.warning("Matrix not square! {} pixels ignored on output ({}x{}) matrix image".format(len(pix)-tot_size, side_size, side_size))

        change_gain_mat = np.reshape(change_gain[:tot_size], (side_size, side_size))
        change_offset_mat = np.reshape(change_offset[:tot_size], (side_size, side_size))

        fig_diff = plot_utils.plot_matrix(change_gain_mat, change_offset_mat)
        fig_diff.suptitle("Matrices of changes of gain and offsets (>3 sigma changes)")
        fig_diff.savefig("{}/change_matrix.png".format(rootname))

        plt.close('all')

        #print(original_gain, original_offset)

        # Development code -- to compare to real values used to generate the simulated input spectra
        if (original_gain is not None) and (original_offset is not None):
            # diff_gains = 100 * (final_gains - original_gain[pix])/final_gains
            # diff_offsets = 100 * (final_offset - original_offset[pix])/final_offset

            diff_gains = (final_gains - original_gain[pix])
            diff_offsets =  (final_offset - original_offset[pix])

            # Histogram of differences between fitted and true/input values
            diff_fig = plot_utils.plot_difference(diff_gains, diff_offsets, bins=30)
            diff_fig.suptitle("Gain and offset differences from input values - {} pixels - Exposure: {} ks".format(len(pix), exp))
            diff_fig.savefig("{}/gain_offset_histograms.png".format(rootname))

            # Reconstruction errors (DIFFERENCE), requirements, and pass fraction
            rec_fig, pix_over_specs = plot_utils.plot_reconstruction(final_gains, final_offset,
                                                     (final_gains - original_gain[pix]),
                                                     (final_offset - original_offset[pix]))
            rec_fig.suptitle("Reconstruction Difference - {} pixels - Exposure: {} ks".format(len(pix), exp))
            rec_fig.savefig("{}/reconstruction_difference.png".format(rootname))

            np.savetxt("{}/pixels_over_specs.txt".format(rootname), pix_over_specs, fmt='%04d', newline='\n')

            # Comparison of best fit gain/offset for all pixels
            comp_fig = plot_utils.plot_comparison(indices,
                                                  final_gains, initial_gains, final_gains_err,
                                                  final_offset, initial_offset, final_offset_err,
                                                  pix_over_specs=pix_over_specs)
            #comp_fig.tight
            comp_fig.suptitle("Best fit gain and offset - {} pixels - Exposure: {} ks".format(len(pix), exp))
            comp_fig.savefig("{}/gain_offset_comparisons.png".format(rootname))
            plt.tight_layout()
            plt.close('all')


if __name__ == '__main__':

    # Working directory root
    workdir = path.dirname(path.realpath(__name__))

    if args.logfile is not None:
        log.basicConfig(filename = path.join(workdir,"LOGS/{}".format(args.logfile)),
                        level = args.loglevel,
                        format = "%(asctime)s: %(name)s : %(funcName)s: %(levelname)s: %(message)s",
                        # format="%(asctime)s [%(levelname)s] %(message)s",
                        )
    else:
        log.basicConfig(level = args.loglevel,
                        format = "%(asctime)s: %(name)s : %(funcName)s: %(levelname)s: %(message)s",
                        # format="%(asctime)s [%(levelname)s] %(message)s",
                        )

    log.getLogger('matplotlib').setLevel(log.WARNING)

    # Input gain-offset matrix
    filerel_in  = path.join(workdir, "RELATION/{}".format(args.matrix))

    # Output gain-offset matrix
    filerel_out = path.join(workdir, "RELATION/AUX-ECL-PIX-CAL-{}.fits".format(date.today().strftime("%Y%m%d")))

    # Input centroids of lines to fit
    #   *** could eventually be a fits file with line information
    filelines   = path.join(workdir, "LINES_INFOS/{}".format(args.lines))

    # FINAL VERSION SHOULD LOOK LIKE THIS:
    inspec = path.join(workdir, "BACKGROUNDS/{}".format(args.spectra))
    outdir = path.join(workdir, "RESULTS")
    inrel = path.join(workdir, "RELATION/exact_Full_matrix_rel_6400pix_1000ks.txt")

    mainrun(filerel_in, filerel_out,
            inspec,
            outdir, args.rootname,
            exp=1000, inputrel=inrel)
