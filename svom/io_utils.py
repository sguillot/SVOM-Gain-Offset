import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import log

# SVOM SPECIFIC imports
from svom.utils import En2Ch
from svom.utils import Ch2En
import svom.plot_utils

#####################################
####### READING FUNCTIONS ###########
#####################################

### READING INPUT GAIN-OFFSET RELATION FROM FITS FILE
def read_rel(filename,rootname, randomize=False,plotmatrix=False):
    
    matrixfile = fits.open(filename)
    rel_data = matrixfile[1].data 
    detIDs  = rel_data['DETNUM']
    gains   = rel_data['GAIN']
    offsets = rel_data['OFFSET']
    log.info("Input gain-offset matrix {} loaded".format(path.basename(filename)))

    ## Add small errors
    ## Dev. matrix has same gain/offset values for all pixels
    if randomize is True:
        gains_errors  = np.random.normal(0.,0.001*np.mean(gains), len(gains))
        offset_errors = np.random.normal(0.,0.001*np.mean(offsets), len(offsets))
        gains   = gains + gains_errors
        offsets = offsets + offset_errors

    if plotmatrix:
        ### DEVLOPEMENT -- ARBITRARY LIST -> MATRIX 
        gain_mat = np.reshape(gains, (80,80))
        offset_mat = np.reshape(offsets, (80,80))
        
        fig_mat = plot_utils.plot_matrix(gain_mat,offset_mat)
        fig_mat.savefig("{}/input_matrix.png".format(rootname))
        
    ### Stacks the DETID, GAIN, OFFSET columns into a single array (needs three arrays of equal lengths)
    try:
        relations_table = np.column_stack((detIDs,gains,offsets))
        return relations_table 
    except:
        log.error("Error stacking the table of gain/offsets from fits file")
        exit()
        

### READING INPUT FITS FILE WITH ALL SPECTRA
def read_spec(filename,rootname,plotspec=False):
    
    evtfile = fits.open(filename)
    evtdata = evtfile[1].data
    hdr  = evtfile[1].header
    log.info("Spectral file {} loaded: {} spectra".format(path.basename(filename),len(evtdata['pixel'])))

    ### Get the exposure
    try:
        Exposure = hdr['EXPOSURE']
    except KeyError:
        log.warning("Can't read exposure from spectrum - Keyword does not exist in header")
        Exposure = None

    ## Plots only one of the input spectra (the one from pixel 0)
    if plotspec:
        log.warning("Plotting only one spectrum (out of {})".format(len(evtdata['pixel'])))
        spec_fig = plot_utils.plot_spec_ch(np.arange(evtdata['channels'][0]),evtdata['spectrum'][0])
        spec_fig.savefig('{}/input_spectrum_channel.png'.format(rootname),dpi=100)
    
    return evtdata, Exposure


### DEVELOPMENT ONLY 
##    INPUT SPECTRUM WITHOUT REDISTRIBUTION
##    THE ONE USED TO GENERATE THE INPUT MATRIX
def read_rawspec(filename,rootname, plotspec=False):
    
    evtfile = fits.open(filename)
    evtdata = evtfile[1].data
    hdr  = evtfile[1].header
    log.info("Raw (no redistribution) spectral file loaded: {} spectra".format(len(evtdata[0])-1))

    if plotspec:
        spec_fig = plot_utils.plot_spec_en(evtdata['energy'],evtdata['pix'])
        spec_fig.set_size_inches(12,6)
        spec_fig.savefig('{}/input_spectrum_energy_NoRedist.png'.format(rootname),dpi=100)
                                   
    return evtdata

### READING INPUT FILE WITH LINE CENTROID INFO 
def read_lines(filename):
    
    all_intervals = []
    all_centroids = []

    ## Currently reads an ASCII file - could be replaced by FITS file with spectral line info.
    ##   File format:  low_limit, centroids, centroids, centroids, upper_limit
    f = open(filename,"r")
    all_lines = f.readlines()
    for l in all_lines:
        if (l.startswith('#'))==False:                                  # Ignores commented lines
            line = l.split()
            interval = np.array([float(line[0]),float(line[-1])])       # interval is (low_limit,upper_limit)
            centroids = np.array([float(x) for x in line[1:-1]])        # centroids are all the values except first and last 
            all_intervals.append(interval)
            all_centroids.append(centroids)
    f.close()
    log.info("Centroid file {} loaded".format(path.basename(filename)))

    if len(all_intervals)!=len(all_centroids):                          # checking numbers of line blocks
        log.error("Error with Nb of blocks. Exiting!")
        exit()
    else:
        return(all_intervals,all_centroids)


#####################################
####### WRITING FUNCTIONS ###########
#####################################

### WRITES OUTPUT GAIN-OFFSET MATRIX
def write_rel(data_idx, data_gain, data_offset,
              file_in, file_out, rootname, clobber=True, plotmatrix=False):

    # data0 = idx
    # data1 = finalgain
    # data2 = finaloffset

    ## Checks if output file exists and if overwritting is allowed
    if path.isfile(file_out):
        if clobber:
            log.info("File {} will be overwritten".format(path.basename(file_out)))
            os.remove(file_out)
        else:
            log.warning("File {} already exists. Set clobber=True to overwrite.".format(path.basename(file_out)))

    ## Open previous matrix to copy column names for the output -- it's not the best way to do it :-(        
    matrix_in = fits.open(file_in)
    hdr_in  = matrix_in[1].header 
    cols_in = matrix_in[1].columns

    ## Creates the FITS columns from the data array and column names 
    # c1 = fits.Column(name=cols_in.names[0], array=np.array(data_idx), format=cols_in.formats[0])
    # c2 = fits.Column(name=cols_in.names[1], array=np.array(data_gain), format=cols_in.formats[1])
    # c3 = fits.Column(name=cols_in.names[2], array=np.array(data_offset), format=cols_in.formats[2])
    c1 = fits.Column(name=cols_in.names[0], array=data_idx, format=cols_in.formats[0])
    c2 = fits.Column(name=cols_in.names[1], array=data_gain, format=cols_in.formats[1])
    c3 = fits.Column(name=cols_in.names[2], array=data_offset, format=cols_in.formats[2])

    ## From columns to FITS table
    output_table = fits.BinTableHDU.from_columns([c1, c2, c3])
    
    if plotmatrix:
        ### DEVELOPEMENT -- TO PERMIT LESS THAN 6400 PIXEL MATRICES
        side_size = int(np.floor(np.sqrt(len(data_idx))))
        tot_size = side_size*side_size

        if (len(data_idx)-tot_size) != 0:
            log.warning("Matrix not square! {} pixels ignored on output ({}x{}) matrix image".format(len(data_idx)-tot_size,side_size,side_size))

        try:
            gain_mat = np.reshape(data_gain[:tot_size], (side_size,side_size))
            offset_mat = np.reshape(data_offset[:tot_size], (side_size,side_size))
            fig_mat = plot_utils.plot_matrix(gain_mat,offset_mat)
            fig_mat.suptitle("Input matrix for gain and offset")
            fig_mat.savefig("{}/output_matrix.png".format(rootname))
        except:
            log.warning("Problem plotting output matrix (check matrix size).")



    
    ## Writes output_table to FITS file
    try:
        output_table.writeto(file_out)
        log.info("Output gain/offset matrix for {} pixels written succesfully".format(len(data_idx)))
    except:
        log.warning("Could not write to file {}".format(path.basename(file_out)))



#####################################
######### OLD FUNCTIONS #############
#####################################

### CHANGED FROM READING EVENT FILE TO READING FITS FILE WITH SPECTRA
## DEPRECATED!!!

def __OLD_read_spec(filename,exptime=None,plotspec=False, rootname='Default'):
    
    evtfile = fits.open(filename)
    evtdata = evtfile[1].data
    hdr  = evtfile[1].header

    try:
        Exposure = hdr['EXPOSURE']
    except KeyError:
        log.warning("Can't read exposure from spectrum - Keyword does not exist in header")
        if exptime is not None:
            log.warning("Will not change exposure of input spectra")
            exptime=None
    
    mask_evts = (evtdata['effMult'] == 1)
    #mask_evts = np.logical_and(evtdata['effMult'] == 1, evtdata['procID'] == 1)
    mask_energy = np.logical_and(evtdata['energy']>0.0,evtdata['energy']<200.0) ## ONLY FOR DEVELOPMENT - reads energy!
    
    all_masks = np.logical_and(mask_evts,mask_energy)
    single_evts = evtdata[all_masks]

    if exptime is not None:
        if exptime>=Exposure:
            log.warning("Chosen exposure is larger than input spectrum (or equal)...")
            log.warning("   Using the full input spectrum exposure ({} sec)".format(Exposure))
            exptime=Exposure
        else:
            log.info("Re-scaling input spectrum from exposure time {} sec to {} sec".format(Exposure,exptime))
            Nb_evts = len(single_evts)
            cts_ratio = int(Nb_evts*(exptime / Exposure))
            random_ind = np.random.choice(Nb_evts, size=cts_ratio, replace=False)
            single_evts = single_evts[random_ind]
            log.info("  {} events selected out of {}".format(len(single_evts),Nb_evts))
      
    counts,  bin_edges = np.histogram(single_evts['energy'],bins=np.arange(4, 150, 0.15))
    energies = bin_edges[:-1]

    if 0 in counts:
        log.error("Spectrum contains bins with 0 events: {}".format(0 in counts))
        log.error("Try increasing the exposure time...")

    fakegain =   0.145  ## 0.145 in Gain/Offset Matrix
    fakeoffset = 3.000  ## 3.000 in Gain/Offset Matrix
    channels, _ = En2Ch(energies,fakegain,fakeoffset)  ## ONLY FOR DEVELOPMENT - to return a Channel spectrum

    if plotspec:
        spec_fig = plot_utils.plot_spec_ch(channels,counts)
        spec_fig.set_size_inches(12,6)
        spec_fig.savefig('{}/input_spectrum.png'.format(rootname),dpi=100)
        
    #return channels, counts
    return np.column_stack((channels, counts))    
    
