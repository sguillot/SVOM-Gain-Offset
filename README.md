# SVOM-Gain-Offset
Code to calculate the gain-offset relation for the 6400 pixels of the ECLAIRs camera onboard SVOM

### Required packages:
- numpy
- scipy
- lmfit
- uncertainties
- matplotlib
- astropy

### The folder NOTEBOOKS contains:

- iPython notebooks to make FITS file with 6400 spectra (one per pixel) for different exposure times from Background event files provided by Sujay Mate. One notebook is for the event file with the Earth in the FOV, one is for a full orbit (to select times when the Earth is not in the field of view).
- The notebook bins the events (in keV) into channel spectra, assuming some (fixed) value of the gain and offset.
- There is the possibility to add small errors to the gains and offsets for the 6400 pixels.
- There is an option to choose the event multiplicity.
- The outputs are:
   - 1 fits file with 6400 spectra, written in BACKGROUNDS/
   - 1 text file with gain/offset values, written in RELATION/. These are the values used to make the spectra, which can be useful to have for development and tests, to see how well the gain and offset can be recovered.

************************************************************************************

### The folder BACKGROUNDS contains:

- FITS files with the spectra of the 6400 pixels. For the moment, these are generated from background event files, and for different exposure times.
- A high-S/N spectrum without any energy redistribution, helpful at the development stage to identify lines easily.

************************************************************************************

### The folder RELATIONS/ contains:

- The FITS file with the input gain and offset for the 6400 pixels. For the moment, the values are just 0.145 keV/channel for the gain, and 3.0 keV for the gain. Later on, this matrix will be the input with the gain/offset values to be updated by the code.
- The output FITS file is written here as well, with an updated date.
- This folder also contains the gain/offset values that were used (in the notebooks, see above) to generate the channel spectra from the event files.

************************************************************************************

### The folder LINES_INFOS/ contains:

- Text files with the block energy intervals and the energies of the lines to fit
- There are multiple text files, for different cases.  For ex., I found that ignoring the 80-90 keV resulted in better fits, because of the line complexity in that range.

************************************************************************************
************************************************************************************


In code/, the main execution is done with:
> python main_pool.py 
>

Use option --help for more details.

************************************************************************************


A VERIFIER / A AMELIORER

- Verifier la convergence du fit (avec LMFIT) et les estimations des barres d'erreur.
- Verifier pourquoi le fit du gain et de l'offset (directement) a des barres d'erreur très large.
- Rajouter les infos sur les pixels (X,Y) à partir du DETNUM.
- Check title (pix Number) in diagnostic plots.


DONE:
- Verifier pourquoi l'offset a environ 0.5 canal de difference avec la valeur originale.
  - REASON:   Fitting a continuous "channel function" to discrete (integer) "channel data".
  - SOLUTION: 0.5 added to the input binned channel data in fit_utils.py -->  xdata = np.array(xdata) + 0.5
- Pour le fit de la relation linéaire Canal-Energie, changer scipy.curvefit en regression lineaire simple.
  - Trials show that this give worse results (see RESULTS/LINEAR_FIT_TESTS).
- Changer la variable "width" -> FWHM
  - DONE
- Fix axis values in Plot_Fit_statistics
  - DONE (ax.set_xticks version change in Matplotlib 3.4 to 3.5)
- Add centroid values to Ticks in Plot_Fit_statistics
  - DONE