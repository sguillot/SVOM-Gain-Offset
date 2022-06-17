import argparse, configparser

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

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_file", type=str, help='Config file')
parser.add_argument("--spectra", help="Input spectra (fits file matrix of 6400 spectra)", type=str, default=None)
parser.add_argument("--matrix", help="Input Gain-Offset matrix (fits file)", type=str, default="AUX-ECL-PIX-CAL-20170101Q01.fits")
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
args = parser.parse_args()

print(args)

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

print(args)
