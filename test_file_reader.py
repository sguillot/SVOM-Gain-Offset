import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from os import path
from datetime import date

from svom.file_reader import FileReader

pars = FileReader("PFILES/test.par").params_list

print(pars)
print(pars[:][0])
