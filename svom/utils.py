import numpy as np
from astropy import log

def En2Ch(En,gain,offset,gain_err=None, offset_err=None):
    ## gain   in keV/chan
    ## offset in keV
    channel = (En-offset)/gain
    
    if np.any(channel<0):
        log.warning("Energy->Channel error! Some channel < 0...Continuing...But there will be problems!")
    
    ## Propagates the uncertainties in the channel (if errors for gain and offset are given)
    if (gain_err is not None) and (offset_err is not None):
        channel_err = channel * np.sqrt(np.power(offset_err/offset,2)+np.power(gain_err/gain,2)) 
        return channel, channel_err
    else:
        return channel, 0

def Ch2En(Ch,gain,offset,gain_err=None, offset_err=None):
    ## gain   in keV/chan
    ## offset in keV
    energy = gain*Ch + offset
    
    ## Propagates the uncertainties in the energy (if errors for gain and offset are given)
    if (gain_err is not None) and (offset_err is not None):
        energy_err = np.sqrt(np.power(Ch*gain_err,2)+np.power(offset_err,2)) 
        return energy, energy_err
    else:
        return energy, 0

## GAIN:    keV/chan -> chan/keV:  1/gain
## OFFSET:    keV    -> chan    :  -(offset/gain)
