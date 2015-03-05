"""
The auralisation module provides generic tools for auralisation and calculating back from the receiver to the source.
"""

#import numpy as np
#np.seterr(all='raise')

from .auralisation import Auraliser, Geometry, get_default_settings
#from signal import Signal
from acoustics.atmosphere import Atmosphere


from auraliser.reverter import Reverter

from .generator import *
from acoustics.directivity import * 

from ism import Wall

from .sinks import mono
