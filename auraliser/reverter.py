"""Reverter.
"""

from .auralisation import DEFAULT_SETTINGS
from .propagation import *


from acoustics._signal import Signal
from acoustics.atmosphere import Atmosphere

class Reverter(object):
    """Class for calculating back from receiver to source.
    """
    
    def __init__(self, source, receiver, atmosphere=None, settings=None):
        
        self.source = source
        """Source position."""
        self.receiver = receiver
        """Receiver position."""
        
        self.atmosphere = atmosphere if atmosphere else Atmosphere()
        
        #self.geometry = geometry if geometry else Geometry()
        
        self.settings = dict()
        """
        Configuration of this auraliser.
        """
        self.settings.update(DEFAULT_SETTINGS)
        if settings:
            self.settings.update(settings)
        
    def revert(self, signal):#, signal, doppler=True, atmospheric_absorption=True, spherical_spreading=True):
        """
        Calculate back from receiver signal to source signal.
        """
        samples = len(signal)
        fs = signal.fs
        
        distance = np.linalg.norm(self.source - self.receiver, axis=1)
        
        
        if self.settings['spreading']['include']:
        #if spherical_spreading:
            signal = unapply_spherical_spreading(signal, distance)
        if self.settings['atmospheric_absorption']['include']:
            signal = unapply_atmospheric_absorption(signal,
                                                    fs,
                                                    self.atmosphere,
                                                    distance,
                                                    taps=self.settings['atmospheric_absorption']['taps'],
                                                    n_d=self.settings['atmospheric_absorption']['unique_distances']
                                                    )[0:samples]
        if self.settings['doppler']['include'] and self.settings['doppler']['frequency']:
            delay = distance / self.atmosphere.soundspeed
            signal = unapply_doppler(signal, delay, fs)
        
        return Signal(signal, fs)
