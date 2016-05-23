"""Reverter.
"""

from .auralisation import _DEFAULT_SETTINGS, recursive_mapping_update
from .propagation import *
import logging

from acoustics._signal import Signal
from acoustics.atmosphere import Atmosphere

import auraliser.tools
logger = auraliser.tools.create_logger(__name__)

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
        recursive_mapping_update(self.settings, _DEFAULT_SETTINGS)
        #self.settings.recursupdate(DEFAULT_SETTINGS)
        if settings:
            recursive_mapping_update(self.settings, settings)
            #self.settings.update(settings)
        
    def revert(self, signal):#, signal, doppler=True, atmospheric_absorption=True, spherical_spreading=True):
        """
        Calculate back from receiver signal to source signal.
        """
        samples = len(signal)
        fs = signal.fs
        
        distance = np.linalg.norm(self.source - self.receiver, axis=1)
        
        
        if self.settings['spreading']['include']:
            logger.info("revert: Unapply spherical spreading.")
            signal = unapply_spherical_spreading(signal, distance)
        
        # Atmospheric attenuation when discarding the reflected path
        if not self.settings['reflections']['include'] and self.settings['atmospheric_absorption']['include']:
            logger.info("revert: Unapply atmospheric absorption.")
            signal = unapply_atmospheric_absorption(signal,
                                                    fs,
                                                    self.atmosphere,
                                                    distance,
                                                    n_blocks=self.settings['atmospheric_absorption']['taps'],
                                                    n_distances=self.settings['atmospheric_absorption']['unique_distances']
                                                    )[0:samples]
        
        # Correction for atmospheric attenuation and reflected path
        if self.settings['reflections']['include'] and self.settings['atmospheric_absorption']['include']:
            pass


        if self.settings['doppler']['include'] and self.settings['doppler']['frequency']:
            logger.info("revert: Unapply Doppler frequency shift.")
            delay = distance / self.atmosphere.soundspeed
            signal = unapply_doppler(signal, 
                                     delay,
                                     fs,
                                     method=self.settings['doppler']['interpolation'],
                                     kernelsize=self.settings['doppler']['kernelsize']
                                     )
            # 
            if self.settings['doppler']['purge_zeros']:
                logger.info("revert: Purge zeros due to initial delay.")
                delay = int(distance[-1]/self.atmosphere.soundspeed * fs)
                signal = signal[0:len(signal)-delay]
        
        
        return Signal(signal, fs)


    def initial_delay(signal):
        """Initial delay
        """
        pass
    
