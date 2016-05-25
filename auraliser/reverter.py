"""Reverter.
"""

from .auralisation import _DEFAULT_SETTINGS, recursive_mapping_update
from .propagation import *
import logging

from acoustics._signal import Signal
from acoustics.atmosphere import Atmosphere

import auraliser.tools
logger = auraliser.tools.create_logger(__name__)

import auraliser
from streaming import Stream

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
        settings = self.settings
        atmosphere = self.atmosphere
        samples = len(signal)
        fs = signal.fs
        
        distance = np.linalg.norm(self.source - self.receiver, axis=1)
        nblock = settings['nblock']

        distance = Stream(iter(distance)).blocks(nblock)
        signal = Stream(iter(signal)).blocks(nblock)
        
        if settings['spreading']['include']:
            logger.info("revert: Unapply spherical spreading.")
            #signal = unapply_spherical_spreading(signal, distance)
            signal = auraliser.realtime.apply_spherical_spreading(signal.blocks(nblock),
                                                                  distance.copy().blocks(nblock),
                                                                  inverse=True)
        
        # Atmospheric attenuation when discarding the reflected path
        if settings['atmospheric_absorption']['include']:
            logger.info("revert: Unapply atmospheric absorption.")

            signal = auraliser.realtime.apply_atmospheric_attenuation(
                signal=signal,
                fs=fs,
                distance=distance.copy(),
                nhop=settings['atmospheric_absorption']['nhop'],
                atmosphere=atmosphere,
                ntaps=settings['atmospheric_absorption']['ntaps'],
                inverse=True,
            )

            #signal = unapply_atmospheric_absorption(signal,
                                                    #fs,
                                                    #self.atmosphere,
                                                    #distance,
                                                    #n_blocks=settings['atmospheric_absorption']['taps'],
                                                    #n_distances=settings['atmospheric_absorption']['unique_distances']
                                                    #)[0:samples]
        
        # Correction for atmospheric attenuation and reflected path
        if settings['reflections']['include'] and settings['atmospheric_absorption']['include']:
            pass


        if settings['doppler']['include'] and settings['doppler']['frequency']:
            logger.info("revert: Unapply Doppler frequency shift.")
            #delay = distance / self.atmosphere.soundspeed
            #signal = unapply_doppler(signal,
                                     #delay,
                                     #fs,
                                     #method=settings['doppler']['interpolation'],
                                     #kernelsize=settings['doppler']['kernelsize']
                                     #)
            signal = auraliser.realtime.apply_doppler(signal=signal,
                                                      delay=distance.copy()/atmosphere.soundspeed,
                                                      fs=fs)
            # 
            #if settings['doppler']['purge_zeros']:
                #logger.info("revert: Purge zeros due to initial delay.")
                #delay = int(distance[-1]/self.atmosphere.soundspeed * fs)
                #signal = signal[0:len(signal)-delay]

        if settings['doppler']['include'] and settings['doppler']['frequency'] and settings['doppler']['purge_zeros']:
            initial_distance = distance.copy().samples().peek()
            # Initial delay in samples
            initial_delay = int(math.ceil(initial_distance/atmosphere.soundspeed * fs))
            signal = signal.samples().drop(initial_delay)
            del initial_distance, initial_delay

        del distance

        return Signal(signal.toarray(), fs)


    def initial_delay(signal):
        """Initial delay
        """
        pass
    
