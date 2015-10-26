"""
This module contains signal generators.

"""
import numpy as np
from scipy.signal import firwin, firwin2
from scipy.signal import fftconvolve as convolve
import abc
from acoustics import Signal
import acoustics.signal
import acoustics.generator
import matplotlib.pyplot as plt
#import logging
import warnings




class Generator(object, metaclass=abc.ABCMeta):
    """
    Abstract class for containing spectral components.
    """
    
    @abc.abstractmethod
    def _output(self, t, fs):
        """
        This method should return the generated signal.
        
        :param t: Duration in seconds.
        :param fs: Sample frequency
        """
        pass
    
    def output(self, t, fs):
        return self._output(t, fs)
    
class Custom(Generator):
    """
    Use custom values for the signal.
    """
    
    def __init__(self, values=None):
        
        self.values = values
        """
        Values
        """
        
    def _output(self, t, fs):
        """
        """
        if not int(np.round(t*fs)) == len(self.values):
            raise ValueError("Custom generator values are wrong amount of samples.")
        return self.values
    

class Sine(Generator):
    """Sine wave generator.
    """
    
    def __init__(self, frequency):
        
        self.frequency = frequency
        """
        Frequency of the sine.
        """
    
    def _output(self, t, fs):
        """
        """
        return np.sin(2.0 * np.pi * self.frequency * np.arange(0.0, t, 1.0/fs)) * np.sqrt(2) # sqrt(2) for leq=94 dB

class AbstractNoise(Generator, metaclass=abc.ABCMeta):
    """Abstract class for noise generators.
    """
    
    def __init__(self, color='pink'):
        self.color = color
        """Color of noise.
        """
    
    #@property
    #def color(self):
        #"""Color of noise.
        #"""
        #return self._color
    
    #@color.setter
    #def color(self, value):
        #self._color = value
        ##if value not in acoustics.generator._noise_generators.keys():
            ##raise ValueError("Noise color is unavailable.")
        ##else:
            ##self._color = value
    
    @property
    def _noise_generator(self):
        return acoustics.generator._noise_generators[self.color]
    

class Noise(AbstractNoise):
    """White noise generator.
    """
    
    def __init__(self, color='pink'):
        super().__init__(color)
    
    def _output(self, t, fs):
        """
        """
        samples = int(np.round(t*fs))
        return self._noise_generator(samples)
        
    
class NoiseBands(AbstractNoise):
    """Bandpass filtered noise.
    
    """
    
    def __init__(self, bands, gains=None, order=8, color='pink'):
        
        super().__init__(color)
        
        self.bands = bands
        """Frequency bands.
        
        See :class:`acoustics.signal.Frequencies`.
        
        """
        
        self.gains = gains if gains is not None else np.zeros_like(self.bands.center)
        """Gain per band.
        """
        
        self.order = order
        """Order of bandpass filters.
        
        :warning: Higher orders can cause significant problems!
        
        """
        
        self.color = color
        """Color of noise.
        """
        
    def _output(self, t, fs):
        samples = int(np.round(t*fs))
        noise = Signal(self._noise_generator(samples), fs)
        signal = noise.bandpass_frequencies(self.bands, order=self.order, zero_phase=True)[1].gain(self.gains).sum(axis=0)
        return signal
        
        #fb = acoustics.signal.Filterbank(self.bands, sample_frequency=fs, order=self.order)
        #noise = self._noise(int(np.round(t*fs)))
        #output = np.zeros_like(noise)
        #try:
            #for band, gain in zip(fb.lfilter(noise), self.gains):
                #output += (band * db_to_lin(gain))
        #except ValueError:
            #warnings.warn("Cornerfrequency was higher than sample rate. Frequency band was not included.")
        #finally:
            #return output
        
        
def plot_generator(generator, t=5.0, fs=44100):
    
    signal = generator.output(t, fs)
    times = np.arange(0, t, 1.0/fs)
    
    frequencies, spectrum = acoustics.signal.ir2fr(signal, fs)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    p1 = ax1.plot(times, signal)
    ax1.set_xlabel('$t$ in s')
    ax1.set_ylabel('x')
    
    ax2 = fig.add_subplot(312)
    p2 = ax2.plot(frequencies, 20.0*np.log10(np.abs(spectrum)))
    ax2.set_xlabel('$f$ in Hz')
    ax2.set_ylabel('$20 \log10{|X|}$')
    ax2.set_xscale('log')
    
    ax3 = fig.add_subplot(313)
    p3 = ax3.plot(frequencies, np.angle(spectrum))
    ax3.set_xlabel('$f$ in Hz')
    ax3.set_ylabel('$\angle X in rad$')
    ax3.set_xscale('log')
    
    return fig
    
    
