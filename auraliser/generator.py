"""
This module contains signal generators.

"""
import numpy as np
from scipy.signal import firwin, firwin2
from scipy.signal import fftconvolve as convolve
import abc
import acoustics.signal
import acoustics.generator
import matplotlib.pyplot as plt

from .auralisation import db_to_lin

class Generator(object):
    __metaclass__ = abc.ABCMeta
    """
    Abstract class for containing spectral components.
    """
    
    @abc.abstractmethod
    def output(self, t, fs):
        """
        This method should return the generated signal.
        
        :param t: Duration in seconds.
        :param fs: Sample frequency
        """
        pass
    

class Custom(Generator):
    """
    Use custom values for the signal.
    """
    
    def __init__(self, values=None):
        
        self.values = values
        """
        Values
        """
        
    def output(self, t, fs):
        """
        """
        assert(t == len(self.values))
        return self.values
    

class Sine(Generator):
    """Sine wave generator.
    """
    
    def __init__(self, frequency):
        
        self.frequency = frequency
        """
        Frequency of the sine.
        """
    
    def output(self, t, fs):
        """
        """
        return np.sin(2.0 * np.pi * self.frequency * np.arange(0.0, t, 1.0/fs))
    
    
class Noise(Generator):
    """White noise generator.
    """
    
    def __init__(self, color='white'):
        
        self.color = color
    
    def output(self, t, fs):
        """
        """
        return acoustics.generator.noise(t*fs, color=self.color)
    
    @property
    def color(self):
        """Color of noise.
        """
        return self._color
    
    @color.setter
    def color(self, value):
        if value not in acoustics.generator.noise_generators.keys():
            raise ValueError("Noise color is unavailable.")
        else:
            self._color = value
        
    
class WhiteBand(Generator):
    """
    Bandpass filtered white noise.
    """
    
    def __init__(self, f1, f2, numtaps):
        #numtaps, cutoff, width=None, window='hamming', pass_zero=True, scale=True, nyq=1.0
        
        self.f1 = f1
        """Lower frequency of passband"""
        
        self.f2 = f2
        """Upper frequency of passband"""
        
        self.numtaps = numtaps
        """Amount of taps."""
    
    def output(self, t, fs):
        """
        """
        h = firwin(self.numtaps, cutoff=[self.f1, self.f2], pass_zero=False, nyq=fs/2.0)
        return convolve(np.random.randn(t*fs), h, mode='same') 
    

class WhiteOctaveBands(Generator):
    """
    Bandpass filtered noise where the pass band is a fractional-octave.
    Contrary to :class:`WhiteOctaveBand` this class allows setting setting the power of several bands.
    """
    
    def __init__(self, centerfrequencies, gains=None, fraction=1):
        """
        
        :param frequencies: List or array of center frequencies.
        :param gains: List or array of gains for each band. The gains are specified in dB.
        :param fraction: Fraction of fractional-octave filter.
        
        """
        
        self.centerfrequencies = centerfrequencies
        """
        Center frequencies.
        """
        
        self.gains = np.array(gains) if len(gains)==len(centerfrequencies) else np.zeros(len(centerfrequencies))
        """
        Gains in dB.
        """
        
        self.fraction = fraction
        """
        Fraction of fractional-octave filter.
        """
    
    def output(self, t, fs):
        """
        """

        nyq = fs/2.0
        f = list(self.centerfrequencies)
        f.insert(0, 0)
        f.append(nyq)
        gains = list(db_to_lin(self.gains))
        gains.insert(0, 0.0)
        gains.append(0.0)
        fil = firwin2(100, f, gains, nyq=nyq)
        r = np.random.randn(t*fs)
        return convolve(r, fil, mode='same')
        
        #f = acoustics.signal.OctaveBand(center=self.centerfrequencies, fraction=self.fraction)
        #fb = acoustics.signal.Filterbank(f, sample_frequency=fs, order=3)
        #r = np.random.randn(t*fs)
        #out = np.array(fb.filtfilt(r))
        #return (self.gains[:,None] * out).sum(axis=0)
        
        
    
    
class WhiteOctaveBand(Generator):
    """
    Bandpass filtered noise where the pass-band is a fractional-octave.
    """
    
    def __init__(self, centerfrequency, fraction=1):
        """
        
        :param centerfrequency: Centerfrequency
        :param fraction: Fraction
        
        """
        
        self.centerfrequency = centerfrequency
        """
        Center frequency of fraction-octave band.
        """
        
        self.fraction = fraction
        """
        Fraction of fractional-octave filter.
        """
        
    def output(self, t, fs):
        """
        """
        f = acoustics.signal.OctaveBand(center=self.centerfrequency, fraction=self.fraction)
        fb = acoustics.signal.Filterbank(f, sample_frequency=fs)
        r = np.random.randn(t*fs)
        return fb.filtfilt(r)[0]
        
        
        
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
    
    
