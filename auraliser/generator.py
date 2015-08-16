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
#import logging
import warnings

from .tools import db_to_lin


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
        return acoustics.signal.normalize(self._output(t, fs))
    
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
    
    def _output(self, t, fs):
        """
        """
        return np.sin(2.0 * np.pi * self.frequency * np.arange(0.0, t, 1.0/fs)) * np.sqrt(2) # sqrt(2) for leq=94 dB

class AbstractNoise(Generator, metaclass=abc.ABCMeta):
    """Abstract class for noise generators.
    """
    
    def __init__(self, color):
        self.color = color
        """Color of noise.
        """
    
    @property
    def color(self):
        """Color of noise.
        """
        return self._color
    
    @color.setter
    def color(self, value):
        if value not in acoustics.generator._noise_generators.keys():
            raise ValueError("Noise color is unavailable.")
        else:
            self._color = value
    
    @property
    def _noise(self):
        return acoustics.generator._noise_generators[self._color]
    


    
    
class Noise(AbstractNoise):
    """White noise generator.
    """
    
    def __init__(self, color='pink'):
        super().__init__(color)
    
    def _output(self, t, fs):
        """
        """
        return self._noise(t*fs)
        

#class NoiseBand(AbstractNoise):
    #"""
    #Bandpass filtered noise.
    #"""
    
    #def __init__(self, f1, f2, order=3, color='pink'):
        ##numtaps, cutoff, width=None, window='hamming', pass_zero=True, scale=True, nyq=1.0
        
        #super().__init__(color)
        
        #self.f1 = f1
        #"""Lower frequency of passband"""
        
        #self.f2 = f2
        #"""Upper frequency of passband"""
        
        #self.order = order
        #"""Filter order.
        
        #..seealso:: :func:`acoustics.signal.bandpass`
        
        #"""
        
    #def output(self, t, fs):
        #"""
        #"""
        #return acoustics.signal.bandpass(self._noise(t*fs), self.f1, self.f2, fs, order=self.order)
        ##h = firwin(self.numtaps, cutoff=[self.f1, self.f2], pass_zero=False, nyq=fs/2.0)
        ##return convolve(self._noise(t*fs), h, mode='same') 
    
    
class NoiseBands(AbstractNoise):
    """Bandpass filtered noise.
    
    """
    
    def __init__(self, bands, gains=None, order=1, color='pink'):
        
        super().__init__(color)
        
        self.bands = bands
        """Frequency bands.
        
        See :class:`acoustics.signal.Frequencies`.
        
        """
        
        self.gains = gains if gains is not None else np.ones_like(self.bands.center)
        """Gain per band.
        """
        
        self.order = order
        """Order of bandpass filters.
        
        :warning: Higher orders can cause significant problems!
        
        """
        
    def _output(self, t, fs):
        
        fb = acoustics.signal.Filterbank(self.bands, sample_frequency=fs, order=self.order)
        noise = self._noise(t*fs)
        output = np.zeros_like(noise)
        try:
            for band, gain in zip(fb.lfilter(noise), self.gains):
                output += (band * db_to_lin(gain))
        except ValueError:
            warnings.warn("Cornerfrequency was higher than sample rate. Frequency band was not included.")
        finally:
            return output
        
    
#class NoiseOctaveBand(AbstractNoise):
    #"""
    #Bandpass filtered noise where the pass-band is a fractional-octave.
    #"""
    
    #def __init__(self, centerfrequency, fraction=1, color='pink', order=3):
        #"""
        
        #:param centerfrequency: Centerfrequency
        #:param fraction: Fraction
        
        #"""
        #super().__init__(color)
        
        #self.centerfrequency = centerfrequency
        #"""
        #Center frequency of fraction-octave band.
        #"""
        
        #self.fraction = fraction
        #"""
        #Fraction of fractional-octave filter.
        #"""
        
        #self.order = order
        #"""Order of Butterworth filters.
        #"""
        
    #def output(self, t, fs):
        #"""
        #"""
        #f = acoustics.signal.OctaveBand(center=self.centerfrequency, fraction=self.fraction)
        #fb = acoustics.signal.Filterbank(f, sample_frequency=fs, order=self.order)
        #r = self._noise(t*fs)
        #return list(fb.filtfilt(r))[0]
        

#class NoiseOctaveBands(AbstractNoise):
    
    #def __init__(self, bands, fraction=1, color='pink', order=3):
        
        #super().__init__(color)



#class NoiseOctaveBands(AbstractNoise):
    #"""
    #Bandpass filtered noise where the pass band is a fractional-octave.
    #Contrary to :class:`WhiteOctaveBand` this class allows setting setting the power of several bands.
    #"""
    
    #def __init__(self, centerfrequencies, gains=None, fraction=1, color='pink', order=3, pass_zero=False, pass_nyq=False, taps=None, stair=True):
        #"""
        
        #:param frequencies: List or array of center frequencies.
        #:param gains: List or array of gains for each band. The gains are specified in dB.
        #:param fraction: Fraction of fractional-octave filter.
        
        #"""
        #super().__init__(color)
        
        #self._centerfrequencies = np.asarray(centerfrequencies)

        #self.gains = gains if len(gains)==len(centerfrequencies) else np.zeros(len(centerfrequencies))
        #"""Gains in dB.
        #"""
        
        #self.fraction = fraction
        #"""Fraction of fractional-octave filter.
        #"""
        
        #self.order = order
        #"""Order of Butterworth filters.
        #"""
        
        #self.pass_zero = pass_zero
        
        #self.pass_nyq = pass_nyq
    
        #self.taps = taps
        
        #self.stair = stair
    
    
    #@property
    #def centerfrequencies(self):
        #"""Center frequencies.
        #"""
        #return self._centerfrequencies
    
    #@property
    #def gains(self):
        #return self._gains
    
    #@gains.setter
    #def gains(self, x):
        #if len(x)==len(self.centerfrequencies):
            #self._gains = np.asarray(x)
        #else:
            #raise ValueError("Incorrect amount of items.")
    
    #def output(self, t, fs):
        #"""
        #"""        
        #if self.stair:
            #noise = self._noise(t*fs)
            #ob = acoustics.signal.OctaveBand(center=self.centerfrequencies, fraction=self.fraction)
            #fb = acoustics.signal.Filterbank(ob, sample_frequency=fs, order=self.order)
            #out = np.zeros(t*fs)
            
            ##print(ob.center)
            #gains = db_to_lin(self.gains)
            #for i, (band, gain) in enumerate(zip( fb.lfilter(noise), gains)):
                ##print(self.centerfrequencies[i])
                ##print(np.isnan(band).any())
                ##print(band)
                ##print(gain)
                #out += (band * gain )
            #return out
        
        #else:
            #nyq = fs/2.0
            #i = self.centerfrequencies < nyq
            #if not i.all():
                #warnings.warn("Not all specified frequencies were below the Nyquest frequency.")
            
            #f = np.zeros(len(self.centerfrequencies[i])+2)
            #gains = np.zeros_like(f)
            #f[1:-1] = self.centerfrequencies[i]
            #gains[1:-1] = db_to_lin(self.gains[i])
            #f[-1] = nyq
            
            #gains[0] = 1.0 if self.pass_zero else 0.0
            #gains[-1] = 1.0 if self.pass_nyq else 0.0
            
            ###print(f)
            ###print(gains)
            
            #fil = firwin2(2047, f, gains, nyq=nyq)
            #r = self._noise(t*fs)
            #return convolve(r, fil, mode='same')
        
        ##f = acoustics.signal.OctaveBand(center=self.centerfrequencies, fraction=self.fraction)
        ##fb = acoustics.signal.Filterbank(f, sample_frequency=fs, order=self.order)
        ##r = acoustics.generator.noise(t*fs, self.color)
        ##out = np.array(list(fb.filtfilt(r)))
        ##return (db_to_lin(self.gains[:,None]) * out).sum(axis=0)
        
        

        
        
        
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
    
    
