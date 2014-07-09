"""
Module with functions and classes for outdoor acoustics.
"""

import numpy as np
import scipy
import scipy.signal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from scipy.interpolate import interp1d as ip
from scipy.interpolate import InterpolatedUnivariateSpline as ip

from signal import Signal as Signal    

from acoustics.atmosphere import Atmosphere    
from acoustics.octave import Octave
from acoustics.signal import convolve

from Base import norm, unit_vector

import geometry as Geo

class Position(object):
    
    def __init__(self, x, y, z):
        self.x = x 
        self.y = y
        self.z = z 


class Transducer(object):#, metaclass=abc.ABCMeta):
    """
    Object
    """
    def __init__(self, position=None, position_directivity=None):
        
        self.position = position
        """
        Position of object.
        """
        
        self.position_directivity = position_directivity
        """
        Position of the source for the calculation of the directivity.
        
        If the source is the original source then this position should be equal to the position of the object.
        If this is a mirror source, then this position should be equal to the first order mirror that this mirror descents from.
        """
        
class Source(Transducer):
    pass

class Receiver(Transducer):
    pass


class Model(object):
    """
    Class describing geometry and environment.
    """
    
    def __init__(self, geometry=None, atmosphere=None, taps=50):
        """
        Constructor
        
        :param geometry: Geometry of the model. An object is created if None is passed.
        :type geometry: :class:`Auraliser.Geometry.Geometry`
        :param atmosphere: Atmosphere of the model. An object is created if None is passed.
        :type atmosphere: :class:`Auraliser.Atmosphere.Atmosphere`
        
        """
        
        self.geometry = geometry if geometry else Geometry()
        """Geometry of the model."""
        self.atmosphere = atmosphere if atmosphere else Atmosphere()
        """Atmosphere of the model."""
        
        self.taps = taps
        
    
    @property
    def delay(self):
        """
        Time delay between source and receiver.
        
        .. math:: t = s / c
        
        where :math:`s` represents the source-receiver distance and :math:`c` the speed of sound.
        """ 
        return self.geometry.distance / self.atmosphere.soundspeed

    def velocity(self, signal):
        """
        Relative velocity :math:`v` between source and receiver.
        
        :param signal: :class:`auraliser.signal.Signal`
        
        .. math:: v = \\vec\\nabla d \cdot f_s
        
        where :math:`d` is the Eucledian distance between the source and receiver calculated through :meth:`Auraliser.Geometry.Geometry.distance`.
        
        """
        return - np.gradient(self.geometry.distance) * signal.sample_frequency
    
    @property
    def directivity_vector(self):
        """
        Directivity vector.
        
        Uses the receiver position and the directivity position of the source.
        """
        return unit_vector(self.geometry.receiver.position - self.geometry.source.position_directivity)
    
    
    def _atmospheric_absorption(self, signal, sign, taps, N, n_d):
        """
        Apply or unapply atmospheric absorption depending on sign.
        """
        d = self.geometry.distance
        
        if n_d is not None:
            d_ir = np.linspace(d.min(), d.max(), n_d, endpoint=True)   # Distances to check
            
            start = (N-taps)//2
            stop = (N+taps)//2 - 1
            
            ir_i = self.atmosphere.ir_attenuation_coefficient(d=d_ir, N=N, fs=signal.sample_frequency, sign=sign)[start:stop+1, :]
            
            indices = np.argmin(np.abs(d.reshape(-1,1) - d_ir), axis=1)
            ir = ir_i[:, indices]
            
        else:
            ir = self.atmosphere.ir_attenuation_coefficient(d=self.geometry.distance, N=N, fs=signal.sample_frequency, sign=sign)[0:taps,:]
        
        return Signal(convolve(signal, ir), sample_frequency=signal.sample_frequency)
        
        
    def apply_atmospheric_absorption(self, signal, taps=10, N=2048, n_d=None):
        """
        Apply atmospheric absorption to ``signal``.
        
        :param signal: Signal
        :type signal: :class:`auraliser.signal.Signal`
        :param taps: Amount of filter taps to keep.
        :param N: Blocks to use for performing the FFT. Determines frequency resolution.
        :param n_d: Amount of unique distances to consider.
        """
        return self._atmospheric_absorption(signal, +1, taps, N, n_d)
        
        
    def unapply_atmospheric_absorption(self, signal, taps=10, N=2048, n_d=None):
        """
        Unapply atmospheric absorption to `signal`.
        
        :param signal: Signal
        :type signal: :class:`auraliser.signal.Signal`
        :param taps: Amount of filter taps to keep.
        :param N: Blocks to use for performing the FFT. Determines frequency resolution.
        :param n_ir: Value between 0 and 1 representing the percentage of unique impulse responses to use.
        """
        return self._atmospheric_absorption(signal, -1, taps, N, n_d)
    
    
    ###@staticmethod
    ###def apply_source_strength(self, signal, ir):
        ###"""
        ###Apply time- and frequency-variant strength to signal.
        ###"""
        
    
    
    def apply_spherical_spreading(self, signal):
        """
        Apply spherical spreading to ``signal``.
        
        .. math:: p_2 = p_1 \\frac{r_2}{r_1}
        
        where :math:`r_2` is 1.0.
        
        :param signal: Signal
        :type signal: :class:`auraliser.signal.Signal`
        
        :rtype: :class:`auraliser.signal.Signal`
        """
        return signal / self.geometry.distance # * 1.0
        
    def unapply_spherical_spreading(self, signal):
        """
        Unapply spherical spreading.
        
        .. math:: p_2 = p_1 \\frac{r_2}{r_1}
        
        where :math:`r_1` is 1.0.
        
        :param signal: Signal
        :type signal: :class:`auraliser.signal.Signal`
        
        :rtype: :class:`auraliser.signal.Signal`
        """
        return signal * self.geometry.distance # / 1.0
    
    #@staticmethod
    #def _interpolate(signal, delay):
        #"""
        #Apply ``delay`` to ``signal`` in-place.
        
        #:param signal: Signal to be delayed.
        #:type signal: :class:`auraliser.signal.Signal`
        #:param delay: Delay time
        #:type delay: :class:`np.ndarray`
        #"""
        
        #k_r = np.arange(0, len(signal), 1)          # Create vector of indices
        #k_e = k_r + delay * signal.sample_frequency# Create vector of warped indices
        ##k_e = k_e - np.min(k_e)                     # While we have an actual delay, we still want the audio file to start at t=0.
        #k_e_floor = np.floor(k_e).astype(int)       # Floor the warped indices. Convert to integers so we can use them as indices.
        
        #k_e_floor = k_e_floor * (k_e_floor >= 0) * (k_e_floor < len(signal)) + -1 * (  ( k_e_floor < 0) + (k_e_floor >= len(signal)) )
        #f = ip(k_r, signal, k=2)
        
        #signal_out = f(k_e_floor)
        
        #"""We want to change the signal in place."""
        #signal *= 0.0           # Set to zeros
        #signal += signal_out    # And add our new signal
        
        #return signal
    
    #@staticmethod
    #def _experiment(signal, delay):
        
        #k_r = np.arange(0, len(signal), 1)          # Create vector of indices
        #k_e = k_r - delay * signal.sample_frequency # Create vector of warped indices
        
        #f = ip(k_r, signal)                    # Function to interpolate
        
        #k_e_floor = np.floor(k_e).astype(int)       # Floor the warped indices. Convert to integers so we can use them as indices.
        
        #truth = (k_e_floor >= np.min(k_r) ) * (k_e_floor < np.max(k_r)) # We're only interested in samples of the emitter that 
        
        #signal_out = f(k_e_floor * truth) * truth   # New signal
        
        #"""We want to change the signal in place."""
        #signal *= 0.0           # Set to zeros
        #signal += signal_out  # And add our new signal
        
        #"""For debugging"""
        #np.savetxt('k_e_floor', k_e_floor, fmt='%i')
        #np.savetxt('signalout', signal_out, fmt='%f')
        
    @staticmethod
    def _apply_doppler_shift(signal, delay):
        """
        Apply ``delay`` to ``signal``.
        
        :param signal: Signal to be delayed.
        :type signal: :class:`auraliser.signal.Signal`
        :param delay: Delay time
        :type delay: :class:`np.ndarray`
        """
        k_e = np.arange(0, len(signal), 1)                  # Time axis emitter
        k_r = k_e + delay * signal.sample_frequency         # Time axis receiver
        
        f = ip(k_r, signal)     # Create a function to interpolate the signal at the receiver.
        
        truth = (k_e >= np.min(k_r) ) * (k_e < np.max(k_r)) # We can only interpolate, not extrapolate...
        signal_out = np.nan_to_num(f(k_e * truth)) * truth                 # Interpolated signal
        
        signal_out = signal_out * (np.abs(signal_out) <= 1.0) + 1.0 * (np.abs(signal_out) > 1.0)    # Remove any pulses (sonic booms)
        
        """We want to change the signal in place."""
        #signal *= 0.0
        #signal += signal_out
        
        """For debugging"""
        #np.savetxt('k_r', k_r, fmt='%f')
        #np.savetxt('k_e', k_e, fmt='%i')
        #np.savetxt('signalout', signal_out, fmt='%f')
        
        return signal_out
        
    @staticmethod
    def _map_source_to_receiver(signal, delay):
        """
        Apply ``delay`` to ``signal`` in-place.
        
        :param signal: Signal to be delayed.
        :type signal: :class:`auraliser.signal.Signal`
        :param delay: Delay time
        :type delay: :class:`np.ndarray`
        """
                
        k_r = np.arange(0, len(signal), 1)          # Create vector of indices
        k_e = k_r - delay * signal.sample_frequency # Create vector of warped indices
        
        k_e_floor = np.floor(k_e).astype(int)       # Floor the warped indices. Convert to integers so we can use them as indices.
        
        truth = ( k_e_floor >= 0 ) * ( k_e_floor < len(signal) )
        
        #k_e_floor = k_e_floor * (k_e_floor >= 0) * (k_e_floor < len(signal)) + -1 * (  ( k_e_floor < 0) + (k_e_floor >= len(signal)) )
        k_e_floor = k_e_floor * truth + -1 * np.negative(truth)
        
        signal_out = ( ( 1.0 - k_e + k_e_floor) * signal[np.fmod(k_e_floor, len(signal))] * ( k_e_floor >= 0) * (k_e_floor < len(signal)) ) +  \
                     (k_e - k_e_floor) * signal[np.fmod(k_e_floor +1, len(signal))] * (k_e_floor+1 >= 0) * (k_e_floor +1 < len(signal)) + np.zeros(len(signal))
        
        signal_out *= truth
        """For debugging"""
        #np.savetxt('k_e_floor', k_e_floor, fmt='%i')
        #np.savetxt('signalout', signal_out, fmt='%f')
        
        """We want to change the signal in place."""
        #signal *= 0.0           # Set to zeros
        #signal += signal_out    # And add our new signal
        
        return signal_out
        
    def apply_doppler(self, signal):
        """
        Apply Doppler shift to ``signal``.
        
        :param signal: Signal to be shifted.
        :type signal: :class:`auraliser.signal.Signal`
        """        
        return Signal(self._apply_doppler_shift(signal, self.delay), sample_frequency=signal.sample_frequency)
        
    def unapply_doppler(self, signal):
        """
        Unapply Doppler shift to ``signal``.
        
        :param signal: Signal to be shifted.
        :type signal: :class:`auraliser.signal.Signal`
        """
        return Signal(self._map_source_to_receiver(signal, - self.delay), sample_frequency=signal.sample_frequency)
        #self._experiment(signal, -self.delay)
        
    def doppler_shift(self, frequency):
        """
        Apply Doppler shift based on this model to ``frequency``.
        
        :param frequency: Frequency to be shifted.
        """
        raise NotImplementedError
    
    def plot_delay(self, filename, signal=None):
        """
        Plot the delay as function of time and write to ``filename``.
        If ``signal`` is specified the time-axis will use the provided sample frequency to convert the x-axis to seconds.
        
        :param filename: Name of file
        :type filename: :class:`string`
        """
        delay = self.delay
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('Delay')
        
        xsignal = np.arange(0.0, float(len(delay)))
        try:
            xsignal /=  signal.sample_frequency
            ax0.set_xlabel(r'$t$ in s')
        except AttributeError:
            ax0.set_xlabel(r'$t$ in samples')
        
        ax0.plot(xsignal, delay)
        ax0.set_xlabel(r'$t$ in s')
        ax0.set_ylabel(r'$d$ in s')
        ax0.grid()
        fig.savefig(filename)

    def plot_time_compression(self, filename, signal=None):
        """
        Plot time-compression factor :math:`Q`.
        """
        
        Q_time = - np.gradient(self.delay)
        
        #Q_freq = v * self.source.receiver - self.source.
        
        
        
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('Time compression')
        
        xsignal = np.arange(0.0, float(len(Q_time)))
        try:
            xsignal /=  signal.sample_frequency
            ax0.set_xlabel(r'$t$ in s')
        except AttributeError:
            ax0.set_xlabel(r'$t$ in samples')
        
        ax0.plot(xsignal, Q_time)
        ax0.set_xlabel(r'$t$ in s')
        ax0.set_ylabel(r'$Q$ in -')
        ax0.grid()
        fig.savefig(filename)

    def plot_atmospheric_absorption_signal(self, filename, signal):
        """
        Plot the atmospheric absorption signal.
        """
        
        alpha = self.atmospheric_absorption(signal)
        
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('Atmospheric absorption')
        
        #xsignal = np.arange(0.0, len(alpha))
        #xsignal /=  signal.sample_frequency
        #ax0.set_xlabel(r'$t$ in s')
        
        #ax0.plot(xsignal, 10.0 * np.log10(np.abs(alpha)))
        
        ax0.specgram(alpha, Fs=signal.sample_frequency)
        
        ax0.set_ylabel(r'$L_{a}$ in dB re 1')
        ax0.grid()
        fig.savefig(filename)

    def export_atmospheric_attenuation_ir(self, filename):
        """
        Export the impulse responses.
        """
        raise NotImplementedError
    
    def plot_velocity(self, filename, signal):
        """
        Plot the relative velocity between source and receiver as function of time and write to ``filename``.
        
        :param filename: Name of file
        :type filename: :class:`string`
        :param signal: Signal
        :type signal: :class:`auraliser.signal.Signal`
        """
        velocity = self.velocity(signal)
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('Relative velocity')
        
        xsignal = np.arange(0.0, float(len(velocity)))
        xsignal /=  signal.sample_frequency
        #velocity *= signal.sample_frequency
        ax0.set_xlabel(r'$t$ in s')
        
        ax0.plot(xsignal, velocity)
        ax0.set_ylabel(r'$v$ in m/s')
        ax0.grid()
        fig.savefig(filename)


class Geometry(object):
    """
    Class describing a geometry.
    """
    
    def __init__(self, source=None, receiver=None):
        """
        Constructor
        
        :param source: Source
        :param receiver: Receiver
        """
        self.source = source if source else Source()
        """Source"""
        
        self.receiver = receiver if receiver else Receiver()
        """Receiver"""
    
    @property
    def distance(self):
        """
        The Euclidean distance :math:`s` between source and receiver.
        
        .. note:: Cannot use :func:`np.linalg.norm` since we want to calculate the norm for each sample, and not a total norm.
        """
        return norm(self.source.position - self.receiver.position)
        #return np.sum(np.abs(self.source.position - self.receiver.position)**2.0, axis=1)**(0.5)

    @property
    def orientation(self):
        """
        Orientation of the `source` from the `receiver` point of view.
        
        
        Return the unit vector
        
        .. math:: \\mathbf{\\hat{u}} = \\frac{\\mathbf{u}}{||\\mathbf{u}||}
        
        where
        
        .. math:: \\mathbf{u} = \\mathbf{x_{source}} - \\mathbf{x_{receiver}}
    
        """
        return unit_vector(self.source.position - self.receiver.position)
        
    
    def plot_position(self, filename):
        """
        Plot position of source and receiver and write to ``filename``.
        
        :param filename: Name of file
        :type filename: :class:`string`
        """
        
        fig = plt.figure()
        
        ax0 = fig.add_subplot(111, projection='3d')
        ax0.set_title('Position')
        
        ax0.plot(xs=self.source.position.x, ys=self.source.position.y, zs=self.source.position.z, label='Source', linewidth=3)
        ax0.plot(xs=self.receiver.position.x, ys=self.receiver.position.y, zs=self.receiver.position.z, label='Receiver', linewidth=3)
        
        ax0.set_xlabel(r'$x$ in m')
        ax0.set_ylabel(r'$y$ in m')
        ax0.set_zlabel(r'$z$ in m')
        
        
        ax0.legend()
        ax0.grid()
        fig.savefig(filename)
    
    def plot_distance(self, filename, signal=None):
        """
        Plot the distance between source and receiver as function of time and write to ``filename``.
        
        :param filename: Name of file
        :type filename: :class:`string`
        :param signal: Signal
        :type signal: :class:`auraliser.signal.Signal`
        """
        distance = self.distance
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('Distance')
        
        xsignal = np.arange(0.0, float(len(distance)))
        try:
            xsignal /=  signal.sample_frequency
            ax0.set_xlabel(r'$t$ in s')
        except AttributeError:
            ax0.set_xlabel(r'$t$ in samples')
        
        ax0.plot(xsignal, distance)
        ax0.set_ylabel(r'$r$ in m')
        ax0.grid()
        fig.savefig(filename)

