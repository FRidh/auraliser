"""
Propagation effects.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ip

try:
    from pyfftw.interfaces.numpy_fft import fft, ifft, rfft, irfft       # Performs much better than numpy's fftpack
    import pyfftw
    #import scipy.signal
    #scipy.signal.signaltools.fftn = pyfftw.interfaces.scipy_fftpack.fftn
    #scipy.signal.signaltools.ifftn = pyfftw.interfaces.scipy_fftpack.ifftn
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import fft, rfft, ifft, irfft
finally:
    #from scipy.signal import fftconvolve
    from scipy.special import erf
    from ._fftconvolve import fftconvolve1D
from acoustics.signal import Filterbank, OctaveBand, convolve
#from turbulence_jens import map_source_to_receiver
from .tools import norm
from scipy.special import gamma
from scipy.special import kv as besselk
from scipy.integrate import cumtrapz
from scipy.signal import fftconvolve, filtfilt

import math
import numba
from acoustics import Signal
import logging


def ir_reflection(spectrum, n_blocks=None):
    """Complex single-sided spectrum to real impulse response.
    """
    spectrum = np.hstack((spectrum, spectrum[..., 0::-1]))
    return np.fft.ifftshift(np.fft.ifft(spectrum, n=n_blocks), axes=1).real

def ir_atmosphere(spectrum, n_blocks=None):
    """Real single-sided spectrum to real impulse response.
    """
    spectrum = np.hstack((spectrum, spectrum[..., 0::-1])) # Apparently not needed since doesn't make any difference.
    return np.fft.ifftshift(np.fft.ifft(spectrum, n=n_blocks), axes=1).real


def apply_spherical_spreading(signal, distance):
    """Apply spherical spreading to ``signal``.

    .. math:: p_2 = p_1 \\frac{r_1}{r_2}

    where :math:`r_1` is 1.0.

    :param signal: Signal
    :type signal: :class:`auraliser.signal.Signal`

    :rtype: :class:`auraliser.signal.Signal`
    """
    return signal / distance # * 1.0

def unapply_spherical_spreading(signal, distance):
    """Unapply spherical spreading.

    .. math:: p_2 = p_2 \\frac{r_1}{r_2}

    where :math:`r_1` is 1.0.

    :param signal: Signal
    :type signal: :class:`auraliser.signal.Signal`

    :rtype: :class:`auraliser.signal.Signal`
    """
    return signal * distance # / 1.0

#def _apply_doppler_shift(signal, delay, fs):
    #"""
    #Apply ``delay`` to ``signal``.

    #:param signal: Signal to be delayed.
    #:type signal: :class:`auraliser.signal.Signal`
    #:param delay: Delay time
    #:type delay: :class:`np.ndarray`
    #"""
    #k_e = np.arange(0, len(signal), 1)                  # Time axis emitter
    #k_r = k_e + delay * fs                              # Time axis receiver

    #f = ip(k_r, signal)     # Create a function to interpolate the signal at the receiver.

    #truth = (k_e >= np.min(k_r) ) * (k_e < np.max(k_r)) # We can only interpolate, not extrapolate...
    #signal_out = np.nan_to_num(f(k_e * truth)) * truth                 # Interpolated signal
    #signal_out = signal_out * (np.abs(signal_out) <= 1.0) + 1.0 * (np.abs(signal_out) > 1.0)    # Remove any pulses (sonic booms)
    #return signal_out

def _map_source_to_receiver(signal, delay, fs):
    """Apply ``delay`` to ``signal`` in-place.

    :param signal: Signal to be delayed.
    :type signal: :class:`auraliser.signal.Signal`
    :param delay: Delay time
    :type delay: :class:`np.ndarray`

    This method is used for back propagation.
    """

    k_r = np.arange(0, len(signal), 1)          # Create vector of indices
    k_e = k_r - delay * fs                      # Create vector of warped indices

    k_e_floor = np.floor(k_e).astype(int)       # Floor the warped indices. Convert to integers so we can use them as indices.

    truth = ( k_e_floor >= 0 ) * ( k_e_floor < len(signal) )

    #k_e_floor = k_e_floor * (k_e_floor >= 0) * (k_e_floor < len(signal)) + -1 * (  ( k_e_floor < 0) + (k_e_floor >= len(signal)) )
    k_e_floor = k_e_floor * truth + -1 * np.negative(truth)

    signal_out = ( ( 1.0 - k_e + k_e_floor) * signal[np.fmod(k_e_floor, len(signal))] * ( k_e_floor >= 0) * (k_e_floor < len(signal)) ) +  \
                    (k_e - k_e_floor) * signal[np.fmod(k_e_floor +1, len(signal))] * (k_e_floor+1 >= 0) * (k_e_floor +1 < len(signal)) + np.zeros(len(signal))
    signal_out *= truth
    return signal_out

#def apply_doppler(signal, delay, fs):
    #"""
    #Apply Doppler shift to ``signal``.

    #:param signal: Signal to be shifted.
    #:type signal: :class:`auraliser.signal.Signal`
    #"""
    #return _apply_doppler_shift(signal, delay, fs)


def interpolation_linear(signal, times, fs):
    """Linear interpolation of `signal` at `times`.

    :param signal: Signal.
    :param times: Times to sample at.
    :param fs: Sample frequency.

    This method is used in forward propagation.

    """
    k_r = np.arange(0, len(signal), 1)          # Create vector of indices
    k = k_r - times * fs                      # Create vector of warped indices
    kf = np.floor(k).astype(int)       # Floor the warped indices. Convert to integers so we can use them as indices.
    R = ( (1.0-k+kf) * signal[kf] + (k-kf) * signal[kf+1] ) * (kf >= 0) #+ 0.0 * (kf<0)
    return R


def apply_doppler(signal, delay, fs, method='linear', kernelsize=10):
    """Apply Doppler shift to ``signal``.

    :param signal: Signal to be shifted.
    :param delay: Delays.
    :param fs: Sample frequency.
    :param kernelsize: Kernelsize in case of `lanczos` method

    """
    if method == 'linear':
        return interpolation_linear(signal, delay, fs)
    elif method == 'lanczos':
        return interpolation_lanczos(signal, delay, fs, kernelsize)

def unapply_doppler(signal, delay, fs, method='linear', kernelsize=10):
    """Unapply Doppler shift to ``signal``.

    :param signal: Signal to be shifted.
    :param delay: Delays.
    :param fs: Sample frequency.
    :param kernelsize: Kernelsize in case of `lanczos` method

    """
    if method == 'linear':
        return _map_source_to_receiver(signal, -delay, fs)
    elif method == 'lanczos':
        return interpolation_lanczos(signal, -delay, fs, kernelsize)

def apply_delay_turbulence(signal, delay, fs):
    """Apply phase delay due to turbulence.

    :param signal: Signal
    :param delay: Delay
    :param fs: Sample frequency
    """

    k_r = np.arange(0, len(signal), 1)          # Create vector of indices
    k = k_r - delay * fs                      # Create vector of warped indices

    kf = np.floor(k).astype(int)       # Floor the warped indices. Convert to integers so we can use them as indices.
    dk = kf - k
    ko = np.copy(kf)
    kf[ko<0] = 0
    kf[ko+1>=len(ko)] = 0
    R = ( (1.0 + dk) * signal[kf] + (-dk) * signal[kf+1] ) * (ko >= 0) * (ko+1 < len(k)) #+ 0.0 * (kf<0)
    return R

#def unapply_doppler(signal, delay, fs):
    #"""Unapply Doppler shift to ``signal``.

    #:param signal: Signal to be shifted.
    #:type signal: :class:`auraliser.signal.Signal`
    #"""
    #return _map_source_to_receiver(signal, -delay, fs)





def apply_doppler_amplitude_using_vectors(signal, mach, unit, multipole):
    """Apply change in pressure due to Doppler shift.

    :param signal: Signal.
    :param mach: Mach.
    :param unit: Unit vector.
    :param multipole: Multipole order.

    """
    #mach = np.gradient(mach)[0]
    #print(mach)
    #print(mach[0], unit[0], np.einsum('ij,ij->i', mach, unit)[0])
    #print((1.0 - np.einsum('ij,ij->i', mach, unit))**-2.0)

    if multipole==0 or multipole==1:
        return signal * (1.0 - np.einsum('ij,ij->i', mach, unit))**-2.0
    elif multipole==2:
        return signal * (1.0 - np.einsum('ij,ij->i', mach, unit))**-3.0
    else:
        raise ValueError("Invalid multipole order.")


def apply_doppler_amplitude(signal, mach, angle, multipole):
    """Apply change in pressure due to Doppler shift.

    :param signal: Signal
    :param mach: Mach number
    :param angle: Angle in radians.
    :param multipole: Multipole order.

    Mach numbers close to one can lead to strong directivity effects.
    """
    if multipole==0 or multipole==1:
        return signal * (1.0 - mach * np.cos(angle))**-2.0
    elif multipole==2:
        return signal * (1.0 - mach * np.cos(angle))**-3.0
    else:
        raise ValueError("Invalid multipole order.")

def unapply_doppler_amplitude(signal, mach, angle, multipole):
    """Unapply change in pressure due to Doppler shift.

    :param signal: Signal
    :param mach: Mach number
    :param angle: Angle in radians.
    :param multipole: Multipole order.

    Mach numbers close to one can lead to strong directivity effects.
    """
    raise signal / apply_doppler_amplitude(signal, mach, angle, multipole)



#from turbulence.vonkarman import covariance_wind as covariance_von_karman

#def covariance_von_karman(f, c0, spatial_separation, distance, scale, Cv, steps=20, initial=0.001):
    #"""Covariance. Wind fluctuations only.

    #:param f: Frequency
    #:param c0: Speed of sound
    #:param spatial_separation: Spatia separation
    #:param distance: Distance
    #:param scale: Correlation length
    #:param Cv: Variance of wind speed
    #:param initial: Initial value


    #"""
    #k = 2.0*np.pi*f / c0
    #K0 = 2.0*np.pi / scale

    #A = 5.0/(18.0*np.pi*gamma(1./3.)) # Equation 11, see text below. Approximate result is 0.033
    #gamma_v = 3./10.*np.pi**2.*A*k**2.*K0**(-5./3.)*4.*(Cv/c0)**2.  # Equation 28, only wind fluctuations

    #kspatial_separation = k * spatial_separation

    #t = kspatial_separation[:,None] * np.linspace(0.00000000001, 1., steps) # Fine discretization for integration

    ##t[t==0.0] = 1.e-20

    ##print( (2.0**(1./6.)*t**(5./6.)/gamma(5./6.) * (besselk(5./6., t) - t/2.0*besselk(1./6., t)) ) )

    ##print(   cumtrapz((2.0**(1./6.)*t**(5./6.)/gamma(5./6.) * (besselk(5./6., t) - t/2.0*besselk(1./6., t)) ), initial=initial)[:,-1]  )

    #B = 2.0*gamma_v * distance / kspatial_separation * cumtrapz((2.0**(1./6.)*t**(5./6.)/gamma(5./6.) * (besselk(5./6., t) - t/2.0*besselk(1./6., t)) ), initial=initial)[:,-1]
    #return B



def _spatial_separation(A, B, C):
    """Spatial separation.

    :param A: Source position as function of time.
    :param B: Reference position, e.g. source position at t=0
    :param C: Receiver position.

    Each row is a sample and each column a spatial dimension.

    """
    a = np.linalg.norm(B-C, axis=1)
    b = np.linalg.norm(A-C, axis=1)
    c = np.linalg.norm(A-B, axis=1)

    gamma = np.arccos((a**2.0+b**2.0-c**2.0) / (2.0*a*b))
    spatial_separation = 2.0 * b * np.sin(gamma/2.0)
    L = b * np.cos(gamma/2.0)

    return spatial_separation, L


from auraliser.scintillations import generate_fluctuations, apply_fluctuations

def _generate_and_apply_fluctuations(signal, fs, frequency, spatial_separation,
                                     distance, soundspeed, scale, include_logamp, include_phase,
                                     state=None, window=None, **kwargs):
    """Apply fluctuations to a signal.
    """
    wavenumber = 2.0 * np.pi * frequency / soundspeed
    samples = len(signal)
    log_amplitude, phase = generate_fluctuations(samples=samples,
                                                 spatial_separation=spatial_separation,
                                                 distance=distance,
                                                 wavenumber=wavenumber,
                                                 scale=scale,
                                                 window=None,
                                                 state=state,
                                                 **kwargs
                                                 )

    if not include_logamp:
        log_amplitude = None
    if not include_phase:
        phase = None
    return apply_fluctuations(signal, fs, frequency=frequency, log_amplitude=log_amplitude, phase=phase).calibrate_to(signal.leq())


def apply_turbulence(signal, fs, fraction, order, spatial_separation, distance, soundspeed,
                     scale, include_logamp, include_phase, state=None, window=None, **kwargs):
    """Apply turbulence to propagation.
    """
    # Upsample data
    factor = 5
    signal = Signal(signal, fs)
    #upsampled = signal.upsample(factor)
    from scipy.interpolate import interp1d
    from acoustics.signal import OctaveBand
    from acoustics.standards.iec_61672_1_2013 import NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES
    from copy import copy

    #spatial_separation = interp1d(signal.times(), spatial_separation)(np.linspace(0.0, signal.times().max(), upsampled.samples))
    #distance = interp1d(signal.times(), distance)(np.linspace(0.0, signal.times().max(), upsampled.samples))

    frequencies = OctaveBand(fstart=NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES[0], fstop=signal.fs/2.0, fraction=fraction)

    #signal = upsampled
    frequencies, signals = signal.bandpass_frequencies(frequencies, order=order, purge=True, zero_phase=True)
    samples = len(signal)
    del signal

    state = state if state else np.random.RandomState()

    modulated = map(lambda frequency, signal: _generate_and_apply_fluctuations(signal, fs, frequency,
                                                                               spatial_separation=spatial_separation,
                                                                               distance=distance,
                                                                               soundspeed=soundspeed,
                                                                               include_logamp=include_logamp,
                                                                               include_phase=include_phase,
                                                                               scale=scale,
                                                                               window=None,
                                                                               state=copy(state),
                                                                               **kwargs), frequencies.center, signals)
    return Signal(sum(modulated), fs)#.decimate(factor, zero_phase=True)



def _ir_attenuation_coefficient(atmosphere, distances, fs=44100.0, n_blocks=2048, sign=-1):
    """
    Calculate the impulse response due to air absorption.

    :param fs: Sample frequency
    :param distances: Distances
    :param blocks: Blocks
    :param sign: Multiply (+1) or divide (-1) by transfer function. Multiplication is used for applying the absorption while -1 is used for undoing the absorption.
    """
    distances = np.atleast_1d(distances)
    f = np.fft.fftfreq(n_blocks, 1./fs)

    tf = np.zeros((len(distances), len(f)), dtype='float64')                          # Transfer function needs to be complex, and same size.
    tf += 10.0**( float(sign) * distances[:,None] * atmosphere.attenuation_coefficient(f) / 20.0  )  # Calculate the actual transfer function.

    return ir_atmosphere(tf, n_blocks=n_blocks)


def _atmospheric_absorption(signal, fs, atmosphere, distance, sign, n_blocks, n_distances=None):
    """Apply or unapply atmospheric absorption depending on sign.

    :param signal: Signal
    :param fs: Sample frequency
    :param atmosphere: Atmosphere
    :param distance: Distance
    :param n_blocks: Amount of filter taps to keep. Blocks to use for performing the FFT. Determines frequency resolution.
    :param n_distances: Amount of unique distances to consider.

    """
    if n_distances is not None:
        distances = np.linspace(distance.min(), distance.max(), n_distances, endpoint=True)   # Distances to check

        # Every row is an impulse response.
        ir_i = _ir_attenuation_coefficient(atmosphere, distances=distances, n_blocks=n_blocks, fs=fs, sign=sign)#[start:stop+1, :]

        # Get the IR of the distance closest by
        indices = np.argmin(np.abs(distance[:,None] - distances), axis=1)
        ir = ir_i[indices, :]
    else:
        ir = _ir_attenuation_coefficient(atmosphere, distances=distance, n_blocks=n_blocks, fs=fs, sign=sign)
    return convolve(signal, ir.T)


def apply_atmospheric_absorption(signal, fs, atmosphere, distance, n_blocks=128, n_distances=None):
    """
    Apply atmospheric absorption to ``signal``.

    :param signal: Signal
    :param fs: Sample frequency
    :param atmosphere: Atmosphere
    :param distance: Distance
    :param n_blocks: Amount of filter taps to keep. Blocks to use for performing the FFT. Determines frequency resolution.
    :param n_distances: Amount of unique distances to consider.
    """
    return _atmospheric_absorption(signal, fs, atmosphere, distance, -1, n_blocks, n_distances)


def unapply_atmospheric_absorption(signal, fs, atmosphere, distance, n_blocks=128, n_distances=None):
    """
    Unapply atmospheric absorption to `signal`.

    :param signal: Signal
    :param fs: Sample frequency
    :param atmosphere: Atmosphere
    :param distance: Distance
    :param n_blocks: Amount of filter taps to keep. Blocks to use for performing the FFT. Determines frequency resolution.
    :param n_distances: Amount of unique distances to consider.
    """
    return _atmospheric_absorption(signal, fs, atmosphere, distance, +1, n_blocks, n_distances)


@numba.jit(nogil=True)
def sinc(x):
    if x == 0:
        return 1.0
    else:
        return math.sin(x*math.pi) / (x*math.pi)

@numba.jit(nogil=True)
def _lanczos_window(x, a):
    if -a < x and x < a:
        return sinc(x) * sinc(x/a)
    else:
        return 0.0

@numba.jit(nogil=True)
def _lanczos_resample(signal, samples, output, a):
    """Sample signal at float samples.
    """
    for index, x in enumerate(samples):
        if x >= 0.0 and x < len(signal):
            for i in range(math.floor(x)-a+1, math.floor(x+a)):
                if i >= 0 and i < len(signal):
                    output[index] += signal[i] * _lanczos_window(x-i, a)
    return output


def interpolation_lanczos(signal, times, fs, a=10):
    """Lanczos interpolation of `signal` at `times`.

    :param signal: Signal.
    :param times: Times to sample at.
    :param fs: Sample frequency.
    :param kernelsize: Size of Lanczos kernel :math:`a`.

    http://en.wikipedia.org/wiki/Lanczos_resampling

    """
    samples = -times * fs + np.arange(len(signal))
    #samples[samples < 0.0] = 0.0 # This is the slowest part.
    return _lanczos_resample(signal, samples, np.zeros_like(signal), a)
