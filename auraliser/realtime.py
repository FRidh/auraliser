import acoustics
import collections
import numpy as np
import operator
from cytoolz import take, partition, concat, count
import itertools
from scintillations import * #get_covariance, impulse_response_fluctuations, tau, correlation_spherical_wave, variance_gaussian
from functools import partial

from streaming._iterator import blocked
from streaming.stream import Stream, BlockStream
from streaming.signal import *


def apply_atmospheric_attenuation(signal, fs, distance, nblock, atmosphere, ntaps, sign=-1, dtype='float64', distance_reducer=np.mean):
    """Apply atmospheric attenuation to signal.

    :param distance: Iterable with distances.
    :param fs: Sample frequency.
    :param atmosphere: Atmosphere.
    :param ntaps: Amount of filter taps.
    :param sign: Sign.
    """
    # Partition `distance` into blocks, and reduce with `distance_reducer`.
    distance = distance.blocks(nblock).map(distance_reducer)
    ir = Stream(atmosphere.impulse_response(d, fs, ntaps=ntaps, sign=sign) for d in distance)
    return convolve(signal=signal, impulse_responses=ir, nblock=nblock, initial_values=None)


def apply_ground_reflection(signal, ir, nblock):
    """Apply ground reflection strength."""

    return convolve(signal=signal, impulse_responses=ir, nblock=nblock)

def apply_doppler(signal, delay, fs, initial_value=0.0):
    """Apply Doppler shift.
    """
    return vdl(signal, times(1./fs), delay, initial_value=initial_value)


def apply_reflection_strength(signal, fs, spectrum):
    """Apply the strength of the mirror source.
    """
    return convolve(blocked(nblock, signal), nblock, ir)


def apply_spherical_spreading(signal, distance, nblock=None):#, nblock):
    """Apply spherical spreading.

    :param signal. Signal. Iterable.
    :param distance: Distance. Iterable.
    :param nblock: Amount of samples in block.

    """
    if nblock is not None:
        signal = signal.blocks(nblock)
        distance = distance.blocks(nblock)
    return signal / distance


def apply_turbulence(signal, fs, fraction, order, spatial_separation, distance, soundspeed,
                     scale, include_logamp, include_phase, state=None, window=None, **kwargs):

    frequency = 1000.0
    wavenumber = 2.0 * np.pi * frequency / soundspeed
    # Generate turbulence fluctuations
    logamp, phase = _turbulence_fluctuations(spatial_separation=spatial_separation, distance=distance, wavenumber=wavenumber, scale=scale, fs=fs, nblock=nblock)

    # Apply fluctuations to signal
    signal = _apply_logamp_fluctuations(signal, logamp, nblock)
    #signal = _apply_phase_fluctuations(signal, phase, frequency, fs)

    return signal

#def noise_generator(nblock, color='white', cycle=False, state=None):

    #if cycle:
        #yield from itertools.cycle(noise(nblock, color=color, state=state))
    #else:
        #while True:
            #yield from noise(nblock, color=color)

#def noise_generator(nblock, color='white', cycle=False, state=None):

    #def _generator(state, cycle):
        #if cycle:
            #return BlockStream(state.randn(nblock), nblock).cycle()
        #else:
            #while True:


    #if state is None:
        #state = np.random.RandomState()

    #if cycle:
        #return convolve(BlockStream(state.randn(nblock)).cycle(), Stream([ir]).cycle())
    #else:
        #state.randn(nblock)


#class Noise(object):

    #def __init__(self, state=None, cycle=False, nblock=False):

        #self._state = state if state is not None else np.random.RandomState()

        #self._cycle = cycle
        #self._nblock = nblock

    #@property
    #def cycle(self):
        #return self._cycle

    #@property
    #def nblock(self):
        #return self._nblock

    #def randn():


#def _turbulence_block(state):

    #variance_logamp = variance_gaussian_with_saturation(distance, wavenumber, correlation_length,
                                                        #mean_mu_squared, include_saturation=include_saturation)
    #variance_phase = variance_gaussian(distance, wavenumber, correlation_length, mean_mu_squared)

    #correlation = correlation_spherical_wave(tau(ntaps, fs_low), correlation_time)

    ## Impulse responses
    #ir = impulse_response_fluctuations(correlation, fs_low, window=window)

    ## We need random numbers.
    #noise = state.randn(nsamples_low)

    ## Modulation signal
    #fluctuations = fftconvolve(noise, ir, mode='same')
    #fluctuations /= fluctuations.std()
    ## Apply correct variance
    #logamp = fluctuations * np.sqrt(variance_logamp)
    #phase  = fluctuations * np.sqrt(variance_phase)

    ## Upsampled modulation signals
    #logamp = np.interp(times, times_low, logamp)
    #phase = np.interp(times, times_low, phase)

    #return logamp, phase


def nextpow2(x):
    return int(2**np.ceil(np.log2(x)))


def _turbulence_fluctuations_low(nblock, ntaps, fs, speed, correlation_length, window=None, state=None):
    """Calculate fluctuations due to turbulence.

    :returns: Fluctuations with variance 1.

    Because the transverse velocity can change over time, the lower sample frequency will vary as well.
    This causes problems and therefore we have to fix the lower sample frequency. However, by doing so,
    we might have too few taps.

    The correct variance needs to be applied to obtain actual logamp and phase fluctuations.

    .. warning:: This is the downsampled version!
    """

    # Correlation as function of time. We need to downsample the speed.
    # We then calculate the correlation in a block, using the mean speed.
    # NOTE: low-pass filter speed before downsampling!
    _tau = tau(ntaps, fs)

    correlation_time = correlation_length / speed

    correlation = correlation_time.map(lambda x: correlation_spherical_wave(_tau, x))

    ir = Stream(impulse_response_fluctuations(c, fs, window=window) for c in correlation)

    # Noise generator. Samples from the standard normal distribution.
    state = state if state is not None else np.random.RandomState()
    noise = BlockStream( (state.randn(nblock) for i in itertools.count()), nblock)

    # Fluctuations at lower sample frequency. We normalise with the standard deviation.
    fluctuations = convolve(noise, ir, nblock, ntaps).blocks(nblock)
    std = fluctuations.copy().std()
    #print(type(std))
    fluctuations = fluctuations / std
    return fluctuations


def turbulence_fluctuations(nblock, fs, fs_low, ntaps, speed, correlation_length, window=None, state=None):
    """Calculate fluctuations due to turbulence.

    :param ntaps: Amount of taps in down-sampled version.

    """

    # We create first a down-sampled fluctuations signal with sample frequency `fs_low`.
    ratio = fs / fs_low
    nblock_low = nblock/ratio
    if nblock_low < ntaps:
        nblock_low = ntaps
    #nblock_low = nextpow2(nblock_low)
    #ntaps = nextpow2(ntaps)

    # Time streams
    t = times(1./fs)
    t_low = times(1./fs_low)

    # Downsample transverse speed
    speed_low = interpolate(t.copy(), speed.copy(), t_low.copy())

    # Calculate fluctuations
    fluctuations = _turbulence_fluctuations_low(nblock_low, ntaps, fs_low, speed_low, correlation_length, window=window, state=state)
    # Upsample fluctuations to requested sample frequency.
    fluctuations = interpolate(t_low, fluctuations, t)

    return fluctuations

def _turbulence_apply_logamp_variance(nblock, fluctuations, distance, wavenumber, correlation_length, mean_mu_squared, include_saturation):
    logamp_func = lambda d : variance_gaussian_with_saturation(d, wavenumber, correlation_length, mean_mu_squared, include_saturation=include_saturation)
    # Variances as function of distance
    variance_logamp = distance.blocks().map(logamp_func)
    # Apply correct variance. Multiply fluctuations with standard deviation element for element.
    logamp = fluctuations.blocks(nblock) * variance_logamp.sqrt()
    return logamp

def _turbulence_apply_phase_variance(nblock, fluctuations, distance, wavenumber, correlation_length, mean_mu_squared):
    phase_func = lambda d : variance_gaussian(d, wavenumber, correlation_length, mean_mu_squared)
    # Variances as function of distance
    variance_phase = distance.blocks().map(phase_func)
    # Apply correct variance. Multiply fluctuations with standard deviation element for element.
    phase = fluctuations.blocks(nblock) * variance_phase.sqrt()
    return phase

def _turbulence_apply_logamp_fluctuations(signal, logamp):
    return signal * np.exp(logamp)

def _turbulence_apply_phase_fluctuations(signal, phase, frequency, fs):
    omega = (2.0*np.pi*frequency)
    delay = phase / omega
    signal = vdl(signal, times(1./fs), delay)
    return signal



def turbulence_fluctuations_spectral(signal, frequencies, nblock, fs, fs_low,
                                     ntaps, ntaps_octaves, speed, correlation_length, window, state,
                                     distance, soundspeed, mean_mu_squared, include_logamp=True,
                                     include_phase=True, include_saturation=True):
    """Gaussian fluctuations with time-variant transverse speed.
    """

    # Frequency bands for which we calculate series of fluctuations
    frequencies = frequencies[frequencies.upper < fs/2.0]
    wavenumbers = 2. * np.pi * frequencies.center / soundspeed
    nbands = len(frequencies)

    distance = distance.blocks(nblock) # Doing this here saves having to block in every copy after using tee

    #----Offline bandpass filtering because IIR filtering is not yet implemented-----------------------------
    signal = acoustics.Signal(signal.toarray(), fs)
    frequencies, signals = signal.bandpass_frequencies(frequencies, order=8, purge=True, zero_phase=True)
    signals = (Stream(signal).blocks(nblock) for signal in signals)

    ## Create bandpass filters to filter our input signal
    #filters = (bandpass_filter(f_low, f_up, fs, ntaps_octaves) for f_low, f_up in zip(frequencies.lower, frequencies.upper))
    ## And split our signal so we have one copy per frequency band
    #signals = signal.blocks(nblock).tee(nbands)
    ## Bandpass the signals
    #signals = (convolve(signal, constant(filt), nblock).blocks(nblock) for filt, signal in zip(filters, signals))
    # --------------------------------------------------------------------------------------------------------

    # Generic series of fluctuations. We need to apply the correct variance later
    fluctuations = turbulence_fluctuations(nblock, fs, fs_low, ntaps, speed, correlation_length, window, state).blocks(nblock)



    if include_logamp:
        # Logamp fluctuations
        logamps = map(lambda w: _turbulence_apply_logamp_variance(nblock, fluctuations.copy(), distance.copy(), w, correlation_length, mean_mu_squared, include_saturation), wavenumbers)
        # and apply to our signal
        signals = map(_turbulence_apply_logamp_fluctuations, signals, logamps)

    if include_phase:
        # Phase fluctuations
        phases  = map(lambda w: _turbulence_apply_phase_variance(nblock, fluctuations.copy(), distance.copy(), w, correlation_length, mean_mu_squared), wavenumbers)
        # and apply to our signal
        signals = map(_turbulence_apply_phase_fluctuations, signals, phases, frequencies.center, itertools.cycle([fs]))

    # Add the bandpassed signals to obtain the final result
    signal = sum(signal.blocks(nblock) for signal in signals )
    return signal

#def _turbulence_fluctuations(speed, distance, wavenumber, scale, fs, nblock,
                             #ntaps, window=None, model='gaussian', **kwargs):
    #"""Calculate fluctuations due to turbulence.

    #:returns: (logamp, phase) with each being an iterable.
    #"""

    #try:
        #include_saturation = kwargs.pop('include_saturation')
    #except KeyError:
        #include_saturation = False

    ## Blocked.
    #distance = distance.blocks(nblock)
    #speed = speed.blocks(nblock)
    ##spatial_separation = spatial_separation.blocks(nblock)

    ## Covariance function to use
    #covariance = get_covariance(model)
    #covariance = partial(covariance, wavenumber=wavenumber, scale=scale, **kwargs)

    ## We need to sample the covariance
    #tau = np.arange(ntaps//2)/fs
    #tau = np.concatenate([tau, tau[::-1]])
    #spatial_separation = tau * speed

    ## Let's calculate the covariance for each block
    #cov = Stream(covariance(spatial_separation=tau*s, distance=d) for s, d in zip(speed.copy().map(np.mean), distance.copy().map(np.mean)))

    ## And the impulse response for each block
    #ir_func = partial(impulse_response_fluctuations, window=window)
    #ir = cov.map(ir_func)

    ## We need IR for both logamp and phase fluctuations
    #ir_logamp, ir_phase = ir.tee()

    ## Logamp fluctuations
    #state_logamp_noise = np.random.RandomState()
    #noise = BlockStream( (state_logamp_noise.randn(nblock) for i in itertools.count()), nblock)
    #logamp = convolve(noise, ir_logamp.copy(), nblock)

    ## Phase fluctuations
    #state_phase_noise = np.random.RandomState()
    #noise = BlockStream( (state_phase_noise.randn(nblock) for i in itertools.count()), nblock)
    #phase = convolve(noise, ir_phase.copy(), nblock)

    ## Logamp fluctuations saturation
    #if include_saturation:
        #sat_distance = saturation_distance(mean_mu_squared, wavenumber, scale)
        #logamp = np.sqrt(1.0/(1.0+distance/sat_distance))
        ##logamp = map(lambda d: np.sqrt(1.0/(1.0+d/sat_distance)), distance)

    ## Return logamp and phase fluctuations generators
    #return logamp, phase, ir_logamp, ir_phase


#def _turbulence_impulse_response(distance, spatial_separation, covariance, window=None):
    #"""Calculate an impulse response for a single block.

    #:param nblock: Amount of samples in block. Scalar.
    #:param ntaps: Amount of filter taps. Scalar.
    #:param distance: Distance. Scalar.
    #:param spatial_separation: Spatial seperation. Vector.
    #:param covariance: Covariance. Scalar.
    #:returns: Impulse response. Vector.

    #"""
    #cov = covariance(spatial_separation=spatial_separation, distance=distance.mean())
    #ir = impulse_response_fluctuations(cov, window=window)
    #return ir

def _apply_logamp_fluctuations(signal, logamp):
    """Apply log-amplitude fluctuations to signal.

    :param signal: Signal. Iterable.
    :param logamp: Log-amplitude fluctuations. Iterable.

    """
    return signal * np.exp(logamp)
    #yield from concat(map(lambda x, y: x*np.exp(y), zip(blocked(nblock, signal), blocked(nblock, logamp))))

def _apply_phase_fluctuations(signal, phase, frequency, fs):
    """Apply phase fluctuations using a variable delay line.
    """
    delay = phase/(2.0*np.pi*frequency)
    return vdl(signal, times(1./fs), delay, initial_value=0.0)


def sound_pressure_level(signal, fs, nblock, time=0.125, method='average'):
    """

    :param time: Integration or averaging time.

    .. note:: Might run out of sync.
    """

    if method=='average':
        func = acoustics.standards.iec_61672_1_2013.time_averaged_sound_level
    elif method=='weighting':
        func = acoustics.standards.iec_61672_1_2013.time_weighted_sound_level
    else:
        raise ValueError("Invalid method")

    if nblock < int(round(fs*time)):
        raise ValueError("Block size is smaller than the samples used for integration/averaging.")

    partitioned_signal = partition(signal, nblock)

    yield from concat(map(acoustics.standards.iso_tr_25417_2007.equivalent_sound_pressure_level, blocked))
