import numpy as np
import itertools
from scintillations.stream import modulate as apply_turbulence
from scintillations.stream import transverse_speed

from streaming.stream import Stream, BlockStream
from streaming.signal import *
import streaming.signal
import logging
from acoustics.signal import impulse_response_real_even

import auraliser.tools
logger = auraliser.tools.create_logger(__name__)

def apply_atmospheric_attenuation(signal, fs, distance, nhop, atmosphere, ntaps, inverse=False, distance_reducer=np.mean):
    """Apply atmospheric attenuation to signal.

    :param distance: Iterable with distances.
    :param fs: Sample frequency.
    :param atmosphere: Atmosphere.
    :param ntaps: Amount of filter taps.
    :param sign: Sign.
    :rtype: :class:`streaming.Stream`

    Compute and apply the attenuation due to atmospheric absorption.
    The attenuation can change with distance. The attenuation is a magnitude-only filter.
    We design a linear-phase filter

    .. note:: The filter delay is compensated by dropping the first `ntaps//2` samples.

    """
    # Partition `distance` into blocks, and reduce with `distance_reducer`.
    distance = distance.blocks(nhop).map(distance_reducer)
    ir = Stream(atmosphere.impulse_response(d, fs, ntaps=ntaps, inverse=inverse) for d in distance)
    signal = convolve_overlap_save(signal, ir, nhop, ntaps)
    signal = signal.samples().drop(int(ntaps//2)) # Linear phase, correct for group delay caused by FIR filter.
    return signal


def apply_reflection_strength(emission, nhop, spectra, effective, ntaps, force_hard):
    """Apply mirror source strength.

    :param signal: Signal.
    :param nblock: Amount of samples per block.
    :param spectra: Spectrum per block.
    :param effective: Whether the source is effective or not.
    :param ntaps: Amount of filter taps.
    :param force_hard: Whether to force a hard ground.
    :returns: Signal with correct strength.

    .. warning:: This operation will cause a delay that may vary over time.

    """
    if effective is not None:
        # We have an effectiveness value for each hop (which is a block of samples)
        emission = BlockStream(map(lambda x,y: x *y, emission.blocks(nhop), effective), nblock=nhop)

    if force_hard:
        logger.info("apply_reflection_strength: Hard ground.")
    else:
        logger.info("apply_reflection_strength: Soft ground.")
        impulse_responses = Stream(impulse_response_real_even(s, ntaps) for s in spectra)
        emission = convolve_overlap_save(emission, impulse_responses, nhop, ntaps)
        # Filter has a delay we need to correct for.
        emission = emission.samples().drop(int(ntaps//2))
    return emission


#def apply_ground_reflection(signal, ir, nblock):
    #"""Apply ground reflection strength.

    #:param signal: Signal before ground reflection strength is applied.
    #:param ir: Impulse response per block.
    #:param nblock: Amount of samples per block.
    #:returns: Signal after reflection strength is applied.
    #:type: :class:`streaming.BlockStream`
    #"""

    #signal = convolve(signal=signal, impulse_responses=ir, nblock=nblock)


def apply_doppler(signal, delay, fs, initial_value=0.0):
    """Apply Doppler shift.

    :param signal: Signal before Doppler shift.
    :param delay: Propagation delay.
    :param fs: Constant sample frequency.
    :returns: Doppler-shifted signal.
    :rtype: :class:`streaming.Stream`

    """
    np.savetxt("/tmp/delay.csv", delay.copy().toarray())
    return vdl(signal, times(1./fs), delay, initial_value=initial_value)

def apply_spherical_spreading(signal, distance, inverse=False):#, nblock):
    """Apply spherical spreading.

    :param signal. Signal. Iterable.
    :param distance: Distance. Iterable.
    :param nblock: Amount of samples in block.

    """
    if inverse:
        return signal * distance
    else:
        return signal / distance


#def apply_turbulence(signal, fs, fraction, order, spatial_separation, distance, soundspeed,
                     #scale, include_logamp, include_phase, state=None, window=None, **kwargs):

    #frequency = 1000.0
    #wavenumber = 2.0 * np.pi * frequency / soundspeed
    ## Generate turbulence fluctuations
    #logamp, phase = _turbulence_fluctuations(spatial_separation=spatial_separation, distance=distance, wavenumber=wavenumber, scale=scale, fs=fs, nblock=nblock)

    ## Apply fluctuations to signal
    #signal = _apply_logamp_fluctuations(signal, logamp, nblock)
    ##signal = _apply_phase_fluctuations(signal, phase, frequency, fs)

    #return signal

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


#def _generate_gaussian_fluctuations(nblock, ntaps, fs, correlation_time, window=None, state=None):
    #"""Calculate fluctuations due to turbulence.

    #:returns: Fluctuations with variance 1.

    #Because the transverse velocity can change over time, the lower sample frequency will vary as well.
    #This causes problems and therefore we have to fix the lower sample frequency. However, by doing so,
    #we might have too few taps.

    #The correct variance needs to be applied to obtain actual logamp and phase fluctuations.

    #.. warning:: This is the downsampled version!
    #"""

    ## Correlation as function of time. We need to downsample the speed.
    ## We then calculate the correlation in a block, using the mean speed.
    ## NOTE: low-pass filter speed before downsampling!
    #_tau = tau(ntaps, fs)

    ##correlation_time = correlation_length / speed

    #correlation = correlation_time.map(lambda x: correlation_spherical_wave(_tau, x))

    #ir = Stream(impulse_response_fluctuations(c, fs, window=window) for c in correlation)

    ## Noise generator. Samples from the standard normal distribution.
    #state = state if state is not None else np.random.RandomState()
    #noise = BlockStream( (state.randn(nblock) for i in itertools.count()), nblock)

    ## Fluctuations at lower sample frequency. We normalise with the standard deviation.
    #fluctuations = convolve(noise, ir, nblock, ntaps).blocks(nblock)
    #std = fluctuations.copy().std()
    ##print(type(std))
    #fluctuations = fluctuations / std
    #return fluctuations


#def turbulence_fluctuations(nblock, fs, fs_low, ntaps, speed, correlation_length, window=None, state=None, time=None):
    #"""Calculate fluctuations due to turbulence.

    #:param ntaps: Amount of taps in down-sampled version.

    #"""

    ## We create first a down-sampled fluctuations signal with sample frequency `fs_low`.
    #ratio = fs / fs_low
    #nblock_low = nblock/ratio
    #if nblock_low < ntaps:
        #nblock_low = ntaps
    ##nblock_low = nextpow2(nblock_low)
    ##ntaps = nextpow2(ntaps)

    ## Time streams
    #if time is None:
        #time = streaming.signal.times(1./fs)
    #time_low = streaming.signal.times(1./fs_low)

    ## Downsample transverse speed
    #speed_low = interpolate(time.copy(), speed.copy(), time_low.copy())

    ## Calculate fluctuations
    #correlation_time = correlation_length/speed_low
    #fluctuations = _generate_gaussian_fluctuations(nblock_low, ntaps, fs_low, correlation_time, window=window, state=state)
    ## Upsample fluctuations to requested sample frequency.
    #fluctuations = interpolate(time_low, fluctuations, time)

    #return fluctuations

#def turbulence_apply_logamp_variance(nblock, fluctuations, distance, wavenumber, correlation_length, mean_mu_squared, include_saturation):
    #logamp_func = lambda d : variance_gaussian(d, wavenumber, correlation_length, mean_mu_squared, include_saturation=include_saturation)
    ## Variances as function of distance
    #variance_logamp = distance.blocks(nblock).map(logamp_func)
    ## Apply correct variance. Multiply fluctuations with standard deviation element for element.
    #logamp = fluctuations.blocks(nblock) * variance_logamp.sqrt()
    #return logamp

#def turbulence_apply_phase_variance(nblock, fluctuations, distance, wavenumber, correlation_length, mean_mu_squared):
    #phase_func = lambda d : variance_gaussian(d, wavenumber, correlation_length, mean_mu_squared)
    ## Variances as function of distance
    #variance_phase = distance.blocks(nblock).map(phase_func)
    ## Apply correct variance. Multiply fluctuations with standard deviation element for element.
    #phase = fluctuations.blocks(nblock) * variance_phase.sqrt()
    #return phase

#def turbulence_apply_logamp_fluctuations(signal, logamp):
    #return signal * np.exp(logamp)

#def turbulence_apply_phase_fluctuations(signal, phase, frequency, fs):
    #omega = (2.0*np.pi*frequency)
    #delay = phase / omega
    #signal = vdl(signal, times(1./fs), delay)
    #return signal



#def turbulence_fluctuations_spectral(signal, frequencies, nblock, fs, fs_low,
                                     #ntaps, ntaps_octaves, speed, correlation_length, window, seed,
                                     #distance, soundspeed, mean_mu_squared, include_logamp=True,
                                     #include_phase=True, include_saturation=True):
    #"""Gaussian fluctuations with time-variant transverse speed.
    #"""

    ## Frequency bands for which we calculate series of fluctuations
    #frequencies = frequencies[frequencies.upper < fs/2.0]
    #wavenumbers = 2. * np.pi * frequencies.center / soundspeed
    #nbands = len(frequencies)

    ## Create PRNG with given seed
    #state = np.random.RandomState(seed)

    #distance = distance.blocks(nblock) # Doing this here saves having to block in every copy after using tee

    ## Generic series of fluctuations. We need to apply the correct variance later
    #fluctuations = turbulence_fluctuations(nblock, fs, fs_low, ntaps, speed, correlation_length, window, state).blocks(nblock)

    #if include_logamp:
        ##----Offline bandpass filtering because IIR filtering is not yet implemented-----------------------------
        #signal = acoustics.Signal(signal.toarray(), fs)
        #frequencies, signals = signal.bandpass_frequencies(frequencies, order=8, purge=True, zero_phase=True)
        #signals = (Stream(signal).blocks(nblock) for signal in signals)

        ### Create bandpass filters to filter our input signal
        ##filters = (bandpass_filter(f_low, f_up, fs, ntaps_octaves) for f_low, f_up in zip(frequencies.lower, frequencies.upper))
        ### And split our signal so we have one copy per frequency band
        ##signals = signal.blocks(nblock).tee(nbands)
        ### Bandpass the signals
        ##signals = (convolve(signal, constant(filt), nblock).blocks(nblock) for filt, signal in zip(filters, signals))
        ## --------------------------------------------------------------------------------------------------------
        ## Logamp fluctuations
        #logamps = map(lambda w: turbulence_apply_logamp_variance(nblock, fluctuations.copy(), distance.copy(), w, correlation_length, mean_mu_squared, include_saturation), wavenumbers)
        ## and apply to our signal
        #signal = sum(map(turbulence_apply_logamp_fluctuations, signals, logamps))


    ### Add the bandpassed signals to obtain the final result
    ##signal = sum(signal.blocks(nblock) for signal in signals )

    ## Phase variance ~ frequency^2, thus phase is linear with frequency, and group delay is constant.

    #if include_phase:
        ## Phase fluctuations
        #frequency = 1.0 # Frequency independent.
        #wavenumber = 2.*np.pi*frequency / soundspeed
        #phase  = turbulence_apply_phase_variance(nblock, fluctuations.copy(), distance.copy(), wavenumber, correlation_length, mean_mu_squared)
        ## and apply to our signal
        #signal = turbulence_apply_phase_fluctuations(signal, phase, frequency, fs).blocks(nblock)

    #return signal



#def turbulence_fluctuations_spectral(signal, frequencies, nblock, fs, fs_low,
                                     #ntaps, ntaps_octaves, speed, correlation_length, window, seed,
                                     #distance, soundspeed, mean_mu_squared, include_logamp=True,
                                     #include_phase=True, include_saturation=True):
    #"""Gaussian fluctuations with time-variant transverse speed.
    #"""

    ## Frequency bands for which we calculate series of fluctuations
    ##frequencies = frequencies[frequencies.upper < fs/2.0]
    #ntaps2 = 128
    #frequencies = np.linspace(0.0, fs/2.0, ntaps2//2)
    #wavenumbers = 2. * np.pi * frequencies / soundspeed
    ##nbands = len(frequencies)

    ## Create PRNG with given seed
    #state = np.random.RandomState(seed)

    #distance = distance.blocks(nblock) # Doing this here saves having to block in every copy after using tee

    ## Generic series of fluctuations. We need to apply the correct variance later
    #fluctuations = turbulence_fluctuations(nblock, fs, fs_low, ntaps, speed, correlation_length, window, state).blocks(nblock)

    #if include_logamp:
        ## Apply fluctuations
        #logamp = fluctuations.copy()
        #signal = signal * np.exp(logamp)
        #print(signal.peek())
        ## Calculate spectral weighting of fluctuations
        #logamp_func = lambda d : variance_gaussian(d, wavenumbers, correlation_length, mean_mu_squared, include_saturation=include_saturation)
        #variance_logamp = distance.copy().mean().map(logamp_func)
        #print(variance_logamp.peek())
        #ir = variance_logamp.map(lambda x: np.fft.ifftshift(np.fft.irfft(np.sqrt(x), n=ntaps2)))#*np.hanning(ntaps2))
        #print(ir.peek())
        ## And apply to our signal
        #signal = streaming.signal.convolve(signal, ir, nblock)
        #print(signal.peek())

    #if include_phase:
        ## Phase variance ~ frequency^2, thus phase is linear with frequency, and group delay is constant.
        ## Phase fluctuations
        #frequency = 1.0 # Frequency independent.
        #wavenumber = 2.*np.pi*frequency / soundspeed
        #phase  = _turbulence_apply_phase_variance(nblock, fluctuations.copy(), distance.copy(), wavenumber, correlation_length, mean_mu_squared)
        ## and apply to our signal
        #signal = _turbulence_apply_phase_fluctuations(signal, phase, frequency, fs).blocks(nblock)

    #return signal





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

#def _apply_logamp_fluctuations(signal, logamp):
    #"""Apply log-amplitude fluctuations to signal.

    #:param signal: Signal. Iterable.
    #:param logamp: Log-amplitude fluctuations. Iterable.

    #"""
    #return signal * np.exp(logamp)
    ##yield from concat(map(lambda x, y: x*np.exp(y), zip(blocked(nblock, signal), blocked(nblock, logamp))))

#def _apply_phase_fluctuations(signal, phase, frequency, fs):
    #"""Apply phase fluctuations using a variable delay line.
    #"""
    #delay = phase/(2.0*np.pi*frequency)
    #return vdl(signal, times(1./fs), delay, initial_value=0.0)


