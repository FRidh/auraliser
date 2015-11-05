"""
Atmospheric turbulence causes fluctuations in the sound speed which in effects causes fluctuations in the amplitude and phase of the sound pressure.

Using :class:`Turbulence` a time series of fluctuations can be generated and applied to a signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from acoustics.signal import zero_crossings
from scipy.signal import resample

from scipy.special import erf

from auraliser._fftconvolve import fftconvolve1D

from turbulence.vonkarman import covariance_wind as covariance_vonkarman_wind


def variance_gaussian(spatial_separation, distance, wavenumber, scale, mean_mu_squared):
    """Variance of Gaussian fluctuations.

    :param spatial_separation: Spatial separation.
    :param distance: Distance.
    :param wavenumber: Wavenumber.
    :param mean_mu_squared: Mean mu squared.
    :param scale: Outer length scale.
    :returns: Variance

    """
    return np.sqrt(np.pi)/2.0 * mean_mu_squared * wavenumber**2.0 * distance * scale

def correlation_spherical_wave(spatial_separation, scale):
    """Correlation of spherical waves.

    :param spatial_separation: Spatial separation.
    :param scale: Outer length scale.
    :returns: Correlation
    """
    x = spatial_separation/scale
    cor = np.sqrt(np.pi) / 2.0 * erf(x)/x
    cor[x==0.0] = 1.0
    return cor


def covariance_gaussian(spatial_separation, distance, wavenumber, scale, mean_mu_squared, **kwargs):
    """Calculate the covariance of a Gaussian turbulence spectrum and spherical waves.

    See Daigle, 1987: equation 2 and 3.

    :param spatial_separation: Spatial separation.
    :param distance: Distance.
    :param wavenumber: Wavenumber.
    :param mean_mu_squared: Mean mu squared.
    :param scale: Outer length scale.
    :returns: Covariance

    .. math:: B{\chi} (\rho) = B{S}(\rho) = \\frac{\\sqrt{\\pi}}{2} \\langle \\mu^2 \\rangle k^2 r L \\frac{\\Phi(\\rho/L) }{\\rho / L}

    """
    #covariance = 0.0
    #covariance += (spatial_separation!=0.0) * \
                  #np.nan_to_num( ( np.pi/4.0 * mean_mu_squared * (wavenumber*wavenumber) * \
                  #distance * scale * (erf(spatial_separation/scale) / \
                  #(spatial_separation/scale) ) ) )

    #covariance += (spatial_separation==0.0) * np.sqrt(np.pi)/2.0 * \
                  #mean_mu_squared * (wavenumber*wavenumber) * distance * scale
    cor = correlation_spherical_wave(spatial_separation, scale)
    var = variance_gaussian(spatial_separation, distance, wavenumber, scale, mean_mu_squared)
    covariance = cor * var

    return covariance


def saturation_distance(mean_mu_squared, wavenumber, scale):
    """Saturation distance according to Wenzel.

    :param mean_mu_squared: Mean mu squared.
    :param wavenumber: Wavenumber.
    :param scale: Outer length scale.

    See Daigle, 1987: equation 5

    .. math:: r_s = \\frac{1}{2 \\langle \mu^2 \\rangle k^2 L}

    """
    return 1.0 / (2.0 * mean_mu_squared * wavenumber*wavenumber * scale)


def impulse_response_fluctuations(covariance, window=None):
    """Impulse response describing fluctuations.

    :param covariance: Covariance vector.
    :param window: Window to apply to impulse response. If passed `None`, no window is applied.
    :returns: Impulse response of fluctuations filter.

    """
    samples = covariance.shape[-1]
    # If we have an uneven amount of samples, then append a zero.
    if samples%2:
        covariance = np.append(covariance, covariance[-1])
    autospectrum = np.abs(np.fft.rfft(covariance)) # Autospectrum
    del covariance

    # The autospectrum is real-valued. Taking the inverse DFT results in complex and symmetric values."""
    ir = np.fft.ifftshift((np.fft.irfft(np.sqrt(autospectrum)).real), axes=-1)
    del autospectrum
    ir[...,1:] *= 2.0 # Conservation of power

    # Apply window
    if window is not None:
        ir *= window(samples)[...,:]
    return ir


def generate_gaussian_fluctuations(samples, spatial_separation, distance, wavenumber,
                                   mean_mu_squared, scale, window=None,
                                   include_saturation=False, state=None):
    """Generate time series of log-amplitude and phase fluctuations.

    :param samples: Length of series of fluctuations.
    :param spatial_separation: Spatial separation.
    :param distance: Distance.
    :param wavenumber: Wavenumber
    :param mean_mu_squared: Mean mu squared.
    :param scale: Outer length scale.
    :param window: Window function. See also :func:`impulse_response_fluctuations`.
    :param include_saturation: Include saturation of log-amplitude.
    :param state: State of random number generator.
    :type: :class:`np.random.RandomState`
    :returns: Log-amplitude array and phase array.

    This generates one serie of each.

    """
    covariance = covariance_gaussian(spatial_separation=spatial_separation,
                                     distance=distance,
                                     wavenumber=wavenumber,
                                     mean_mu_squared=mean_mu_squared,
                                     scale=scale,
                                     )

    ir = impulse_response_fluctuations(covariance, window=window)

    state = state if state else np.random.RandomState()

    noise = state.randn(samples*2-1)
    log_amplitude = fftconvolve(noise, ir, mode='valid')

    if include_saturation:
        sat_distance = saturation_distance(mean_mu_squared, wavenumber, scale)
        log_amplitude *=  (np.sqrt( 1.0 / (1.0 + distance/sat_distance) ) )

    noise = state.randn(samples*2-1)
    phase = fftconvolve(noise, ir, mode='valid')

    return log_amplitude, phase


COVARIANCES = {
        'gaussian' : covariance_gaussian,
        'vonkarman_wind' : covariance_vonkarman_wind,
    }

def generate_fluctuations(samples, spatial_separation, distance, wavenumber,
                          scale, state=None, window=None,
                          covariance='gaussian', **kwargs):

    try:
        covariance_func = COVARIANCES[covariance]
    except KeyError:
        raise ValueError("Unknown covariance function.")

    try:
        include_saturation = kwargs.pop('include_saturation')
    except KeyError:
        include_saturation = False

    # Determine the covariance
    spatial_separation = np.ones(samples) * spatial_separation
    cov = covariance_func(spatial_separation=spatial_separation,
                                 distance=distance,
                                 wavenumber=wavenumber,
                                 scale=scale, **kwargs)

    #cov /= cov[0]
    # Create an impulse response using this covariance
    ir = impulse_response_fluctuations(cov, window=window)

    # We need random numbers.
    state = state if state else np.random.RandomState()

    # Calculate log-amplitude fluctuations
    noise = state.randn(samples*2-1)
    log_amplitude = fftconvolve(noise, ir, mode='valid')
    #log_amplitude -= cov[0]

    # Include log-amplitude saturation
    if include_saturation:
        if covariance == 'gaussian':
            mean_mu_squared = kwargs['mean_mu_squared']
            sat_distance = saturation_distance(mean_mu_squared, wavenumber, scale)
            log_amplitude *=  (np.sqrt( 1.0 / (1.0 + distance/sat_distance) ) )
        else:
            raise ValueError("Cannot include saturation for given covariance function.")
    # Calculate phase fluctuations
    noise = state.randn(samples*2-1)
    phase = fftconvolve(noise, ir, mode='valid')

    return log_amplitude, phase


def apply_log_amplitude(signal, log_amplitude):
    """Apply log-amplitude fluctuations.

    :param signal: Pressure signal.
    :param log_amplitude: Log-amplitude fluctuations.

    .. math:: p_m = p \\exp{\\chi}

    """
    return signal * np.exp(log_amplitude) # exp(2*log_amplitude) if intensity


def apply_phase(signal, phase, frequency, fs):
    """Apply phase fluctuations.

    :param signal: Pressure signal.
    :param phase: Phase fluctuations.
    :param frequency: Frequency of tone.
    :param fs: Sample frequency.

    Phase fluctuations are applied through a resampling.

    """
    delay = phase/(2.0*np.pi*frequency)
    signal = _apply_delay_turbulence(signal, delay, fs)
    return signal

def apply_fluctuations(signal, fs, frequency=None, log_amplitude=None, phase=None):
    """Apply log-amplitude and/or phase fluctuations.
    """
    if log_amplitude is not None:
        signal = apply_log_amplitude(signal, log_amplitude)
    if phase is not None:
        signal = apply_phase(signal, phase, frequency, fs)
    return signal


def _apply_delay_turbulence(signal, delay, fs):
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


def logamp_variance(amp):
    """Variance of log-amplitude fluctuations.

    :param amp: Time-series of amplitude fluctuations, NOT log-amplitude.

    See Daigle, 1983: equation 15, 16 and 19.
    """
    return (np.log(amp/(amp.mean(axis=-1)[...,None]))**2.0).mean(axis=-1)
    #return (( np.log(amp) - np.log(amp.mean(axis=-1) )[...,None])**2.0).mean(axis=-1)

def phase_variance(phase):
    """Variance of phase fluctuations.

    :param phase: Time-series of phase fluctuations.

    See Daigle, equation 20.
    """
    return ((phase - (phase.mean(axis=-1)[...,None]))**2.0).mean(axis=-1)


#def generate_many_gaussian_fluctuations(samples, spatial_separation, distance, wavenumber,
                                        #mean_mu_squared, scale, window=np.hamming,
                                        #include_saturation=False, seed=None):
    #"""Generate time series of log-amplitude and phase fluctuations.

    #:param samples: Length of series of fluctuations.
    #:param spatial_separation: Spatial separation.
    #:param distance: Distance.
    #:param wavenumber: Wavenumber
    #:param mean_mu_squared: Mean mu squared.
    #:param scale: Outer length scale.
    #:param window: Window function.
    #:param include_saturation: Include saturation of log-amplitude.
    #:param seed: Seed.
    #:returns: Log-amplitude array and phase array.

    #This function performs better when many series need to be generated.

    #"""

    ## Calculate correlation
    ##B = (spatial_separation!=0.0) * np.nan_to_num( ( np.pi/4.0 * mean_mu_squared * (k*k)[:,None] * r[None,:] * L * (erf(spatial_separation/L) / (spatial_separation/L))[None,:] ) )
    ##B = (spatial_separation==0.0)[None,:] * (np.sqrt(np.pi)/2.0 * mean_mu_squared * (k*k)* r * L )[:,None]

    #spatial_separation = np.atleast_1d(spatial_separation)
    #distance = np.atleast_1d(distance)
    #wavenumber = np.atleast_1d(wavenumber)
    #mean_mu_squared = np.atleast_1d(mean_mu_squared)
    #scale = np.atleast_1d(scale)
    #covariance = covariance_gaussian(spatial_separation[None,:], distance[:,None],
                                          #wavenumber[:,None], mean_mu_squared[:,None], scale[:,None])

    #if covariance.ndim==2:
        #N = covariance.shape[-2]
    #elif covariance.ndim==1:
        #N = 1
    #else:
        #raise ValueError("Unsupported amount of dimensions.")

    ## Seed random numbers generator.
    #np.random.seed(seed)
    #n = samples * 2 - 1

    #ir = impulse_response_fluctuations(covariance, window=window)

    #noise = np.random.randn(N,n)
    #log_amplitude = fftconvolve1D(noise, ir, mode='valid') # Log-amplitude fluctuations
    #del noise

    #if include_saturation:
        #sat_distance = saturation_distance(mean_mu_squared, wavenumber, scale)
        #log_amplitude *=  (np.sqrt( 1.0 / (1.0 + distance/sat_distance) ) )[...,None]
        #del sat_distance

    #noise = np.random.randn(N,n)
    #phase = fftconvolve1D(noise, ir, mode='valid')           # Phase fluctuations

    #return log_amplitude, phase




def gaussian_fluctuations_variances(samples, f0, fs, mean_mu_squared,
                                    distance, scale,
                                    spatial_separation, soundspeed,
                                    include_saturation=True, state=None):
    """Calculate the variances of fluctuations in the time series of amplitude and phase fluctuations.

    :param samples: Amount of samples to take.
    :param f0: Frequency for which the fluctuations should be calculated.
    :param fs: Sample frequency.
    :param mean_mu_squared: Mean of refractive-index squared.
    :param r: Distance.
    :param L: Outer length scale.
    :param rho: Spatial separation.
    :param soundspeed: Speed of sound.
    :param include_phase: Include phase fluctuations.
    :param state: State of numpy random number generator.

    """
    spatial_separation *= np.ones(samples)
    wavenumber = 2.0 * np.pi * f0 / soundspeed
    a, p = generate_gaussian_fluctuations(samples, spatial_separation, distance,
                                               wavenumber, mean_mu_squared, scale,
                                               include_saturation=include_saturation,
                                               state=state)

    return logamp_variance(np.exp(a)), phase_variance(p)


#def fluctuations_variance(signals, fs, N=None):
    #"""
    #Determine the variance in log-amplitude and phase by ensemble averaging.

    #:param signals: List of signals or samples.
    #:param fs: Sample frequency

    #The single-sided spectrum is calculated for each signal/sample.

    #The log-amplitude of the :math:`n`th sample is given by

    #.. math:: \\chi^2 = \\ln{\\frac{A_{n}}{A_{0}}}

    #where :math:`A_{n}` is the amplitude of sample :math:`n` and :math:`A_{0}` is the ensemble average

    #.. math:: A_{0} = \\frac{1}{N} \\sum_{n=1}^{N} \\chi_{n}^2

    #"""

    #s = np.array(signals) # Array of signals

    ##print s

    #f, fr = ir2fr(s, fs, N) # Single sided spectrum
    #amp = np.abs(fr)
    #phase = np.angle(fr)
    #logamp_squared_variance = (np.log(amp/amp.mean(axis=0))**2.0).mean(axis=0)
    #phase_squared_variance = ((phase - phase.mean(axis=0))**2.0).mean(axis=0)

    #return f, logamp_squared_variance, phase_squared_variance


def plot_variance(frequency, logamp, phase):
    """
    Plot variance.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(frequency, logamp, label=r"$\langle \chi^2 \rangle$", color='b')
    ax.scatter(frequency, phase, label=r"$\langle S^2 \rangle$", color='r')
    ax.set_xlim(100.0, frequency.max())
    ax.set_ylim(0.001, 10.0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid()
    ax.set_xlabel(r"$f$ in Hz")
    ax.set_ylabel(r"$\langle X \rangle$")

    return fig

