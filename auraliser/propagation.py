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

import math
import numba


#def ir_real_signal(spectrum, n_blocks=None):
    #"""Take a single-sided spectrum `tf` and convert it to an impulse response of `N` samples.
    
    #:param spectrum: Single-sided spectrum real.
    #:param n_blocks: Amount of blocks to use for the FFT and thus IR taps.
    
    #.. note:: This function should work for multiple tf's in one array. In the input every row is a spectrum. In the output every row is an IR.
    
    #"""
    #spectrum = np.hstack((spectrum, spectrum[..., 1::-1]))
    #return np.fft.ifftshift(np.fft.ifft(spectrum, n=n_blocks), axes=0).real


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
    
    .. math:: p_2 = p_1 \\frac{r_2}{r_1}
    
    where :math:`r_2` is 1.0.
    
    :param signal: Signal
    :type signal: :class:`auraliser.signal.Signal`
    
    :rtype: :class:`auraliser.signal.Signal`
    """
    return signal / distance # * 1.0
    
def unapply_spherical_spreading(signal, distance):
    """Unapply spherical spreading.
    
    .. math:: p_2 = p_1 \\frac{r_2}{r_1}
    
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
    


from turbulence.vonkarman import covariance_wind as covariance_von_karman

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
    
    """
    #B = B[None,:]
    #print(type(A))
    #print(type(B))
    #print(type(C))
    #print(A.shape, B.shape, C.shape)
    a = norm(B-C)#.norm()
    #A = A.asarray()
    #B = B.asarray()
    #C = C.asarray()
    b = norm(A-C)
    #print(A, B)
    c = norm(A-B)
    #print((A-).shape)
    #print(a)
    #print(b)
    #print((A-B).shape)

    #print(a, b, c)
    gamma = np.arccos((a**2.0+b**2.0-c**2.0) / (2.0*a*b))
    #print(gamma)
    #spatial_separation = 2.0 * b * np.sin(gamma/2.0)
    spatial_separation = 2.0 * a * np.tan(gamma/2.0)
    dr = a - b
    return spatial_separation, dr


from scipy.signal import fftconvolve, butter, filtfilt

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



def apply_turbulence_vonkarman(signal, fs, mean_mu_squared, r, scale, spatial_separation, Cv, soundspeed=343.0, fraction=3, order=1, include_saturation=True, include_amplitude=True, include_phase=True, seed=None):
    
    #cutoff = fs/2.0 * 0.3
    #signal = lowpass(signal, fs, cutoff)
    
    #spatial_separation /= 1000000.0
    
    #print(len(spatial_separation))
    #n = 2000
    #spatial_separation = moving_average(spatial_separation, n=n) 
    #spatial_separation = np.hstack((spatial_separation, np.ones(n-1)*spatial_separation[-1]))
    #print(len(spatial_separation))
    
    #print(spatial_separation)
    #spatial_separation = np.abs( np.random.randn(len(signal)) * 0.0001 )
    #print(spatial_separation)
    
    modulation_frequencies = 10.0
    samples = len(signal)
    B = covariance_von_karman(modulation_frequencies, soundspeed, spatial_separation, r, scale, Cv, steps=50)
    auto = np.abs(fft(B))#, n=samples))).real # Autospectrum, however, still double-sided
    ir = np.fft.ifftshift((ifft(np.sqrt(auto)).real), axes=-1) #
    ir *= 2.0
    n = samples * 2 - 1
    
    np.random.seed(seed)
    
    from acoustics.signal import power_spectrum
    
    if include_amplitude:
        n1 = np.random.randn(n)
        logamp = fftconvolve(n1, ir, mode='valid')

        #n = 500
        #logamp = moving_average(logamp, n=n) 
        #logamp = np.hstack((logamp, np.ones(n-1)*logamp[-1]))

        #logamp = lowpass(logamp, fs, cutoff, order=1)
        #print(power_spectrum(logamp, fs)[1])
        signal = signal * logamp
        
    if include_phase:
        n2 = np.random.randn(n)
        phase = fftconvolve(n2, ir, mode='valid')
        delay = phase/(2.0*np.pi*modulation_frequencies)
        signal = apply_delay_turbulence(signal, delay, fs)
    return signal

#def apply_turbulence(signal, fs, mean_mu_squared, r, scale, spatial_separation, Cv, soundspeed=343.0, fraction=3, order=1, include_saturation=True, include_amplitude=True, include_phase=True, seed=None):
    #"""Apply turbulence to signal.
    
    #:param signal: Original signal
    #:param fs: Sample frequency
    #:param mean_mu_squared: Dynamic refractive index squared.
    #:param r: Distance.
    #:param scale: Correlation length / Outer length scale.
    #:param spatial_separation: Spatial separation.
    #:param soundspeed: Speed of sound.
    #:param fraction: Fraction of octaves.
    #:param order: Order of bandpass filters.
    #:param include_saturation: Include saturation.
    #:param include_amplitude: Include amplitude modulations.
    #:param include_phase: Include phase modulations.
    #:param seed: Seed for random number generator.
    
    #"""
    #samples = len(signal)
    #ob = OctaveBand(fstart=5.0, fstop=fs / 2.0, fraction=fraction)
    #ob = OctaveBand.from_bands( ob[ ob.upper < fs / 2.0 ] ) 
    #print(ob)

    #fb = Filterbank(ob, sample_frequency=fs, order=order)
    
    ## Modulation frequencies.
    #modulation_frequencies = ob.center
    
    ## Amount of signals.
    #N = len(modulation_frequencies)

    ## Bandpass filtered input signal.
    #signals = np.empty((N, samples), dtype='float64')
    #for i, s in enumerate(fb.filtfilt(signal)):
        #signals[i,:] = s
    #del signal#del ob, fb
    ## Wavenumber
    #k = 2.0 * np.pi * modulation_frequencies / soundspeed
        
    ## Calculate correlation
    #B = np.empty_like(signals)
    #for i, f in enumerate(ob.center):
        #B[i,:] = covariance_von_karman(f, soundspeed, spatial_separation, r, L, Cv)
    ##B = covariance_von_karman(ob.center, soundspeed, spatial_separation, r, L, Cv)
    
    ##print(B)
    ##B += (spatial_separation!=0.0) * np.nan_to_num( ( np.pi/4.0 * mean_mu_squared * (k*k)[:,None] * r[None,:] * L * (erf(spatial_separation/L) / (spatial_separation/L))[None,:] ) ) 
    ##B += (spatial_separation==0.0) * np.sqrt(np.pi)/2.0 * mean_mu_squared * (k*k)[:,None]* r[None,:] * L
    #np.save("B.npy", B)
    ## Seed random numbers generator.
    #np.random.seed(seed)
    #n = samples * 2 - 1
    ##n = samples

    ## Autospectrum of correlation.
    #auto = np.abs(fft(B))#, n=samples))).real # Autospectrum, however, still double-sided
    #del B, spatial_separation
    
    #np.save("auto.npy", auto)


    ## The autospectrum is real-valued. Taking the inverse DFT results in complex and symmetric values."""
    #ir = np.fft.ifftshift((ifft(np.sqrt(auto)).real), axes=-1) #
    #ir = 2.0 * ir[:, 0:samples] # Only take half the IR to have right amount of samples.
    ##ir *= np.hanning(ir.shape[-1])[None,:]
    ##ir[0] /= 2.0
    #del auto
    #np.save("ir.npy", ir)
    
    #if include_amplitude:
        ## Generate random numbers.
        #n1 = np.random.randn(n)
        #n1 = np.tile(n1, (N,1))
        ##ir[:,samples/2] = ir[:,samples/2+1]
        #log_amplitude = fftconvolve1D(n1, ir, mode='valid') # Log-amplitude fluctuations
        ##log_amplitude = fftconvolve1D(n1, ir[:,n/2-5:n/2+5], mode='same')
        ##print(log_amplitude.shape)
        #del n1
        #if not include_phase:
            #del ir
        ## Apply amplitude saturation
        ##if include_saturation:
            ##saturation_distance = 1.0 / (2.0 * mean_mu_squared * k*k * L)
            ##log_amplitude *=  np.sqrt( 1.0 / (1.0 + r[None,:]/saturation_distance[:,None]) )
            ##del saturation_distance
        ## Apply fluctuations
        #print(np.exp(log_amplitude).max())
        #signals *= np.exp(log_amplitude)
        #del log_amplitude
        
    #if include_phase:
        ## Generate random numbers.
        #n2 = np.random.randn(n)
        #n2 = np.tile(n2, (N,1))
        
        #phase = fftconvolve1D(n2, ir, mode='valid')           # Phase fluctuations
        ##phase = fftconvolve1D(n2, ir[:,n/2-50:n/2+50], mode='same')
        #del n2, ir
        ## Apply fluctuations
        #delay = phase/(2.0*np.pi*modulation_frequencies)[:,None]
        #print(delay.max())
        #print(delay.min())
        #for i in range(N):
            #signals[i] = apply_delay_turbulence(signals[i], delay[i], fs)
            ##signals[i] = map_source_to_receiver(signals[i], delay[i], fs)
        #del delay
        
    #return signals.sum(axis=0)
    

def apply_turbulence_gaussian(signal, fs, mean_mu_squared, distance, scale, 
                              spatial_separation, soundspeed=343.0, 
                              fraction=3, order=1, include_saturation=True, 
                              include_amplitude=True, include_phase=True, 
                              seed=None):
    """Apply turbulence to signal.
    
    :param signal: Original signal
    :param fs: Sample frequency
    :param mean_mu_squared: Dynamic refractive index squared.
    :param distance: Distance.
    :param scale: Correlation length / Outer length scale.
    :param spatial_separation: Spatial separation.
    :param soundspeed: Speed of sound.
    :param fraction: Fraction of octaves.
    :param order: Order of bandpass filters.
    :param include_saturation: Include saturation.
    :param include_amplitude: Include amplitude modulations.
    :param include_phase: Include phase modulations.
    :param seed: Seed for random number generator.
    :returns: The original signal but with fluctuations applied to it.
    
    """
    samples = len(signal)
    
    # All fraction octaves band in range
    bands = OctaveBand(fstart=5.0, fstop=fs/2., fraction=fraction) 
    
    nyq = fs/2.0 # Nyquist frequency
    
    # Assure cornerfrequencies of the fractional-octaves are below the Nyquist frequency
    if (bands.upper >= nyq).any():
        index = np.where( bands.upper < nyq)[0].max()
        bands = OctaveBand(fstart=5.0, fstop=bands.center[index], fraction=fraction) 
    
    filterbank = Filterbank(bands, sample_frequency=fs, order=order)
    
    # Modulation frequencies.
    modulation_frequencies = bands.center
    
    # Amount of signals.
    n_signals = len(modulation_frequencies)

    # Bandpass filtered input signal.
    signals = np.empty((n_signals, samples), dtype='float64')
    for i, signal_i in enumerate(filterbank.filtfilt(signal)):
        signals[i,:] = signal_i
    
    del signal, bands, filterbank
    
    # Wavenumber
    k = 2.0 * np.pi * modulation_frequencies / soundspeed
        
    # Calculate covariance
    covariance = np.zeros_like(signals)
    
    covariance += (spatial_separation!=0.0) * \
                  np.nan_to_num( ( np.pi/4.0 * mean_mu_squared * (k*k)[:,None] * \
                  distance[None,:] * scale * (erf(spatial_separation/scale) / \
                  (spatial_separation/scale))[None,:] ) ) 
              
    covariance += (spatial_separation==0.0) * np.sqrt(np.pi)/2.0 * \
                  mean_mu_squared * (k*k)[:,None]* distance[None,:] * scale
    
    # Autospectrum of correlation.
    auto = np.abs(fft(covariance)) # Autospectrum, however, still double-sided
    del covariance, spatial_separation

    # The autospectrum is real-valued. Taking the inverse DFT results in complex and symmetric values."""
    ir = np.fft.ifftshift((ifft(np.sqrt(auto)).real), axes=-1) #
    ir = 2.0 * ir[:, 0:samples] # Only take half the IR to have right amount of samples.
    
    del auto
    
    # Seed random numbers generator.
    np.random.seed(seed)
    
    # Amount of noise samples required for the convolution
    n_noise_samples = samples * 2 - 1
    
    if include_amplitude:
        # Generate random numbers.
        noise = np.random.randn( n_noise_samples)
        noise = np.tile(noise, (n_signals, 1))
        
        # Log-amplitude fluctuations. Convolution of noise with impulse response
        log_amplitude = fftconvolve1D(noise, ir, mode='valid')
        del noise
        
        if not include_phase:
            del ir
        
        # Apply amplitude saturation
        if include_saturation:
            saturation_distance = 1.0 / (2.0 * mean_mu_squared * k*k * scale)
            log_amplitude *=  np.sqrt( 1.0 / (1.0 + r[None,:]/saturation_distance[:,None]) )
            del saturation_distance
        
        # Apply fluctuations
        signals *= np.exp(log_amplitude)
        del log_amplitude
        
    if include_phase:
        # Generate random numbers.
        noise = np.random.randn( n_noise_samples)
        noise = np.tile(noise, (n_signals, 1))
        phase = fftconvolve1D(noise, ir, mode='valid')           # Phase fluctuations
        del noise, ir
        # Apply fluctuations
        delay = phase/(2.0*np.pi*modulation_frequencies)[:,None]
        for i in range(n_signals):
            # Apply delay by resampling the signal. Uses linear interpolation.
            signals[i] = apply_delay_turbulence(signals[i], delay[i], fs) 
        del delay
        
    return signals.sum(axis=0) # Sum over the contribution of every band

    
    
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
    

#@numba.jit(nogil=True)    
#def _lanczos_resample(signal, delay, output, a, fs):
    #"""Sample signal at float samples.
    #"""
    #for index in range(len(signal)):#, x in enumerate(samples):
        #x = -delay[index] * fs + index
        #if x < 0.0:
            #x = 0.0
        #for i in range(math.floor(x)-a+1, math.floor(x+a)):
            #output[index] += signal[i] * _lanczos_window(x-i, a)
    #return output
    
    
    

    
    
