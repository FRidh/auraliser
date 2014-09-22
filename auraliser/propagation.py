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
from turbulence_jens import map_source_to_receiver

def ir_real_signal(tf, N=None):
    """
    Take a single-sided spectrum `tf` and convert it to an impulse response of `N` samples.
    
    :param tf: Single-sided spectrum.
    :param N: Amount of samples to use for the impulse response.
    
    .. note:: This function should work for multiple tf's in one array.
    
    """
    return np.fft.ifftshift(irfft(tf, n=N)).real


def apply_spherical_spreading(signal, distance):
    """
    Apply spherical spreading to ``signal``.
    
    .. math:: p_2 = p_1 \\frac{r_2}{r_1}
    
    where :math:`r_2` is 1.0.
    
    :param signal: Signal
    :type signal: :class:`auraliser.signal.Signal`
    
    :rtype: :class:`auraliser.signal.Signal`
    """
    return signal / distance # * 1.0
    
def unapply_spherical_spreading(signal, distance):
    """
    Unapply spherical spreading.
    
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
    """
    Apply ``delay`` to ``signal`` in-place.
    
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
    
def apply_doppler(signal, delay, fs):
    """
    Apply Doppler shift to ``signal``.
    
    :param signal: Signal to be shifted.
    :type signal: :class:`auraliser.signal.Signal`
    """    
    k_r = np.arange(0, len(signal), 1)          # Create vector of indices
    k = k_r - delay * fs                      # Create vector of warped indices
    kf = np.floor(k).astype(int)       # Floor the warped indices. Convert to integers so we can use them as indices.
    
    R = ( (1.0-k+kf) * signal[kf] + (k-kf) * signal[kf+1] ) * (kf >= 0) #+ 0.0 * (kf<0)
    return R
    
def unapply_doppler(signal, delay, fs):
    """
    Unapply Doppler shift to ``signal``.
    
    :param signal: Signal to be shifted.
    :type signal: :class:`auraliser.signal.Signal`
    """
    return _map_source_to_receiver(signal, -delay, fs)


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
    

from scipy.special import gamma
from scipy.special import kv as besselk
from scipy.integrate import cumtrapz

def correlation_von_karman(f, c0, rho, distance, L, Cv):
    """
    
    """
    k = 2.0*np.pi*f / c0
    K0 = 2.0*np.pi / L
    
    A = 5.0/(18.0*np.pi*gamma(1./3.)) # Equation 11, see text below. Approximate result is 0.033
    gamma_v = 3./10.*np.pi**2.*A*k**2.*K0**(-5./3.)*4.*(Cv/c0)**2.  # Equation 28, only wind fluctuations
    
    krho = k * rho
    t = krho
    t[t==0.0] = 1.e-20
    B = 2.0*gamma_v * distance / krho * cumtrapz((2.0**(1./6.)*t**(5./6.)/gamma(5./6.) * (besselk(5./6., t) - t/2.0*besselk(1./6., t)) ), krho, dx=None, initial=0.001)

    return B

def _spatial_separation(A, B, C):
    """Spatial separation.
    Source a, reference source b and receiver c.
    """
    a = (B-C).norm()
    A = A.asarray()
    B = B.asarray()
    C = C.asarray()
    b = norm(A-C)
    #print(A, B)
    c = norm(A-B)

    #print(a, b, c)
    gamma = np.arccos((a**2.0+b**2.0-c**2.0) / (2.0*a*b))
    #print(gamma)
    rho = 2.0 * b * np.sin(gamma/2.0)
    dr = a - b
    return rho, dr

def apply_turbulence(signal, fs, mean_mu_squared, r, L, rho, Cv, soundspeed=343.0, fraction=3, order=1, include_saturation=True, include_amplitude=True, include_phase=True, seed=None):
    """Apply turbulence to signal.
    
    :param signal: Original signal
    :param fs: Sample frequency
    :param mean_mu_squared: Dynamic refractive index squared.
    :param r: Distance.
    :param L: Correlation length / Outer length scale.
    :param rho: Spatial separation.
    :param soundspeed: Speed of sound.
    :param fraction: Fraction of octaves.
    :param order: Order of bandpass filters.
    :param include_saturation: Include saturation.
    :param include_amplitude: Include amplitude modulations.
    :param include_phase: Include phase modulations.
    :param seed: Seed for random number generator.
    
    """
    samples = len(signal)
    ob = OctaveBand(fstart=5.0, fstop=fs / 2.5, fraction=fraction)
    fb = Filterbank(ob, sample_frequency=fs, order=order)
    
    # Modulation frequencies.
    f0 = ob.center
    
    # Amount of signals.
    N = len(f0)

    # Bandpass filtered input signal.
    signals = np.empty((N, samples), dtype='float64')
    for i, s in enumerate(fb.filtfilt(signal)):
        signals[i,:] = s
    #del ob, fb
    # Wavenumber
    k = 2.0 * np.pi * f0 / soundspeed
        
    # Calculate correlation
    B = np.empty_like(signals)
    for i, f in enumerate(ob.center):
        B[i,:] = correlation_von_karman(f, soundspeed, rho, r, L, Cv)
        #print(B)
    #B += (rho!=0.0) * np.nan_to_num( ( np.pi/4.0 * mean_mu_squared * (k*k)[:,None] * r[None,:] * L * (erf(rho/L) / (rho/L))[None,:] ) ) 
    #B += (rho==0.0) * np.sqrt(np.pi)/2.0 * mean_mu_squared * (k*k)[:,None]* r[None,:] * L
    np.save("B.npy", B)
    # Seed random numbers generator.
    np.random.seed(seed)
    n = samples * 2 - 1

    # Autospectrum of correlation.
    auto = np.abs(fft(B))#, n=samples))).real # Autospectrum, however, still double-sided
    del B, rho
    
    np.save("auto.npy", auto)


    # The autospectrum is real-valued. Taking the inverse DFT results in complex and symmetric values."""
    ir = np.fft.ifftshift((ifft(np.sqrt(auto)).real), axes=-1) #
    ir = 2.0 * ir[:, 0:samples] # Only take half the IR to have right amount of samples.
    #ir[0] /= 2.0
    del auto
    np.save("ir.npy", ir)
    
    if include_amplitude:
        # Generate random numbers.
        n1 = np.random.randn(n)
        n1 = np.tile(n1, (N,1))
        
        log_amplitude = fftconvolve1D(n1, ir, mode='valid') # Log-amplitude fluctuations
        del n1
        if not include_phase:
            del ir
        # Apply amplitude saturation
        if include_saturation:
            saturation_distance = 1.0 / (2.0 * mean_mu_squared * k*k * L)
            log_amplitude *=  np.sqrt( 1.0 / (1.0 + r[None,:]/saturation_distance[:,None]) )
            del saturation_distance
        # Apply fluctuations
        signals *= np.exp(log_amplitude)
        del log_amplitude
        
    if include_phase:
        # Generate random numbers.
        n2 = np.random.randn(n)
        n2 = np.tile(n2, (N,1))
        
        phase = fftconvolve1D(n2, ir, mode='valid')           # Phase fluctuations
        del n2, ir
        # Apply fluctuations
        delay = phase/(2.0*np.pi*f0)[:,None]
        for i in range(N):
            signals[i] = map_source_to_receiver(signals[i], delay[i], fs)
        del delay
        
    return signals.sum(axis=0)
    

#def apply_turbulence(signal, fs, mean_mu_squared, r, L, rho, soundspeed=343.0, fraction=3, order=1, include_saturation=True, include_amplitude=True, include_phase=True, seed=None):
    #"""Apply turbulence to signal.
    
    #:param signal: Original signal
    #:param fs: Sample frequency
    #:param mean_mu_squared: Dynamic refractive index squared.
    #:param r: Distance.
    #:param L: Correlation length / Outer length scale.
    #:param rho: Spatial separation.
    #:param soundspeed: Speed of sound.
    #:param fraction: Fraction of octaves.
    #:param order: Order of bandpass filters.
    #:param include_saturation: Include saturation.
    #:param include_amplitude: Include amplitude modulations.
    #:param include_phase: Include phase modulations.
    #:param seed: Seed for random number generator.
    
    #"""
    #samples = len(signal)
    #ob = OctaveBand(fstart=5.0, fstop=fs / 2.5, fraction=fraction)
    #fb = Filterbank(ob, sample_frequency=fs, order=order)
    
    ## Modulation frequencies.
    #f0 = ob.center
    
    ## Amount of signals.
    #N = len(f0)

    ## Bandpass filtered input signal.
    #signals = np.empty((N, samples), dtype='float64')
    #for i, s in enumerate(fb.filtfilt(signal)):
        #signals[i,:] = s
    #del ob, fb
    ## Wavenumber
    #k = 2.0 * np.pi * f0 / soundspeed
        
    ## Calculate correlation
    #B = np.zeros_like(signals)
    #B += (rho!=0.0) * np.nan_to_num( ( np.pi/4.0 * mean_mu_squared * (k*k)[:,None] * r[None,:] * L * (erf(rho/L) / (rho/L))[None,:] ) ) 
    #B += (rho==0.0) * np.sqrt(np.pi)/2.0 * mean_mu_squared * (k*k)[:,None]* r[None,:] * L
    #np.save("B.npy", B)
    ## Seed random numbers generator.
    #np.random.seed(seed)
    #n = samples * 2 - 1

    ## Autospectrum of correlation.
    #auto = np.abs(fft(B))#, n=samples))).real # Autospectrum, however, still double-sided
    #del B, rho
    
    #np.save("auto.npy", auto)


    ## The autospectrum is real-valued. Taking the inverse DFT results in complex and symmetric values."""
    #ir = np.fft.ifftshift((ifft(np.sqrt(auto)).real), axes=-1) #
    #ir = 2.0 * ir[:, 0:samples] # Only take half the IR to have right amount of samples.
    ##ir[0] /= 2.0
    #del auto
    #np.save("ir.npy", ir)
    
    #if include_amplitude:
        ## Generate random numbers.
        #n1 = np.random.randn(n)
        #n1 = np.tile(n1, (N,1))
        
        #log_amplitude = fftconvolve1D(n1, ir, mode='valid') # Log-amplitude fluctuations
        #del n1
        #if not include_phase:
            #del ir
        ## Apply amplitude saturation
        #if include_saturation:
            #saturation_distance = 1.0 / (2.0 * mean_mu_squared * k*k * L)
            #log_amplitude *=  np.sqrt( 1.0 / (1.0 + r[None,:]/saturation_distance[:,None]) )
            #del saturation_distance
        ## Apply fluctuations
        #signals *= np.exp(log_amplitude)
        #del log_amplitude
        
    #if include_phase:
        ## Generate random numbers.
        #n2 = np.random.randn(n)
        #n2 = np.tile(n2, (N,1))
        
        #phase = fftconvolve1D(n2, ir, mode='valid')           # Phase fluctuations
        #del n2, ir
        ## Apply fluctuations
        #delay = phase/(2.0*np.pi*f0)[:,None]
        #for i in range(N):
            #signals[i] = map_source_to_receiver(signals[i], delay[i], fs)
        #del delay
        
    #return signals.sum(axis=0)

    
def _atmospheric_absorption(signal, fs, atmosphere, distance, sign, taps, N, n_d):
    """
    Apply or unapply atmospheric absorption depending on sign.
    """    
    if n_d is not None:
        d_ir = np.linspace(distance.min(), distance.max(), n_d, endpoint=True)   # Distances to check
        
        start = (N-taps)//2
        stop = (N+taps)//2 - 1
        
        ir_i = atmosphere.ir_attenuation_coefficient(d=d_ir, N=N, fs=fs, sign=sign)[start:stop+1, :]
        
        indices = np.argmin(np.abs(distance.reshape(-1,1) - d_ir), axis=1)
        ir = ir_i[:, indices]
        
    else:
        ir = atmosphere.ir_attenuation_coefficient(d=distance, N=N, fs=fs, sign=sign)[0:taps,:]
    return convolve(signal, ir)
    
    
def apply_atmospheric_absorption(signal, fs, atmosphere, distance, taps=128, N=2048, n_d=None):
    """
    Apply atmospheric absorption to ``signal``.
    
    :param signal: Signal
    :param fs: Sample frequency
    :param atmosphere: Atmosphere
    :param distance: Distance
    :param taps: Amount of filter taps to keep.
    :param N: Blocks to use for performing the FFT. Determines frequency resolution.
    :param n_d: Amount of unique distances to consider.
    """
    return _atmospheric_absorption(signal, fs, atmosphere, distance, +1, taps, N, n_d)
    
    
def unapply_atmospheric_absorption(self, signal, taps=128, N=2048, n_d=None):
    """
    Unapply atmospheric absorption to `signal`.
    
    :param signal: Signal
    :param fs: Sample frequency
    :param atmosphere: Atmosphere
    :param distance: Distance
    :param taps: Amount of filter taps to keep.
    :param N: Blocks to use for performing the FFT. Determines frequency resolution.
    :param n_ir: Value between 0 and 1 representing the percentage of unique impulse responses to use.
    """
    return _atmospheric_absorption(signal, fs, atmosphere, distance, -1, taps, N, n_d)
