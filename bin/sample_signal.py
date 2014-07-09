
import sys
sys.path.append('..')

import Geometry as Geo

import numpy as np

from auraliser.model import Model
from auraliser.signal import Signal

import matplotlib as mpl

mpl.rc('figure', figsize=(12,10))

import scipy.signal


def main():
    
    """Construct Model"""
    model = Model()
    
    """Construct a signal."""
    fs = 44100.0            # Sample frequency
    duration = 30.         # Duration in seconds.
    f = np.array([70, 350, 800, 830])              # Emission signal consists of a tone of 1000.0 Hz
    Af = np.array([0.09, 0.02, 0.01, 0.01]) * 20.0
    
    assert(np.all(fs >= 2.0 * f))     # Nyquist
    
    velocity = 80.0                        # Velocity of source in meters per second.
    #velocity = 1.5 * model.atmosphere.soundspeed
    
    distance = velocity * duration          # Total distance covered
    
    dt = 1.0/fs                               # Seconds per sample
    t = np.arange(0.0, duration, dt)        # Time vector
    signal = np.sum( Af.reshape((-1,1)) * np.sin(2.0 * np.pi * f.reshape((-1,1)) * t) , axis=0)   # Emission signal
    
    signal = np.random.randn(len(t))/2.0       # Let's add some random noise.
    
    #signal += np.random.randn(len(t))/2.0       # Let's add some random noise.
    #signal /= np.abs(signal).max()
    
    signal = Signal(signal, sample_frequency=fs)    # Emission signal as the right type
    
    x = velocity * (t - duration/2.0)    # Source moves along the x-axis.
    y = 0.0
    z = 0.01   # Altitude of source
    
    
    src_x = np.ones(len(t)) * x
    src_y = np.ones(len(t)) * y   # Source does not move along the y-axis.
    src_z = np.ones(len(t)) * z   # Source does not change in altitude.
    model.geometry.source.position = Geo.PointList(np.vstack((src_x,src_y,src_z)))
    
    
    rcv_x = np.zeros(len(t))
    rcv_y = np.zeros(len(t))
    rcv_z = np.zeros(len(t))
    model.geometry.receiver.position = Geo.PointList(np.vstack((src_x,src_y,src_z)))
    
    
    #signal.plot_spectrogram('source.png')
    #signal.to_wav('source.wav')
    
    #signal = model.apply_doppler(signal)
    #signal = model.apply_spherical_spreading(signal)
    signal = model.apply_atmospheric_absorption(signal, n_d=100, taps=100)
    
    signal.plot_spectrogram('receiver.png')
    signal.to_wav('receiver.wav')
    
    
    #s1.plot_scalo('after_attenuation_scalo.png')
    
    
    
    model.atmosphere.plot_attenuation_coefficient('alpha.png', np.logspace(1.0, 4.0, 1000))
    
    #print s1.sample_frequency
    
    #model.plot_attenuation_impulse_response('ir_attenuation.png', signal)
    
    #signal = model.apply_atmospheric_absorption(signal)
    
    #signal.to_wav('after_attenuation.wav')
    #signal.plot_spectrogram('after_attenuation.png')
    
    #signal.plot_signal('Signal.png', 0, 44)
    #model.plot_time_compression('compression.png', signal=signal)
    #model.plot_delay('delay.png', signal=signal)
    #model.geometry.plot_distance('distance.png', signal=signal)
    #model.geometry.plot_position('position.png')
    
    #model.plot_velocity('velocity.png', signal=signal)
    
    #signal.to_wav('before_shift.wav')
    #signal.plot_spectrogram('spectrogram_before_shift.png')
    
    #model.apply_doppler(signal)
    
    #signal.to_wav('after_shift.wav')
    #signal.plot_spectrogram('spectrogram_after_shift.png')
    
    #model.unapply_doppler(signal)
    
    #signal.to_wav('after_reverting_shift.wav')
    #signal.plot_spectrogram('spectrogram_after_reverting.png')
    
    
if __name__=="__main__":
    main()