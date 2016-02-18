import numpy as np
import pytest
from acoustics import Signal
from acoustics.atmosphere import Atmosphere
#from auraliser.propagation import apply_atmospheric_absorption

@pytest.fixture
def fs():
    return 44100.

@pytest.fixture
def frequency():
    return 1000.

#def test_atmospheric_attenuation(fs, frequency):
    #"""Test whether the attenuation is correct.
    #"""

    #duration = 5.0
    #distance = 100.0
    #n_blocks = 256

    #atmosphere = Atmosphere()
    #samples = int(fs*duration)
    #t = np.arange(samples) / fs

    #signal = Signal( np.sin(2.0*np.pi*frequency*t) , fs)

    #out = Signal( apply_atmospheric_absorption(signal, fs, atmosphere, distance*np.ones(samples), n_blocks=512, n_distances=20) , fs)

    #attenuation = atmosphere.attenuation_coefficient(frequency) * distance

    #signal_difference = signal.leq() - out.leq()

    ## Assert we have attenuation by testing.
    #assert( signal_difference > 0.0 )

    #assert( np.abs(signal_difference - attenuation) < 1.0)

# Test has been replaced by realtime version
#def test_atmospheric_attenuation_n_distances():
    #"""Test whether the attenuation is identical when using two different
    #values for `n_distances`.
    #"""

    #fs = 4000.0
    #frequency = 1000.0
    #duration = 1.0
    #distance = 100.0

    #n_blocks = 256

    #atmosphere = Atmosphere()
    #samples = int(fs*duration)
    #t = np.arange(samples) / fs

    #signal = Signal( np.sin(2.0*np.pi*frequency*t) , fs)

    #out1 = Signal( apply_atmospheric_absorption(signal, fs, atmosphere, distance*np.ones(samples), n_blocks=n_blocks, n_distances=20) , fs)

    #out2 = Signal( apply_atmospheric_absorption(signal, fs, atmosphere, distance*np.ones(samples), n_blocks=n_blocks, n_distances=40) , fs)

    #assert(out1.leq()==out2.leq())

    #assert(np.all(out1==out2))




##def test_atmospheric_attenuation_varying_distance(duration):
    ##"""Test whether a varying distance works correctly.

    ##Note that the error strongly depends on the relation between the sample frequency,
    ##the signal frequency, duration and the amount of blocks that are used.
    ##"""

    ##fs = 4000.0
    ##frequency = 1000.0

    ##duration = 2.0

    ##atmosphere = Atmosphere()
    ##samples = int(fs*duration)
    ##t = np.arange(samples) / fs

    ##distance = np.linspace(100.0, 1000.0, samples)

    ##signal = Signal( np.sin(2.0*np.pi*frequency*t) , fs)

    ##out = Signal( apply_atmospheric_absorption(signal, fs, atmosphere, distance, n_blocks=512, n_distances=50) , fs)

    ##attenuation = atmosphere.attenuation_coefficient(frequency) * distance

    ##signal_difference = signal.leq() - out.leq()

    ### Assert we have attenuation by testing.
    ##assert( signal_difference > 0.0 )

    ##assert( np.all( np.abs(signal_difference - attenuation) < 1.0) )