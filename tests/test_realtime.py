"""
Tests for :mod:`acoustics.realtime`.

"""
import pytest
from acoustics import Signal
from acoustics.atmosphere import Atmosphere
from acoustics.generator import noise
from auraliser.realtime import *
from itertools import cycle, chain
from streaming.stream import Stream, BlockStream


@pytest.fixture
def frequency():
    return 1000.0

@pytest.fixture
def tone(frequency):
    fs = 8000.
    duration = 10.0
    samples = int(fs*duration)
    times = np.arange(samples)/fs
    signal = Signal(np.sin(2.*np.pi*frequency*times), fs)
    return signal

signal = tone


def test_apply_spherical_spreading(signal):
    """Test whether spherical spreading is applied correctly."""

    distance = np.linspace(100, 1000, signal.samples)

    time = 0.125
    nblock = np.floor(signal.fs*time)

    # Determine levels of original signal and signal after spreading
    signal_levels = signal.levels(time=time)[1]
    attenuated_levels = Signal.from_iter(apply_spherical_spreading(Stream(signal), Stream(distance)), signal.fs).levels(time=time)[1]
    # We're interested in the difference, which is a negative value.
    actual_difference = attenuated_levels - signal_levels

    # We partition the distance in chunks corresponding to the averaging/integration time
    # We take the closest point as reference (NOT mean value!).
    distance = np.reshape(distance[0:signal.samples//nblock*nblock], (-1, nblock)).min(axis=-1)

    # Sound is expected to decrease with
    expected_difference = - 20.0*np.log10(distance)

    # There is a relatively high error for the points nearby
    assert np.all( np.abs(actual_difference - expected_difference) <  0.5)


def test_apply_atmospheric_attenuation(signal, frequency):

    atmosphere = Atmosphere()

    distance = np.linspace(100., 1000., signal.samples)
    nblock = 8192

    # Apply atmospheric attenuation
    attenuated = apply_atmospheric_attenuation(Stream(signal), signal.fs, Stream(distance), nblock=nblock, atmosphere=atmosphere, ntaps=128)
    # Create instance of Signal
    attenuated = Signal.from_iter(attenuated.samples(), signal.fs)

    # To determine sound pressure level as function of time. Note that we have to remove a tail (incomplete block)
    attenuated_levels = attenuated.levels()[1]
    signal_levels = signal.levels()[1][:len(attenuated_levels)]

    obtained_attenuation = signal_levels - attenuated_levels

    assert np.all(obtained_attenuation > 0.0)

    # FIXME: determine correct blocked_distance
    #blocked_distance = np.fromiter(map(np.mean, partition(nblock, iter(distance))), dtype='float64')

    #expected_attenuation = atmosphere.attenuation_coefficient(frequency) * blocked_distance

    #assert ( (obtained_attenuation - expected_attenuation) < 0.1 ).all()


#class TestTurbulence(object):


    #def test_turbulence_fluctuations():

        #soundspeed = 343.
        #scale = 1.1
        #nblock = 8192
        #ntaps = 128
        #mean_mu_squared = 3.0e-6

        #frequency = 1000.
        #wavenumber = 2.*np.pi*frequency / soundspeed

        #spatial_separation = np.ones(samples) * 0.006
        #distance = np.ones(samples) * 1000.

        #logamp, phase = _turbulence_fluctuations()

