"""
Tests for :mod:`acoustics.realtime`.

"""
import pytest
from acoustics import Signal
from acoustics.atmosphere import Atmosphere
from acoustics.generator import noise
from auraliser.realtime import *
from itertools import cycle, chain
import streaming
from streaming.stream import Stream, BlockStream
from geometry import Point, Vector


@pytest.fixture
def duration():
    return 15.

@pytest.fixture
def fs():
    return 8000.

@pytest.fixture
def frequency():
    return 1000.0

@pytest.fixture
def nsamples(fs, duration):
    return int(fs*duration)

@pytest.fixture
def times(nsamples, fs):
    return np.arange(nsamples) / fs

@pytest.fixture
def tone_sequence(times, frequency):
    return np.sin(2.*np.pi*frequency*times)

@pytest.fixture
def tone_stream(tone_sequence):
    return Stream(tone_sequence)

@pytest.fixture
def speed_source():
    return streaming.signal.constant(100.)

@pytest.fixture
def velocity_source(speed_source):
    theta = np.random.randn(1)
    x = np.sin(theta) * speed_source.copy()
    y = np.cos(theta) * speed_source.copy()
    z = streaming.signal.constant(0.1)
    return Stream(map(Vector, zip(x,y,z)))

@pytest.fixture
def position_receiver():
    return streaming.stream.constant(Point(0., 0., 0.))

@pytest.fixture
def position_source(velocity_source):
    return Point(-200, 0., 100.) + velocity_source


@pytest.fixture
def distance()

signal = tone_stream


#def test_apply_spherical_spreading(signal):
    #"""Test whether spherical spreading is applied correctly."""

    ## NOTE: This test hangs pytest and is therefore disabled.

    #distance = np.linspace(100, 1000, signal.samples)

    #time = 0.125
    #nblock = np.floor(signal.fs*time)

    ## Determine levels of original signal and signal after spreading
    #signal_levels = signal.levels(time=time)[1]
    #attenuated_levels = Signal(apply_spherical_spreading(Stream(signal), Stream(distance)).toarray(), signal.fs).levels(time=time)[1]
    ## We're interested in the difference, which is a negative value.
    #actual_difference = attenuated_levels - signal_levels

    ## We partition the distance in chunks corresponding to the averaging/integration time
    ## We take the closest point as reference (NOT mean value!).
    #distance = np.reshape(distance[0:signal.samples//nblock*nblock], (-1, nblock)).min(axis=-1)

    ## Sound is expected to decrease with
    #expected_difference = - 20.0*np.log10(distance)

    ## There is a relatively high error for the points nearby
    #assert np.all( np.abs(actual_difference - expected_difference) <  0.5)


def test_apply_atmospheric_attenuation(signal, frequency):

    atmosphere = Atmosphere()

    distance = np.linspace(100., 1000., signal.samples)
    nblock = 8192

    # Apply atmospheric attenuation
    attenuated = apply_atmospheric_attenuation(Stream(signal), signal.fs, Stream(distance), nblock=nblock, atmosphere=atmosphere, ntaps=128)
    # Create instance of Signal
    attenuated = Signal(attenuated.samples().toarray(), signal.fs)

    # To determine sound pressure level as function of time. Note that we have to remove a tail (incomplete block)
    attenuated_levels = attenuated.levels()[1]
    signal_levels = signal.levels()[1][:len(attenuated_levels)]

    obtained_attenuation = signal_levels - attenuated_levels

    assert np.all(obtained_attenuation > 0.0)

    # FIXME: determine correct blocked_distance
    #blocked_distance = np.fromiter(map(np.mean, partition(nblock, iter(distance))), dtype='float64')

    #expected_attenuation = atmosphere.attenuation_coefficient(frequency) * blocked_distance

    #assert ( (obtained_attenuation - expected_attenuation) < 0.1 ).all()


#def test_apply_doppler(tone_stream, ):




class TestTurbulence(object):
    """Test whether :func:`apply_turbulence` works.
    """

    @pytest.fixture
    def duration(self):
        return 3.0

    @pytest.fixture
    def nsamples(self, fs, duration):
        return int(fs*duration)

    @pytest.fixture(params=[44100., 8000.])
    def fs(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def include_saturation(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def include_amplitude(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def include_phase(self, request):
        return request.param

    @pytest.fixture(params=[343., streaming.signal.constant(343.)])
    def soundspeed(self, request):
        return request.param

    @pytest.fixture(params=[1000., streaming.signal.constant(1000.)])
    def distance(self, request):
        return request.param

    @pytest.fixture(params=[100., streaming.signal.constant(100.)])
    def speed(self, request):
        return request.param

    def test_turbulence_fluctuations(self, nsamples, fs, speed, soundspeed, distance, include_saturation, include_amplitude, include_phase):

        nhop = 256
        correlation_length = 1.1
        mean_mu_squared = 3.0e-6
        frequency = 100.
        signal = streaming.signal.sine(frequency, fs).take(nsamples)

        modulated = apply_turbulence(signal, fs, nhop, correlation_length, speed,
                                     distance, soundspeed, mean_mu_squared, ntaps_corr=8192,
                                     ntaps_spectra=128, window=None, include_saturation=include_saturation,
                                     state=None,
                                     include_amplitude=include_amplitude,
                                     include_phase=include_phase)

        modulated = modulated.toarray()
        assert len(modulated) > 0
