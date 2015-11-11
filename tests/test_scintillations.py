import pytest
import numpy as np

from auraliser.scintillations import *

@pytest.fixture(params=[44100, 44101])
def samples(request):
    return request.param

@pytest.fixture
def spatial_separation(samples):
    return np.ones(samples) * 0.006

@pytest.fixture(params=[10.0, 100.0, 1000.0])
def distance(request):
    return request.param

@pytest.fixture
def frequency():
    return 1000.0

@pytest.fixture
def soundspeed():
    return 343.0

@pytest.fixture
def wavenumber(frequency, soundspeed):
    return 2.0*np.pi*frequency / soundspeed

@pytest.fixture
def mean_mu_squared():
    return 3.0e-6

@pytest.fixture
def scale():
    return 1.0

@pytest.fixture(params=[True, False])
def include_saturation(request):
    return request.param

@pytest.fixture(params=[None, np.random.RandomState()])
def state(request):
    return request.param

@pytest.fixture(params=[None, np.hanning])
def window(request, samples):
    win = request.param
    if win is not None:
        return win(samples)
    else:
        return None


def test_covariance_gaussian(spatial_separation, distance, wavenumber, scale, mean_mu_squared):

    covariance_gaussian(spatial_separation, distance, wavenumber, scale, mean_mu_squared)


def test_impulse_response_fluctuations(samples, spatial_separation, distance, wavenumber, scale, mean_mu_squared):
    cov = covariance_gaussian(spatial_separation, distance, wavenumber, scale, mean_mu_squared)
    assert len(impulse_response_fluctuations(cov)) == samples


def test_generate_gaussian_fluctuations(samples, spatial_separation, distance, wavenumber,
                                        mean_mu_squared, scale, include_saturation, state, window):

    logamp, phase = generate_gaussian_fluctuations(samples, spatial_separation,
                                                   distance, wavenumber, mean_mu_squared,
                                                   scale, include_saturation=True,
                                                   state=state, window=None)
    assert len(logamp)==samples
    assert len(phase)==samples

    logamp_variance(np.exp(logamp))
    phase_variance(phase)


def test_generate_fluctuations(samples, spatial_separation, distance, wavenumber,
                               mean_mu_squared, scale, include_saturation, state, window):

    logamp, phase = generate_fluctuations(samples, spatial_separation, distance, wavenumber,
                                          scale, mean_mu_squared=mean_mu_squared,
                                          include_saturation=True,
                                          state=state, window=None)
    assert len(logamp)==samples
    assert len(phase)==samples

    logamp_variance(np.exp(logamp))
    phase_variance(phase)

#def test_variance_fluctuations(samples, spatial_separation, distance, wavenumber,
                               #mean_mu_squared, scale, include_saturation, state, window):

    #var = variance_gaussian(spatial_separation, distance, wavenumber, scale, mean_mu_squared)

    #logamp, phase = generate_gaussian_fluctuations(samples, spatial_separation,
                                                   #distance, wavenumber, mean_mu_squared,
                                                   #scale, include_saturation=False,
                                                   #state=state, window=None)

    #assert var == logamp_variance(np.exp(logamp))
