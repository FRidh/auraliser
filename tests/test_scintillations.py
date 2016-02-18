###import pytest
###import numpy as np

###from auraliser.scintillations import *

###@pytest.fixture(params=[44100, 44101])
###def nsamples(request):
    ###return request.param

###@pytest.fixture
###def ntaps():
    ###return 128

###@pytest.fixture
###def fs():
    ###return 44100

###@pytest.fixture
###def speed():
    ###return 3.0

###@pytest.fixture(params=[10.0, 1000.0])
###def distance(request):
    ###return request.param

###@pytest.fixture
###def frequency():
    ###return 1000.0

###@pytest.fixture
###def soundspeed():
    ###return 343.0

###@pytest.fixture
###def wavenumber(frequency, soundspeed):
    ###return 2.0*np.pi*frequency / soundspeed

###@pytest.fixture
###def mean_mu_squared():
    ###return 3.0e-6

###@pytest.fixture
###def scale():
    ###return 1.0

###@pytest.fixture(params=[True, False])
###def include_saturation(request):
    ###return request.param

####@pytest.fixture(params=[None, np.random.RandomState()])
####def state(request):
    ####return request.param

####@pytest.fixture(params=[None, np.hanning])
####def window(request, nsamples):
    ####win = request.param
    ####if win is not None:
        ####return win(nsamples)
    ####else:
        ####return None

###@pytest.fixture
###def state():
    ###return np.random.RandomState()

###@pytest.fixture
###def window():
    ###return None

####def test_covariance_gaussian(spatial_separation, distance, wavenumber, scale, mean_mu_squared):

    ####covariance_gaussian(spatial_separation, distance, wavenumber, scale, mean_mu_squared)


####def test_impulse_response_fluctuations(nsamples, spatial_separation, distance, wavenumber, scale, mean_mu_squared):
    ####cov = covariance_gaussian(spatial_separation, distance, wavenumber, scale, mean_mu_squared)
    ####assert len(impulse_response_fluctuations(cov)) == nsamples


###def test_generate_gaussian_fluctuations(nsamples, ntaps, fs, speed, distance, frequency, soundspeed,
                                        ###mean_mu_squared, scale, include_saturation, state, window):
    ###"""Test function."""

    ###logamp, phase = generate_gaussian_fluctuations(nsamples, ntaps, fs, speed,
                                                   ###distance, frequency, soundspeed, mean_mu_squared,
                                                   ###scale, include_saturation=True,
                                                   ###state=state, window=None)
    ###assert len(logamp)==nsamples
    ###assert len(phase)==nsamples

    ###logamp_variance(np.exp(logamp))
    ###phase_variance(phase)


###def test_generate_fluctuations(nsamples, ntaps, fs, speed, distance, frequency, soundspeed,
                               ###mean_mu_squared, scale, include_saturation, state, window):
    ###"""Test function."""

    ###logamp, phase = generate_fluctuations(nsamples, ntaps, fs, speed, distance, frequency, soundspeed,
                                          ###scale, mean_mu_squared=mean_mu_squared,
                                          ###include_saturation=True,
                                          ###state=state, window=None)
    ###assert len(logamp)==nsamples
    ###assert len(phase)==nsamples

    ###logamp_variance(np.exp(logamp))
    ###phase_variance(phase)


###from functools import partial

###class TestVariances(object):

    ###@pytest.fixture
    ###def fs(self):
        ###return 16000

    ###@pytest.fixture
    ###def duration(self):
        ###return 120.

    ###@pytest.fixture
    ###def nsamples(self, fs, duration):
        ###return int(fs*duration)

    ###@pytest.fixture(params=[10., 10000.])
    ###def frequency(self, request):
        ###return request.param

    ###@pytest.fixture(params=[10., 10000.])
    ###def distance(self, request):
        ###return request.param

    ###def test_variance(self, nsamples, ntaps, fs, speed, distance, frequency, soundspeed,
                      ###mean_mu_squared, scale, include_saturation, state, window):

        ###logamp, phase = generate_gaussian_fluctuations(nsamples, ntaps, fs, speed,
                                                    ###distance, frequency, soundspeed, mean_mu_squared,
                                                    ###scale, include_saturation=True,
                                                    ###state=state, window=None)
        #### Obtained variances
        ###obtained_logamp_var = logamp_variance(np.exp(logamp))
        ###obtained_phase_var = phase_variance(phase)

        #### Expected variances
        ###wavenumber =  2.0*np.pi*frequency / soundspeed
        ###expected_var = variance_gaussian(distance, wavenumber, scale, mean_mu_squared)
        ###expected_logamp_var = expected_var
        ###expected_phase_var = expected_var
        ###if include_saturation:
            ###print(expected_logamp_var)
            ###expected_logamp_var = expected_logamp_var * saturation_factor(distance, wavenumber, scale, mean_mu_squared)

        ###assert np.allclose(obtained_logamp_var, expected_logamp_var)
        ###assert np.allclose(obtained_phase_var, expected_logamp_var)

####class TestVariances(object):

    ####@pytest.fixture
    ####def fs(self):
        ####return 16000

    ####@pytest.fixture
    ####def duration(self):
        ####return 2.0

    ####@pytest.fixture
    ####def samples(self):
        ####return int(fs*duration)

    ####@pytest.fixture
    ####def npoints(self):
        ####"""Amount of unique measurements points/samples."""
        ####return 100

    ####@pytest.fixture
    ####def arguments(self, nsamples, ntaps, fs, speed, distance, frequency, soundspeed,
                  ####mean_mu_squared, scale, include_saturation, window):
        ####args = {'nsamples'         : nsamples,
                ####'ntaps'            : ntaps,
                ####'fs'               : fs,
                ####'speed'            : speed,
                ####'distance'         : distance,
                ####'frequency'        : frequency,
                ####'soundspeed'       : soundspeed,
                ####'mean_mu_squared'  : mean_mu_squared,
                ####'scale'            : scale,
                ####'include_saturation':include_saturation,
                ####'window'           : window,
                ####}
        ####return args

    ####def test_variances_function_of_distance(self, npoints, arguments):

        ##### Remove the variable we would like to investigate
        ####del arguments['distance']
        ####fluctuations_func = partial(generate_fluctuations, **arguments)
        ####print(fluctuations_func)

        ####variable = np.logspace(1, 4, npoints)

        ####logamps, phases = zip(*list(map(fluctuations_func, variable)))

        ####logamp_variances = map(logamp_variance, logamps)
        ####phase_variances = map(phase_variance, phases)

        ####np.array(list(logamp_variances))
pass
