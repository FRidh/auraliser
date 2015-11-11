import pytest
import numpy as np
import tempfile
import pickle

import auraliser

from auraliser import Auraliser, Geometry

from auraliser.propagation import apply_atmospheric_absorption

from acoustics import Signal
from acoustics.atmosphere import Atmosphere

# Samples related

@pytest.fixture(params=[12000.0, 22050.0])
def fs(request):
    return request.param

@pytest.fixture(params=[2.0, 10.0])
def duration(request):
    return request.param

@pytest.fixture
def samples(duration, fs):
    return int(np.round(duration * fs))

# Propagation related

#@pytest.fixture
#def model(duration, fs):
    #return Auraliser(duration, sample_frequency=fs)

@pytest.fixture(params=[None])#, Geometry(), Geometry.ground()])
def geometry(request):
    return request.param

@pytest.fixture
def model(duration, fs, geometry):
    return Auraliser(duration, sample_frequency=fs, geometry=geometry)


#@pytest.fixture
#def models(duration, fs):

#@pytest.fixture
#def model_with_ground(duration, fs):
    #return Auraliser(duration, sample_frequency=fs, geometry=Geometry.ground())

#@pytest.fixture
#def model_propagation(model):
    #return model

# Emission related

@pytest.fixture(params=[500.0, 999.95, 4000.0])
def frequency(request):
    return request.param

@pytest.fixture(params=['white', 'pink'])
def color(request):
    return request.param


@pytest.fixture
def model_full():
    pass

#@pytest.fixture(params=possible_settings.items())
#def settings(request):
    #return request.param


#possible_settings = {
    #'default' : get_default_settings(),
    #'propagation_disabled' : recursive_mapping_update(get_default_settings().update({'reflections'}),
    #}

class TestGeometry:

    def test_pickle(self, geometry):

        with tempfile.TemporaryDirectory() as tmpdirname:
            with open('obj.pickle', mode='w+b') as f:
                pickle.dump(geometry, f)

            with open('obj.pickle', mode='r+b') as f:
                geometry2 = pickle.load(f)

            assert geometry2 == geometry


class TestModel:

    def test_remove_objects(self, model):

        model.remove_objects()
        assert len(list(model.objects)) == 0


    def test_pickle(self, model):

        with tempfile.TemporaryDirectory() as tmpdirname:
            with open('obj.pickle', mode='w+b') as f:
                pickle.dump(model, f)

            with open('obj.pickle', mode='r+b') as f:
                model2 = pickle.load(f)

            assert model2 == model

class TestEmission:
    pass


class TestGenerator:

    def test_custom(self, model, fs, duration, samples):

        # When using correct amount of samples
        signal = np.random.randn(samples)
        generator = auraliser.generator.Custom(values=signal)
        assert (generator.output(duration, fs) == signal).all()

        # When using incorrect amount of samples
        with pytest.raises(ValueError):
            signal = np.random.randn(samples+1)
            generator = auraliser.generator.Custom(values=signal)
            assert (generator.output(duration, fs) == signal).all()


    def test_sine(self, model, fs, duration, samples, frequency):

        generator = auraliser.generator.Sine(frequency)
        assert len(generator.output(duration, fs)) == samples

    def test_noise(self, model, fs, duration, samples, color):

        generator = auraliser.generator.Noise(color=color)
        assert len(generator.output(duration, fs)) == samples


    def test_noisebands(self, model, fs, duration, samples, color):
        pass


#class TestPropagation:



    #def test_doppler():
        #pass

    #def test_atmospheric_attenuation():
        #pass

    #def test_turbulence():
        #pass


def test_atmospheric_attenuation(fs, frequency):
    """Test whether the attenuation is correct.
    """

    duration = 5.0
    distance = 100.0
    n_blocks = 256
    
    atmosphere = Atmosphere()
    samples = int(fs*duration)
    t = np.arange(samples) / fs

    signal = Signal( np.sin(2.0*np.pi*frequency*t) , fs)

    out = Signal( apply_atmospheric_absorption(signal, fs, atmosphere, distance*np.ones(samples), n_blocks=512, n_distances=20) , fs)

    attenuation = atmosphere.attenuation_coefficient(frequency) * distance

    signal_difference = signal.leq() - out.leq()
    
    # Assert we have attenuation by testing.
    assert( signal_difference > 0.0 )
    
    assert( np.abs(signal_difference - attenuation) < 1.0)


def test_atmospheric_attenuation_n_distances():
    """Test whether the attenuation is identical when using two different 
    values for `n_distances`.
    """
    
    fs = 4000.0
    frequency = 1000.0
    duration = 1.0
    distance = 100.0
    
    n_blocks = 256
    
    atmosphere = Atmosphere()
    samples = int(fs*duration)
    t = np.arange(samples) / fs

    signal = Signal( np.sin(2.0*np.pi*frequency*t) , fs)

    out1 = Signal( apply_atmospheric_absorption(signal, fs, atmosphere, distance*np.ones(samples), n_blocks=n_blocks, n_distances=20) , fs)
    
    out2 = Signal( apply_atmospheric_absorption(signal, fs, atmosphere, distance*np.ones(samples), n_blocks=n_blocks, n_distances=40) , fs)

    assert(out1.leq()==out2.leq())

    assert(np.all(out1==out2))




def test_atmospheric_attenuation_varying_distance(duration):
    """Test whether a varying distance works correctly.
    
    Note that the error strongly depends on the relation between the sample frequency,
    the signal frequency, duration and the amount of blocks that are used.
    """
    
    fs = 4000.0
    frequency = 1000.0

    duration = 2.0
    
    atmosphere = Atmosphere()
    samples = int(fs*duration)
    t = np.arange(samples) / fs

    distance = np.linspace(100.0, 1000.0, samples)

    signal = Signal( np.sin(2.0*np.pi*frequency*t) , fs)

    out = Signal( apply_atmospheric_absorption(signal, fs, atmosphere, distance, n_blocks=512, n_distances=50) , fs)

    attenuation = atmosphere.attenuation_coefficient(frequency) * distance

    signal_difference = signal.leq() - out.leq()
        
    # Assert we have attenuation by testing.
    assert( signal_difference > 0.0 )
    
    assert( np.all( np.abs(signal_difference - attenuation) < 1.0) )
    

#@pytest.fixture(params=[10.0, 40.0, 160.0])
#def velocity(request):
    #return request.param

#from auraliser.propagation import apply_doppler, apply_delay_turbulence
#from acoustics.doppler import frequency_shift
#from acoustics.signal import instantaneous_frequency

#def test_doppler(fs, frequency, velocity):
    #"""Test Doppler shift
    #"""
    #soundspeed = 343.0
    #duration = 20.0
    
    #samples = int(fs*duration)
    #t = np.arange(samples) / fs
    #x = velocity * t 
    #x -= x.max()/2.0
    
    
    #signal = Signal( np.sin(2.0*np.pi*frequency*t) , fs)
    #delay = x / soundspeed
    
    #out = Signal( apply_delay_turbulence(signal, delay, fs), fs)
    
    #shift = frequency_shift(frequency, velocity*np.sign(x), 0.0, soundspeed=soundspeed)
    
    ##print(shift)

    
    #actual_shift = np.unwrap( instantaneous_frequency(out, out.fs) )
        
    #print(out)    
    #print(shift.shape)
    #print(actual_shift.shape)
    
    #assert(np.all(shift == actual_shift))
    
