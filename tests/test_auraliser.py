import auraliser
import copy
import logging
import numpy as np
import pickle
import pytest
import tempfile

from auraliser.auralisation import *
from auraliser.generator import *
from acoustics.directivity import *
from auraliser.sinks import *
from auraliser import Auraliser, Geometry, get_default_settings
from acoustics import Signal
from geometry import Point


#logging.basicConfig(level=logging.DEBUG)

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

@pytest.fixture
def times(samples, fs):
    return np.arange(samples) / fs

# Propagation related

#@pytest.fixture
#def model(duration, fs):
    #return Auraliser(duration, sample_frequency=fs)

@pytest.fixture(params=[None])#, Geometry(), Geometry.ground()])
def geometry(request):
    return request.param

@pytest.fixture
def source_position(duration, times):
    velocity = 60.0                         # Velocity of source in meters per second.
    distance = velocity * duration          # Total distance covered
    x = np.ones_like(times) * velocity * (times - duration/2.0)    # Source moves along the x-axis.
    y = np.ones_like(times) * 1.0
    z = np.ones_like(times) * 100.0   # Altitude of source
    return np.vstack((x, y, z)).T

@pytest.fixture
def receiver_position():
    return Point(0.0,0.0,1.6)

@pytest.fixture
def full_model(duration, fs, geometry, source_position, receiver_position):
    model = Auraliser(duration, geometry=geometry)
    model.settings['fs'] = fs

    rcv = model.add_receiver(name='receiver', position=receiver_position)

    src = model.add_source(name='source0', position=source_position)
    subsrc = src.add_subsource('subsource0')
    sine0 = subsrc.add_virtualsource('sine0', signal=Sine(1000.0), directivity=Omni())

    return model

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


possible_settings = {
    'default'                   : get_default_settings(),
    'reflections_soft'          : {'reflections': {'force_hard' : False}},
    'reflections_hard'          : {'reflections': {'force_hard' : True}},
    'doppler_linear'            : {'doppler'    : {'interpolation': 'linear'}},
    'doppler_lanczos'           : {'doppler'    : {'interpolation':  'lanczos'}},
    'turbulence_gaussian'       : {'turbulence' : {'covariance' : 'gaussian'}},
    'turbulence_vonkarman'      : {'turbulence' : {'covariance' : 'vonkarman'}},
    }

@pytest.fixture(params=possible_settings.values())
def settings(request):
    return request.param


class _TestBase(object):

    def deepcopy(self, item):
        assert copy.deepcopy(item) == item

    def test_equality(self, item):
        assert item == item

    # Fails with Model. no idea yet why
    #def test_pickle(self, item):

        #with tempfile.TemporaryDirectory() as tmpdirname:
            #with open('obj.pickle', mode='w+b') as f:
                #pickle.dump(item, f)

            #with open('obj.pickle', mode='r+b') as f:
                #item2 = pickle.load(f)

            #assert item2 == item


class TestReceiver(_TestBase):

    @pytest.fixture
    def item(self, receiver_position):
        return Receiver(None, 'rcv', receiver_position)

class TestSource(_TestBase):

    @pytest.fixture
    def item(self, source_position):
        return Source(None, 'src', source_position)

#class TestSubsource(_TestBase):

    #@pytest.fixture
    #def item(self, source_position):
        #src = Source(None, 'src', source_position)
        #return Subsource(None, 'sub', src, 0.0)

class TestGeometry(_TestBase):

    @pytest.fixture
    def item(self, geometry):
        return geometry

class TestModel(_TestBase):

    # NOTE
    # Pickling seems to fail when model has a source and/or receiver.
    # Geometry pickles fine.

    @pytest.fixture
    def item(self, model):
        return model

    @pytest.fixture(params=[True, False])
    def include_receiver(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def include_source(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def include_geometry(self, request):
        return request.param

    @pytest.fixture
    def model(self, duration, fs, geometry, source_position, include_receiver, include_source, include_geometry):

        model = Auraliser(duration, geometry=geometry)
        model.settings['fs'] = fs
        if include_receiver:
            rcv = model.add_receiver(name='receiver', position=Point(0.0,0.0,1.6))

        if include_source:
            src = model.add_source(name='source0', position=source_position)
            subsrc = src.add_subsource('subsource0')
            sine0 = subsrc.add_virtualsource('sine0', signal=Sine(1000.0), directivity=Omni())

        if include_geometry:
            model.geometry = geometry

        return model


    def test_remove_object(self, model):
        objects = list(model.objects)
        for obj in objects:
            model.remove_object(obj.name)
        assert len(model._objects) == 0
        assert len(list(model.objects)) == 0


    def test_remove_objects(self, model):
        model.remove_objects()
        assert len(model._objects) == 0
        assert len(list(model.objects)) == 0


class TestEmission(object):
    pass


class TestGenerator(object):

    def test_custom(self, fs, duration, samples):

        # When using correct amount of samples
        signal = np.random.randn(samples)
        generator = auraliser.generator.Custom(values=signal)
        assert (generator.output(duration, fs) == signal).all()

        # When using incorrect amount of samples
        with pytest.raises(ValueError):
            signal = np.random.randn(samples+1)
            generator = auraliser.generator.Custom(values=signal)
            assert (generator.output(duration, fs) == signal).all()


    def test_sine(self, fs, duration, samples, frequency):

        generator = auraliser.generator.Sine(frequency)
        assert len(generator.output(duration, fs)) == samples

    def test_noise(self, fs, duration, samples, color):

        generator = auraliser.generator.Noise(color=color)
        assert len(generator.output(duration, fs)) == samples


    def test_noisebands(self, fs, duration, samples, color):
        pass



def test_full_model(full_model, settings):
    model = full_model

    rcv = list(model.receivers)[0]

    model.update_settings(settings)
    fs = model.sample_frequency

    signal = Signal(mono(rcv.auralise()).toarray(), fs)

    nblock = model.settings['nblock']

    assert signal.samples == (model.samples // nblock * nblock)


#class TestPropagation(object):



    #def test_doppler():
        #pass

    #def test_atmospheric_attenuation():
        #pass

    #def test_turbulence():
        #pass


    

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
    
