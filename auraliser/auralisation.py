"""
The auralisation module provides tools to perform an auralisation.

The :class:`Auraliser` is a powerful interface for performing auralisations.
It allows one to define a soundscape consisting of several sources and
receivers and performing auralisations at the receivers due to the specified sources.

The `class:`Auralisar` supports several propagation effects:

-  **geometrical spreading** resulting in a decrease in sound pressure with increase in source-receiver distance
-  **Doppler shift** due to relative motion between the moving aircraft and the non-moving receiver
-  **atmospheric absorption** due to relaxation processes
-  **reflections** at the ground and facades due to a sudden change in impedance
-  **modulations and decorrelation** due to fluctuations caused by atmospheric turbulence

"""
import abc
import acoustics
import auraliser
import collections
import geometry
import ism
import itertools
import math
import numpy as np
import weakref

from numtraits import NumericalTrait
from acoustics import Signal
from acoustics.signal import convolve

# To render the geometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from multipledispatch import dispatch

import streaming
from streaming.stream import Stream, BlockStream, repeat_each
from streaming.signal import constant
import cytoolz
import copy

import auraliser.tools
logger = auraliser.tools.create_logger(__name__)

@dispatch(object, object)
def unequality(a, b):
    if a is None and b is None:
        return False
    elif a is None or b is None:
        return True

    elif a.__class__ != b.__class__:
        return True
    else:
        return unequality(a.__dict__, b.__dict__)

#@dispatch(object, None)
#def unequality(a, b):
    #return True

#@dispatch(None, None)
#def unequality(a, b):
    #return False

@dispatch(int, int)
def unequality(a, b):
    return a!=b

@dispatch(float, float)
def unequality(a, b):
    return a!=b

@dispatch(str, str)
def unequality(a, b):
    return a!=b

@dispatch(object, np.ndarray)
def unequality(a, b):
    return True

@dispatch(np.ndarray, np.ndarray)
def unequality(a, b):
    return not np.allclose(a, b) # We use all(), not any()

@dispatch(dict, dict)
def unequality(a, b):
    try:
        return a!=b
    except ValueError:
        if set(a.keys()) != set(b.keys()):
            return True
        else:
            for key, value in a.items():
                if unequality(b[key], value):
                    return True
            else:
                return False

def equality(a, b):
    return not unequality(a, b)


#def _equalityity_with_arrays(a, b):
    #"""Test equalityity between two objects.
    #"""
    #if self.__class__ == other.__class__:
        #for key, value in self.__dict__:
            #if key not in other:
                #return False
            #equality = other[key] != value
            ##try:
            #unequality = not equality
            #print(equality)
            ##except ValueError:
                ##print(equality)
                ##unequality = not np.all(equality)
            #if unequality:
                #return False
        #else:
            #return True
    #return False

def recursive_mapping_update(d, u):
    """Recursively update a mapping/dict.

    :param d: Target mapping.
    :param u: Source mapping.

    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = recursive_mapping_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def position_for_directivity(mirror):
    """
    Get the position to be used for the directivity.

    Directivity is given by vector from the first order to the original source after.
    However, this directivity is mirrored to

    """
    m = mirror
    while m.order > 1:
        m = getattr(m, 'mother')
    return m.position

#class PositionDescriptor(object):
    #"""Descriptor that enforces lists of geometry.Points.
    #"""

    #def __init__(self):
        #pass

    #def __get__(self, instance, owner):
        ## we get here when someone calls x.d, and d is a NonNegative instance
        ## instance = x
        ## owner = type(x)
        #return self.data.get(instance, self.default)

    #def __set__(self, instance, value):
        ## we get here when someone calls x.d = val, and d is a NonNegative instance
        ## instance = x
        ## value = val
        #if value < 0:
            #raise ValueError("Negative value not allowed: %s" % value)
        #self.data[instance] = value

class PositionDescriptor(object):
    """Descriptor that enforces positions.
    """

    def __init__(self, attr):
        self.attr = attr
        self.default = (0.0, 0.0, 0.0)

    def __get__(self, instance, owner):
        pos = instance.__dict__.get(self.attr, self.default)
        pos = np.asarray(pos)
        if pos.shape==(1,3):
            pos = np.tile(pos, (instance._auraliser.samples, 1))
        #pos = pos[instance._auraliser.samples, :]
        return pos

    def __set__(self, instance, value):

        if value is None:
            value = self.default

        if isinstance(value, geometry.Point):# or isinstance(value, geometry.Vector):
            instance.__dict__[self.attr] = np.array(value)[None,:]
        elif isinstance(value, tuple):
            if len(value)==3:
                instance.__dict__[self.attr] = np.array(value)[None,:]
            else:
                raise ValueError("Tuple of wrong size.")
        elif isinstance(value, np.ndarray):
            if value.ndim!=2:
                raise ValueError("Array has wrong amount of dimensions.")
            elif value.shape[-1]!= 3:
                raise ValueError("Array should have three columns.")
            else:
                instance.__dict__[self.attr] = np.asarray(value)
        else:
            raise ValueError("Cannot set value, invalid type '{}'.".format(type(value)))

class Auraliser(object):
    """
    An auraliser object contains the model for simulating how :attr:`sources` sound at :attr:`receivers` for a given :attr:`atmosphere`.
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._objects = list()
        """List with all objects in model.
        """
        return obj

    def __init__(self, duration, atmosphere=None, geometry=None, settings=None):
        """
        Constructor.

        :param duration: Duration of the signal to auralise given in seconds.
        :param sample_frequency: Sample frequency
        :param atmosphere: Atmosphere
        :param geometry: Geometry
        :param settings: Dictionary with configuration.

        """
        self.settings = get_default_settings()
        """
        Configuration of this auraliser.
        """
        if settings:
            recursive_mapping_update(self.settings, settings)

        self.atmosphere = atmosphere if atmosphere else acoustics.atmosphere.Atmosphere()
        """
        Atmosphere described by :class:`Auraliser.Atmosphere.Atmosphere`.
        """

        self.duration = duration
        """
        Duration of auralisation.
        """

        #self.sample_frequency = sample_frequency
        #"""Sample frequency.
        #"""

        self.geometry = geometry if geometry else Geometry()
        """
        Geometry of the model described by :class:`Geometry`.
        """

    def __eq__(self, other):
        return equality(self, other)
        #return (self.__dict__ == other.__dict__) and (self.__class__ == other.__class__)

    def __del__(self):
        del self._objects[:]

    @property
    def sample_frequency(self):
        return self.settings['fs']

    def _get_real_object(self, name):
        """Get real object by name.

        :param name: Name of `object`.

        :returns: Real `object`.
        """
        name = name if isinstance(name, str) else name.name
        for obj in self._objects:
            if name == obj.name:
                return obj
        else:
            raise ValueError("Cannot retrieve object. Unknown name {}. ".format(name))

    def get_object(self, name):
        """Get object by name.

        :param name: Name of `object`.

        :returns: Proxy to `object`.

        """
        return weakref.proxy(self._get_real_object(name))

    def remove_object(self, name):
        """Delete object from model."""
        name = name if isinstance(name, str) else name.name
        for obj in self._objects:
            if name == obj.name:
                log.debug('Removing object with name "{}"'.format(name))
                self._objects.remove(obj)

    def remove_objects(self):
        """Delete all objects from model."""
        log.debug('Removing all objects from model.')
        del self._objects[:]

    def _add_object(self, name, model, *args, **kwargs):
        """Add object to model."""
        log.debug('Adding object with name "{}" to model.'.format(name))
        obj = model(weakref.proxy(self), name, *args, **kwargs)   # Add hidden hard reference
        self._objects.append(obj)
        return self.get_object(obj.name)


    def add_source(self, name, position):#)*args, **kwargs):
        """
        Add source to auraliser.
        """
        return self._add_object(name, Source, position)#*args, **kwargs)

    def add_receiver(self, name, position):#*args, **kwargs):
        """
        Add receiver to auraliser.
        """
        return self._add_object(name, Receiver, position=position)#*args, **kwargs)

    def update_settings(self, settings):
        """Recursively update :attr:`settings` with `settings` using :func:`recursive_mapping_update`.

        :param settings: New settings to use.

        .. note:: If you want to assign a dictionary with new settings, replacing all old settings, you should just reassign :meth:`settings`.
        """
        self.settings = recursive_mapping_update(self.settings, settings)

    @property
    def time(self):
        """Time vector.
        """
        return np.arange(0.0, samples)/ fs

    @property
    def samples(self):
        """Amount of samples.
        """
        return int(np.round(self.duration * self.sample_frequency))

    @property
    def objects(self):
        """Objects.
        """
        yield from (self.get_object(obj.name) for obj in self._objects)

    @property
    def sources(self):
        """Sources.
        """
        yield from (obj for obj in self.objects if isinstance(obj, Source) )

    @property
    def receivers(self):
        """Receivers.
        """
        yield from (obj for obj in self.objects if isinstance(obj, Receiver) )

    @property
    def subsources(self):
        """Subsources.
        """
        yield from (obj for obj in self.objects if isinstance(obj, Subsource) )

    @property
    def virtualsources(self):
        """Virtual sources.
        """
        yield from (obj for obj in self.objects if isinstance(obj, Virtualsource) )

    def can_auralise(self):
        """Test whether all sufficient information is available to perform an auralisation.
        """

        if not self.sources:
            raise ValueError('No sources available')

        if not self.receivers:
            raise ValueError('No receivers available')

        if not self.atmosphere:
            raise ValueError('No atmosphere available.')

        if not self.geometry:
            raise ValueError('No geometry available.')

        return True

    #def _get_mirror_sources_from_ism(self, source, receiver):
        #"""Determine the mirror sources for the given source and receiver.

        #:param source: Source position
        #:param receiver: Receiver position

        #Use the image source method to determine the mirror sources.
        #Return only the amount of mirrors as specified in the settings.

        #.. note:: Mirror receivers are calculated instead of mirror sources.
        #"""
        #log.info("_get_mirror_sources_from_ism: Determining mirrors sources.")
        #model = ism.Model(self.geometry.walls, source=receiver, receiver=source, max_order=self.settings['reflections']['order_threshold'])
        #mirrors = model.determine(strongest=self.settings['reflections']['mirrors_threshold'])
        #yield from mirrors

    @staticmethod
    def _auralise_subsource(subsource, receiver, settings, geometry, atmosphere):
        """Synthesize the signal of a subsource.

        We check whether reflections are included or not.
        """

        # Generate the emission signals.
        log.info("_auralise_subsource: Generating subsource emission signals.")
        subsource.generate_signals()

        nblock = settings['nblock']

        # Make sure the positions are streams
        subsource_position = Stream(iter(subsource.position)).blocks(nblock)
        receiver_position = Stream(iter(receiver.position)).blocks(nblock)

        # Determine mirrors
        log.info("_auralise_subsource: Determine mirrors.")
        if settings['reflections']['include'] and len(geometry.walls) > 0: # Are reflections possible?
            log.info("_auralise_subsource: Searching for mirrors. Reflections are enabled, and we have walls.")
            #resolution = settings['reflections']['update_resolution']

            mirrors = _ism_mirrors(subsource_position, receiver_position, subsource.signal, geometry.walls, settings)

        else: # No walls, so no reflections. Therefore the only source is the real source.
            log.info("_auralise_subsource: Not searching for mirror sources. Either reflections are disabled or there are no walls.")
            #emission = subsource.signal( unit_vector(receiver.position - subsource.position)) # Use non-Stream here for now...
            emission = subsource.signal(unit_vector_stream(receiver_position.copy() - subsource_position.copy()))
            # Final sources
            mirrors = [ Mirror(subsource_position, receiver_position, emission) ]

        # Yield contribution of each mirror source.
        for mirror in mirrors:
            #print(mirror.receiver_position.peek())
            signal = _apply_propagation_effects(mirror.source_position.copy(), mirror.receiver_position.copy(), mirror.emission, settings, settings['fs'], atmosphere)
            # Orientation is the unit vector from source to receiver
            orientation = unit_vector_stream(mirror.source_position.copy() - mirror.receiver_position.copy())
            yield signal, orientation


    @staticmethod
    def _auralise_source(source, receiver, settings, geometry, atmosphere):
        """Synthesize the signal at `receiver` due to `source`. This includes all subsources and respective mirror sources.
        """
        log.info("_auralise_source: Auralising source {}".format(source.name))

        for subsource in source.subsources:
            signals_and_orientations = Auraliser._auralise_subsource(subsource, receiver, settings, geometry, atmosphere)
            yield from signals_and_orientations

        log.info("_auralise_source: Finished auralising source {}".format(source.name))

    def auralise(self, receiver, sources=None):
        """Synthesise the signal due to one or multiple sources at `receiver`. All subsources are included.

        :param receiver: Receiver.
        :param sources: Iterable of sources.

        """
        receiver = self.get_object(receiver)
        log.info("auralise: Auralising at {}".format(receiver.name))
        if self.can_auralise():
            log.info("auralise: Can auralise.")

        sources = sources if sources else self.sources
        sources = (self.get_object(source) for source in sources)

        # We don't want to be able to update the settings during an auralization
        settings = copy.deepcopy(self.settings)
        geometry = copy.deepcopy(self.geometry)
        atmosphere = copy.deepcopy(self.atmosphere)

        for source in sources:
            yield from Auraliser._auralise_source(source, receiver, settings, geometry, atmosphere)


    def plot(self, **kwargs):
        """Plot model.

        :seealso: :func:`plot_model`.
        """
        return plot_model(self, **kwargs)


def unit_vector_stream(stream):
    """Calculate unit vector for each element in stream.
    :type stream: BlockStream
    """
    return stream.map(lambda x: x / np.linalg.norm(x, axis=-1)[...,None])


##def _apply_mirror_source_strength(emission, effective, strength, force_hard, ntaps):
    ##"""Apply mirror source strength"""
    ##if effective is not None:
        ##emission = emission * effective
    ###if force_hard or np.all(strength == 1.0):
    ##if force_hard:
        ##log.info("_apply_source_effects: Hard ground.")
    ##else:
        ### Apply mirror source strength. Only necessary when we have a soft surface.
        ##log.info("_apply_source_effects: Soft ground.")
        ###impulse_responses = map(auraliser.propagation.impulse_response, strength)
        ##impulse_responses = strength.map(auraliser.propagation.impulse_response)
        ##emission = convolve(emission, impulse_responses, nblock)
        ###emission = emission.samples().drop(int(ntaps//2))

    ##return emission


def _ism_get_mirror_sources(source, receiver, walls, order_threshold, mirrors_threshold):
    """Determine the mirror sources for the given source and receiver.

    :param source: Source position
    :param receiver: Receiver position

    Use the image source method to determine the mirror sources.
    Return only the amount of mirrors as specified in the settings.

    .. note:: Mirror receivers are calculated instead of mirror sources.
    """
    log.info("_ism_get_mirror_sources: Determining mirror sources.")
    model = ism.Model(walls=walls, source=source, receiver=receiver, max_order=order_threshold)
    mirrors = model.determine(strongest=mirrors_threshold)
    yield from mirrors

def _ism_mirrors(subsource_position, receiver_position, emission, walls, settings):
    """Determine mirror sources and their emissions.

    :param subsource_position: Position of the subsource.
    :type subsource_position: :class:`Stream`
    :param receiver_position: Position of the subsource.
    :type receiver_position: :class:`Stream`
    :param emission: Emission generator.
    :param settings: Settings.
    :type settings: :func:`dict`
    :returns: Yields mirror sources with emission signals.
    """
    log.info("_ism_mirrors: Determining mirror sources and their emissions.")
    resolution = settings['reflections']['nhop']
    nblock = settings['nblock']


    subsource_position_resolution = subsource_position.copy().blocks(resolution)
    subsource_position = subsource_position.blocks(nblock)

    # Obtain mirrors from Image Source Method.
    # Determine mirror sources every `n` samples. We pick the first sample of each block.
    _subsource_position = list(subsource_position_resolution.copy().map(lambda x: geometry.Point(*x[0])))
    _receiver_position = [ geometry.Point(*receiver_position.copy().samples().peek()) ]

    mirrors = _ism_get_mirror_sources(source=_receiver_position,
                                      receiver=_subsource_position,
                                      walls=walls,
                                      order_threshold=settings['reflections']['order_threshold'],
                                      mirrors_threshold=settings['reflections']['mirrors_threshold'])

    for mirror in mirrors:
        # To determine the directivity correction we need to know the orientation from mirror receiver to source.
        # What matters however is not the position of the mirror, but the position of the first order mirror.
        # Since we consider a non-moving receiver, the position is a constant. See also above, _receiver_position.
        receiver_position_directivity = position_for_directivity(mirror)
        # Emission given a vector pointing from receiver to source.
        orientation_directivity = unit_vector_stream(receiver_position_directivity - subsource_position.copy())
        signal = emission(orientation_directivity).blocks(resolution)

        # Apply correct directivity.
        #effective = BlockStream(mirror.effective, resolution)
        #strength = BlockStream(mirror.strength, resolution)

        # Single values as function of time indicating whether the source was effective or not.
        effective = Stream(iter(mirror.effective)).repeat_each(resolution).blocks(resolution)
        # Spectra as function of time
        strength = Stream(iter(mirror.strength)).repeat_each(resolution).blocks(resolution)

        # Signal after applying mirror source strength and effectiveness
        signal = auraliser.realtime.apply_reflection_strength(signal, nblock, strength, effective, settings['reflections']['ntaps'], settings['reflections']['force_hard'])
        #signal = _apply_mirror_source_strength(signal, effective, strength, settings['reflections']['force_hard'], settings['reflections']['ntaps'])

        mirror_receiver_position = constant(mirror.position)
        yield Mirror(subsource_position.copy().blocks(nblock), mirror_receiver_position.blocks(nblock), signal.blocks(nblock))



def _apply_propagation_effects(source, receiver, signal, settings, fs, atmosphere):
    """Apply the propagation filters for this specific source-mirror-receiver combination.

    :param source: Source position
    :type source: :class:`Stream` of :class:`Point`
    :param receiver: Receiver position
    :type receiver: :class:`Stream` of :class:`Point`
    :param signal: Initial signal
    :type signal: :class:`Stream` or :class:`BlockStream`
    :param settings: Settings
    :type settings: :func:`dict`
    :param fs: Sample frequency
    :param atmosphere: Atmosphere
    :type atmosphere: :class:`acoustics.atmosphere.Atmosphere`
    :returns: Single-channel signal.
    :rtype: :class:`Stream` or :class:`BlockStream`

    Propagation effects included are:
        - Amplitude decrease due to spherical spreading (:func:`auraliser.realtime.apply_spreading`).
        - Delay due to spherical spreading (Doppler shift).
        - Atmospheric turbulence.
        - Atmospheric attenuation.
    """
    log.info("_apply_propagation_effects: Auralising mirror")

    nblock = settings['nblock']

    source = source.blocks(nblock)
    receiver = receiver.blocks(nblock)

    # Distance vector pointing from receiver to source.
    distance_vector = source.copy() - receiver.copy()

    # Norm of distance vector pointing from receiver to source.
    distance = distance_vector.copy().map(lambda x: np.linalg.norm(x, axis=-1))

    # Source velocity vector
    #velocity = diff(source.copy()).blocks(nblock) * fs # streaming.signal.diff works only on scalars!
    velocity = source.copy().map(lambda x: np.diff(x, axis=0)) * fs

    # Source speed
    speed = velocity.copy().map(lambda x: np.linalg.norm(x, axis=-1))

    # Unit vector pointing from receiver to source.
    #orientation_sr = distance_vector.copy() / distance.copy()

    #print("Distance: {}".format(distance))
    #print("Source:{}".format(source.peek()))
    #print("Receiver: {}".format(receiver.peek()))


    # Apply spherical spreading.
    if settings['spreading']['include']:
        log.info("_apply_propagation_effects: Applying spherical spreading.")
        signal = auraliser.realtime.apply_spherical_spreading(signal.blocks(nblock), distance.copy().blocks(nblock))

    # Apply delay due to spreading (Doppler shift)
    if settings['doppler']['include'] and settings['doppler']['frequency']:
        log.info("_apply_propagation_effects: Applying Doppler frequency shift.")
        signal = auraliser.realtime.apply_doppler(signal=signal,
                                                  delay=distance.copy()/atmosphere.soundspeed,
                                                  fs=fs)

    # Apply atmospheric turbulence
    if settings['turbulence']['include']:
        log.info("_apply_propagation_effects: Applying turbulence.")

        # We first need to compute the transverse speed
        # We also only want to have a single value per block.
        # Therefore we pick the first values.
        nhop = settings['turbulence']['nhop']
        _distance = Stream(distance.copy().blocks(nhop).map(cytoolz.first))
        _velocity = Stream(velocity.copy().blocks(nhop).map(cytoolz.first))
        _orientation = unit_vector_stream(Stream(distance_vector.copy().blocks(nhop).map(cytoolz.first)))
        #_orientation = Stream(_orientation_sr.copy().blocks(nhop).map(lambda x: x[0]))
        _transverse_speed = Stream(iter(map(auraliser.realtime.transverse_speed, _velocity, _orientation)))
        #print(next(_distance))
        #print(next(_velocity))
        #print(next(_orientation))
        #print(next(_transverse_speed))

        state = np.random.RandomState(seed=settings['turbulence']['seed'])

        signal = auraliser.realtime.apply_turbulence(signal=signal,
                                                     fs=fs,
                                                     nhop=nhop,
                                                     correlation_length=settings['turbulence']['correlation_length'],
                                                     speed=_transverse_speed.copy(),
                                                     distance=_distance.copy(),
                                                     soundspeed=atmosphere.soundspeed,
                                                     mean_mu_squared=settings['turbulence']['mean_mu_squared'],
                                                     fmin=settings['turbulence']['fs_minimum'],
                                                     ntaps_corr=settings['turbulence']['ntaps_corr'],
                                                     ntaps_spectra=settings['turbulence']['ntaps_spectra'],
                                                     window=settings['turbulence']['window'],
                                                     include_saturation=settings['turbulence']['saturation'],
                                                     state=state,
                                                     include_amplitude=settings['turbulence']['amplitude'],
                                                     include_phase=settings['turbulence']['phase'],
                                                     )

        del _distance, _velocity, _orientation, _transverse_speed, state

    # Apply atmospheric absorption.
    if settings['atmospheric_absorption']['include']:
        log.info("_apply_propagation_effects: Applying atmospheric absorption.")

        signal = auraliser.realtime.apply_atmospheric_attenuation(
            signal=signal,
            fs=fs,
            distance=distance.copy(),
            nhop=nblock,
            atmosphere=atmosphere,
            ntaps=settings['atmospheric_absorption']['ntaps'],
            sign=-1,
            dtype='float64',
            )

    # Force zeros until first real sample arrives. Should only be done when the time delay (Doppler) is applied.
    # But force to zero why...? The responsible function should make it zero.
    if settings['doppler']['include'] and settings['doppler']['frequency'] and settings['doppler']['purge_zeros']:
        initial_distance = distance.copy().samples().peek()
        # Initial delay in samples
        initial_delay = int(math.ceil(initial_distance/atmosphere.soundspeed * fs))
        signal = signal.samples().drop(initial_delay)
        del initial_distance, initial_delay

    # Clear memory to prevent memory leaks
    del source, receiver, distance_vector, distance, velocity, speed


    return signal


class Base(object):
    """
    Base class
    """

    def __init__(self, auraliser, name, description=None):
        """
        Constructor of base class.
        """

        self._auraliser = auraliser
        """Auraliser.
        """

        self.name = name
        """Name of this object.
        """

        self.description = description
        """Description of object.
        """

    def __del__(self):
        del self._auraliser


    def __str__(self):
        return "({})".format(self.name)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, str(self))

    def __eq__(self, other):
        return equality(self, other)

    @property
    @abc.abstractmethod
    def position(self):
        """Position
        """

    def plot_position(self, interval=1):
        """Plot position.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*self.position[::interval].T)
        ax.set_xlabel(r'$x$ in m')
        ax.set_ylabel(r'$y$ in m')
        ax.set_zlabel(r'$z$ in m')
        return fig


class Receiver(Base):
    """Receiver
    """

    position = PositionDescriptor('position')
    """Position of object.
    """

    def __init__(self, auraliser, name, position):
        """Constructor."""

        super().__init__(auraliser, name=name)
        self.position = position

    def auralise(self, sources=None):
        """Auralise the scene at receiver location.

        :param sources: List of sources to include. By default all the sources in the scene are included.

        """
        return self._auraliser.auralise(self, sources)


class Source(Base):
    """
    Class for modelling a source.

    """

    position = PositionDescriptor('position')
    """Position of object.
    """

    def __init__(self, auraliser, name, position):
        """Constructor.
        """

        super().__init__(auraliser, name)#, description=description)

        self.position = position

    @property
    def subsources(self):
        """Subsources.
        """
        yield from (obj for obj in self._auraliser.subsources if obj.source.name==self.name)

    def get_subsource(self, name):
        """Get object by name.
        """
        return self._auraliser.get_object(name)

    def add_subsource(self, name):
        """Add subsource to auraliser.
        """
        return self._auraliser._add_object(name, Subsource, self.name)#, **kwargs)

    def remove_subsource(self, name):
        """Remove subsource.
        """
        self._auraliser.remove_object(name)



class Subsource(Base):
    """
    Class for modelling a subsource. A subsource is a component of source having a different position.
    """

    position_relative = PositionDescriptor('position_relative')
    """Position of object.
    """

    def __init__(self, auraliser, name, source, position_relative=None):
        """Constructor.
        """
        super().__init__(auraliser, name)

        self.source = source
        self.position_relative = position_relative# if position_relative is not None else np.arraygeometry.Vector(0.0, 0.0, 0.0)#geometry.Pointgeometry.Vector()

    _source = None


    #_position_relative = None

    @property
    def position(self):
        """Absolute position of subsource.
        """
        return self.source.position + self.position_relative

    @property
    def source(self):
        """Source.
        """
        return self._auraliser.get_object(self._source)

    @source.setter
    def source(self, x):
        x = self._auraliser.get_object(x)
        self._source = x.name

    @property
    def virtualsources(self):
        """Generator returning the virtual sources.
        """
        yield from (obj for obj in self._auraliser.virtualsources if obj.subsource.name==self.name)

    def generate_signals(self):
        """Generate the signal.
        """
        for src in self.virtualsources:
            src.generate_signal()

    def get_virtualsource(self, name):
        """Get object by name.
        """
        return self._auraliser.get_object(name)

    def add_virtualsource(self, name, **kwargs):
        """Add virtualsource to auraliser.
        """
        return self._auraliser._add_object(name, Virtualsource, self.name, **kwargs)
        #obj = Virtualsource(self, name=name)
        #self._virtualsources.append(obj)
        #return self.get_virtualsource(obj.name)


    def remove_virtualsource(self, name):
        """Remove a virtual source from this source.
        """
        self._auraliser.remove_object(name)

    def signal(self, orientation):
        """Return the signal of this subsource as function of SubSource - Receiver orientation.

        :param orientation: Orientation.

        The signal is the sum of all VirtualSources corrected for directivity.
        """
        #return np.array([src.emission(orientation) for src in self.virtualsources]).sum(axis=0)
        #signal = 0.0
        #for src in self.virtualsources:
            #signal += src.emission(orientation)
        #return signal
        #print(orientation)
        return sum((src.emission(orientation.copy()) for src in self.virtualsources))


class Virtualsource(Base):
    """
    Class for modelling specific spectral components within a :class:`Auraliser.SubSource` that have the same directivity.
    """

    def __init__(self, auraliser, name, subsource, signal, rotation=None, directivity=None, level=94.0, multipole=0):
        """Constructor.
        """

        super().__init__(auraliser, name)

        self.subsource = subsource
        self.signal = signal

        self.directivity = directivity if directivity else acoustics.directivity.Omni()
        """Directivity of the signal.
        """
        #self.modulation = modulation
        """
        Amplitude modulation of the signal in Hz.
        """
        self.level = level
        """Level of the signal in decibel.
        """

        self._signal_generated = None
        """Generated signal

        This value is generated and stored at the beginning of an auralization.

        Includes gain, but not yet directivity!
        """

        self.multipole = multipole
        """Multipole order.

        Valid values are 0 for a monopole, 1 for a dipole and 2 for a quadrupole.
        """

    _subsource = None


    @property
    def position(self):
        """Position.
        """
        return self._subsource.position

    @property
    def subsource(self):
        """Subsource.
        """
        return self._auraliser.get_object(self._subsource)

    @subsource.setter
    def subsource(self, x):
        x = self._auraliser.get_object(x)
        self._subsource = x.name

    def generate_signal(self):
        t = self.subsource.source._auraliser.duration
        fs = self.subsource.source._auraliser.sample_frequency
        self._signal_generated = Signal(self.signal.output(t, fs), fs).calibrate_to(self.level)

    def emission(self, orientation):
        """The signal this :class:`Virtualsource` emits as function of orientation.

        :param orientation: A vector of cartesian coordinates.
        """
        #signal = self._signal_generated * self.directivity.using_cartesian(orientation[:,0], orientation[:,1], orientation[:,2])
        #if self._auraliser.settings['doppler']['include'] and self._auraliser.settings['doppler']['amplitude']: # Apply change in amplitude.
            #mach = np.gradient(self.subsource.position)[0] * self._auraliser.sample_frequency / self._auraliser.atmosphere.soundspeed
            #signal = auraliser.propagation.apply_doppler_amplitude_using_vectors(signal, mach, orientation, self.multipole)
        #return signal
        settings = self._auraliser.settings
        nblock = settings['nblock']
        signal = Stream(self._signal_generated).blocks(nblock=nblock)
        orientation = orientation.blocks(nblock)
        directivity = orientation.copy().map(lambda x: self.directivity.using_cartesian(*(x.T)).T)
        # Signal corrected with directivity
        signal = signal * directivity
        if settings['doppler']['include'] and settings['doppler']['amplitude']:
            mach = np.gradient(self.subsource.position)[0] * self._auraliser.sample_frequency / self._auraliser.atmosphere.soundspeed
            mach = Stream(mach).blocks(nblock)
            signal = BlockStream((auraliser.propagation.apply_doppler_amplitude_using_vectors(s, m, o, self.multipole) for s, m, o in zip(signal, mach, orientation)), nblock)
        return signal

#_Mirror = collections.namedtuple('Mirror', ['source_position', 'receiver_position',
                                            #'emission', 'settings', 'samples',
                                            #'sample_frequency', 'atmosphere'])

Mirror = collections.namedtuple('Mirror', ['source_position', 'receiver_position', 'emission'])
"""Mirror container.
"""


def get_default_settings():
    d = dict()
    d = recursive_mapping_update(d, _DEFAULT_SETTINGS)
    return d

_DEFAULT_SETTINGS = {

    'nblock'                :   8192,   # Default blocksize
    'fs'                    :   44100,  # Default sample frequency

    'reflections':{
        'include'           :   True,   # Include reflections
        'mirrors_threshold' :   2,      # Maximum amount of mirrors to include
        'order_threshold'   :   3,      # Maximum order of reflections
        'nhop'              :   4096,   # Update effectiveness every N samples.
        'ntaps'             :   4096,   # Amount of filter taps for ifft mirror strength.
        'force_hard'        :   True,   # Force hard reflections.
        },
    'doppler':{
        'include'           :   True,   # Include Doppler shift
        'frequency'         :   True,   # Include the frequency shift.
        'amplitude'         :   True,   # Include the change in intensity.
        'purge_zeros'       :   False,  # Purge the (initial) zeros due to the delay in arrival.
        'interpolation'     :   'linear',   # Lanczos interpolation
        'kernelsize'        :   10,
        },
    'spreading':{
        'include'           :   True,   # Include spherical spreading
        },
    'atmospheric_absorption':{
        'include'           :   True,   # Include atmospheric absorption
        'ntaps'              :   4096,    # Amount of filter taps to use for ifft
        #'unique_distances'  :   100,    # Calculate the atmospheric for N amount unique distances.
        },
    'turbulence':{
        'include'           :   False,  # Include modulations and decorrelation due to atmospheric turbulence.
        'mean_mu_squared'   :   3.0e-7,
        'correlation_length':   20.0,
        'fs_minimum'        :   100.,    # Sample frequency at which to compute the fluctuations.
        'saturation'        :   True,  # Include log-amplitude saturation
        'amplitude'         :   True,   # Amplitude modulations
        'phase'             :   True,   # Phase modulations
        'seed'              :   100,   # By setting this value to an integer, the 'random' values will be similar for each auralisation.
        #'state'             :   np.random.RandomState(), # Use same state for all propagation paths
        'ntaps_corr'        :   8192,
        'ntaps_spectra'     :   512,
        'nhop'              :   128,
        'window'            :   None,
        },
    'plot':{
        'general':{
            'linewidth'     :   2.0,
            'markersize'    :   10.0,
            'marker'        :   '+',
            'interval'      :   1000,
            },
        'sources':{
            'include'       :   True,
            'color'         :   'r',
            },
        'subsources':{
            'include'       :   False,
            'color'         :   'y',
            },
        'receivers':{
            'include'       :   True,
            'color'         :   'b',
            },
        'subreceivers':{
            'include'       :   False,
            'color'         :   'g',
            },
        'walls':{
            'include'       :   True,
            'alpha'         :   0.5,    # Transparancy of walls.
            'normal'        :   True,   # Show normal vectors.
            'color'         :   'b',
            },
        },
    }
"""
Default settings of :class:`Auraliser'.

All possible settings are included.
"""


class Geometry(object):
    """
    Class describing the geometry of the model.
    """

    def __init__(self, walls=None):


        self.walls = walls if walls else list()
        """List of walls or faces.
        """

    def __eq__(self, other):
        return equality(self, other)
        #return (self.__dict__ == other.__dict__) and (self.__class__ == other.__class__)

    def render(self):
        """Render the geometry.
        """
        return render_geometry(self)

    @classmethod
    def ground(cls, nbins=get_default_settings()['reflections']['ntaps']):
        corners = [geometry.Point(-1e5, -1e5, 0.0),
                   geometry.Point(+1e5, -1e5, 0.0),
                   geometry.Point(+1e5, +1e5, 0.0),
                   geometry.Point(-1e5, +1e5, 0.0)
                   ]
        center = geometry.Point(0.0, 0.0, 0.0)
        impedance = np.zeros(nbins, dtype='complex128')
        g = ism.Wall(corners, center, impedance)
        return cls(walls=[g])

#def render_geometry(geometry):
    #"""
    #Render geometry.
    #"""

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #_render_geometry(geometry, ax)
    #return fig


def _render_geometry(geometry, ax):

    polygons = Poly3DCollection( [wall.points for wall in geometry.walls] )
    #polygons.set_color(colors.rgb2hex(sp.rand(3)))
    #polygons.tri.set_edgecolor('k')

    ax.add_collection3d( polygons )
    #ax.relim() # Does not support Collections!!! So we have to manually set the view limits...
    #ax.autoscale()#_view()

    #coordinates = np.array( [wall.points for wall in geometry.walls] ).reshape((-1,3))
    #minimum = coordinates.min(axis=0)
    #maximum = coordinates.max(axis=0)

    #ax.set_xlim(minimum[0], maximum[0])
    #ax.set_ylim(minimum[1], maximum[1])
    #ax.set_zlim(minimum[2], maximum[2])

    return ax


def plot_model(model, **kwargs):
    """Plot model geometry.
    """
    settings = model.settings['plot']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Transducers
    transducers = ['sources', 'subsources', 'receivers']#, 'subreceivers']

    def _get_plot_items(obj, attr):
        return ({'name':item.name, 'position':item.position} for item in getattr(obj, attr))

    data = {sort: _get_plot_items(model, sort) for sort in transducers}

    for sort in transducers:
        if settings[sort]['include']:
            for item in data[sort]:
                ax.plot(*item['position'][::settings['general']['interval']].T,
                        label=item['name'],
                        color=settings[sort]['color'],
                        marker=settings['general']['marker'],
                        markersize=settings['general']['markersize']
                        )

    # Walls
    ax = _render_geometry(model.geometry, ax)

    ax.set_title('Overview')
    ax.set_xlabel(r'$x$ in m')
    ax.set_ylabel(r'$y$ in m')
    ax.set_zlabel(r'$z$ in m')

    #ax.relim()

    ## Sources
    #for src in model.sources:
        #if sources:
            #ax.plot(*src.position[::interval].T, label=src.name, color='r', marker='+', markersize=markersize, linewidth=linewidth)
        #if subsources:
            #for subsrc in src.subsources:
                #ax.plot(*subsrc.position[::interval].T, label=subsrc.name, color='y', markersize=markersize, linewidth=linewidth)

    ## Receivers
    #for rcv in model.receivers:
        #if receivers:
            #ax.plot(*rcv.position[::interval].T, label=rcv.name, color='r', markersize=markersize, linewidth=linewidth)
        #if subreceivers:
            #for subsrc in rcv.subreceivers:
                #ax.plot(*subrcv.position[::interval].T, label=subrcv.name, color='y', markersize=markersize, linewidth=linewidth)

    # Walls


    return fig

