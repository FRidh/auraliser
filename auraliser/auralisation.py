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

import weakref
import numpy as np
import logging
import abc
from collections import namedtuple
import itertools
from multiprocessing import Pool

from geometry import Point, Vector
import ism

from acoustics.atmosphere import Atmosphere
from acoustics.directivity import Omni
from acoustics import Signal
from acoustics.signal import convolve

# To render the geometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Local
#from ._signal import Signal
import auraliser
import auraliser.propagation
from .generator import Noise
from .tools import norm, unit_vector

import collections

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
    
    Directivity is given by vector from the first order to the original source after. However, this directivity is mirrored to 
    
    """
    m = mirror
    while m.order > 1:
        m = getattr(m, 'mother')
    return m.position

#class PositionDescriptor(object):
    #"""Descriptor that enforces lists of Points.
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
        
        if isinstance(value, Point) or isinstance(value, Vector):
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


class Auraliser(object):
    """
    An auraliser object contains the model for simulating how :attr:`sources` sound at :attr:`receivers` for a given :attr:`atmosphere`.
    """
    
    def __init__(self, duration, sample_frequency=44100.0, atmosphere=None, geometry=None, settings=None):
        """
        Constructor.
        
        :param duration: Duration of the signal to auralise given in seconds.
        :param sample_frequency: Sample frequency
        :param atmosphere: Atmosphere
        :param geometry: Geometry
        :param settings: Dictionary with configuration.
        
        """
        
        self.settings = dict()
        """
        Configuration of this auraliser.
        """
        recursive_mapping_update(self.settings, _DEFAULT_SETTINGS)
        #self.settings.recursupdate(DEFAULT_SETTINGS)
        if settings:
            recursive_mapping_update(self.settings, settings)
            #self.settings.update(settings)
        
        self.atmosphere = atmosphere if atmosphere else Atmosphere()
        """
        Atmosphere described by :class:`Auraliser.Atmosphere.Atmosphere`.
        """
        
        self._objects = list()
        """
        List of :class:`Auraliser.Auraliser.Source`s.
        """
            
        self.duration = duration
        """
        Duration of auralisation.
        """
        
        self.sample_frequency = sample_frequency
        """Sample frequency.
        """
    
        self.geometry = geometry if geometry else Geometry()
        """
        Geometry of the model described by :class:`Geometry`.
        """
    
    
    def __del__(self):
        del self._objects[:]
    
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
            raise ValueError("Unknown name. Cannot get object.")
        
    def get_object(self, name):
        """Get object by name.
        
        :param name: Name of `object`.
        
        :returns: Proxy to `object`.
        
        """
        name = name if isinstance(name, str) else name.name
        for obj in self._objects:
            if name == obj.name:
                return weakref.proxy(obj) 
        else:
            raise ValueError("Unknown name. Cannot get object.")
    
    def remove_object(self, name):
        """Delete object from model."""
        for obj in self._objects:
            if name == obj.name:
                self._objects.remove(obj)
    
    def remove_objects(self):
        """Delete all objects from model."""
        for obj in self._objects:
            self._objects.remove(obj)
    
    def _add_object(self, name, model, *args, **kwargs):
        """Add object to model."""
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
    
    
    @property
    def time(self):
        """Time vector.
        """
        return np.arange(0.0, float(self.duration), 1.0/self.sample_frequency)

    @property
    def samples(self):
        """Amount of samples.
        """
        return self.duration * self.sample_frequency
    
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
    
    def _get_mirror_sources_from_ism(self, source, receiver):
        """Determine the mirror sources for the given source and receiver.
        
        :param source: Source position
        :param receiver: Receiver position
        
        Use the image source method to determine the mirror sources.
        Return only the amount of mirrors as specified in the settings.
        
        .. note:: Mirror receivers are calculated instead of mirror sources.
        """
        logging.info("_get_mirror_sources_from_ism: Determining mirrors sources.")
        model = ism.Model(self.geometry.walls, receiver, source, max_order=self.settings['reflections']['order_threshold'])
        mirrors = model.determine(strongest=self.settings['reflections']['mirrors_threshold'])
        yield from mirrors
    
    
    def _auralise_subsource(self, subsource, receiver):
        """Synthesize the signal of a subsource.
        """
        
        # Generate the emission signals.
        logging.info("_auralise_subsource: Generating subsource emission signals.")
        subsource.generate_signals()
        
        # Determine mirrors
        logging.info("_auralise_subsource: Determine mirrors.")
        if self.settings['reflections']['include'] and len(self.geometry.walls) > 0: # Are reflections possible?
            logging.info("_auralise_subsource: Searching for mirrors. Reflections are enabled, and we have walls.")
            resolution = self.settings['reflections']['update_resolution']
            subsource_position = [Point(*src) for src in subsource.position[::resolution]]# subsource.position[::n]
            receiver_position = [Point(*receiver.position[0])]
            
            ## Mirror receivers
            #mirrors1, mirrors2 = itertools.tee( self._get_mirror_sources_from_ism(subsource_position, receiver_position) )
            #del subsource_position, receiver_position

            #emissions = (_apply_source_effects(mirror, subsource, self.settings, self.samples, self.sample_frequency, self.atmosphere) for mirror in mirrors1)
            
            ## pre-zip mirrors for propagation effects calculation
            #mirrors = (_Mirror(subsource.position, np.array(mirror.position), emission, self.settings, self.samples, self.sample_frequency, self.atmosphere ) for mirror, emission in zip(mirrors2, emissions))
            #del emissions
            
            mirrors = self._get_mirror_sources_from_ism(subsource_position, receiver_position)
            mirrors = (_Mirror(subsource.position, np.array(mirror.position), _apply_source_effects(mirror, subsource, self.settings, self.samples, self.sample_frequency, self.atmosphere), self.settings, self.samples, self.sample_frequency, self.atmosphere ) for mirror in mirrors)

        else: # No walls, so no reflections. Therefore the only source is the real source.
            logging.info("_auralise_subsource: Not searching for mirror sources. Either reflections are disabled or there are no walls.")
            emission = subsource.signal( unit_vector(receiver.position - subsource.position))
            mirrors = [ _Mirror(np.array(subsource.position), receiver.position, emission, self.settings, self.samples, self.sample_frequency, self.atmosphere) ]
            del emission

        mirrors1, mirrors2 = itertools.tee(mirrors)
        # Apply propagation effects.
        signals = itertools.starmap(_apply_propagation_effects, mirrors1)
        signals = (Signal(signal, self.sample_frequency) for signal in signals) # Convert to signal classes
        
        # Determine orientations for receiver encoding
        orientations = ( unit_vector(mirror.source_position - mirror.receiver_position) for mirror in mirrors2)
        
        yield from zip(signals, orientations)

        
    def _auralise_source(self, source, receiver):
        """Synthesize the signal at `receiver` due to `source`. This includes all subsources and respective mirror sources.
        """
        logging.info("_auralise_source: Auralising source {}".format(source.name))

        for subsource in source.subsources:
            signals_and_orientations = self._auralise_subsource(subsource, receiver)
            yield from signals_and_orientations
            
        logging.info("_auralise_source: Finished auralising source {}".format(source.name))

    def auralise(self, receiver, sources=None):
        """Synthesise the signal due to one or multiple sources at `receiver`. All subsources are included.
        
        :param receiver: Receiver.
        :param sources: Iterable of sources.
        
        """
        receiver = self.get_object(receiver)
        logging.info("auralise: Auralising at {}".format(receiver.name))
        if self.can_auralise():
            logging.info("auralise: Can auralise.")
        
        sources = sources if sources else self.sources
        sources = (self.get_object(source) for source in sources)
        
        signal = 0.0
        for source in sources:
            yield from self._auralise_source(source, receiver)

    
    def plot(self, **kwargs):
        """Plot model.
        
        :seealso: :func:`plot_model`.
        """
        return plot_model(self, **kwargs)


def _apply_source_effects(mirror, subsource, settings, samples, fs, atmosphere):
    """Apply source effects to source.
    
    Includes:
        - Directivity of source
        - Effectiveness and strength of mirror
    
    """
    logging.info("_apply_source_effects: Applying source effects...")

    resolution = settings['reflections']['update_resolution'] 
    
    # Emission signal with directivity.
    orientation = unit_vector(np.array(position_for_directivity(mirror)) - subsource.position) # Orientation from receiver to source.
    signal = subsource.signal( orientation ) # Signal for given orientation.
    del orientation
    
    if mirror.effective is not None:
        signal *= np.repeat(mirror.effective, resolution, axis=0)[0:samples]
        del mirror.effective

    # Apply correct source strength due to reflections.
    if not settings['reflections']['force_hard'] and not np.all(mirror.strength == 1.0): # Save computations, for direct source there is no need.
        
        logging.info("_apply_source_effects: Soft ground.")
        signal = convolve(signal, np.repeat(  (auraliser.propagation.ir_reflection(mirror.strength, settings['reflections']['taps'])), resolution, axis=0)[0:samples].T)[0:samples]
        
        # We cannot yet delete the strength of the object since a child mirror source might need it.
        # Same for the position, as the child needs to know it for the directivity.
        # But, we can at least delete the strength of this mother source. (also not true?!)
        #del mirror.strength 
        #del mirror.mother.strength, mirror.mother.effective, mirror.mother.distance
        #print(signal)
    else:
        logging.info("_apply_source_effects: Hard ground.")
    del mirror
    return signal

    
##def _apply_receiver_effects(source_position, receiver, signal, settings):
    ##"""Apply receiver effects to receiver.
    
    ##:param source: Source
    ##:param receiver: Receiver
    ##:param signal: Signal
    ##:returns: Multi-channel signal adjusted for receiver effects.
    
    ##Includes:
        ##- Directivity of receiver.
    
    ##"""
    
    ### Orientation from source to receiver.    
    ##orientation = unit_vector(source_position - receiver.position) 
    
    ##return signal, orientation
    
    #### Multichannel receiver
    ###signals = np.tile(signal, (len(receiver.channels), 1))
    
    ###resolution = settings['directivity']['update_resolution']
    
    #### Apply directivity correction to channel
    ###for i, channel in enumerate(receiver.channels):
        #### Sample directivity only every `resolution` samples.
        ###d = np.repeat(channel.directivity.using_cartesian(*orientation[::resolution].T), resolution)[0:len(signal)]
        ###signals[i,:] *= d
    ###return signals

def _apply_propagation_effects(source, receiver, signal, settings, samples, fs, atmosphere):
    """Apply the propagation filters for this specific source-mirror-receiver combination.    
    
    :param source: Source position
    :param receiver: Receiver position
    :param signal: Initial signal
    :param settings: Settings dictionary.
    :param samples: Amount of samples.
    :param fs: Sample frequency
    :param atmosphere: Atmosphere
    :returns: Single-channel signal.
    
    Propagation effect included are:
        - Amplitude decrease due to spherical spreading.
        - Delay due to spherical spreading (Doppler shift).
        - Atmospheric turbulence.
        - Atmospheric attenuation.
    """
    logging.info("_apply_propagation_effects: Auralising mirror")

    
    distance = norm(source - receiver)

    # Apply delay due to spreading (Doppler shift)
    if settings['doppler']['include'] and settings['doppler']['frequency']:
        logging.info("_apply_propagation_effects: Applying Doppler frequency shift.")
        signal = auraliser.propagation.apply_doppler(signal, 
                                                     distance/atmosphere.soundspeed, 
                                                     fs, 
                                                     method=settings['doppler']['interpolation'],
                                                     kernelsize=settings['doppler']['kernelsize'],
                                                     )
    
    # Apply atmospheric turbulence.
    if settings['turbulence']['include']:
        logging.info("_apply_propagation_effects: Applying turbulence.")
        
        if settings['turbulence']['spatial_separation']:
            rho, dr = auraliser.propagation._spatial_separation(source, (np.random.randn(3))[None,:], receiver)
            rho, dr = auraliser.propagation._spatial_separation(source, (source[0]+np.random.randn(3))[None,:], receiver)
            #rho, dr = auraliser.propagation._spatial_separation(source, source[0][None,:], receiver[None,:])
            #rho = norm(source.asarray() - source.asarray()[0])
            turbulence_distance = distance #+ dr
            rho = np.linalg.norm( np.gradient(source)[0], axis=1 )
        else:
            rho = np.zeros_like(signal)
            turbulence_distance = distance
        
        ##print(rho)
        ###rho = np.ones_like(rho) * rho.mean()
        #rho = np.abs(rho)
        ##rho = np.ones_like(rho) 
        ##print(np.abs(rho))
        #from .propagation import moving_average
        #n = 1000
        #rho = moving_average(rho, n=n)
        #rho = np.hstack((rho, np.ones(n-1)*rho[-1]))
        #print(rho)
        
        if settings['turbulence']['model'] == 'gaussian':
            signal = auraliser.propagation.apply_turbulence_gaussian(signal,
                                                            fs,
                                                            settings['turbulence']['mean_mu_squared'],
                                                            turbulence_distance,
                                                            settings['turbulence']['correlation_length'],
                                                            rho,
                                                            atmosphere.soundspeed,
                                                            settings['turbulence']['fraction'],
                                                            settings['turbulence']['order'],
                                                            settings['turbulence']['saturation'],
                                                            settings['turbulence']['amplitude'],
                                                            settings['turbulence']['phase'],
                                                            settings['turbulence']['random_seed'],
                                                            )
            del rho                            
        
        elif settings['turbulence']['model'] == 'vonkarman':
            rho = np.ones_like(rho)
            signal = auraliser.propagation.apply_turbulence_vonkarman(signal,
                                                            fs,
                                                            settings['turbulence']['mean_mu_squared'],
                                                            turbulence_distance,
                                                            settings['turbulence']['correlation_length'],
                                                            rho,
                                                            settings['turbulence']['variance_windspeed'],
                                                            atmosphere.soundspeed,
                                                            settings['turbulence']['fraction'],
                                                            settings['turbulence']['order'],
                                                            settings['turbulence']['saturation'],
                                                            settings['turbulence']['amplitude'],
                                                            settings['turbulence']['phase'],
                                                            settings['turbulence']['random_seed'],
                                                            )
        
        else:
            raise ValueError("Turbulence model '{}' does not exist".format(settings['turbulence']['model']))
        
        
    # Apply atmospheric absorption.
    if settings['atmospheric_absorption']['include']:
        logging.info("_apply_propagation_effects: Applying atmospheric absorption.")
        signal = auraliser.propagation.apply_atmospheric_absorption(signal,
                                              fs,
                                              atmosphere,
                                              distance,
                                              n_blocks=settings['atmospheric_absorption']['taps'],
                                              n_distances=settings['atmospheric_absorption']['unique_distances']
                                              )[0:samples]

    # Apply spherical spreading.
    if settings['spreading']['include']:
        logging.info("_apply_propagation_effects: Applying spherical spreading.")
        signal = auraliser.propagation.apply_spherical_spreading(signal, distance)

    # Force zeros until first real sample arrives. Should only be done when the time delay (Doppler) is applied.
    if settings['doppler']['include'] and settings['doppler']['frequency']:
        initial_delay = int(distance[0]/atmosphere.soundspeed * fs)
        if settings['doppler']['purge_zeros']:
            signal = signal[initial_delay:]
        else:
            signal[0:initial_delay] = 0.0
            
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


#class Receiver(Base):
    #"""Receiver
    #"""
    
    #position = PositionDescriptor('position')
    #"""Position of object.
    #"""
    
    #def __init__(self, auraliser, name, position):
        #"""
        #Constructor.
        #"""
        
        #super().__init__(auraliser, name=name)
        
        #self.position = position
    
    
    #def auralise(self, sources=None):
        #"""
        #Auralise the scene at receiver location.
        
        #:param sources: List of sources to include. By default all the sources in the scene are included.
        
        #"""
        #return self._auraliser.auralise(self, sources)


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
    
#class Receiver(Base):
    #"""Receiver
    #"""
    
    #position = PositionDescriptor('position')
    #"""Position of object.
    #"""
    
    
    #def __init__(self, auraliser, name, position, channels=None):
        #"""
        #Constructor.
        #"""
        
        #super().__init__(auraliser, name=name)
        
        #self.position = position
    
        #self.channels = [Channel(directivity=Omni())]
    
    ##@channels.setter
    ##def channels(self, x):
        ##self._channels = x
    
    ##@property
    ##def channels(self):
        ##"""Channels.
        ##"""
        ##return self._channels
    
    #def auralise(self, sources=None):
        #"""
        #Auralise the scene at receiver location.
        
        #:param sources: List of sources to include. By default all the sources in the scene are included.
        
        #"""
        #return self._auraliser.auralise(self, sources)


class Channel(object):
    """Channel for receiver.
    
    """
    
    def __init__(self, directivity=None):
        
        self.directivity = directivity if directivity else Omni()

    def __repr__(self):
        return "Channel({}".format(str(self))
    
    def __str__(self):
        return "({})".format(str(self.directivity))

#class Mono(AbstractReceiver):
    #"""Mono signal receiver.
    #"""
    #pass

#class Stereo(AbstractReceiver):
    #"""Stereo signal receiver.
    #"""
    #pass

#class Custom(AbstractReceiver):
    #pass


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
        self.position_relative = position_relative# if position_relative is not None else np.arrayVector(0.0, 0.0, 0.0)#PointVector()

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
        signal = 0.0
        for src in self.virtualsources:
            signal += src.emission(orientation)
        return signal
        

class Virtualsource(Base):
    """
    Class for modelling specific spectral components within a :class:`Auraliser.SubSource` that have the same directivity.
    """

    def __init__(self, auraliser, name, subsource, signal, rotation=None, directivity=None, gain=0.0, multipole=0):
        """Constructor.
        """
        
        super().__init__(auraliser, name)
        
        self.subsource = subsource
        self.signal = signal
        
        self.directivity = directivity if directivity else Omni()
        """Directivity of the signal.
        """
        #self.modulation = modulation
        """
        Amplitude modulation of the signal in Hz.
        """
        self.gain = gain
        """Gain of the signal in decibel.
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
        self._signal_generated = Signal(self.signal.output(t, fs), fs).gain(self.gain)

    def emission(self, orientation):
        """The signal this :class:`Virtualsource` emits as function of orientation.
        
        :param orientation: A vector of cartesian coordinates.
        """
        signal = self._signal_generated * self.directivity.using_cartesian(orientation[:,0], orientation[:,1], orientation[:,2])
        if self._auraliser.settings['doppler']['include'] and self._auraliser.settings['doppler']['amplitude']: # Apply change in amplitude.
            mach = np.gradient(self.subsource.position)[0] * self._auraliser.sample_frequency / self._auraliser.atmosphere.soundspeed 
            signal = auraliser.propagation.apply_doppler_amplitude_using_vectors(signal, mach, orientation, self.multipole)
        return signal


_Mirror = namedtuple('Mirror', ['source_position', 'receiver_position', 
                               'emission', 'settings', 'samples', 
                               'sample_frequency', 'atmosphere'])
"""Mirror container.
"""

class Geometry(object):
    """
    Class describing the geometry of the model.
    """
    
    def __init__(self, walls=None):
        
        
        self.walls = walls if walls else list()
        """List of walls or faces.
        """
        
    def render(self):
        """Render the geometry.
        """
        return render_geometry(self)
        


def get_default_settings():
    d = dict()
    d = recursive_mapping_update(d, _DEFAULT_SETTINGS)
    return d 
    

_DEFAULT_SETTINGS = {
    
    'reflections':{
        'include'           :   True,   # Include reflections
        'mirrors_threshold' :   2,      # Maximum amount of mirrors to include
        'order_threshold'   :   3,      # Maximum order of reflections
        'update_resolution' :   100,    # Update effectiveness every N samples.
        'taps'              :   256,     # Amount of filter taps for ifft mirror strength.
        'force_hard'        :   False,  # Force hard reflections.
        },
    'doppler':{
        'include'           :   True,   # Include Doppler shift
        'frequency'         :   True,   # Include the frequency shift.
        'amplitude'         :   True,   # Include the change in intensity.
        'purge_zeros'       :   False,  # Purge the (initial) zeros due to the delay in arrival.
        'interpolation'     :   'lanczos',   # Lanczos interpolation
        'kernelsize'        :   10,
        },
    'spreading':{
        'include'           :   True,   # Include spherical spreading
        },
    'atmospheric_absorption':{
        'include'           :   True,   # Include atmospheric absorption
        'taps'              :   256,    # Amount of filter blocks to use for ifft
        'unique_distances'  :   100,    # Calculate the atmospheric for N amount unique distances.
        },
    'turbulence':{
        'include'           :   False,  # Include modulations and decorrelation due to atmospheric turbulence.
        'mean_mu_squared'   :   3.0e-6,
        'correlation_length':   1.0,
        'variance_windspeed':   0.03,
        'saturation'        :   False,  # Include log-amplitude saturation
        'amplitude'         :   True,   # Amplitude modulations
        'phase'             :   True,   # Phase modulations
        'spatial_separation':   False,
        'fraction'          :   1,      # Fraction of filter
        'order'             :   8,      # Order of filter
        'random_seed'       :   100,   # By setting this value to an integer, the 'random' values will be similar for each auralisation.
        'model'             :   'gaussian',
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
            }
        
        },
    'directivity': {
        'update_resolution' :   100,    # Sample directivity every N samples. Needed for spherical harmonics.
        },
        
        
        
        #'receivers' : {
            #'mono'  :   [
                #('M', 'Omni', ()),
                #],
            #'stereo':   [
                #('L', 'Cardioid', ()),
                #('
                #],
                       
                    
                    #],
                #}
            #'stereo' :
                
                
            
            
            #}
    
    }
"""
Default settings of :class:`Auraliser'.

All possible settings are included.
"""

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
    
