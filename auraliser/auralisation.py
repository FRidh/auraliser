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
from functools import partial
from multiprocessing import Pool
from copy import deepcopy
import logging
import ism

import acoustics
from acoustics.atmosphere import Atmosphere
from acoustics.directivity import Omni

# To render the geometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Local
from ._signal import Signal
import auraliser
import auraliser.propagation

#from .pointvector import PointVector
from geometry import Point, Vector
from geometry import PointList as PointVector

from .Base import norm, unit_vector
import itertools

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
        self.settings.update(DEFAULT_SETTINGS)
        if settings:
            self.settings.update(settings)
        
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
        return self._add_object(name, Receiver, position)#*args, **kwargs)
    
    
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
        logging.info("Determining mirrors sources.")
        model = ism.Model(self.geometry.walls, receiver, source, max_order=self.settings['reflections']['order_threshold'])
        yield from model.determine(strongest=self.settings['reflections']['mirrors_threshold'])
    
    
    
    def _auralise_subsource(self, subsource, receiver):
        """Synthesize the signal of a subsource.
        """
        subsource.generate_signals() # Generate the signals.
        
        if self.settings['reflections']['include'] and len(self.geometry.walls) > 0: # Are reflections possible?
            n = self.settings['reflections']['update_resolution']
            subsource_position = [Point(*src) for src in subsource.position[::n]]# subsource.position[::n]
            receiver_position = Point(*receiver.position[0])
            
            #print(subsource_position._data.shape)
            
            mirrors = self._get_mirror_sources_from_ism(subsource_position, receiver_position)
            del subsource_position, receiver_position
            
            mirrors = _prepare_mirrors(mirrors, subsource, self.settings, self.samples, self.sample_frequency, self.atmosphere)
            
            signal = np.zeros(self.samples)

            #with Pool() as pool:
                #for mirror in pool.starmap(_auralise_mirror, mirrors):
            for mirror in itertools.starmap(_auralise_mirror, mirrors):    
                    signal += mirror
            return signal
    
        else: # No walls, so no reflections. Therefore the only source is the real source.
            signal = subsource.signal( unit_vector(receiver.position - subsource.position))
            #mirror = Mirror(receiver.position, None, None, unit_vector(receiver.position.asarray()-subsource.position.asarray()))
            return _auralise_mirror(subsource.position, receiver.position, signal, self.settings, self.samples, self.sample_frequency, self.atmosphere)
            
    def _auralise_source(self, source, receiver):
        """
        Synthesize the signal due to a source. This includes all subsources and respective mirror sources.
        """
        #signals = list()
        #for sub_src in source.subsources:
            #signals.append(self._auralise_subsource(sub_src, receiver))
        #return Signal(np.sum(signals, axis=0), sample_frequency=self.sample_frequency)   
        logging.info("Auralising source {}".format(source.name))
        signal = 0.0
        for subsource in source.subsources:
            signal += self._auralise_subsource(subsource, receiver)
        logging.info("Finished auralising source {}".format(source.name))
        return signal    

    def auralise(self, receiver, sources=None):
        """
        Synthesise the signal due to one or multiple sources. All subsources are included.
        
        :param receiver: Receiver.
        :param sources: Iterable of sources.
        
        """
        receiver = self.get_object(receiver)
        logging.info("Auralising at {}".format(receiver.name))
        self.can_auralise()
        logging.info("Can auralise.")
        
        sources = sources if sources else self.sources
        sources = (self.get_object(source) for source in sources)
        
        signal = 0.0
        for source in sources:
            signal += self._auralise_source(source, receiver)
        #return signal
        return Signal(signal, sample_frequency=self.sample_frequency)
    

def _prepare_mirrors(mirrors, subsource, settings, samples, fs, atmosphere):
    """Prepare mirrors.
    """
    logging.info("Preparing mirrors...")

    n = settings['reflections']['update_resolution'] 
    for mirror in mirrors:
        
        orientation = unit_vector(np.array(position_for_directivity(mirror)) - subsource.position) # Orientation from receiver to source.
        signal = subsource.signal( orientation ) # Signal for given orientation.
        del orientation
        
        print(len(signal))
        print(mirror.effective.shape)
        
        
        if mirror.effective is not None:
            signal *= np.repeat(mirror.effective, n, axis=0)[0:samples]
            del mirror.effective
    
        # Apply correct source strength due to reflections.
        if not settings['reflections']['force_hard']:
            if mirror.strength is not None:
                #signal = acoustics.signal.convolve(signal, auraliser.propagation.ir_real_signal(np.repeat(mirror.strength, n, axis=0), settings['reflections']['taps']).T)[0:samples]
                signal = acoustics.signal.convolve(signal, np.repeat(  (auraliser.propagation.ir_real_signal(mirror.strength)[:,0:settings['reflections']['taps']]), n, axis=0).T[:,0:samples])[0:samples]
                del mirror.strength
        
        #out = (subsource.position, mirror.position, signal, np.repeat(mirror.effective, n, axis=0), np.repeat(mirror.strength, n, axis=0), settings, samples, fs, atmosphere )
        out = (subsource.position, mirror.position, signal, settings, samples, fs, atmosphere )
        del mirror
        yield out



def _auralise_mirror(source, receiver, signal, settings, samples, fs, atmosphere):
    """
    Synthesize the signal due to a single mirror.
    
    :param source: Source position
    :param receiver: Receiver position
    :param signal: Initial signal
    :param mirror_effective: Effectiveness of mirror
    :param mirror_strength: Strength of mirror
    :param settings: Settings dictionary.
    :param samples: Amount of samples.
    :param fs: Sample frequency
    :param atmosphere: Atmosphere
    """
    logging.info("Auralising mirror")
    
    distance = norm(source - receiver)
            
    # Apply Doppler shift.
    if settings['doppler']['include'] and settings['doppler']['frequency']:
        logging.info("Applying Doppler frequency shift.")
        signal = auraliser.propagation.apply_doppler(signal, distance/atmosphere.soundspeed, fs)
    
    # Apply turbulence.
    if settings['turbulence']['include']:
        logging.info("Applying turbulence.")
        
        if settings['turbulence']['spatial_separation']:
            rho, dr = auraliser.propagation._spatial_separation(source, source[0], receiver)
            #rho = norm(source.asarray() - source.asarray()[0])
            turbulence_distance = distance #+ dr
        else:
            rho = np.zeros_like(signal)
            turbulence_distance = distance
        Cv = 0.001  
        signal = auraliser.propagation.apply_turbulence(signal,
                                                        fs,
                                                        settings['turbulence']['mean_mu_squared'],
                                                        turbulence_distance,
                                                        settings['turbulence']['correlation_length'],
                                                        rho,
                                                        Cv,
                                                        atmosphere.soundspeed,
                                                        settings['turbulence']['fraction'],
                                                        settings['turbulence']['order'],
                                                        settings['turbulence']['saturation'],
                                                        settings['turbulence']['amplitude'],
                                                        settings['turbulence']['phase'],
                                                        settings['turbulence']['random_seed'],
                                                        )
        
        #signal = auraliser.propagation.apply_turbulence(signal,
                                                        #fs,
                                                        #settings['turbulence']['mean_mu_squared'],
                                                        #turbulence_distance,
                                                        #settings['turbulence']['correlation_length'],
                                                        #rho,
                                                        #atmosphere.soundspeed,
                                                        #settings['turbulence']['fraction'],
                                                        #settings['turbulence']['order'],
                                                        #settings['turbulence']['saturation'],
                                                        #settings['turbulence']['amplitude'],
                                                        #settings['turbulence']['phase'],
                                                        #settings['turbulence']['random_seed'],
                                                        #)
        del rho                            
        
    # Apply atmospheric absorption.
    if settings['atmospheric_absorption']['include']:
        logging.info("Applying atmospheric absorption.")
        signal = auraliser.propagation.apply_atmospheric_absorption(signal,
                                              fs,
                                              atmosphere,
                                              distance,
                                              taps=settings['atmospheric_absorption']['taps'],
                                              n_d=settings['atmospheric_absorption']['unique_distances']
                                              )[0:samples]

    # Apply spherical spreading.
    if settings['spreading']['include']:
        logging.info("Applying spherical spreading.")
        signal = auraliser.propagation.apply_spherical_spreading(signal, distance)
    
    # Force zeros until first real sample arrives.
    initial_delay = int(distance[0]/atmosphere.soundspeed * fs)
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

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.name)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.name)

#class NamedObject(object):
    #"""Descriptor to access named object.
    #"""
    
class Receiver(Base):
    """
    Receiver
    """
    
    position = PositionDescriptor('position')
    """Position of object.
    """
    
    def __init__(self, auraliser, name, position):
        """
        Constructor.
        """
        
        super().__init__(auraliser, name=name)
        
        self.position = position
    
    def auralise(self, sources=None):
        """
        Auralise the scene at receiver location.
        
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
        self.position_relative = position_relative# if position_relative is not None else np.arrayVector(0.0, 0.0, 0.0)#PointVector()

    _source = None
    
    
    #_position_relative = None

    @property
    def position(self):
        """Absolute position of subsource.
        """
        return self.source.position + self.position_relative

    #@property
    #def position_relative(self):
        #"""Relative position of subsource.
        #"""
        #return self._position_relative
     
    #@position_relative.setter
    #def position_relative(self, x):
        #self._position_relative = x
        
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
        #return np.array([src.signal_using_orientation(orientation) for src in self.virtualsources]).sum(axis=0)
        signal = 0.0
        for src in self.virtualsources:
            signal += src.signal_using_orientation(orientation)
        return signal
        

class Virtualsource(Base):
    """
    Class for modelling specific spectral components within a :class:`Auraliser.SubSource` that have the same directivity.
    """

    def __init__(self, auraliser, name, subsource, rotation=None, directivity=None, gain=0.0, multipole=0):
        """Constructor.
        """
        
        super().__init__(auraliser, name)
        
        self.subsource = subsource

        
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
        """
        
        self.multipole = multipole
        """Multipole order.
        
        Valid values are 0 for a monopole, 1 for a dipole and 2 for a quadrupole.
        """
    
    _subsource = None
    
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
        self._signal_generated = self.signal.output(t, fs)

    def signal_using_orientation(self, orientation):
        """The signal this :class:`VirtualSource` emits as function of orientation.
        
        :param orientation: A vector of cartesian coordinates.
        """
        mach = np.gradient(self.subsource.position)[0] * self._auraliser.sample_frequency / self._auraliser.atmosphere.soundspeed 
        signal = self._signal_generated * self.directivity.using_cartesian(orientation[:,0], orientation[:,1], orientation[:,2]) * db_to_lin(self.gain)
        if self._auraliser.settings['doppler']['include'] and self._auraliser.settings['doppler']['amplitude']: # Apply change in amplitude.
            signal = auraliser.propagation.apply_doppler_amplitude_using_vectors(signal, mach, orientation, self.multipole)
        return signal

class Mirror(object):
    """
    Mirror source container.
    
    This structure contains the information necessary for auralising the mirror.
    """
    
    def __init__(self, position, effective, strength, signal):
        
        self.position = position
        """Actual position of the mirror source.
        """
        self.effective = effective
        """Effectiveness of the source.
        """
        self.strength = strength
        """Strength of the source.
        """
        #self.order = order
        #"""
        #Order
        #"""
        self.signal = signal
        """Initial signal.
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
        raise NotImplementedError()
        


DEFAULT_SETTINGS = {
    
    'reflections':{
        'include'           :   True,   # Include reflections
        'mirrors_threshold' :   2,      # Maximum amount of mirrors to include
        'order_threshold'   :   3,      # Maximum order of reflections
        'update_resolution' :   50,    # Update effectiveness every N samples.
        'taps'              :   512,     # Amount of filter taps for ifft mirror strength.
        'force_hard'        :   False,  # Force hard reflections.
        },
    'doppler':{
        'include'           :   True,   # Include Doppler shift
        'frequency'         :   True,   # Include the frequency shift.
        'amplitude'         :   True,   # Include the change in intensity.
        },
    'spreading':{
        'include'           :   True,   # Include spherical spreading
        },
    'atmospheric_absorption':{
        'include'           :   True,   # Include atmospheric absorption
        'taps'              :   128,     # Amount of filter taps for ifft 
        'unique_distances'  :   100,    # Calculate the atmospheric for N amount unique distances.
        },
    'turbulence':{
        'include'           :   True,   # Include modulations and decorrelation due to atmospheric turbulence.
        'mean_mu_squared'   :   3.0e-6,
        'correlation_length':   12.0,
        'saturation'        :   True,
        'amplitude'         :   True,   # Amplitude modulations
        'phase'             :   True,   # Phase modulations
        'spatial_separation':   False,
        'fraction'          :   1,
        'order'             :   1,
        'random_seed'       :   None,   # By setting this value to an integer, the 'random' values will be similar for each auralisation.
        },
    }
"""
Default settings of :class:`Auraliser'.

All possible settings are included.
"""

def db_to_lin(db):
    """
    Convert gain in dB to linear.
    """
    return 10.0**(db/20.0)

def lin_to_db(lin):
    """
    Convert gain from linear to dB
    """
    return 20.0 * np.log10(lin/10.0)
    
    

def render_geometry(geometry, filename=None):
    """
    Render geometry.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    polygons = Poly3DCollection( [wall.points for wall in geometry.walls] )
    #polygons.set_color(colors.rgb2hex(sp.rand(3)))
    #polygons.tri.set_edgecolor('k')
    
    ax.add_collection3d( polygons )
    #ax.relim() # Does not support Collections!!! So we have to manually set the view limits...
    #ax.autoscale()#_view()

    coordinates = np.array( [wall.points for wall in geometry.walls] ).reshape((-1,3))
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    
    ax.set_xlim(minimum[0], maximum[0])
    ax.set_ylim(minimum[1], maximum[1])
    ax.set_zlim(minimum[2], maximum[2])

    if filename:
        fig.savefig(filename)
    else:
        return fig


    
    
    
    
