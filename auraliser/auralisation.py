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

import geometry
import ism

import acoustics
from acoustics.atmosphere import Atmosphere
from acoustics.directivity import Omni

import auraliser.signal
import turbulence_jens

import model as signal_model

import weakref
import numpy as np
import scipy.signal

# To render the geometry
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


try:
    from pyfftw.interfaces.numpy_fft import ifft       # Performs much better than numpy's fftpack
except ImportError:
    from numpy.fft import ifft

###def pointlist_of_point(point, n):
    ###"""
    ###Return PointList ``n`` ``point``s.
    ###"""
    ###return geometry.PointList(np.vstack((point.x*n, point.y*n, point.z*n)))
   

def ir_real_signal(tf, N):
    """
    Take a single-sided spectrum `tf` and convert it to an impulse response of `N` samples.
    
    :param tf: Single-sided spectrum.
    :param N: Amount of samples to use for the impulse response.
    
    .. note:: This function should work for multiple tf's in one array.
    
    """
    #tf = np.hstack( (tf, np.conj(tf[::-1]) ))
    tf = np.hstack(( tf, np.conj(np.fliplr(tf))) )

    #ir = ifft(tf, n=N)
    #ir = np.hstack((ir[N/2:N], ir[0:N/2]))
    ir = np.fft.ifftshift(np.fft.ifft(tf, n=N))
    
    return ir.real
    


def position_for_directivity(mirror):
    """
    Get the position to be used for the directivity.
    
    Directivity is given by vector from the first order to the original source after. However, this directivity is mirrored to 
    
    """
    m = mirror
    while m.order > 1:
        m = getattr(m, 'mother')
    """Now mirror versus plane of reflection."""

    return m.position

##class PositionDescriptor(object):

    ##"""A descriptor that forbids negative values"""
    ##def __init__(self):
        ##pass
        
    ##def __get__(self, instance, owner):
        ### we get here when someone calls x.d, and d is a NonNegative instance
        ### instance = x
        ### owner = type(x)
        ##return self.data.get(instance, self.default)
    
    ##def __set__(self, instance, value):
        ### we get here when someone calls x.d = val, and d is a NonNegative instance
        ### instance = x
        ### value = val
        ##if value < 0:
            ##raise ValueError("Negative value not allowed: %s" % value)
        ##self.data[instance] = value

class Auraliser(object):
    """
    An auraliser object contains the model for simulating how :attr:`sources` sound at :attr:`receivers` for a given :attr:`atmosphere`.
    """
    
    def __init__(self, duration, sample_frequency=44100.0, atmosphere=None, geometry=None, settings=None, **kwargs):
        """
        Constructor.
        
        :param duration: Duration of the signal to auralise given in seconds.
        :param sample_frequency: Sample frequency
        :param atmosphere: Atmosphere
        :param geometry: Geometry
        :param settings: Dictionary with configuration.
        
        """
        
        self.settings = DEFAULT_SETTINGS
        """
        Configuration of this auraliser.
        """
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
        self.sample_frequency = sample_frequency
    
    
        #self._mirror_sources = dict()
        """
        Dictionary containing lists of mirror sources.
        """
    
        self.geometry = geometry if geometry else Geometry()
        """
        Geometry of the model described by :class:`Geometry`.
        """
    
    
    def getObject(self, name):
        """Get object by name."""
        for obj in self.objects:
            if name == obj.name:
                return obj
        else:
            return None
    
    def removeObject(self, name):
        """Delete object from model."""
        for obj in self._objects:
            if name == obj.name:
                self._objects.remove(obj)
    
    def _addObject(self, name, model, **properties):
        """Add object to model."""
        #if name not in self._objects:
        obj = model(weakref.proxy(self), name, **properties)   # Add hidden hard reference
        self._objects.append(obj)
        return self.getObject(obj.name)
    
    
    def addSource(self, name, **properties):
        """
        Add source to auraliser.
        """
        return self._addObject(name, Source, **properties)
    
    def addReceiver(self, name, **properties):
        """
        Add receiver to auraliser.
        """
        return self._addObject(name, Receiver, **properties)
    
    
    @property
    def time(self):
        return np.arange(0.0, float(self.duration), 1.0/self.sample_frequency)

    @property
    def samples(self):
        return self.duration * self.sample_frequency
    
    @property
    def objects(self):
        return [weakref.proxy(obj) for obj in self._objects]
    
    @property
    def sources(self):
        return [weakref.proxy(obj) for obj in self._objects if isinstance(obj, Source) ]
    
    @property
    def receivers(self):
        return [weakref.proxy(obj) for obj in self._objects if isinstance(obj, Receiver) ] 


    def can_auralise(self):
        """
        Test whether all sufficient information is available to perform an auralisation.
        """
        
        if not self.sources:
            raise ValueError('No sources available')
        
        if not self.receivers:
            raise ValueError('No receivers available')
        
        if not self.atmosphere:
            raise ValueError('No atmosphere available.')
        
        if not self.geometry:
            raise ValueError('No geometry available.')
        
    
    def _obtain_mirror_sources(self, source, receiver):
        """
        Determine the mirror sources for the given source and receiver.
        
        :param source: Source position
        :param receiver: Receiver position
        
        Use the image source method to determine the mirror sources.
        Return only the amount of mirrors as specified in the settings.
        """
        
        """Create an image source model for this source and receiver. Note that we use an inverse model!"""
        model = ism.Model(self.geometry.walls, 
                          receiver, 
                          source, 
                          max_order=self.settings['reflections']['order_threshold'])
        
        model.determine_mirrors()
        model.determine_effectiveness()
        mirrors = model.max(N=self.settings['reflections']['mirrors_threshold'], effective=True)
        
        return mirrors
    
    def _auralise_mirror(self, mirror, source_position, signal):
        """
        Synthesize the signal due to a single mirror.
        
        :param mirror: :class:`Mirror` describing the real or mirror receiver.
        :param source_position: Position of source.
        :param signal: Signal.
        """
        n = self.samples
        
        
        """Represent all positions as 2D arrays."""
        
        if isinstance(source_position, geometry.Point):
            src_pos = np.ones((n, 3)) * source_position
        elif isinstance(source_position, geometry.PointList):
            src_pos = source_position.as_array()
        else:
            raise TypeError
            
        rcv_pos = np.ones((n, 3)) * mirror.position
        dir_pos = np.ones((n, 3)) * mirror.position_directivity
                      
        src = signal_model.Source(position=rcv_pos, position_directivity=dir_pos)               
        rcv = signal_model.Receiver(position=src_pos) 
        
        geo = signal_model.Geometry(source=src, receiver=rcv)
        model = signal_model.Model(atmosphere=self.atmosphere, geometry=geo)
        
        
        """Create signal for the specific orientation (directivity)."""
        signal = auraliser.signal.Signal(signal( model.directivity_vector ), 
                                         sample_frequency=self.sample_frequency)
        
        assert not np.isnan(signal).any()
        
        #print mirror.position
        
        """Turn source on and off depending on whether there is an obstacle in between."""
        if mirror.effective is not None:
            signal *= mirror.effective
        
        """Apply correct source strength due to reflections."""
        if not self.settings['reflections']['force_hard']:
            if mirror.strength is not None:
                ir = ir_real_signal(mirror.strength, self.settings['reflections']['taps']).T
                #print ir.shape
                #print signal.shape
                signal = auraliser.signal.Signal(acoustics.signal.convolve(signal, ir)[0:self.samples],
                                                 signal.sample_frequency)
        
        
        """Apply Doppler shift."""
        if self.settings['doppler']['include']:
            signal = model.apply_doppler(signal)


        #print type(signal)
        """Apply turbulence"""
        if self.settings['turbulence']['include']:
            #print type(signal)
            SPACING = np.zeros_like(signal)
            DISTANCE = 2000.0
            DISTANCE = model.geometry.distance
            MODULATION_FREQ = 200.0# self.sample_frequency/10.0
            turbulence = turbulence_jens.Turbulence(SPACING, 
                                                    self.atmosphere.soundspeed, 
                                                    self.sample_frequency/2.0,
                                                    self.settings['turbulence']['mean_mu_squared'],
                                                    DISTANCE,
                                                    MODULATION_FREQ,
                                                    self.settings['turbulence']['correlation_length'],
                                                    len(signal),
                                                    self.settings['turbulence']['depth']
                                                    )
            turbulence.randomize().generate()
            #signal = turbulence.apply_modulation(signal)
            signal = auraliser.signal.Signal(turbulence.apply_modulation(signal), 
                                             sample_frequency=self.sample_frequency)
        
        """Apply atmospheric absorption."""
        if self.settings['atmospheric_absorption']['include']:
            #print signal._data.shape
            signal = model.apply_atmospheric_absorption(signal, 
                                                        taps=self.settings['atmospheric_absorption']['taps'],
                                                        n_d=self.settings['atmospheric_absorption']['unique_distances']
                                                        )[0:self.samples]
        #print "Signal absorption: {}".format(signal)
        
        
        """Apply spherical spreading."""
        if self.settings['spreading']['include']:
            signal = model.apply_spherical_spreading(signal)
        
        #print "Signal spreading: {}".format(signal)
        
        
        if self.settings['roundzero']:
            signal[np.abs(signal)<self.settings['roundzero']] = 0.0
        
        
        return signal
    
        
    def _auralise_subsource(self, subsource, receiver):
        """
        Synthesize the signal of a subsource, due to all of its subsources.
        """
        
        subsource.generate_signals() # Generate the signals.
        
        if self.settings['reflections']['include'] and len(self.geometry.walls) > 0: # Are reflections possible?
            
            n = self.settings['reflections']['update_resolution']
            s = self.samples
            
            #print "N {}".format(len(subsource.position))
            source_position = geometry.PointList(subsource.position.as_array()[::n])
            mirrors = self._obtain_mirror_sources(source_position, receiver.position)
            
            #print "N {}".format(len(source_position))
            
            signal = np.zeros(s)
            
            #signals = list()
            for mirror in mirrors:
                
                """Create source object for synthesis. Repeat effectiveness for positions that were not tested."""
                m = _Mirror(mirror.position, np.repeat(mirror.effective,n, axis=0), np.repeat(mirror.strength,n, axis=0), position_for_directivity(mirror))
               
                signal += self._auralise_mirror(m, subsource.position, subsource.signal)
            return signal
    
        else: # No walls, so no reflections. Therefore the only source is the real source.
            mirror = _Mirror(receiver.position, None, None, receiver.position)
            
            return self._auralise_mirror(mirror, subsource.position, subsource.signal)
            
        
        
        
    def _auralise_source(self, source, receiver):
        """
        Synthesize the signal due to a source. This includes all subsources and respective mirror sources.
        """
        
        self.can_auralise()
        
        source = self.getObject(source)
        receiver = self.getObject(receiver)
        
        
        signals = list()
        for sub_src in source.subsources:
            signals.append(self._auralise_subsource(sub_src, receiver))
        return auraliser.signal.Signal(np.sum(signals, axis=0), sample_frequency=self.sample_frequency)
    
    ###def auralise(self, source, receiver, doppler=True, atmospheric_absorption=True, spherical_spreading=True):
        ###"""
        ###Auralise the sound ``source`` generates at ``receiver`` using this model.
        
        
        ###:param source: Source
        ###:param receiver: Receiver
        
        ###"""
        ###source = self.getObject(source)
        ###receiver = self.getObject(receiver)
        
        ###signals = list()
        ###"""Calculate all the effects for each subsource."""
        ###for sub_src in source.subsources:
            
            ###"""Create calculation model"""
            ###model = SignalModel(atmosphere=self.atmosphere, geometry=Model.Geometry(source=sub_src, receiver=receiver))
        
            ###"""Create signal for the specific orientation."""
            ###signal = sub_src.signal( - model.geometry.orientation )
            ###if doppler:
                ###signal = model.apply_doppler(signal)
            ###if atmospheric_absorption:
                ###signal = model.apply_atmospheric_spreading(signal)
            ###if spherical_spreading:
                ###signal = model.apply_spherical_spreading(signal)
            
            ###signals.append(signal)
            
        ###return np.sum(signals, axis=0)
    
    
class Base(object):
    """
    Base class
    """
    
    def __init__(self, name=None, description=None):
        """
        Constructor of base class.
        """
        
        self.name = name
        """
        Name of this object.
        """
        
        self.description = description
        """
        Description of object.
        """
    
class Source(Base):
    """
    Class for modelling a source.
    
    """
    def __init__(self, auraliser, name=None, description=None, position=None):
        """
        Constructor.
        """
        
        super(Source, self).__init__(name=name, description=description)
        self._auraliser = auraliser
        """
        Auraliser.
        """
        
        self._subsources = list()
        """
        List of sub sources that represent the source.
        """
        
        self.position = position if position else geometry.PointList()
        """
        Position of object.
        """
    
    @property
    def subsources(self):
        return [weakref.proxy(obj) for obj in self._subsources]
    
    
    def getSubsource(self, name):
        """Get object by name."""
        for obj in self.subsources:
            if name == obj.name:
                return obj
        return None
    
    def addSubsource(self, name=None):
        """
        Add subsource to auraliser.
        """
        obj = Subsource(self, name=name)
        self._subsources.append(obj)
        return self.getSubsource(obj.name)
        
    
    def removeSubsource(self, name):
        """
        Remove :class:`Auraliser.Source` from auraliser.
        """
        for obj in self._subsources:
            if obj.name == name:
                self._subsources.remove(obj)


class Receiver(Base):
    """
    Receiver
    """
    
    def __init__(self, auraliser, name=None, position=None):
        """
        Constructor.
        """
        
        super(Receiver, self).__init__(name=name)
    
        self._auraliser = auraliser
        self.position = position if position else geometry.PointList()
        """
        Position of object.
        """
    
    def auralise(self, sources=None):
        """
        Auralise the scene at receiver location.
        
        :param sources: List of sources to include. By default all the sources in the scene are included.
        
        """
        if not sources:
            sources = [source.name for source in self._auraliser.sources]
        
        signals = list()
        for source in sources:
            signals.append( self._auraliser._auralise_source(source, self.name) )
        return np.sum(signals, axis=0)
    
    
class Subsource(Base):
    """
    Class for modelling a subsource. A subsource is a component of source having a different position.
    """
    def __init__(self, source, position=None, name=None):
        """
        Constructor.
        """
        
        super(Subsource, self).__init__(name=name)
        
        self._source = source
        self._virtualsources = list()
        """
        List of virtual sources that represent this subsource.
        """
        
        self.position = position if position else geometry.PointList()
        """
        Position of object.
        """
    
    @property
    def virtualsources(self):
        return [weakref.proxy(obj) for obj in self._virtualsources]
    
    
    def generate_signals(self):
        """
        Generate the signal.
        """
        for src in self.virtualsources:
            src.generate_signal()
    
    def getVirtualsource(self, name):
        """Get object by name."""
        for obj in self.virtualsources:
            if name == obj.name:
                return obj
        return None
    
    def addVirtualsource(self, name=None):
        """
        Add virtualsource to auraliser.
        """
        obj = Virtualsource(self, name=name)
        self._virtualsources.append(obj)
        return self.getVirtualsource(obj.name)
        
    
    def removeVirtualsource(self, name):
        """
        Remove :class:`Auraliser.Source` from auraliser.
        """
        for obj in self._virtualsources:
            if obj.name == name:
                self._virtualsources.remove(obj)
                
    def signal(self, orientation):
        """
        Return the signal of this subsource as function of SubSource - Receiver orientation.
        
        The signal is the sum of all VirtualSources corrected for directivity.
        """
        return np.sum(np.array([src.signal_using_orientation(orientation) for src in self.virtualsources]), axis=0)


class Virtualsource(Base):
    """
    Class for modelling specific spectral components within a :class:`Auraliser.SubSource` that have the same directivity.
    """

    def __init__(self, subsource, name=None, rotation=None, directivity=None, gain=0.0):
        """
        Constructor.
        """
        
        super(Virtualsource, self).__init__(name)
        
        self._subsource = subsource
        """
        Subsource.
        """
        
        #self.signal = signal
        #"""
        #Signal.
        #"""
        
        self.directivity = directivity if directivity else Omni()
        """
        Directivity of the signal.
        """
        #self.modulation = modulation
        """
        Amplitude modulation of the signal in Hz.
        """
        self.gain = gain
        """
        Gain of the signal in decibel.
        """
        
        self.signal_generated = None
        """
        Generated signal
        """
        
    @property
    def position(self):
        """
        Position
        """
        return self._subsource.position
    
    def generate_signal(self):
        t = self._subsource._source._auraliser.duration
        fs = self._subsource._source._auraliser.sample_frequency
        
        self.signal_generated = self.signal.output(t, fs)
        
    
    def signal_using_orientation(self, orientation):
        """
        The signal this :class:`VirtualSource` emits as function of orientation.
        
        :param orientation: A vector of cartesian coordinates.
        """
        return self.signal_generated * self.directivity.using_cartesian(orientation[:,0], orientation[:,1], orientation[:,2]) * db_to_lin(self.gain)


class _Mirror(object):
    """
    Mirror source container.
    
    This structure contains the information necessary for auralising the mirror.
    """
    
    def __init__(self, position, effective, strength, position_directivity):
        
        self.position = position
        """
        Actual position of the mirror source.
        """
        self.effective = effective
        """
        Effectiveness of the source.
        """
        self.strength = strength
        """
        Strength of the source.
        """
        #self.order = order
        #"""
        #Order
        #"""
        self.position_directivity = position_directivity
        """
        Position used for directivity orientation.
        """
    
class Geometry(object):
    """
    Class describing the geometry of the model.
    """
    
    def __init__(self, walls=None):
        
        
        self.walls = walls if walls else list()
        """
        List of walls or faces.
        """
        
    def render(self):
        """
        Render the geometry.
        """
        raise NotImplementedError()
        


DEFAULT_SETTINGS = {
    
    'reflections':{
        'include'           :   True,   # Include reflections
        'mirrors_threshold' :   2,      # Maximum amount of mirrors to include
        'order_threshold'   :   3,      # Maximum order of reflections
        'update_resolution' :   20,    # Update effectiveness every N samples.
        'taps'              :   32,     # Amount of filter taps for ifft mirror strength.
        'force_hard'        :   False,  # Force hard reflections.
        },
    'doppler':{
        'include'           :   True,   # Include Doppler shift
        },
    'spreading':{
        'include'           :   True,   # Include spherical spreading
        },
    'atmospheric_absorption':{
        'include'           :   True,   # Include atmospheric absorption
        'taps'              :   32,     # Amount of filter taps for ifft 
        'unique_distances'  :   100,    # Calculate the atmospheric for N amount unique distances.
        },
    'turbulence':{
        'include'           :   True,   # Include modulations and decorrelation due to atmospheric turbulence.
        'depth'             :   1.0,    # Modulation depth/index.
        'mean_mu_squared'   :   6.0e-6,
        'correlation_length':   12.0,
        },
    'roundzero'             :   1.0e-15,
    }
"""
Default settings of :class:`Auraliser'.

All possible settings are included.
"""

def db_to_lin(db):
    """
    Convert gain in dB to linear.
    """
    return 10.0**(db/10.0)

def lin_to_db(lin):
    """
    Convert gain from linear to dB
    """
    return 10.0 * np.log10(lin/10.0)
    
    

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

    
    
