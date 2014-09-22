cimport cython
cimport numpy 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import acoustics
from acoustics.signal import power_spectrum

#cdef op(str operator, a, b):
    #cdef int equal
    #try:
        #equal = a.sample_frequency==b.sample_frequency
    #except AttributeError:
        #equal = True
    #finally:
        #if equal:
            #return a.__class__(getattr(super(type(a),a), operator)(b), sample_frequency=a.sample_frequency)
        #else:
            #raise ValueError("Sample frequencies do not match. a: {} b: {}".format(a.sample_frequency, b.sample_frequency))
   

#cdef iop(str, operator, a, b):
    #cdef int equal
    #try:
        #equal = a.sample_frequency ==b.sample_frequency
    #except AttributeError:
        #equal = True
    #finally:
        #if equal:
            #getattr(super(type(a),a), operator)(b)
            #return a
        #else:
            #raise ValueError("Sample frequencies do not match. a: {} b: {}".format(a.sample_frequency, b.sample_frequency))
        
class Signal(numpy.ndarray):

    sample_frequency = 0.0
    
    def __new__(cls, data, sample_frequency):
        
        if np.asarray(data).ndim!=1:
            raise ValueError("Incorrect amount of dimensions. One dimension is required.")
        obj = np.asarray(data).view(cls)
        obj.sample_frequency = sample_frequency
        return obj
    
    def __array_prepare__(self, array, context=None):
        try:
            a = context[1][0]
            b = context[1][1]
        except IndexError:
            return array
        
        if hasattr(a, 'sample_frequency') and hasattr(b, 'sample_frequency'):
            if a.sample_frequency == b.sample_frequency:
                return array
            else:
                raise ValueError("Sample frequencies do not match.")
        else:
            return array
        
    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)
    
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: 
            return
        self.sample_frequency = getattr(obj, 'sample_frequency', None)
        
        
   
    #def __add__(self, other):
        #return op('__add__', self, other)

    #def __sub__(self, other):
        #return op('__sub__', self, other)
    
    #def __mul__(self, other):
        #return op('__mul__', self, other)
    
    #def __div__(self, other):
        #return op('__div__', self, other)
    
    #def __mod__(self, other):
        #return op('__mod__', self, other)
    
    #def __iadd__(self, other):
        #return op('__iadd__', self, other)

    #def __isub__(self, other):
        #return op('__isub__', self, other)
    
    #def __imul__(self, other):
        #return op('__imul__', self, other)
    
    #def __idiv__(self, other):
        #return op('__idiv__', self, other)
    
    #def __imod__(self, other):
        #return op('__imod__', self, other)
    

    
###class Signal(object):
    ###"""
    ###Class for containing a signal in time-domain.
    
    ###This class provides methods for plotting the signal in time- and/or frequency-domain.
    ###Signals can be created from a WAV file or written to one.
    ###"""
    
    
    ####@property
    ####def data(self):
        ####return self._data
    
    ####@data.setter
    ####def data(self, data):
        ####if data.ndim==1:
            ####self._data = data
        ####else:
            ####raise ValueError("Wrong shape.")

    ###def __init__(self, data, sample_frequency=44100):
        ###"""
        ###Constructor
        
        ###:param input_array: Array describing the time data.
        ###:param sample_frequency: Sample frequency :math:`f_s`
        ###:type sample_frequency: int
        ###"""
        
        
        ###try:
            ###data = data._data
        ###except AttributeError:
            ###pass
        ###if data.ndim==1:
            ###self._data = data
        ###else:
            ###raise ValueError("Wrong shape.")
        
        ###self.sample_frequency = sample_frequency
        ###"""Sample frequency"""
    
    ###def __add__(self, other):
        ###return op('__add__', self, other)

    ###def __sub__(self, other):
        ###return op('__sub__', self, other)
    
    ###def __mul__(self, other):
        ###return op('__mul__', self, other)
    
    ###def __div__(self, other):
        ###return op('__div__', self, other)
    
    #def __mod__(self, other):
        #if isinstance(other, Signal):
            #return self.__class__(self._data % other._data, sample_frequency=self.sample_frequency)
        #else:
            #return self.__class__(self._data % other, sample_frequency=self.sample_frequency)
    
    #def __iadd__(self, other):
        #if isinstance(other, Signal):
            #if self.sample_frequency == other.sample_frequency:
                #self._data += other._data
        #else:
            #self._data += other
        #return self
    
    #def __isub__(self, other):
        #if isinstance(other, Signal):
            #if self.sample_frequency == other.sample_frequency:
                #self._data -= other._data
        #else:
            #self._data -= other
        #return self
    
    #def __imul__(self, other):
        #if isinstance(other, Signal):
            #if self.sample_frequency == other.sample_frequency:
                #self._data *= other._data
        #else:
            #self._data *= other
        #return self
    
    #def __idiv__(self, other):
        #if isinstance(other, Signal):
            #if self.sample_frequency == other.sample_frequency:
                #self._data /= other._data
        #else:
            #self._data /= other
        #return self
    
    #def __abs__(self):
        #return self * self
    
    #def __pos__(self):
        #return self.__class__(+self._data, sample_frequency=self.sample_frequency)
    
    #def __neg__(self):
        #return self.__class__(-self._data, sample_frequency=self.sample_frequency)
    
    #def __len__(self):
        #return len(self._data)
    
    #def __getitem__(self, key):
        #return self._data[key]
        ##return self.__class__(self._data[key], self.sample_frequency)
        
    #def __setitem__(self, key, value):
        #self._data[key] = value
    
    #def __repr__(self):
        #return "Signal({})".format(self._data.__str__())
    
    #def __str__(self):
        #return self._data.__str__()
    
    #def __iter__(self):
        #return self._data.__iter__()
    
    #@property
    #def real(self):
        #return self.__class__(self._data.real, self.sample_frequency)
    
    #@property
    #def imag(self):
        #return self.__class__(self._data.imag, self.sample_frequency)
    
    @property
    def size(self):
        """Amount of items in signal."""
        return len(self)
    
    #def min(self):
        #"""Minimum value."""
        #return self._data.min()
    
    #def max(self):
        #"""Maximum value."""
        #return self._data.max()
    
    #def argmin(self):
        #"""Index of minimum value."""
        #return self._data.argmin()
    
    #def argmax(self):
        #"""Index of maximum value."""
        #return self._data.argmax()
    
    #def conjugate(self):
        #"""Complex conjugate."""
        #return self._data.conjugate()
    
    #def mean(self):
        #"""
        #Signal mean value.
        
        #.. math:: \\mu = \\frac{1}{N} \\sum_{n=0}^{N-1} x_n
        
        #"""
        #return self._data.mean()
    
    def energy(self):
        """
        Signal energy.
        
        .. math:: E = \\sum_{n=0}^{N-1} |x_n|^2
        
        """
        return (self**2.0).sum()
    
    def power(self):
        """
        Signal power.
        
        .. math:: P = \\frac{1}{N} \\sum_{n=0}^{N-1} |x_n|^2
        """
        return self.energy() / len(self)
    
    def rms(self):
        """
        RMS signal power.
        
        .. math:: P_{RMS} = \\sqrt{P}
        
        """
        return np.sqrt(self.power())

    #def std(self):
        #"""
        #Standard deviation.
        #"""
        #return self._data.std()
    
    #def var(self):
        #"""
        #Signal variance.
        
        #.. math:: \\sigma^2 = \\frac{1}{N} \\sum_{n=0}^{N-1} |x_n - \\mu |^2
        
        #"""
        #return self._data.var()
    
    def spectrum(self):
        """
        Create spectrum.
        """
        return power_spectrum(self, self.sample_frequency)
    
    def plot_spectrum(self, filename=None):
        """
        Plot spectrum of signal.
        
        :param filename: Name of file.
        """
        f, o = self.spectrum()
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('SPL')
        ax0.semilogx(f, 10.0*np.log10(o))
        ax0.set_ylabel(r"$L_{p}$ in dB")
        ax0.set_xlabel(r"$f$ in Hz")
        if filename:
            fig.savefig(filename)
        else:
            return fig
    
    def spectrogram(self, filename=None):
        """
        Plot spectrograms of the signals.
        
        :param filename: Name of file.
        """
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('Spectrogram')
        f = ax0.specgram(self, Fs=self.sample_frequency, noverlap=128, NFFT=4096)#, vmin=self._data.min(), vmax=self._data.max())
        cb = fig.colorbar(mappable=f[3])
        ax0.set_xlabel(r'$t$ in s')
        ax0.set_ylabel(r'$f$ in Hz')
        if filename:
            fig.savefig(filename)
        else:
            return fig
    
    
    def leq(self):
        """
        Equivalent level.
        """
        return acoustics.standards.iec_61672_1_2013.fast(self, self.sample_frequency)
    
    
    def plot_leq(self, filename=None):
        """
        Plot equivalent level.
        """
        t, L = self.leq()
        L_masked = np.ma.masked_where(np.isinf(L), L)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Equivalent level')
        ax.plot(t, L_masked)
        ax.set_xlabel('$t$ in s')
        ax.set_ylabel('$L_{eq,F}$ in dB')
        
        if filename:
            fig.savefig(filename)
        else:
            return fig
    
    def octaves(self):
        """
        Calculate time-series of octaves.
        """
        return acoustics.signal.octaves(self, self.sample_frequency)
    
    def plot_octaves(self, filename=None):
        """
        Plot octaves.
        """
        f, o = self.octaves()
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('SPL')
        ax0.semilogx(f.center, o)
        ax0.set_ylabel(r"$L_{p}$ in dB")
        ax0.set_xlabel(r"$f$ in Hz")
        
        if filename:
            fig.savefig(filename)
        else:
            return fig
    
    def third_octaves(self):
        """
        Calculate time-series of octaves.
        """
        return acoustics.signal.third_octaves(self, self.sample_frequency)
    
    def plot_third_octaves(self, filename=None):
        """
        Plot octaves.
        """
        f, o = self.third_octaves()
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('SPL')
        ax0.semilogx(f.center, o)
        ax0.set_ylabel(r"$L_{p}$ in dB")
        ax0.set_xlabel(r"$f$ in Hz")
        if filename:
            fig.savefig(filename)
        else:
            return fig
    
    def fractional_octaves(self, fraction):
        """Fractional octaves.
        """
        return acoustics.signal.fractional_octaves(self, self.sample_frequency, fraction=fraction)
    
    def plot_fractional_octaves(self, fraction, filename=None):
        """
        Plot octaves.
        """
        f, o = self.fractional_octaves(fraction)
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('SPL')
        ax0.semilogx(f.center, o)
        ax0.set_ylabel(r"$L_{p}$ in dB")
        ax0.set_xlabel(r"$f$ in Hz")
        if filename:
            fig.savefig(filename)
        else:
            return fig
    
    
    def plot(self, filename=None, start=None, stop=None):
        """
        Plot signal as function of time.
        By default the entire signal is plotted.
        
        :param filename: Name of file.
        :param start: First sample index.
        :type start: int
        :param stop: Last sample index.
        :type stop: int
        """
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        ax0.set_title('Signal')
        ax0.plot(self._signal[start:stop])
        ax0.set_xlabel(r'$t$ in n')
        ax0.set_ylabel(r'$x$ in -') 
        if filename:
            fig.savefig(filename)
        else:
            return fig
        
    #def plot_scalo(self, filename=None):
        #"""
        #Plot scalogram 
        #"""
        #from scipy.signal import ricker
        
        #wavelet = ricker
        #widths = np.logspace(-1, 3.5, 100)
        #x = cwt(self._signal, wavelet, widths)
        
        #interpolation = 'nearest'
        
        #from matplotlib.ticker import LinearLocator, AutoLocator, MaxNLocator
        #majorLocator = LinearLocator()
        #majorLocator = MaxNLocator()
        
        
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.set_title('Scaleogram')
        ##ax.set_xticks(np.arange(0, x.shape[1])*self.sample_frequency)
        ##ax.xaxis.set_major_locator(majorLocator)
        
        ##ax.imshow(10.0 * np.log10(x**2.0), interpolation=interpolation, aspect='auto', origin='lower')#, extent=[0, 1, 0, len(x)])
        #ax.pcolormesh(np.arange(0.0, x.shape[1])/self.sample_frequency, widths, 10.0*np.log(x**2.0))
        #if filename:
            #fig.savefig(filename)
        #else:
            #return fig
    
    #def plot_scaleogram(self, filename):
        #"""
        #Plot scaleogram
        #"""
        #import pywt
        
        #wavelet = 'dmey'
        #level = pywt.dwt_max_level(len(self), pywt.Wavelet(wavelet))
        #print level
        #level = 20
        #order = 'freq'
        #interpolation = 'nearest'
        
        #wp = pywt.WaveletPacket(self, wavelet, 'sym', maxlevel=level)
        #nodes = wp.get_level(level, order=order)
        #labels = [n.path for n in nodes]
        #values = np.abs(np.array([n.data for n in nodes], 'd'))
        
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.set_title('Scaleogram')
        #ax.imshow(values, interpolation=interpolation, aspect='auto', origin='lower', extent=[0, 1, 0, len(values)])
        ##ax.set_yticks(np.arange(0.5, len(labels) + 0.5))
        ##ax.set_yticklabels(labels)
        
        #fig.savefig(filename)
    
    def to_wav(self, filename, normalize=True, depth=16):
        """
        Save signal as WAV file.
        
        .. warning:: The WAV file will have 16-bit depth!
        
        :param filename: Filename
        """
        data = self
        dtype = data.dtype if not depth else 'int'+str(depth)
        if normalize:
            data = data / data.max() * 0.5
        #if depth:
            #data = (data * 2.0**depth).astype(dtype)
        #print(data)
        wavfile.write(filename, int(self.sample_frequency), data)
        #wavfile.write(filename, int(self.sample_frequency), self._data/np.abs(self._data).max() *  0.5)
        #wavfile.write(filename, int(self.sample_frequency), np.int16(self._data/(np.abs(self._data).max()) * 32767) )
    
    @classmethod
    def from_wav(cls, filename):
        """
        Create signal from WAV file.
        
        :param filename: Filename
        """
        sample_frequency, data = wavfile.read(filename)  
        data = data.astype(np.float32, copy=False)
        data /= np.max(np.abs(data))
        return cls(data, sample_frequency=sample_frequency)
    
    def to_mat(filename):
        """
        Save signal to MAT file.
        """
        raise NotImplementedError
    
    @classmethod
    def from_mat(cls, filename):
        """
        Load signal from MAT file.
        """
        raise NotImplementedError
