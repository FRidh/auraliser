"""Sinks.

This module specifies encoding methods.

Each encoding function takes at least the argument contributions.
Contributions is an iterable of tuples. Each tuple consist of (signal, orientation).
"""
import abc
import itertools
from acoustics import Signal
from acoustics.ambisonics import acn
from acoustics.directivity import SphericalHarmonic


def encode(contributions, directivities, resolution=1000):
    """Encode contributions with directivities.
    
    :param contributions: Contributions is an iterable containing tuples of two elements, 
    with the first element the signal and the second element the Cartesian orientation.
    :param directivities: Iterable containing directivities. Each item results in a signal to be outputted.
    
    """
    contributions = list(contributions)
    
    for directivity in directivities:
        yield sum( ( (signal * np.repeat(channel.directivity.using_cartesian(*orientation[::resolution].T), resolution)[0:len(signal)]) for signal, orientation in contributions) ) 

def mono(contributions):
    """Encode contributions as mono signal.

    :param contributions: Contributions
    :returns: Signal

    """
    return sum(signal for signal, orientation in contributions)

def ambisonics(contributions, order, resolution=1000):
    """Encode signal as ambisonics using ACN format.
    """
    directivities = (SphericalHarmonic(m=m, n=n) for n, m in acn)
    #yield from encode(contributions, directivities, order)
    results = list(encode(contributions, directivities, order))
    return Signal(results, results[0].fs)
 
 



#class _Encoder(object):
    #"""Abstract encoder."""
        
    #@abc.abstractmethod
    #def encode(self, contributions):
        #"""Encode contributions with."""
        #pass
    
#class Mono(Encoder):
    #"""Mono encoder."""
   
    #def __init__(self):
        #super().__init__(directivities=[Mono()])
        


       
   
    #def encode(self, contributions):
        #return sum((signal for signal, contribution in contributions))
    
#class Stereo(Encoder):
    #pass


#class Encoder(object):
    
    #def __init__(self, directivities):
        
        #self._directivities = directivities

    #def encode(self, contributions):
        #yield from encode(contributions, directivities, resolution)

#class Ambisonics(Encoder):
    #"""Ambisonics encoder."""

    #def __init__(self, order):
        
        #self.order = order
        #"""Order."""
        
    #def acn(self):
        #"""ACN."""
        #yield from acn(self.order)
        
    #def encode(self, contributions):
        
        
    
    
    
