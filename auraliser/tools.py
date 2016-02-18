import numpy as np
from functools import singledispatch
import numba

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


@singledispatch
def norm(z):
    """
    Returns the Eucledian norm :math:`||\\mathbf{z}||` of `z`.
    
    The Eucledian  norm is given by
    
    #.. math:: \\mathbf{}
    """
    return np.linalg.norm(z, axis=1)
    #return np.sum(np.abs(z)**2.0, axis=1)**(0.5)


@singledispatch
def unit_vector(z):
    """
    Return the unit vector :math:`\\mathbf{\\hat{u}}`.
    
    The unit vector is given by
    
    .. math:: \\mathbf{\\hat{u}} = \\frac{\\mathbf{u}}{||\\mathbf{u}||}
    
    """
    return  z / norm(z)[...,None]


#@numba.jit
#def _norm(x):
    #return x[0]**2.0 + y[1]**2.0

#@numba.jit
#def _unit_vector(x):
    #return x / norm(x)

#@norm.register(AbstractStream)
#def _(stream):
    #return stream.samples().map(norm)

#@unit_vector.register(AbstractStream)
#def _(stream):
    #return stream.samples().map(_unit_vector)

#@norm.register(BlockStream)
#def _(stream):
    #return stream

#@unit_vector.register(BlockStream):
