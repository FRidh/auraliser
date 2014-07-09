import numpy as np


def norm(z):
    """
    Returns the Eucledian norm :math:`||\\mathbf{z}||` of `z`.
    
    The Eucledian  norm is given by
    
    #.. math:: \\mathbf{}
    """
    return np.sum(np.abs(z)**2.0, axis=1)**(0.5)
   
def unit_vector(z):
    """
    Return the unit vector :math:`\\mathbf{\\hat{u}}`.
    
    The unit vector is given by
    
    .. math:: \\mathbf{\\hat{u}} = \\frac{\\mathbf{u}}{||\\mathbf{u}||}
    
    """
    return  z / norm(z)[:,None]


class Vector(object):
    """
    XYZ vectors.
    """
    
    def __init__(self, x=None, y=None, z=None):
        
        self.x = x if x else np.array(0.0)
        """X vectors vector."""
        self.y = y if y else np.array(0.0)
        """Y vectors vector."""
        self.z = z if z else np.array(0.0)
        """Z vectors vector."""
    
    def as_array(self):
        """
        Return the vectors as a 2-dimensional array.
        """
        return np.vstack( (self.x, self.y, self.z) ).transpose()

