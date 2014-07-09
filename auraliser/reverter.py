from model import Geometry


class Reverter(object):
    """
    Class for calculating back from receiver to source.
    """
    
    def __init__(self, atmosphere=None, geometry=None):
        
        self.atmosphere = atmosphere if atmosphere else Atmosphere()
        
        self.geometry = geometry if geometry else Geometry()
        
    def revert(self, signal, doppler=True, atmospheric_absorption=True, spherical_spreading=True):
        """
        Calculate back from receiver signal to source signal.
        """
        
        model = Model(atmosphere=self.atmosphere, geometry=self.geometry)
        
        if spherical_spreading:
            signal = model.unapply_spherical_spreading(signal)
        if atmospheric_absorption:
            signal = model.unapply_atmospheric_absorption(signal)
        if doppler:
            signal = model.unapply_doppler(signal)
        
        return signal