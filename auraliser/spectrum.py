import numpy as np

class Spectrum(np.ndarray):
    """
    Spectrum.
    """
    
    def __new__(cls, input_array, sort=None, frequency=None):
        obj = np.asarray(input_array).view(cls)
        return obj
    
    def __init__(self, input_array, frequency, sort='linear'):
        """
        
        :param sort: 'oct' or 'linear'.
        :type sort: str
        
        """
        assert len(input_array) == len(frequency)
        
        self.frequency = frequency
        self.sort = sort
        
    def view_as(self, sort, fmin, fmax, interval):
        """
        View the spectrum with different bandwidth settings.
        
        When requested resolution is higher than available, interpolation will be used.
        
        :param sort:
        :param fmin: Minimum frequency
        :param fmax: Maximum frequency
        :param interval: Bandwidth when sort='linear'. Octave interval when sort='interval'.
        """
        pass
    
    
