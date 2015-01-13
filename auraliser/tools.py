import numpy as np

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
    
