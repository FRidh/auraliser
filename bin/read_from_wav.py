"""
Example that shows how to unapply the Doppler shift to an existing wav file.
"""


import sys
sys.path.append('..')


import numpy as np

#from auraliser import Signal

from auraliser.signal import Signal
from auraliser.model import Model, Source, Receiver

import matplotlib as mpl
mpl.rc('figure', figsize=(12,10))



def main():
    
    filename = '../data/recording.wav'
    
    signal = Signal.from_wav(filename)
    signal.plot_spectrogram('../data/spec.png')
    
    """Let's create a model. We ultimately need position vectors for the source and receiver."""
    model = Model()
    fs = signal.sample_frequency
    
    duration = fs * len(signal)
    
    velocity = 65.0  # Velocity of the source in m/s. Guess
    
    distance_per_sample = velocity / fs    # Velocity vector expressed in meters per sample
    
    
    xyz_strip = np.array([1600.0, 0.0, -10.0])  # Touchdown coordinate
    xyz_receiver = np.array([0.0, 0.0, 0.0])    # Receiver
    
    xyz_passage = np.array([0.0, 0.0, 65.0])    # Height of source at passage
    
    distance = xyz_strip - xyz_passage          # Distance between passage and touchdown 
    orientation = distance / np.linalg.norm(distance)   # This results in the bearing of the aircraft.
    
    print( np.degrees(np.arctan(orientation[2]/orientation[0])))
    
    assert(orientation[2] < 0.0)                # The aircraft should decrease in altitude...
    
    dxds = distance_per_sample * orientation    # Let's create a velocity vector instead!!
   
    t_passage = 23.8   # Time after start of recording that the aircraft passes the receiver location
    
    source = np.outer( dxds, np.arange(0, len(signal)))   # Position of source, not yet given the right offset.
    
    s_passage = t_passage * fs
    
    source = source.transpose()
    
    source = source - source[s_passage,:] + xyz_passage.transpose()
    
    #print ( (  - xyz_receiver ).transpose() )
    
    #source = Source(
    
    
    model.geometry.source.position.x = source[:,0]
    model.geometry.source.position.y = source[:,1]
    model.geometry.source.position.z = source[:,2]
    
    receiver = np.ones((len(signal), 3)) * xyz_receiver
    model.geometry.receiver.position.x = receiver[:,0]
    model.geometry.receiver.position.y = receiver[:,1]
    model.geometry.receiver.position.z = receiver[:,2]
    
    #print model.geometry.source.position.as_array()[-1,:]
    
    ###model.geometry.plot_position('../data/position.png')
    ###model.geometry.plot_distance('../data/distance.png', signal=signal)
    ###model.plot_velocity('../data/velocity.png', signal=signal)
    ###model.plot_delay('../data/delay.png', signal=signal)
    ###model.plot_time_compression('../data/time_compression.png')
    
    assert source[-1,2] > xyz_strip[2]  # That we have no crash...
    
    signal.plot_spectrogram('../data/original.png')
    
    l = len(signal)
    
    signal = model.unapply_atmospheric_absorption(signal, taps=200, n_d=50)[0:l]
    signal = model.unapply_spherical_spreading(signal)
    signal = model.unapply_doppler(signal)
    
    signal.plot_spectrogram('../data/after_absorption.png')
    
    
    
    
    signal.to_wav('../data/after_absorption.wav')
    
    ###model.unapply_spherical_spreading(signal)
    ####signal.to_wav('../data/after_spreading.wav')
    
    
    ###model.unapply_doppler(signal)
    ###signal.to_wav('../data/after_doppler_and_spreading.wav')
    ###signal.plot_spectrogram('../data/after_doppler_and_spreading.png')
    
    
    #signal.plot_scaleogram('../data/scaleogram.png')
    
    
if __name__ == "__main__":
    main()