
from auraliser import Reverter


##def create_geometry(velocity, xyz_strip, xyz_receiver, xyz_passage, t_passage):
    ##"""
    ##Create geometry. Assume linear ascend or descend.
    ##"""
    


def main():
    
    """Settings"""
    filename = '../data/recording.wav'
    
    
    velocity = 65.0  # Velocity of the source in m/s. Guess
    xyz_strip = np.array([1600.0, 0.0, -20.0])  # Touchdown coordinate
    xyz_receiver = np.array([0.0, 0.0, 0.0])    # Receiver
    xyz_passage = np.array([0.0, 0.0, 80.0])    # Height of source at passage
    t_passage = 23.8   # Time after start of recording that the aircraft passes the receiver location
    
    
    
    """Load signal"""
    signal = Signal.from_wav(filename)
    signal.plot_spectrogram('../data/spec.png')
    
    """Let's create a model. We ultimately need position vectors for the source and receiver."""
    fs = signal.sample_frequency
    
    duration = fs * len(signal)
    distance_per_sample = velocity / fs    # Velocity vector expressed in meters per sample
    distance = xyz_strip - xyz_passage          # Distance between passage and touchdown 
    orientation = distance / np.linalg.norm(distance)   # This results in the bearing of the aircraft.
    
    print np.degrees(np.arctan(orientation[2]/orientation[0]))
    
    assert(orientation[2] < 0.0)                # The aircraft should decrease in altitude...
    
    dxds = distance_per_sample * orientation    # Let's create a velocity vector instead!!
    source = np.outer( dxds, np.arange(0, len(signal)))   # Position of source, not yet given the right offset.
    s_passage = t_passage * fs
    source = source.transpose()
    source = source - source[s_passage,:] + xyz_passage.transpose()
    receiver = np.ones((len(signal), 3)) * xyz_receiver
    
    assert source[-1,2] > xyz_strip[2]  # That we have no crash...
    
    
    model = Reverter()
    
    #print model.geometry
    
    #model.geometry.source.position.x = source[:,0]
    #model.geometry.source.position.y = source[:,1]
    #model.geometry.source.position.z = source[:,2]
    
    
    #model.geometry.receiver.position.x = receiver[:,0]
    #model.geometry.receiver.position.y = receiver[:,1]
    #model.geometry.receiver.position.z = receiver[:,2]
    
    #rev_signal = model.revert(signal)
    
    
    #rev_signal.plot_spectrogram('../data/after_absorption.png')
    #rev_signal.to_wav('../data/after_absorption.wav')
    



if __name__ == '__main__':
    main()