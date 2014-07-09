
import sys
sys.path.append('..')

import numpy as np

from Auraliser import Signal



def main():
    
    ###"""Construct a signal."""
    ###fs = 2000.0            # Sample frequency
    ###duration = 1.0         # Duration in seconds.
    ###f = 100.0              # Emission signal consists of a tone of 1000.0 Hz
    
    ###assert(fs >= 2.0 * f)     # Nyquist
    
    ###dt = 1.0/fs                               # Seconds per sample
    ###t = np.arange(0.0, duration, dt)        # Time vector
    ###signal = np.sin(2.0 * np.pi * f * t)    # Emission signal
    
    ####signal = signal + np.random.randn(len(t))/2.0       # Let's add some random noise.
    
    ###signal = Signal(signal, sample_frequency=fs)    # Emission signal as the right type

    signal = Signal(np.arange(0.0, 10.0), sample_frequency=100)
    
    print signal
    
    x =  signal - 2. * np.arange(0, 10)
    
    print x
    
    #print signal.sample_frequency
    
    #print x.sample_frequency
    
    #filename = '../data/short.wav'
    
    #signal = Signal.from_wav(filename)
    
    #print signal.describe()
    
    
    #signal.dB().plot()
    #signal.plot_scalo('../scalo.png')

    #signal.plot_scaleogram('../data/scaleogram.png')
    
    #signal.plot_spectrogram('../data/spectrogram.png')

if __name__ == '__main__':
    main()