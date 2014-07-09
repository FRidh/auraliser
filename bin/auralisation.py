from auraliser import Auraliser
import numpy as np
from auraliser.generator import Sine, WhiteNoise, WhiteBand, WhiteOctaveBands
from acoustics.directivity import Cardioid, FigureEight, Omni
from geometry import Point, PointList
from ism import Wall

def auralisation():
    
    
    fs = 44100.0            # Sample frequency
    duration = 12.         # Duration in seconds.
    
    df = 500.0
    
    """Preparing a geometry."""
    velocity = 60.0                        # Velocity of source in meters per second.
    
    distance = velocity * duration          # Total distance covered
    dt = 1.0/fs                             # Seconds per sample
    t = np.arange(0.0, duration, dt)        # Time vector
    
    x = np.ones_like(t) * velocity * (t - duration/2.0)    # Source moves along the x-axis.
    y = np.ones_like(t) * 1.0
    z = np.ones_like(t) * 100.0   # Altitude of source
    #print x[-1]
    """Create object to perform auralisation with."""
    auraliser = Auraliser(duration=duration, sample_frequency=fs)
    
    #auraliser.settings['spreading']['include']= False
    
    """The main source is the aircraft."""
    src = auraliser.addSource(name='aircraft')
    
    """A primary source on the aircraft is one of the jet engines."""
    jet = src.addSubsource(name='jet')
    jet.position = PointList(np.vstack((x,y,z)))
    
    """The jet has a slightly different position and radiates components with different directivities."""
    rear1 = jet.addVirtualsource('rear1')
    
    """The jet radiates a sine wave with a certain frequency and a directivity."""
    rear1.signal = Sine(frequency=70.0)
    rear1.directivity = FigureEight()
    rear1.gain = -18.0
    
    rear2 = jet.addVirtualsource('rear2')
    rear2.signal = Sine(frequency=350.0)
    rear2.directivity = FigureEight()
    rear2.gain = -23.0
    
    rear3 = jet.addVirtualsource('rear3')
    rear3.signal = Sine(frequency=800.0)
    rear3.directivity = FigureEight()
    rear3.gain = -26.0
    
    rear4 = jet.addVirtualsource('rear4')
    rear4.signal = Sine(frequency=830.0)
    rear4.directivity = Omni()
    rear4.gain = -26.0
    
    #"""White noise can also be added."""
    #rear5 = jet.addVirtualsource('rear5')
    #rear5.signal = WhiteNoise()
    #rear5.directivity = Omni()
    #rear5.gain = 0.1
    
    """Noise in a band can also be added."""
    #rear6 = jet.addVirtualsource('rear6')
    #rear6.signal = WhiteBand(f1=1.0, f2=250.0, numtaps=100)
    #rear6.directivity = Omni()
    #rear6.gain = 0.1
    
    rear7 = jet.addVirtualsource('rear7')
    rear7.signal = Sine(frequency=1800.0)
    rear7.directivity = FigureEight()
    rear7.gain = -27.0
    
    
    rear8 = jet.addVirtualsource('rear8')
    rear8.signal = Sine(frequency=1850.0)
    rear8.directivity = FigureEight()
    rear8.gain = -27.0
    
    noise1 = jet.addVirtualsource('noise1')
    noise1.signal = WhiteOctaveBands([16.0, 31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0],
                                     [+5.0, -2.0, -7.7, -8.9, -9.0, -10.5, -11.0, -17.0, -23.0]
                                     )
    noise1.directivity = FigureEight()
    noise1.gain = -4.0
    
    """We want to auralise at a receiver location and therefore add a receiver to the model."""
    #rcv = auraliser.addReceiver(name='receiver', position=PointList([Point(0,0,0)]))
    
    rcv = auraliser.addReceiver(name='receiver', position=Point(0.0,0.0,0.1))
    
    rcv.position = Point(0,0,1.6)
    
    
    """Let's add a ground surface so we get reflections."""
    impedance = 400000.0 * np.linspace(1000, 100, fs/2/df) + 200 * np.linspace(1000, 100, fs/2/df)
    
    #groundcorners1 = [ Point(-100.0, -100.0, 0.0), Point(100.0, -100.0, 0.0), Point(100.0, 100.0, 50.0), Point(-100.0, 100.0, 50.0) ]
    #ground1 = Wall(groundcorners1, impedance, Point(0.0, 0.0, 25.0))
    
    groundcorners1 = [ Point(-100.0, -100.0, 0.0), Point(100.0, -100.0, 0.0), Point(100.0, 100.0, 0.0), Point(-100.0, 100.0, 0.0) ]
    ground1 = Wall(groundcorners1, impedance, Point(0.0, 0.0, 0.0))
    
    
    corners1 = [ Point(2.0, 0.0, 0.0), Point(2.0, 10.0, 0.0), Point(2.0, 10.0, 10.0), Point(2.0, 0.0, 10.0)]
    wall1 = Wall(corners1, impedance, Point(2.0, 5.0, 5.0))
    
    corners1m = [ Point(2.0, 0.0, 10.0), Point(2.0, 10.0, 10.0), Point(2.0, 10.0, 0.0), Point(2.0, 0.0, 0.0)]
    wall1m = Wall(corners1m, impedance, Point(2.0, 5.0, 5.0))

    roof = Wall([Point(-100.0, -100.0, 10.0), Point(100.0, -100.0, 10.0), Point(100.0, 100.0, 10.0), Point(-100.0, 100.0, 10.0) ], 
                impedance, 
                Point(0.0, 0.0, 10.0))

    roof_inverted = Wall([Point(+100.0, +100.0, 10.0), Point(+100.0, -100.0, 10.0), Point(-100.0, -100.0, 10.0), Point(-100.0, +100.0, 10.0) ], 
                         impedance, 
                         Point(0.0, 0.0, 10.0))
    
    auraliser.geometry.walls = [ground1]#, roof, roof_inverted]#, wall1_mirrored]
    
    
    auraliser.settings['spreading']['include'] = True
    auraliser.settings['doppler']['include'] = True
    auraliser.settings['atmospheric_absorption']['include'] = True
    auraliser.settings['reflections']['include'] = True
    auraliser.settings['turbulence']['include'] = False
    
    
    """Finally, we perform the auralisation and obtain a time-domain signal."""
    signal = auraliser._auralise_source(source='aircraft', receiver='receiver')
    signal.to_wav('auralisation.wav')
    
    signal.spectrogram('spectrogram.png')
    
    
    
    
    

if __name__ == '__main__':
    auralisation()



