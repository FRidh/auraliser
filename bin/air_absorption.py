
import sys
sys.path.append('..')

import numpy as np

from Auraliser import Atmosphere


def main():
    
    
    atmosphere = Atmosphere(temperature=303.0, pressure=101.325, relative_humidity=0.0)
    
    f = np.logspace(1.0, 5.0,  1000)
    
    #print model.molar_concentration_water_vapour
    
    #print atmosphere.attenuation_coefficient(f)
    
    atmosphere.plot_attenuation_coefficient('absorption.png', f)
    
    







if __name__=="__main__":
    main()