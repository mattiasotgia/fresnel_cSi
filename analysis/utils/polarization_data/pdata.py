import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

import numpy as np
from jacobi import propagate


__all__ = ['DataPolarization']


def hundred2rad(hundred):
    return hundred * np.pi / 50
def rad2hundred(rad):
    return rad * 50 / np.pi

class DataPolarization:
    
    def __init__(self,filenames_glob: str = None, skiprows = 23):
        
        self.mean = np.zeros_like(filenames_glob, dtype=np.float64)
        self.std = np.zeros_like(filenames_glob, dtype=np.float64)
        self.angle = np.zeros_like(filenames_glob, dtype=np.float64)
        
        self.ID = filenames_glob[0].split('/')[1].split('_')[-1]
        self.maximum = 0
        self.max_std = 0
        
        for i, filename in enumerate(filenames_glob):
            data = np.loadtxt(filename, skiprows=skiprows, dtype=np.float64)
            self.mean[i] = data.mean()
            self.std[i] = data.std()
            
            hundred = np.float64(filename.split('.')[0].split('/')[-1])
            self.angle[i] = hundred2rad(hundred) ## self.angle will always be in radians ;) 
    
    def plot(self, rad: bool = True, angle_bias: np.float64 = 0, costh_bias: np.float64 = 0):
        
        angle = self.angle
        ylabel = r'Light intensity ($I/I_0=\cos^2\theta$)'
        xlabel = r'Angle/rad'
        if not rad:
            angle = rad2hundred(self.angle) ## Angle is in HUNDREDS!
            xlabel = r'Angle/grad/4'
        
        plt.errorbar(angle-angle_bias, self.mean-costh_bias, yerr=self.std, 
                     ecolor='k', fmt='k.', mfc='w', label=f'Data ({self.ID})', markersize=8)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
    
    def model(x, p):
        norm, phase, offset = p
        return offset + norm * np.cos(x + phase)**2
    
    def fit(self, verbose = False, norm = 5, angle_bias = -2.82, costh_bias = 0):
        
        fcn = LeastSquares(self.angle, self.mean, self.std, model)
        minimizer = Minuit(fcn, (norm, angle_bias, costh_bias))
        minimizer.migrad()
        if verbose: print(minimizer)
        
        
        angles = np.linspace(self.angle.min(), self.angle.max(),100)
        y, ycov = propagate(
            lambda p: model(angles, p), minimizer.values, minimizer.covariance)
        yerr_prop = np.diag(ycov)**0.5
        
        return minimizer.values, minimizer.errors, angles, y, yerr_prop
    