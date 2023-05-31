import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def get_data(filenames, skiprows=23, normalization=None):
    return np.array([
        AngleData(file, skiprows=skiprows, normalization=normalization) for file in filenames
    ])

class AngleData:
    def __init__(self, file = None, skiprows = 24, normalization = None):
        data = np.loadtxt(file, skiprows=skiprows)
        self.value = np.abs(data.mean())
        self.std = data.std()
        filename_split = file.split('/')[-1].split('.')
        self.angle = np.float64(filename_split[2])*np.pi/180.0 + np.float64(filename_split[3])*np.pi/180.0/60.0
        self.polarization = 'p' if int(filename_split[-4]) == 1 else 's'
        self.inverted = True if 'inv' in filename_split else False
        self.normalized: bool = False if normalization is None else True

class SingleRun:
    def __init__(self,data: np.ndarray, normalization: AngleData = None, polarization: str = None):
        
        norm = AngleData
        
        if normalization is None:
            norm.value = 1
            norm.std = 0
        else:
            norm = normalization
        
        if polarization is None:
            self.polarization = data[0].polarization
        else:
            self.polarization = polarization
        
        self.angles = np.array([
            di.angle for di in data
        ])
        
        self.reflectance = np.array([
            di.value/norm.value for di in data
        ])
        
        self.reflectance_std = np.array([
            np.sqrt( (di.std/norm.value)**2  + (di.value * norm.std / norm.value**2)**2 ) for di in data
        ])
        
        self.IDs = np.array([
            di.inverted for di in data
        ])
    
    def plot(self, plot_asone=True, color='k', markersize=5, marker='o'):
        
        if plot_asone:
            plt.errorbar(self.angles, self.reflectance, yerr=self.reflectance_std, 
                         color=color, ecolor=color, fmt=marker, mfc='w', markersize=markersize, label=f'Data ({self.polarization}-plane)')
        else:
            angles = self.angles[self.IDs == False]
            reflectance = self.reflectance[self.IDs == False]
            reflectance_std = self.reflectance_std[self.IDs == False]
            
            angles_inv = self.angles[self.IDs == True]
            reflectance_inv = self.reflectance[self.IDs == True]
            reflectance_std_inv = self.reflectance_std[self.IDs == True]
            
            plt.errorbar(angles, reflectance, yerr=reflectance_std, 
                         color=color, ecolor=color, fmt='o', mfc='w', markersize=markersize, label=f'Data ({self.polarization}-plane, non inverted)')
            
            plt.errorbar(angles_inv, reflectance_inv, yerr=reflectance_std_inv, 
                         color=color, ecolor=color, fmt='^', mfc='w', markersize=markersize, label=f'Data ({self.polarization}-plane, inverted)')
            
        plt.xlabel('Normal incidence angle/rad')
        plt.ylabel('Reflectance/%')
    