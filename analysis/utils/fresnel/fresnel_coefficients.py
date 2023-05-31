import numpy as np

def r_s(theta, n1, n2):
    '''Compute r_s fresnel coefficient
    
    Parameters:
    `theta`: incident angle, expressed in radians
    `n1`: complex refractive indices for material 1 (n = n + k*1j)
    `n2`: complex refractive indices for material 2 (n = n + k*1j)
    '''
    
    rtarg = 1 - ((n1/n2) * np.sin(theta))**2
    return (
        (n1 * np.cos(theta) - n2 * np.sqrt(rtarg)) / 
        (n1 * np.cos(theta) + n2 * np.sqrt(rtarg))
    )

def r_p(theta, n1, n2):
    '''Compute r_p fresnel coefficient
    
    Parameters:
    `theta`: incident angle, expressed in radians
    `n1`: complex refractive indices for material 1 (n = n + k*1j)
    `n2`: complex refractive indices for material 2 (n = n + k*1j)
    '''
    rtarg = 1 - ((n1/n2) * np.sin(theta))**2
    return (
        (- n1 * np.sqrt(rtarg) + n2 * np.cos(theta)) / 
        (n1 * np.sqrt(rtarg) + n2 * np.cos(theta))
    )
    
def phi(theta, n1, n2, d): 
    ct2 = np.sqrt(1 - (n1 * np.sin(theta) / n2)**2)
    return 2 * np.pi * d * n2 * ct2 / 532e-9
    
def construct_three_layer_r(r: callable, theta, n1, n2, n3, d):
    '''Compute r_p fresnel coefficient
    
    Parameters:
    `theta`: incident angle, expressed in radians
    `n1`: complex refractive indices for material 1 (n = n + k*1j)
    `n2`: complex refractive indices for material 2 (n = n + k*1j)
    `n3`: complex refractive indices for material 3 (n = n + k*1j)
    `d` : inner layer width
    '''
    
    return (
        (r(theta, n1, n2) + r(theta, n2, n3) * np.exp(-2j * phi(theta, n1, n2, d)))/
        (1 + r(theta, n1, n2) * r(theta, n2, n3) * np.exp(-2j * phi(theta, n1, n2, d)))
    )