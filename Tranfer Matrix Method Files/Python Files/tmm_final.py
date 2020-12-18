import numpy as np
from numpy.lib.scimath import arcsin

import sys

EPSILON = sys.float_info.epsilon

# A simple function for determining whether a wave is propagating forward

def theta_forward(n, theta):
       
    assert n.real >= 0
    assert n.imag >= 0
    
    n_cos_theta = n * np.cos(theta)    
    if np.absolute(n_cos_theta.imag) > 100 * EPSILON:
        value = (n_cos_theta.imag > 0)
    else:
        value = (n_cos_theta.real > 0)
   
    return value

# Snell's law as given in Eq.(3.10)

def snells_law(n_1, n_2, theta_1):

    theta_2 = arcsin(n_1 * np.sin(theta_1) / n_2)
    if theta_forward(n_2, theta_2):
        return theta_2
    else:
        return np.pi - theta_2

# Create a list of incident and refracted angles for all layers
    
def snells_law_list(n_list, theta_incident):
   
    theta_refracted_list = arcsin(n_list[0] * np.sin(theta_incident) / n_list)
    if not theta_forward(n_list[0], theta_refracted_list[0]):
        theta_refracted_list[0] = np.pi - theta_refracted_list[0]
    if not theta_forward(n_list[-1], theta_refracted_list[-1]):
        theta_refracted_list[-1] = np.pi - theta_refracted_list[-1]
    return theta_refracted_list

# Reflection amplitude as given in Eq.(3.22) and Eq.(3.24)

def reflection_amplitude(polarization, n_i, n_f, theta_i, theta_f):

    if polarization == 's':
        return ((n_i * np.cos(theta_i) - n_f * np.cos(theta_f)) /
                (n_i * np.cos(theta_i) + n_f * np.cos(theta_f)))
    elif polarization == 'p':
        return ((n_f * np.cos(theta_i) - n_i * np.cos(theta_f)) /
                (n_f * np.cos(theta_i) + n_i * np.cos(theta_f)))
    else:
        raise ValueError("Polarization must be 's' or 'p'")
        
def amplitude_r(polarization, n_i, n_f, theta_i, theta_f):

    r = reflection_amplitude(polarization, n_i, n_f, theta_i, theta_f)
    return reflected_power(r)

# Tranmission amplitude as given in Eq.(3.23) and Eq.(3.25)

def transmission_amplitude(polarization, n_i, n_f, theta_i, theta_f):

    if polarization == 's':
        return ((2 * n_i * np.cos(theta_i)) / 
                (n_i * np.cos(theta_i) + n_f * np.cos(theta_f)))
    elif polarization == 'p':
        return ((2 * n_i * np.cos(theta_i)) / 
                (n_f * np.cos(theta_i) + n_i * np.cos(theta_f)))
    else:
        raise ValueError("Polarization must be 's' or 'p'")
        
def amplitude_t(polarization, n_i, n_f, theta_i, theta_f):
  
    t = transmission_amplitude(polarization, n_i, n_f, theta_i, theta_f)
    return transmitted_power(polarization, t, n_i, n_f, theta_i, theta_f)

# Reflection and Transmission powers obtained from Eqs.(4.12 - 4.14)

def reflected_power(r):

    return np.absolute(r)**2

def transmitted_power(polarization, t, n_i, n_f, theta_i, theta_f):
  
    if polarization == 's':
        return np.absolute(t**2) * (((n_f * np.cos(theta_f)).real) / (n_i * np.cos(theta_i)).real)
    elif polarization == 'p':
        return np.absolute(t**2) * (((n_f * np.conj(np.cos(theta_f))).real) / (n_i * np.conj(np.cos(theta_i))).real)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

# Template for our transfer matrices        
        
def matrix_array(a, b, c, d, dtype=float):
    
    template_array = np.empty((2,2), dtype=dtype)
    template_array[0,0] = a
    template_array[0,1] = b
    template_array[1,0] = c
    template_array[1,1] = d
    return template_array

# The main function of this program; it takes the previous functions as inputs and returns a TMM

def transfer_matrix_method(polarization, n_list, d_list, theta_incident, wavelength):

# A series of input tests    
    
    n_list = np.array(n_list)
    
    d_list = np.array(d_list, dtype=float) 
  
    if ((hasattr(wavelength, 'size') and wavelength.size > 1)
          or (hasattr(theta_incident, 'size') and theta_incident.size > 1)):
        raise ValueError("Only 1 wavelenght and 1 incident angle can be entered at a time")
    if ((n_list.ndim != 1) 
          or (d_list.ndim != 1) 
          or (n_list.size != d_list.size)):
        raise ValueError("Problem with n_list or d_list!")
        
    assert d_list[0] == d_list[-1] == np.inf
    assert np.absolute((n_list[0] * np.sin(theta_incident)).imag) < 100*EPSILON
    assert theta_forward(n_list[0], theta_incident), 'Are the angles in radians?'
    
    num_layers = n_list.size

    theta_list = snells_law_list(n_list, theta_incident)

    k_list = 2 * np.pi * n_list * np.cos(theta_list) / wavelength

    phase_angle = k_list * d_list
    
# Define arrays for the reflection and transmission amplitudes

    r_list = np.zeros((num_layers, num_layers), dtype=complex)
    t_list = np.zeros((num_layers, num_layers), dtype=complex) 
    for i in range(num_layers-1):
        r_list[i,i+1] = reflection_amplitude(polarization, n_list[i], n_list[i+1],
                                    theta_list[i], theta_list[i+1])
        t_list[i,i+1] = transmission_amplitude(polarization, n_list[i], n_list[i+1],
                                    theta_list[i], theta_list[i+1])

# The transfer matrix as defned in Eq.(4.5)        
        
    Transfer1_list = np.zeros((num_layers, 2, 2), dtype=complex)
    for i in range(1, num_layers-1):
        Transfer1_list[i] = (1/t_list[i,i+1]) * np.dot(
            matrix_array(np.exp(-1j * phase_angle[i]), 0, 0, np.exp(1j * phase_angle[i]),
                           dtype=complex),
            matrix_array(1, r_list[i,i+1], r_list[i,i+1], 1, dtype=complex))

# The transfer matrix as defined in Eq.(4.9)
        
    Transfer2_list = matrix_array(1, 0, 0, 1, dtype=complex)
    for i in range(1, num_layers-1):
        Transfer2_list = np.dot(Transfer2_list, Transfer1_list[i])
    
    Transfer2_list = np.dot(matrix_array(1, r_list[0,1], r_list[0,1], 1,
                                   dtype=complex)/t_list[0,1], Transfer2_list)

    r = Transfer2_list[1,0]/Transfer2_list[0,0]
    t = 1/Transfer2_list[0,0]

    R = reflected_power(r)
    T = transmitted_power(polarization, t, n_list[0], n_list[-1], theta_incident, theta_list[-1])

    return {'r': r, 't': t, 'R': R, 'T': T,
            'k_list': k_list, 'theta_list': theta_list,
            'polarization': polarization, 'n_list': n_list, 'd_list': d_list, 'theta_incident': theta_incident,
            'wavelenght': wavelength}

# A simple function for an unpolarized beam of light

def unpolarized(n_list, d_list, theta_incident, wavelength):

    s_data = transfer_matrix_method('s', n_list, d_list, theta_incident, wavelength)
    p_data = transfer_matrix_method('p', n_list, d_list, theta_incident, wavelength)
    R = (s_data['R'] + p_data['R']) / 2.
    T = (s_data['T'] + p_data['T']) / 2.
    return {'R': R, 'T': T}
