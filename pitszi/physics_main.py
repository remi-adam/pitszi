"""
This file contain the Physics class. 
It is dedicated to derive physical constraints on the 
cluster given the Pk3d inference.
"""

#==================================================
# Requested imports
#==================================================

import os
import numpy as np
import astropy.units as u
import astropy.constants as cst
import astropy.cosmology
from scipy.ndimage import gaussian_filter
import scipy.stats as stats
from scipy.optimize import brentq
import pprint
import copy
import pickle
import dill

from minot.ClusterTools import cluster_global
from pitszi import title


#==================================================
# Mean and std for lognormal distribution
#==================================================

def stat_from_paramLN(mu, sigma):
    """
    Compute the mean and standard deviation of a variable that follows 
    a lognormal distribution:

    f(X) = 1/(X sigma sqrt(2 pi)) . exp( -(ln X - mu)^2 / (2 sigma^2))
        
    In practice, ln (X == delta P / P + 1) follows a normal distribution, 
    centered on mu=0.

    Parameters
    ----------
    - mu (quantity): gaussian mean parameter
    - sigma (float): gaussian sigma parameter
            
    Outputs
    ----------
    - mean (float): mean of the variable
    - std (float): standard deviation of the variable

    """

    mean = np.exp(mu + sigma**2/2)
    std = ((np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2))**0.5

    return mean, std


#==================================================
# Sigma and mu parameter from mean and std of lognormal variable
#==================================================

def paramLN_from_stat(mean, std):
    """
    Compute the sigma and mu parameter of a lognormal distribution
    given the mean and standard deviation of the parameter X, for which ln X 
    is normally distributed.
        
    Parameters
    ----------
    - mean (float): mean of the varible that follow LN PDF
    - std (float): std of the variable that follow LN PDF
            
    Outputs
    ----------
    - mu (float): mu parameter of the lognormal function
    - sigma (float): sigma parameter of the lognormal function

    """

    sigma = (np.log(1 + std**2 / mean**2))**0.5
    mu    = np.log(mean) - 0.5*sigma**2

    return mu, sigma


#==================================================
# Std of lognormal distribution from sigma parameter
#==================================================

def std_from_sigmaLN(sigma):
    """
    Compute the standard deviation of a variable that follows 
    a lognormal distribution with mu=0:

    f(X) = 1/(X sigma sqrt(2 pi)) . exp( -(ln X)^2 / (2 sigma^2))
        
    In practice, ln (X == delta P / P + 1) follows a normal distribution, 
    centered on mu=0.

    Parameters
    ----------
    - sigma (float): gaussian sigma parameter
            
    Outputs
    ----------
    - std (float): standard deviation of X, followed by the lognormal variable

    """
    return ((np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2))**0.5


#==================================================
# Sigma parameter of lognormal distribution from std
#==================================================

def sigmaLN_from_std(std):
    """
    Compute the sigma parameter of a lognormal distribution with mu=0
    given the standard deviation of the parameter X, for which ln X 
    is normally distributed. 
        
    Parameters
    ----------
    - std (float): gaussian sigma parameter
            
    Outputs
    ----------
    - sigma (float): sigma parameter of the lognormal function

    """
    return np.sqrt(np.log((1+np.sqrt(1+4*std**2))/2))


#==================================================
# Sound speed
#==================================================

def sound_speed(kbT, gamma=5.0/3, mu_g=0.61):
    """
    Compute the sound speed given the temperature.
        
    Parameters
    ----------
    - kbT (quantity): the gas temperature homogeneous to keV. 
    Can be a np array.
    - gamma (float): adiabatic index
    - mu_g (float): mean molecular weight of the gas
            
    Outputs
    ----------
    - c_s (quantity): sound speed homogeneous to km/s

    """

    c_s = np.sqrt((gamma*kbT)/(mu_g*cst.m_p))
    
    return c_s.to('km/s')


#==================================================
# Compute sigma_v
#==================================================

def sigma_v(c_s, M3d):
    """
    Compute sigma_v = sound speed x 3D mach number

    Parameters
    ----------
    - c_s (quantity): sound speed homogeneous to km/s
    - M3d(float): 3D Mach number
            
    Outputs
    ----------
    - sigma_v (quantity): homogeneous to km/s

    """

    sigma_v = c_s * M3d
    
    return sigma_v.to('km/s')


#==================================================
# Turbulent to thermal energy ratio
#==================================================

def Eturb_over_Etherm(M3d, gamma=5/3.0):
    """
    Compute turbulent to thermal energy ratio from Mach number.
    This is also the turbulent to thermal pressure ratio

    Eturb = 1/2 rho_gas sigma_v**2
          = 1/2 rho_gas c_s**2 * M3d**2
          = 1/2 rho_gas gamma P_gas / rho_gas * M3d**2
          = 1/2 gamma (gamma-1) Eth * M3d**2
    ==> Eturb / Eth = 1/2 gamma (gamma-1) M3d**2
    
    Parameters
    ----------
    - M3d(float): 3D Mach number
    - gamma (float): adiabatic index
            
    Outputs
    ----------
    - X (float): turbulent to thermal energy ratio

    """

    X = 0.5 * gamma*(gamma-1) * M3d**2
    
    return X


#==================================================
# Mach number from A(Linj) adiabatic Gaspari & Churazov (2013)
#==================================================

def Apeak_to_M3d_G13(A_kpeak, Linj=500*u.kpc, alpha=0.25, gamma=5/3.0):
    """
    Gaspari and Churazov 2013 A&A 559, A78 (2013)
    Implement the 3D Mach number from the pressure 
    fluctuation power spectrum peak amplitude 
    This uses the case of density perturbations and 
    assumes adiabatic perturbations 
    delta P/P = gamma delta n / n
        
    A_rho = 0.25 M3d (Linj/500)^alpha_H (l/Linj)^alpha_c

    Parameters
    ----------
    - A_kpeak (float): the characteristic amplitude of the power spectrum 
    taken at the peak: sqrt(4 pi k_peak^3 P(k_peak)) ; k_peak = 1/Linj
    - Linj (quantity): injection scale, homogeneous to kpc
    - alpha (float): dependence with Linj (~0.2-0.3), see Gaspari & Churazov (2013)
    - gamma (float): adiabatic index
            
    Outputs
    ----------
    - M3d(float): 3D Mach number

    """

    scale = 1/(0.25*gamma)
    M3d = scale * A_kpeak * (500 / Linj.to_value('kpc'))**alpha
    
    return M3d


#==================================================
# Mach number from \int Pk from Zhuravleva + 2023
#==================================================

def sigma_to_M3d_M20(sigma, gamma=5.0/3):
    """
    Rajsekhar Mohapatra +2020 MNRAS 493, 5838–5853 (2020) 
    Rajsekhar Mohapatra +2021 MNRAS 500, 5072–5087 (2021)
    Rajsekhar Mohapatra +2022 MNRAS 510, 3778–3793 (2022)
    Implement the result from their scaling (Figure 6) for the pressure.

    Parameters
    ----------
    - sigma (float): the square root of the integral of Pk
    sigma^2 = \int 4 pi k^2 Pk dk. sigma is directly the norm of 
    pitszi power spectrum
    - gamma (float): adiabatic index

    Outputs
    ----------
    - M3d(float): 3D Mach number

    """

    # b = 1/3 for solenoidal forcing
    # 
    
    # Convert to what would be the sigma of a lognormal pdf for delta P/P + 1
    sigmaLN = sigmaLN_from_std(sigma)

    # Apply the scaling (eq 22)
    M3d = ((np.exp(sigmaLN**2) - 1) / ((1/3.0)**2 * gamma**2))**(1./4)
    
    return M3d


#==================================================
# Mach number from \int Pk from Zhuravleva + 2023
#==================================================

def sigma_to_M3d_Z23(sigma, state='in-between', ell=True, ret_err=False):
    """
    Zhuravleva + 2023 (MNRAS 520, 5157–5172 (2023))
    Implement the result from their Figure 6 for the pressure.
    See also Zhuravleva+ MNRAS 428, 3274–3287 (2013) 

    Parameters
    ----------
    - sigma (float): the sqaure root of the integral of Pk
    sigma^2 = \int 4 pi k^2 Pk dk. sigma is directly the norm of 
    pitszi power spectrum
    - state (str): 'relaxed', 'in-between', 'unrelaxed' according to 
    the table in Zhuravleva + 2023
    - ell (bool): if true, take the elliptical case from Zhuravleva + 2023,
    otherwise take the spherical case
    - ret_err (bool): set to true to return errors
    
    Outputs
    ----------
    - M3d(float): 3D Mach number
    - M3d_err(float): 3D Mach number error, from simple error propagation

    """

    # Convert to what would be the sigma of a lognormal pdf for delta P/P + 1
    sigmaLN = sigmaLN_from_std(sigma)

    # Apply the definition of Zhuravleva+23
    dxi_xi = 2*np.sqrt(2*np.log(2)) / np.log(10) * sigmaLN

    # Apply the scaling for different cases
    if ell: #----- Elliptical case
        if state == 'relaxed':
            M1d     = dxi_xi / 1.0 
            M1d_err = dxi_xi / 1.0**2 * 0.2
        elif state == 'in-between':
            M1d     = dxi_xi / 1.2
            M1d_err = dxi_xi / 1.2**2 * 0.4
        elif state == 'unrelaxed':        
            M1d     = dxi_xi / 1.5
            M1d_err = dxi_xi / 1.5**2 * 0.5
        else:
            raise ValueError('Only relaxed, in-between and unrelaxed states are available')
        
    else: #----- Spherical case
        if state == 'relaxed':
            M1d     = dxi_xi / 1.4
            M1d_err = dxi_xi / 1.4**2 * 0.3
        elif state == 'in-between':
            M1d     = dxi_xi / 1.4
            M1d_err = dxi_xi / 1.4**2 * 0.3
        elif state == 'unrelaxed':        
            M1d     = dxi_xi / 1.5
            M1d_err = dxi_xi / 1.5**2 * 0.3
        else:
            raise ValueError('Only relaxed, in-between and unrelaxed states are available')

    # Convert to 3d Mach number
    M3d     = np.sqrt(3)*M1d
    M3d_err = np.sqrt(3)*M1d_err

    # Return
    if ret_err:
        return M3d, M3d_err
    else:
        return M3d


#==================================================
# Mach number from \int Pk from Zhuravleva + 2023 Table1
#==================================================

def sigma_rad_to_M3d_Z23(sigma, r_scaled, state='in-between', case='prod', ret_err=False):
    """
    Zhuravleva + 2023 (MNRAS 520, 5157–5172 (2023))
    Implement the result from Tab. 1, in the case both radial dependence
    and fluctuations are considered.
    See also Zhuravleva+ MNRAS 428, 3274–3287 (2013) 

    Parameters
    ----------
    - sigma (float): the sqaure root of the integral of Pk
    sigma^2 = \int 4 pi k^2 Pk dk. sigma is directly the norm of 
    pitszi power spectrum
    - r_scaled (float): radius divided by R500
    - state (str): 'relaxed', 'in-between', 'unrelaxed' according to 
    the table in Zhuravleva + 2023
    - case (str): 'sum' (M1d = alpha + beta r/r500 + gamma dxi/xi) 
    or 'prod' (M1d = alpha (r/r500)^beta (dxi/xi)^gamma)
    - ret_err (bool): set to true to return errors

    Outputs
    ----------
    - M3d (array): 3D Mach number as a function of radius

    """
    
    # Convert to what would be the sigma of a lognormal pdf for delta P/P + 1
    sigmaLN = sigmaLN_from_std(sigma)

    # Apply the definition of Zhuravleva+23
    dxi_xi = 2*np.sqrt(2*np.log(2)) / np.log(10) * sigmaLN

    #----- Define the parameters
    # Case sum, see Tab. 1
    if case == 'sum':
        if state == 'relaxed':
            a, b, c = 0.06, 0.11, 0.35
            rms = 0.04
        elif state == 'in-between':
            a, b, c = 0.13, 0.11, 0.24
            rms = 0.07
        elif state == 'unrelaxed':
            a, b, c =  0.21, 0.14, 0.14
            rms = 0.07
        else:
            raise ValueError('Only relaxed, in-between and unrelaxed states are available')

        M3d     = (a + b*r_scaled + c*dxi_xi) *np.sqrt(3)
        M3d_err = M3d * 0 + rms
        
    # Case prod, see Tab. 1
    elif case == 'prod':
        if state == 'relaxed':
            a, b, c = 0.4 , 0.42, 0.29
            rms = 0.04
        elif state == 'in-between':
            a, b, c = 0.45, 0.26, 0.28
            rms = 0.07
        elif state == 'unrelaxed':
            a, b, c = 0.50, 0.21, 0.27
            rms = 0.07
        else:
            raise ValueError('Only relaxed, in-between and unrelaxed states are available')

        M3d     = a * r_scaled**b * dxi_xi**c * np.sqrt(3)
        M3d_err = M3d * 0 + rms

    else:
        raise ValueError('Only sum and prod are possible cases')

    if ret_err:
        return M3d, M3d_err
    else:
        return M3d

    
#==================================================
# Expected NT pressure profile: Shaw et al (2010)
#==================================================

def pnt_over_ptot_r_S10(r, R500, z):
    """
    Implement the nonthermal to thermal pressure from 
    L. Shaw + 2010 (The Astrophysical Journal, 725:1452–1465, 2010 December 20)
    See their Eq 16

    Parameters
    ----------
    - r (quantity): radius homogeneous to kpc
    - R500 (quantity): R500 homogeneous to kpc
    - z (float): cluster redshift

    Outputs
    ----------
    - Pnt_Ptot (array): the ratio between nonthermal to total pressure as a function of radius

    """

    r_scale500 = r.to_value('kpc') / R500.to_value('kpc')
    
    alpha_0 = 0.18 # pm 0.06
    n_nt    = 0.8  # pm 0.25
    beta    = 0.5
    
    fmax = 4**(-n_nt)/alpha_0
    comp1 = (1+z)**beta
    comp2 = (fmax-1)*np.tanh(beta*z)+1
    f_z = np.amin(np.array([comp1, comp2]))
    alpha_z = alpha_0 * f_z
    
    Pnt_Ptot = alpha_z * r_scale500**n_nt
    
    return Pnt_Ptot


#==================================================
# Expected NT pressure profile: Battaglia et al (2011)
#==================================================

def pnt_over_ptot_r_B11(r, R500, z, c500=3.0, cosmo=astropy.cosmology.Planck15):
    """
    Implement the nonthermal to thermal pressure from 
    N. Battaglia + 2012 (The Astrophysical Journal, 758:74 (23pp), 2012 October 20)
    See their Eq 16

    Parameters
    ----------
    - r (quantity): radius homogeneous to kpc
    - R500 (quantity): R500 homogeneous to kpc
    - z (float): cluster redshift
    - c500 (float): NFW concentration parameter
    - cosmo (astropy.cosmology): cosmology object to use

    Outputs
    ----------
    - Pnt_Ptot (array): the ratio between nonthermal to total pressure as a function of radius

    """

    # convert R500 to get M200
    M500 = cluster_global.Rdelta_to_Mdelta(R500.to_value('kpc'), z, delta=500, cosmo=cosmo)
    M200 = cluster_global.Mdelta1_to_Mdelta2_NFW(M500, delta1=500, delta2=200, c1=c500, redshift=z, cosmo=cosmo)

    # Apply functional form
    r_scale500 = r.to_value('kpc') / R500.to_value('kpc')

    alpha_0 = 0.18
    n_nt    = 0.8
    beta    = 0.5
    alpha_z = alpha_0 * (1+z)**beta
    Pnt_Ptot = alpha_z * r_scale500**n_nt * (M200 / 3e14)**(1.0/5)
    
    return Pnt_Ptot


#==================================================
# Expected NT pressure profile: Nelson et al (2014)
#==================================================

def pnt_over_ptot_r_N14(r, R500, z, c500=3.0, cosmo=astropy.cosmology.Planck15):
    """
    Implement the nonthermal to thermal pressure from 
    Nelson +14 (The Astrophysical Journal, 792:25 (8pp), 2014 September)
    See their Eq 7.
    We need to convert from R500 to R200m assuming NFW profile in the process.

    Parameters
    ----------
    - r (quantity): radius homogeneous to kpc
    - R500 (quantity): R500 homogeneous to kpc
    - z (float): cluster redshift
    - c500 (float): NFW concentration parameter
    - cosmo (astropy.cosmology): cosmology object to use

    Outputs
    ----------
    - Pnt_Ptot (array): the ratio between nonthermal to total pressure as a function of radius

    """
    
    #----- Compute R200m
    # Get M500 critical
    M500c = cluster_global.Rdelta_to_Mdelta(R500.to_value('kpc'), z, delta=500, cosmo=cosmo) # Msun

    # NFW profile
    Rs   = R500.to_value('kpc') / c500      # kpc
    rho0 = M500c / (4*np.pi * Rs**3) / (np.log(1 + c500) - c500/(1+c500)) # Msun/kpc^3
    
    # Compute R200m
    delta = 200
    delta_rho_c = delta * cosmo.critical_density(z).to_value('Msun kpc-3')
    delta_rho_m = delta_rho_c * (cosmo.Om0*(1+z)**3) / (cosmo.Om0*(1+z)**3 + cosmo.Ode0)
    def enclosed_mass_difference(radius):
        mass_nfw    = (4*np.pi * Rs**3) * rho0 * (np.log(1 + radius/Rs) - radius/(radius+Rs))
        vol = 4.0/3*np.pi*radius**3
        return  mass_nfw/vol - delta_rho_m
    
    R200m = brentq(enclosed_mass_difference, R500.to_value('kpc')*1e-5, R500.to_value('kpc')*1e5)
    r_scale = r.to_value('kpc') / R200m
    
    #----- Apply the functional form
    A = 0.452
    B = 0.841
    gamma = 1.628
    Pnt_Ptot = 1 - A * (1+ np.exp(-(r_scale/B)**gamma))
    
    return Pnt_Ptot


#==================================================
# Inference class
#==================================================

class Physics():
    """ Physics class
    This class derives physical constraints from 
    Pk3d sampling.

    Attributes
    ----------
        - something: contain the sampling information

        # Admin
        - silent (bool): set to False to give information
        - output_dir (str): the output directory
        
    Methods
    ----------
    - print_param: print the current inference parameters
    """    

    #==================================================
    # Initialization
    #==================================================

    def __init__(self,
                 something,
                 silent=False,
                 output_dir='./pitszi_output',
    ):
        """
        Initialize the inference object. 
        All parameters can be changed on the fly.

        Parameters
        ----------
        - something: the sampling information
 
        - silent (bool): set to False for printing information
        - output_dir (str): directory where outputs are saved
        
        """

        if not silent:
            title.show_physics()
        
        #----- Somthing that contains sampling information
        self.something = something

        #----- Admin
        self.silent     = silent
        self.output_dir = output_dir

        
    #==================================================
    # Print parameters
    #==================================================
    
    def print_param(self):
        """
        Print the current parameters describing the inference.
        
        Parameters
        ----------
        
        Outputs
        ----------
        The parameters are printed in the terminal
        
        """

        print('=====================================================')
        print('============= Current Inference() state =============')
        print('=====================================================')
        
        pp = pprint.PrettyPrinter(indent=4)
        
        par = self.__dict__
        keys = list(par.keys())
        
        for k in range(len(keys)):
            print(('--- '+(keys[k])))
            print(('    '+str(par[keys[k]])))
            print(('    '+str(type(par[keys[k]]))+''))
        print('=====================================================')


    #==================================================
    # Save parameters
    #==================================================
    
    def save_physics(self):
        """
        Save the current physics object.
        
        Parameters
        ----------
            
        Outputs
        ----------
        The parameters are saved in the output directory

        """

        # Create the output directory if needed
        if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)

        # Save
        with open(self.output_dir+'/physics_parameters.pkl', 'wb') as pfile:
            #pickle.dump(self.__dict__, pfile, pickle.HIGHEST_PROTOCOL)
            dill.dump(self.__dict__, pfile)

        # Text file for user
        par = self.__dict__
        keys = list(par.keys())
        with open(self.output_dir+'/physics_parameters.txt', 'w') as txtfile:
            for k in range(len(keys)):
                txtfile.write('--- '+(keys[k])+'\n')
                txtfile.write('    '+str(par[keys[k]])+'\n')
                txtfile.write('    '+str(type(par[keys[k]]))+'\n')

                
    #==================================================
    # Load parameters
    #==================================================
    
    def load_physics(self, param_file):
        """
        Read the a given parameter file to re-initialize the physics object.
        
        Parameters
        ----------
        param_file (str): the parameter file to be read
            
        Outputs
        ----------
            
        """

        with open(param_file, 'rb') as pfile:
            par = dill.load(pfile)
            
        self.__dict__ = par
        
