"""
This file contains the Data class. It is dedicated to the construction of a
Data object, definined by the observational properties of the data.
"""

#==================================================
# Requested imports
#==================================================

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import constants as const
from astropy.wcs import WCS

#==================================================
# Cluster Model class
#==================================================

class Data():
    """ Data class.
    This class defines a data object. 

    Attributes
    ----------
    - image

    ToDo
    ----------  
    
    """

    #==================================================
    # Initialize the cluster object
    #==================================================

    def __init__(self,
                 image,
                 header,
                 instrument='NIKA2',
    ):
        """
        Initialize the data object.
        
        Parameters
        ----------
        - image (2d np.array): data
        - header (str): header of the data
        
        """
        
        # The name of the instrument (NIKA, NIKA2, MUSTANG, Planck, etc)
        self.instrument = instrument
        
        # Image data as a 2D np array, in Compton parameter unit
        self.image = image # ==> Here we should have the possibility of having 2 maps for cross spectra
        
        # Header of the image data
        self.header = header

        # Mask associated with the data
        self.mask = None

        # Noise realization of the data
        self.noise_rms = None
        self.noise_pk  = None
        self.noise_MC  = None

        # The PSF. Could be a single number, a function, a dictionary with {k, PSF}
        self.psf_fwhm = 18*u.arcsec

        # Transfer function, i.e. filtering as a function of k
        self.transfer_function = {'k':np.linspace(0, 1, 1000)*u.arcsec**-1,
                                  'TF':np.zeros(1000) + 1}


    #==================================================
    # Initialize the cluster object
    #==================================================

    def set_nika2_reference_tf(self, k=np.linspace(0, 1, 1000)*u.arcsec**-1):
        """
        Set the transfer function to a parameteric NIKA2 baseline TF
        
        Parameters
        ----------
        - k (1d np quantity array ): the wavenumber
        
        """
        
        kfov = 1/(6*u.arcmin)
        TF = 1 - np.exp(-(karcsec/kfov).to_value(''))
        
        self.transfer_function = {'k':k, 'TF':TF}

    
    #==================================================
    # Generate Monte Carlo noise cube
    #==================================================

    def set_noise_monte_carlo(self, Nmc=100):
        """
        Set the noise Monte Carlo realization according to the noise Pk
        
        Parameters
        ----------
        - Nmc (int): the number of Monte Carlo realizations
        
        """

        Nx, Ny = self.image.shape
        noise_cube = np.zeros((Nmc, Nx, Ny))

        self.noise_mc = noise_cube



        
# Noise
def noise_k(k_invkpc): return (0.05*1e-3/12)**2 + (0.1*1e-3/12)**2 * (k_invkpc*60)**-0.7
def noise_r(r_arcsec): return 1 + np.exp((r_arcsec-150)/60)




    

#==================================================
# Compute noise MC realization
#==================================================

def make_noise_mc(header, center, noise_k, noise_r, seed=None):
    """
    Compute a noise MC realization given a noise Pk
    and a radial dependence
    
    Parameters
    ----------
    - header (str): the maps header
    - center (SkyCoord): the reference center for the noise
    - noise_k (function): a function that takes the wavnumber in 1/arcsec
    as input and returns the noise Pk
    - noise_r (function): a function that takes the radius in arcsec
    as input and returns the noise amplification

    Outputs
    ----------
    - G_k (np array): the power spectrum of a gaussian
    """

    # Set a seed
    np.random.seed(seed)
        
    # Grid info    
    Nx = header['Naxis1']
    Ny = header['Naxis2']
    reso_arcsec = np.abs(header['CDELT2'])*3600

    # Define the wavevector
    k_x = np.fft.fftfreq(Nx, reso_arcsec) # 1/kpc
    k_y = np.fft.fftfreq(Ny, reso_arcsec)
    k2d_x, k2d_y = np.meshgrid(k_x, k_y, indexing='ij')
    k2d_norm = np.sqrt(k2d_x**2 + k2d_y**2)
    k2d_norm_flat = k2d_norm.flatten()

    # Compute Pk noise
    P2d_k_grid = noise_k(k2d_norm)
    P2d_k_grid[k2d_norm == 0] = 0

    noise = np.random.normal(0, 1, size=(Nx, Ny))
    noise = np.fft.fftn(noise)
    noise = np.real(np.fft.ifftn(noise * np.sqrt(P2d_k_grid)))

    # Account for a radial dependence
    ramap, decmap = map_tools.get_radec_map(header)
    dist_map = map_tools.greatcircle(ramap, decmap,
                                     center.icrs.ra.to_value('deg'), center.icrs.dec.to_value('deg'))
    noise = noise * noise_r((dist_map*u.deg).to_value('arcsec'))

    return noise





    #==================================================
    # Generate Mock data
    #==================================================

