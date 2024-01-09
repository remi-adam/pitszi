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
    ):
        """
        Initialize the data object.
        
        Parameters
        ----------
        - image (2d np.array): data
        - header (str): header of the data
        
        """
        
        # The name of the instrument (NIKA, NIKA2, MUSTANG, Planck, etc)
        self.instrument = 'NIKA2'
        
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
        self.transfer_function = None
        self.set_nika2_basline_tf()


    #==================================================
    # Initialize the cluster object
    #==================================================

    def set_nika2_basline_tf(self, k=np.linspace(0, 1, 1000)*u.arcsec**-1):
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


    #==================================================
    # Generate Mock data
    #==================================================

    
