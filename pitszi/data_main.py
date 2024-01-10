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
import copy

from pitszi import utils


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
                 mask=None,
                 psf_fwhm=18*u.arcsec,
                 jackknife=None,
                 raw_rms=None,
                 silent=False,
    ):
        """
        Initialize the data object.
        
        Parameters
        ----------
        - image (2d np.array): data
        - header (str): header of the data
        
        """

        kref = np.linspace(0, 1, 1000)*u.arcsec**-1
        
        # Talk to user?
        self.silent = silent
        
        # Image data as a 2D np array, in Compton parameter unit
        self.image = image
        
        # Header of the image data
        self.header = header

        # Mask associated with the data
        if mask is None:
            self.mask = image * 0 + 1
        else:
            self.mask = mask

        # The PSF. Could be a single number, a function, a dictionary with {k, PSF}
        self.psf_fwhm = psf_fwhm

        # Transfer function, i.e. filtering as a function of k
        self.transfer_function = {'k':kref,
                                  'TF':kref.value*0 + 1}

        # Description of the noise
        self.noise_jackknife = jackknife
        self.noise_rawrms    = raw_rms
        self.noise_model_pk_center = lambda k_arcmin: (0.05*1e-3/12)**2 + (0.1*1e-3/12)**2 * (k_arcmin*60)**-0.7
        self.noise_model_radial    = lambda r_arcsec: 1 + np.exp((r_arcsec-150)/60)
        

    #==================================================
    # Set the Transfer function to a reference model
    #==================================================

    def set_nika2_reference_tf(self, k=np.linspace(0, 1, 1000)*u.arcsec**-1):
        """
        Set the transfer function to a parameteric NIKA2 baseline TF
        
        Parameters
        ----------
        - k (1d np quantity array ): the wavenumber
        
        """
        
        kfov = 1/(6*u.arcmin)
        TF = 1 - np.exp(-(k/kfov).to_value(''))
        
        self.transfer_function = {'k':k, 'TF':TF}



    #==================================================
    # Generate Mock data
    #==================================================

    def set_image_to_mock(self, model_input, use_model_header=False):
        """
        Set the data image to a mock realization given a model
        
        Parameters
        ----------
        - model (class Model object): the model
        - use_model_header (bool): set to true to replacde data header 
        with the current model header. Otherwise the model header is 
        set to the one of the data.
        
        """

        # copy the model so that the input is not modified
        model = copy.deepcopy(model_input)

        # Match the header as requested
        if use_model_header:
            self.header = model.get_map_header()
        else:
            model.map_header = self.header
            
        # Get the raw image model
        input_model = model.get_sz_map()
        
        # Convolve with instrument response function
        TF = {'k_arcsec':self.transfer_function['k'].to_value('arcsec-1'),
              'TF':self.transfer_function['TF']}
        convolved_model = utils.apply_transfer_function(input_model,
                                                        model.get_map_reso().to_value('arcsec'),
                                                        self.psf_fwhm.to_value('arcsec'),
                                                        TF,
                                                        apps_TF_LS=True, apps_beam=True)
        
        # Add noise realization
        #noise_mc = self.get_noise_monte_carlo()
        image_mock = convolved_model# + noise_mc

        # Set the image and the mask if not correct anymore
        self.image = image_mock
        if (self.mask.shape[0] != self.image.shape[0]) or (self.mask.shape[1] != self.image.shape[1]):
            self.mask = self.image * 0 + 1
            if not self.silent:
                print('The mask is set to 1 everywhere since it did not match the data shape anymore')

        return image_mock

        

    #==================================================
    # Compute noise MC realization
    #==================================================
    
    def get_noise_monte_carlo(center=None, Nmc=1, seed=None):
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
        header = self.header
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





