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
import pickle
import pprint

from minot.ClusterTools import map_tools
from pitszi import utils


#==================================================
# Data class
#==================================================

class Data():
    """ Data class.
    This class defines a data object. 

    Attributes
    ----------
    - image (2d array): the data image
    - header (str): the header associated with image
    - mask (2d array): the mask associated with the image data
    - psf_fwhm (quantity): the FWHM of the PSF as a quantity homogeneous to arcsec
    - transfer_function (dict): the transfer function as {'k':k*u.arcsec, 'TF':TF}
    - jackknife (2d array): a representative jacknife map associated with the data
    - noise_covariance (2d array): the covariance matrix associated with the noise
    - silent (bool): set to True to give information

    Methods
    ----------
    - set_nika2_reference_tf: set the transfer fucntion to a NIKA2 reference model
    - get_noise_monte_carlo_from_model: compute noise monte carlo given noise model
    - set_image_to_mock: set the data to a mock realization given the model

    ToDo
    ----------
    - compute noise monte carlo from covariance matrix
    - extract noise model from jackknife map

    """

    #==================================================
    # Initialize the data object
    #==================================================

    def __init__(self,
                 image,
                 header,
                 psf_fwhm=0*u.arcsec,
                 transfer_function=None,
                 mask=None,
                 noise_jackknife=None,
                 noise_rms=None,
                 noise_covmat=None,
                 noise_model_pk_center=None,
                 noise_model_radial=None,
                 silent=False,
                 output_dir='./pitszi_output',
    ):
        """
        Initialize the data object.
        
        Parameters
        ----------
        - image (2d np.array): data
        - header (str): header of the data
        - psf_fwhm=0*u.arcsec,
        - transfer_function=None,
        - mask (2d np.array): mask associated with image
        - noise_jackknife (2d np.array): jacknife map associated with image
        - noise_rms (2d np.array): rms associated with the image
        - noise_covmat (2d np.array): noise covariance matrix associated with flattened image
        - noise_model_pk_center (function): function that give the noise Pk as a function of k in arcsec^-1
        - noise_model_radial (function):  function that give the noise amplitude as a function of radial
        distance from the center r in arcsec (make sense to have 1 in the center)
        - silent (bool)
        - output_dir (str): directory where saving outputs

        """

        kref = np.linspace(0, 1, 1000)*u.arcsec**-1
        
        # Admin
        self.silent     = True
        self.output_dir = output_dir
        
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
        if transfer_function is None:
            self.transfer_function = {'k':kref,
                                      'TF':kref.value*0 + 1}
        else:
            self.transfer_function = transfer_function

        # Description of the noise
        self.noise_jackknife       = noise_jackknife
        self.noise_rms             = noise_rms
        self.noise_covmat          = noise_covmat
        self.noise_model_pk_center = noise_model_pk_center # output in [y] x arcsec^2
        self.noise_model_radial    = noise_model_radial    # output unitless


    #==================================================
    # Print parameters
    #==================================================
    
    def print_param(self):
        """
        Print the current parameters describing the data.
        
        Parameters
        ----------
        
        Outputs
        ----------
        The parameters are printed in the terminal
        
        """

        print('=====================================================')
        print('=============== Current Data() state ================')
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
    # Set reference noise model
    #==================================================

    def set_nika2_reference_noise_model(self):
        """
        Set the noise model according to a typical baseline model.
        
        Parameters
        ----------
        
        """

        # output in [y] x arcsec^2
        self.noise_model_pk_center = lambda k_arcsec: 5e-9 + 15e-9 * (k_arcsec*60)**-1

        # output unitless
        self.noise_model_radial    = lambda r_arcsec: 1 + np.exp((r_arcsec-200)/80)    
        
        
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
        transfer_function = {'k':k, 'TF':TF}
        self.transfer_function = transfer_function

        return transfer_function
    
    
    #==================================================
    # Compute noise MC realization from a noise model
    #==================================================
    
    def get_noise_monte_carlo_from_model(self,
                                         center=None,
                                         Nmc=1,
                                         seed=None):
        """
        Compute a noise MC realization given a noise Pk
        and a radial dependence
        
        Parameters
        ----------
        - center (SkyCoord): the reference center for the noise
        - Nmc (int): the number of monte carlo
        - seed (int): the numpy random seed for reproducible results
    
        Outputs
        ----------
        - noise (nd array): the noise monte carlo cube

        """

        # Check that the covariance exists
        if self.noise_model_pk_center is None or self.noise_model_radial is None:
            if not self.silent:
                print('The noise model (noise_model_pk_center, noise_model_radial) is undefined')
                print('while it is requested for get_noise_monte_carlo_from_model')
            return
        
        # Set a seed
        np.random.seed(seed)

        # Grid info
        header = self.header
        Nx = header['Naxis1']
        Ny = header['Naxis2']
        reso_arcsec = np.abs(header['CDELT2'])*3600
    
        # Define the wavevector
        k_x = np.fft.fftfreq(Nx, reso_arcsec) # 1/arcsec
        k_y = np.fft.fftfreq(Ny, reso_arcsec)
        k2d_x, k2d_y = np.meshgrid(k_x, k_y, indexing='ij')
        k2d_norm = np.sqrt(k2d_x**2 + k2d_y**2)
        k2d_norm_flat = k2d_norm.flatten()
    
        # Compute Pk noise
        P2d_k_grid = self.noise_model_pk_center(k2d_norm)
        P2d_k_grid[k2d_norm == 0] = 0

        amplitude =  np.sqrt(P2d_k_grid / reso_arcsec**2)
        
        noise = np.random.normal(0, 1, size=(Nmc, Nx, Ny))
        noise = np.fft.fftn(noise, axes=(1,2))
        noise = np.real(np.fft.ifftn(noise * amplitude, axes=(1,2)))

    
        # Account for a radial dependence
        ramap, decmap = map_tools.get_radec_map(header)
        if center is None:
            RA0 = np.mean(ramap)
            Dec0 = np.mean(decmap)
            center = SkyCoord(RA0*u.deg, Dec0*u.deg, frame="icrs")
        
        dist_map = map_tools.greatcircle(ramap, decmap,
                                         center.icrs.ra.to_value('deg'), center.icrs.dec.to_value('deg'))
        noise_radial_dep = self.noise_model_radial(dist_map*3600)
        noise = noise * np.transpose(np.dstack([noise_radial_dep]*Nmc), axes=(2,0,1))

        # Remove first axis if Nmc is one
        if Nmc == 1:
            noise = noise[0]
        
        return noise


    #==================================================
    # Compute noise MC realization from the covariance
    #==================================================
    
    def get_noise_monte_carlo_from_covariance(self,
                                              Nmc=1,
                                              seed=None):
        """
        Compute a noise MC realization from the noise covariance
        matrix.
        
        Parameters
        ----------
        - Nmc (int): the number of map to compute in the cube
        - seed (int): the 
        
        Outputs
        ----------
        - noise (nd array): noise MC cube
        """

        # Check that the covariance exists
        if self.noise_covmat is None:
            if not self.silent:
                print('The noise covariance matrix is undefined')
                print('while it is requested for get_noise_monte_carlo_from_covariance')
            return

        # Set a seed
        np.random.seed(seed)

        # Get the flat noise realization
        target_shape = self.image.shape
        noise_mc_flat = np.random.multivariate_normal(np.zeros(self.noise_covmat.shape[0]),
                                                      self.noise_covmat,
                                                      size=Nmc)
        # Reshape it to maps
        if Nmc == 1:
            noise_mc = noise_mc_flat.reshape(target_shape[0], target_shape[1])
        else:
            noise_mc = noise_mc_flat.reshape(Nmc, target_shape[0], target_shape[1])

        return noise_mc

    
    #==================================================
    # Set the noise covariance from MC realizations
    #==================================================

    def set_noise_covariance_from_model(self,
                                        center=None,
                                        Nmc=1000,
                                        seed=None):
        """
        Set the noise covariance from the noise model
        
        Parameters
        ----------
        - center (SkyCoord): the reference center for the noise
        - Nmc (int): the number of monte carlo
        - seed (int): the numpy random seed for reproducible results

        Outputs
        ----------
        - covmat (nd array): the noise covariance matrix

        """

        if not self.silent:
            print('Start computing the covariance matrix from MC simulations')
            print('This can take significant time')
            
        noise_mc = self.get_noise_monte_carlo_from_model(center=center,
                                                         Nmc=Nmc,
                                                         seed=seed)
        covmat = 0
        for i in range(Nmc):
            mflat = noise_mc[i,:,:].flatten()
            covmat += np.matmul(mflat[:,None], mflat[None,:])
        covmat /= Nmc

        self.noise_covmat = covmat

        return covmat
    

    #==================================================
    # Save the noise covariance matrix
    #==================================================
    
    def save_noise_covariance(self):
        """
        Save the noise covariance matrix to avoid long
        computation time
        
        Parameters
        ----------
        
        Outputs
        ----------
        - file is saved
        """
        
        with open(self.output_dir+'/data_noise_covariance_matrix.pkl', 'wb') as pfile:
            pickle.dump(self.noise_covmat, pfile, pickle.HIGHEST_PROTOCOL)


    #==================================================
    # Read saved noise covariance matrix
    #==================================================
    
    def load_noise_covariance(self):
        """
        Read the noise covariance matrix
        
        Parameters
        ----------
        
        Outputs
        ----------
        
        """
        
        with open(self.output_dir+'/data_noise_covariance_matrix.pkl', 'rb') as pfile:
            par = pickle.load(pfile)
            
        self.noise_covmat = par
    

    #==================================================
    # Measure noise from JackKnife
    #==================================================
    
    def set_noise_properties_from_jackknife(self):
        """
        Compute a noise MC realization from the noise covariance
        matrix.
        
        Parameters
        ----------
        - Nmc (int): the number of map to compute in the cube
        - seed (int): the 
        
        Outputs
        ----------
        - noise (nd array): noise MC cube
        """


        
    #==================================================
    # Generate Mock data
    #==================================================

    def set_image_to_mock(self,
                          model_input,
                          model_seed=None,
                          model_no_fluctuations=False,
                          use_model_header=False,
                          noise_origin='model',
                          noise_center=None,
                          noise_seed=None):
        """
        Set the data image to a mock realization given a model
        
        Parameters
        ----------
        - model (class Model object): the model
        - model_seed (bool): set to a number for reproducible fluctuations
        - model_no_fluctuations (bool): set to true when the pure spherical model is requested
        - use_model_header (bool): set to true to replacde data header 
        with the current model header. Otherwise the model header is 
        set to the one of the data.
        - noise_origin (str): can be 'covariance', 'model', or 'none'
        - noise_center (SkyCoord): the reference center for the noise
        - model_seed (bool): set to a number for reproducible noise

        """

        # copy the model so that the input is not modified
        model = copy.deepcopy(model_input)

        # Match the header as requested
        if use_model_header:
            self.header = model.get_map_header()
        else:
            model.map_header = self.header
            
        # Get the raw image model
        input_model = model.get_sz_map(seed=model_seed, no_fluctuations=model_no_fluctuations)
        
        # Convolve with instrument response function
        convolved_model = utils.apply_transfer_function(input_model,
                                                        model.get_map_reso().to_value('arcsec'),
                                                        self.psf_fwhm.to_value('arcsec'),
                                                        self.transfer_function,
                                                        apps_TF_LS=True, apps_beam=True)
        
        # Add noise realization
        if noise_origin == 'model':
            noise_mc = self.get_noise_monte_carlo_from_model(center=noise_center, Nmc=1, seed=noise_seed)
        elif noise_origin == 'covariance':
            noise_mc = self.get_noise_monte_carlo_from_covariance()
        elif noise_origin == 'none':
            noise_mc = 0
        else:
            raise ValueError('noise_origin should be "model", "covariance", or "none"')
        image_mock = convolved_model + noise_mc

        # Set the image and the mask if not correct anymore
        self.image = image_mock
        if (self.mask.shape[0] != self.image.shape[0]) or (self.mask.shape[1] != self.image.shape[1]):
            self.mask = self.image * 0 + 1
            if not self.silent:
                print('A new mask is set with 1 everywhere since the previous one did not match the data shape anymore')

        return image_mock
        
