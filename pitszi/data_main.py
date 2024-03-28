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
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import copy
import pickle
import pprint

from minot.ClusterTools import map_tools
from pitszi import utils_pk


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
    - noise_rms (2d np.array): rms associated with the image
    - noise_covmat (2d np.array): noise covariance matrix associated with flattened image
    - noise_model (list of 2 element): 
        b) returns the noise amplitude as a function of position. This can be a map, or a 
           function that depends on radius from the center in arcsec.
        a) function that give the noise power spectrum, once normalized by its amplitude,
           as a function of k in arcsec^-1
    - noise_mc (3d np array): Monte Carlo noise realizations. Shape = (Nmc, Nx, Ny)
    - silent (bool): set to True to give information
    - output_dir (str): directory where saving outputs

    Methods
    ----------
    - print_param: print the current Data parameters
    - set_nika2_reference_noise_model: set the noise model to a baseline
    - set_nika2_reference_tf: set the transfer fucntion to a NIKA2 reference model
    - get_noise_monte_carlo_from_model: compute noise monte carlo given the noise model
    - get_noise_monte_carlo_from_covariance: compute noise MC realization from the covariance matrix
    - get_noise_covariance_from_model: compute the covariance given a model
    - get_noise_covariance_from_noise_mc: compute the covariance given a input noise MC
    - save_noise_covariance: save the covariance matrix
    - load_noise_covariance: load a pre-existing covariance matrix
    - set_noise_model_from_jackknife: use a jackknife to compute fir the noise model
    - set_noise_model_from_mc: use noise MC to define a noise model
    - set_noise_model_from_covariance: use noise covariance matrix to define a noise model
    - get_noise_rms_from_mc: get the noise rms given the noise MC realizations
    - get_noise_rms_from_covariance: get the noise rms given the noise covariance matrix
    - get_noise_rms_from_model: get the noise rms given the noise model
    - set_image_to_mock: set the data to a mock realization given a model from Model()

    ToDo
    ----------
    - Add functions to degrade / reproject the data while dealing 
    with power spectrum properties
    - Add checks of the data format, consistency, type etc (e.g. header agree with data shape, 
    loaded covriance agrees with data shape, etc)

    """

    #==================================================
    # Initialize the data object
    #==================================================

    def __init__(self,
                 image,
                 header,
                 mask=None,
                 psf_fwhm=0*u.arcsec,
                 transfer_function=None,
                 noise_rms=None,
                 noise_covmat=None,
                 noise_model=None,
                 noise_mc=None,
                 silent=False,
                 output_dir='./pitszi_output',
    ):
        """
        Initialize the data object.
        
        Parameters
        ----------
        - image (2d np.array): data
        - header (str): header of the data
        - mask (2d np.array): mask associated with image
        - psf_fwhm (quantity): the FWHM of the PSF as a quantity homogeneous to arcsec
        - transfer_function (dict): the transfer function as {'k':k*u.arcsec, 'TF':TF}
        - noise_rms (2d np.array): rms associated with the image
        - noise_covmat (2d np.array): noise covariance matrix associated with flattened image
        - noise_model (list of 2 element): 
          b) returns the noise amplitude as a function of position. This can be a map, or a 
             function that depends on radius from the center in arcsec.
          a) function that give the noise power spectrum, once normalized by its amplitude,
             as a function of k in arcsec^-1
        - noise_mc (3d np array): Monte Carlo noise realizations
        - silent (bool): set to true to avoid printing information
        - output_dir (str): directory where saving outputs

        """
        
        kref = np.linspace(0, 1, 1000)*u.arcsec**-1
        
        # Image data as a 2D np array, in Compton parameter unit
        self.image = image
        
        # Header of the image data
        self.header = header

        # Mask associated with the data
        if mask is None:
            self.mask = image * 0 + 1
        else:
            self.mask = mask

        # The PSF
        self.psf_fwhm = psf_fwhm

        # Transfer function, i.e. filtering as a function of k
        if transfer_function is None:
            self.transfer_function = {'k':kref,
                                      'TF':kref.value*0 + 1}
        else:
            self.transfer_function = transfer_function

        # Description of the noise
        self.noise_rms    = noise_rms
        self.noise_covmat = noise_covmat
        self.noise_model  = noise_model
        self.noise_mc     = noise_mc

        # Admin
        self.silent     = silent
        self.output_dir = output_dir

        
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
        
        # output unitless
        self.noise_model[0] = lambda r_arcsec: 1 + np.exp((r_arcsec-200)/80)
        
        # output in [y] x arcsec^2
        self.noise_model[1] = lambda k_arcsec: 5e-9 + 15e-9 * (k_arcsec*60)**-1
        
        
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

        # Check that the model was filled
        if type(self.noise_model) != list:
            if not self.silent:
                print('The noise model [noise_model_radial, noise_model_pk] should be a list')
                print('Cannot run get_noise_monte_carlo_from_model.')
            return

        #---------- Dependence in terms of k
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
        P2d_k_grid = self.noise_model[1](k2d_norm)

        # Clean Pk
        wneg = P2d_k_grid < 0
        P2d_k_grid[wneg] = 0   # Negative Pk can happen for interpolation of poorly defined models
        if np.sum(wneg)>0 and not self.silent:
            print('----- The Pk noise model function implies some Pk<0. These points are set to 0.')
        P2d_k_grid[k2d_norm == 0] = 0

        # Generate the normalized noise
        amplitude =  np.sqrt(P2d_k_grid / reso_arcsec**2)
        noise = np.random.normal(0, 1, size=(Nmc, Nx, Ny))
        noise = np.fft.fftn(noise, axes=(1,2))
        noise = np.real(np.fft.ifftn(noise * amplitude, axes=(1,2)))

        #---------- Account for a radial dependence
        # Case of a function of radius
        if type(self.noise_model[0]) == type(lambda x: 1):
            ramap, decmap = map_tools.get_radec_map(header)
            if center is None:
                RA0 = np.mean(ramap)
                Dec0 = np.mean(decmap)
                center = SkyCoord(RA0*u.deg, Dec0*u.deg, frame="icrs")
            
            dist_map = map_tools.greatcircle(ramap, decmap,
                                             center.icrs.ra.to_value('deg'), center.icrs.dec.to_value('deg'))
            noise_radial_dep = self.noise_model[0](dist_map*3600)
            noise = noise * np.transpose(np.dstack([noise_radial_dep]*Nmc), axes=(2,0,1))

        # Case of a map
        if type(self.noise_model[0]) == type(self.image):
            if self.noise_model[0].shape != (Nx, Ny):
                raise ValueError('The radial dependence of the model (a map) does not match the data shape')
            noise = noise * np.transpose(np.dstack([self.noise_model[0]]*Nmc), axes=(2,0,1))
        
        #---------- Remove first axis if Nmc is one
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
            raise ValueError('The noise covariance matrix is undefined, but needed here')

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
    # Get the noise covariance from model MC realizations
    #==================================================

    def get_noise_covariance_from_model(self,
                                        center=None,
                                        Nmc=1000,
                                        seed=None):
        """
        Get the noise covariance from the noise model
        
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

        return covmat


    #==================================================
    # Get the noise covariance from user MC realizations
    #==================================================

    def get_noise_covariance_from_noise_mc(self):
        """
        Get the noise covariance from the noise MC realizations
        
        Parameters
        ----------

        Outputs
        ----------
        - covmat (nd array): the noise covariance matrix

        """

        if self.noise_mc is None:
            raise ValueError('The noise_mc was not definied, but it is required for getting the covariance here')
        
        if not self.silent:
            print('Start computing the covariance matrix from MC simulations')
            print('This can take significant time')
            
        noise_mc = self.noise_mc
        Nmc = noise_mc.shape[0]
        
        covmat = 0
        for i in range(Nmc):
            mflat = noise_mc[i,:,:].flatten()
            covmat += np.matmul(mflat[:,None], mflat[None,:])
        covmat /= Nmc

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
            cov = pickle.load(pfile)
            
        self.noise_covmat = cov
    

    #==================================================
    # Measure noise from JackKnife
    #==================================================
    
    def set_noise_model_from_jackknife(self,
                                       jkmap,
                                       normmap,
                                       reso,
                                       Nbin=30,
                                       scale='log'):
        """
        Derive the noise model from a jacknife interpolating 
        the power spectrum of the homogeneized image.
        
        Parameters
        ----------
        - jkmap (np array): a jack knife map
        - normmap (np array): the normalization map. jkmap/normmap should have homogeneous noise
        - reso (float): the map resolution in arcsec
        - Nbin (int): number of bin for the Pk
        - scale (str): log or lin, to define the scale used in Pk
        
        Outputs
        ----------
        The noise model is set from the jackknife
        """

        # General info
        Nx, Ny = jkmap.shape
        
        w = (normmap > 0) * ~np.isnan(normmap) * ~np.isinf(normmap)
        if not self.silent:
            if np.sum(~w) > 0:
                print('WARNING: some pixels are bad. This may affect the recovered noise model')
        
        # Extract the normalized map
        img = jkmap/normmap
        img[~w] = 0
        
        # Extract and fit the Pk
        k, pk = utils_pk.extract_pk2d(img, reso, Nbin=Nbin, scalebin=scale)
        w = ~np.isnan(pk)
        spec_mod = interp1d(k[w], pk[w], bounds_error=False, fill_value="extrapolate", kind='linear')
        
        # Fix the model
        self.noise_model = [normmap, lambda k_arcsec: spec_mod(k_arcsec)]
        

    #==================================================
    # Set noise model from MC
    #==================================================
    
    def set_noise_model_from_mc(self,
                                Nbin=30,
                                scale='log'):
        """
        Derive the noise model by interpolating the noise PK from MC realizations
        after homogeneizing the noise.
        
        Parameters
        ----------
        - Nbin (int): number of bin for the Pk
        - scale (str): log or lin, to define the scale used in Pk
        
        Outputs
        ----------
        The noise model is set from the MC
        """

        #----- Sanity check
        if self.noise_mc is None:
            raise ValueError('The noise_mc should be defined for this function')
        
        #----- General info
        Nmc, Nx, Ny = self.noise_mc.shape
        if Nmc < 2: raise ValueError('The number of MC should be large, at least larger than 1.')
        reso = np.abs(self.header['CDELT1'])*3600
        normmap = np.std(self.noise_mc, axis=0)
        
        w = (normmap > 0) * ~np.isnan(normmap) * ~np.isinf(normmap)
        if not self.silent:
            if np.sum(~w) > 0:
                print('WARNING: some pixels are bad. This may affect the recovered noise model')
        
        #----- Extract the pk for each normalized MC
        pk_mc = np.zeros((Nmc,Nbin))
        for i in range(Nmc):
            img = self.noise_mc[i,:,:] / normmap
            img[~w] = 0
        
            k, pk_i = utils_pk.extract_pk2d(img, reso, Nbin=Nbin, scalebin=scale)
            pk_mc[i, :] += pk_i

        pk = np.mean(pk_mc, axis=0)
            
        w = ~np.isnan(pk)
        spec_mod = interp1d(k[w], pk[w], bounds_error=False, fill_value="extrapolate", kind='linear')
        
        #----- Fix the model
        self.noise_model = [normmap, lambda k_arcsec: spec_mod(k_arcsec)]
        


    #==================================================
    # Set noise model from MC
    #==================================================
    
    def set_noise_model_from_covariance(self,
                                        Nmc=1000,
                                        Nbin=30,
                                        scale='log'):
        """
        Derive the noise model by interpolating the noise PK from MC realizations
        after homogeneizing the noise, derive from the covariance matrix
        
        Parameters
        ----------
        - Nmc (int): number of MC to do
        - Nbin (int): number of bin for the Pk
        - scale (str): log or lin, to define the scale used in Pk
        
        Outputs
        ----------
        The noise model is set from the MC
        """

        #----- Sanity check
        if self.noise_covmat is None:
            raise ValueError('The noise_covmat should be defined for this function')

        #----- Get the noise MC
        noise_mc = self.get_noise_monte_carlo_from_covariance(Nmc=Nmc)
        
        #----- General info
        Nmc, Nx, Ny = noise_mc.shape
        if Nmc < 2: raise ValueError('The number of MC should be large, at least larger than 1.')
        reso = np.abs(self.header['CDELT1'])*3600
        normmap = np.std(noise_mc, axis=0)
        
        w = (normmap > 0) * ~np.isnan(normmap) * ~np.isinf(normmap)
        if not self.silent:
            if np.sum(~w) > 0:
                print('WARNING: some pixels are bad. This may affect the recovered noise model')
        
        #----- Extract the pk for each normalized MC
        pk_mc = np.zeros((Nmc,Nbin))
        for i in range(Nmc):
            img = noise_mc[i,:,:] / normmap
            img[~w] = 0
        
            k, pk_i = utils_pk.extract_pk2d(img, reso, Nbin=Nbin, scalebin=scale)
            pk_mc[i, :] += pk_i

        pk = np.mean(pk_mc, axis=0)
            
        w = ~np.isnan(pk)
        spec_mod = interp1d(k[w], pk[w], bounds_error=False, fill_value="extrapolate", kind='linear')
        
        #----- Fix the model
        self.noise_model = [normmap, lambda k_arcsec: spec_mod(k_arcsec)]
        
        
    #==================================================
    # Get the noise rms given a noise MC
    #==================================================
    
    def get_noise_rms_from_mc(self,
                              smooth_fwhm=0*u.arcsec):
        """
        Compute the noise rms from given noise MC
        
        Parameters
        ----------
        - smooth_fwhm (quantity): the gaussian FWHM to smooth the MC prior rms 

        Outputs
        ----------
        - rms (np array): The noise rms
        """
        
        #----- First get noise MC
        noise_mc = copy.copy(self.noise_mc)

        #----- Have the possibility to smooth the noise MC
        if smooth_fwhm>0:
            sigma2fwhm = 2 * np.sqrt(2*np.log(2))
            reso = np.abs(self.header['CDELT1'])
            sigma = smooth_fwhm.to_value('deg')/sigma2fwhm/reso
            noise_mc = gaussian_filter(noise_mc, sigma=[0,sigma,sigma])
        
        #----- Compute the rms
        rms = np.std(noise_mc, axis=0)
        
        #----- Get the rms
        return rms
        

    #==================================================
    # Get the noise rms given a noise MC
    #==================================================
    
    def get_noise_rms_from_covariance(self,
                                 Nmc=1000,
                                 smooth_fwhm=0*u.arcsec,
                                 noise_seed=None):
        """
        Compute the noise rms from the covariance matrix
        
        Parameters
        ----------
        - Nmc (int): number of MC to do
        - smooth_fwhm (quantity): the gaussian FWHM to smooth the MC prior rms 
        - noise_seed (int): the numpy random seed for reproducible results

        Outputs
        ----------
        - rms (np array): The noise rms
        """
        
        #----- First get noise MC
        noise_mc = self.get_noise_monte_carlo_from_covariance(Nmc=Nmc, seed=noise_seed)

        #----- Have the possibility to smooth the noise MC
        if smooth_fwhm>0:
            sigma2fwhm = 2 * np.sqrt(2*np.log(2))
            reso = np.abs(self.header['CDELT1'])
            sigma = smooth_fwhm.to_value('deg')/sigma2fwhm/reso
            noise_mc = gaussian_filter(noise_mc, sigma=[0,sigma,sigma])
        
        #----- Compute the rms
        rms = np.std(noise_mc, axis=0)
        
        #----- Get the rms
        return rms
    

    #==================================================
    # Get the noise rms given a model
    #==================================================
    
    def get_noise_rms_from_model(self,
                                 Nmc=1000,
                                 smooth_fwhm=0*u.arcsec,
                                 center=None,
                                 noise_seed=None):
        """
        Compute the noise rms from the noise model
        
        Parameters
        ----------
        - Nmc (int): number of MC to do
        - smooth_fwhm (quantity): the gaussian FWHM to smooth the MC prior rms 
        - center (SkyCoord): the reference center for the noise
        - noise_seed (int): the numpy random seed for reproducible results
        
        Outputs
        ----------
        - rms (np array): The noise rms
        """
        
        #----- First get noise MC
        noise_mc = self.get_noise_monte_carlo_from_model(center=center,
                                                         Nmc=Nmc,
                                                         seed=noise_seed)

        #----- Have the possibility to smooth the noise MC
        if smooth_fwhm>0:
            sigma2fwhm = 2 * np.sqrt(2*np.log(2))
            reso = np.abs(self.header['CDELT1'])
            sigma = smooth_fwhm.to_value('deg')/sigma2fwhm/reso
            noise_mc = gaussian_filter(noise_mc, sigma=[0,sigma,sigma])
        
        #----- Compute the rms
        rms = np.std(noise_mc, axis=0)
        
        #----- Get the rms
        return rms

    
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
        - noise_origin (str): can be 'covariance', 'model', 'MC', or 'none'
        - noise_center (SkyCoord): the reference center for the noise
        - model_seed (bool): set to a number for reproducible noise

        """
        
        # copy the model so that the input is not modified
        model = copy.deepcopy(model_input)

        # Match the header as requested
        if use_model_header:
            self.header = model.get_map_header()
            if not self.silent:
                msg1 = 'WARNING: the data header is fixed to the model header.'
                msg2 = '         This may affect consistency with other data properties (mask, noise, etc)'
                print(msg1)
                print(msg2)
        else:
            model.map_header = self.header
            
        # Get the raw image model
        input_model = model.get_sz_map(seed=model_seed, no_fluctuations=model_no_fluctuations)
        
        # Convolve with instrument response function
        convolved_model = utils_pk.apply_transfer_function(input_model,
                                                           model.get_map_reso().to_value('arcsec'),
                                                           self.psf_fwhm.to_value('arcsec'),
                                                           self.transfer_function,
                                                           apps_TF_LS=True, apps_beam=True)
        
        # Add noise realization
        if noise_origin == 'model':
            noise_mc = self.get_noise_monte_carlo_from_model(center=noise_center, Nmc=1, seed=noise_seed)
        elif noise_origin == 'covariance':
            noise_mc = self.get_noise_monte_carlo_from_covariance(Nmc=1, seed=noise_seed)
        elif noise_origin == 'MC':
            if len(self.noise_mc.shape) == 3:
                noise_mc = self.noise_mc[0,:,:]
            elif len(self.noise_mc) == 2:
                noise_mc = self.noise_mc
            else:
                raise ValueError('The noise_mc is not correct')
        elif noise_origin == 'none':
            noise_mc = 0
        else:
            raise ValueError('noise_origin should be "model", "covariance", or "none"')
        image_mock = convolved_model + noise_mc

        # Set the image and the mask if not correct anymore
        self.image = image_mock     

        return image_mock
        
