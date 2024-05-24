"""
This file contain the Inference class. 
It is dedicated to extract constraints on model parameters.
"""

#==================================================
# Requested imports
#==================================================

import os
import numpy as np
import astropy.units as u
import astropy.constants as cst
from scipy.ndimage import gaussian_filter
import scipy.stats as stats
import pprint
import copy
import pickle
import dill

from pitszi import utils
from pitszi import utils_pk
from pitszi import title

from pitszi.inference_fitting import InferenceFitting


#==================================================
# Inference class
#==================================================

class Inference(InferenceFitting):
    """ Inference class
        This class infer the profile and power spectrum properties 
        given input data (from class Data()) and model (from class Model())
        that are attributes

    Attributes
    ----------
        - data (object from Class Data()): the data object
        - model (object from class Model()): the model object

        # Nuisance parameters
        self.nuisance_ZL (float): nuisance parameter -- the map zero level
        self.nuisance_Anoise (float): nuisance parameter -- the amplitude of the noise

        # Binning in k
        - kbin_min (quantity): minimum value of the bins edges
        - kbin_max (quantity): maximum value of the bin edges
        - kbin_Nbin (int): number of bins
        - kbin_scale (str): log or lin, the k bin scaling scheme
        
        # Analysis methodology
        - method_use_covmat (bool): use covariance matrix in fit?
        - method_parallel (bool): use parallelization in fit
        - method_w8 (2d array): the weight map
        - method_data_deconv (bool): perform beam and TF data deconvolution, or process the model with beam and TF

        # MCMC parameters
        - mcmc_nwalkers (int): number of MCMC walkers
        - mcmc_nsteps (int): number of MCMC steps to run
        - mcmc_burnin (int): number of burnin steps
        - mcmc_reset (bool): restart MCMC from scratch even if sampler exists?
        - mcmc_run (bool): run MCMC?
        - mcmc_Nresamp (int): number of resampling model in post-MCMC analysis

        # Admin
        - silent (bool): set to False to give information
        - output_dir (str): the output directory
        
        # Useful hiden parameters computed by running the setup() function
        - _pk_setup_done (bool): tell us if the setup was already done
        - _reso_arcsec (float): the map resolution in arcsec
        - _reso_kpc (float): the map resolution in kpc
        - _kpc2arcsec (float): the conversion from kpc to arcsec
        - _Kmnmn (nd complex array): the mode mixing matrix
        - _dy_image (2d array): the image used for Pk calculation
        - _ymap_sph1 (2d array): the smooth ymap model (numerator)
        - _ymap_sph2 (2d array): the smooth ymap model (denominator)
        - _conv_wf (2d array): the pk3d to pk2d conversion window function
        - _conv_pk2d3d (float): the pk3d to pk2d conversion over the considered area
        - _k2d_norm (2d array): the norm of k in a grid
        - _kedges_kpc (1d array): the edges of the k bin in kpc
        - _kedges_arcsec (1d array): the edges of the k bins in arcsec
        - _kctr_kpc (1d array): the center of the k bins in kpc
        - _kctr_arcsec (1d array): the center of the k bins in arcsec
        - _kcount (1d array): the number of pk counts in each k bin
        - _pk2d_data (1d array): the data pk2d
        - _pk2d_noise (1d array): the mean noise pk2d 
        - _pk2d_noise_rms (1d array): the noise rms pk2d
        - _pk2d_noise_cov (2d array): the noise pk2d covarariance matrix
        - _pk2d_modref (1d array): the reference input model mean
        - _pk2d_modref_rms (1d array): the reference input model rms
        - _pk2d_modref_cov (2d array): the reference input model covariance pk2d matrix
    
    Methods
    ----------
    - print_param: print the current inference parameters
    - setup: perform the setup and defines the usefull hidden attributes once and for all
    - set_method_w8: usefull function to help defining the weight map
    - get_k_grid: compute the grid of k
    - get_kedges: compute the edges of the k bins
    - get_kbin_counts: compute the number of counts in each k bin
    - get_pk2d_data_image: compute the image used for Pk analysis
    - get_pk2d_data: compute the Pk2d data from the data
    - get_pk2d_noise_statistics: compute the noise Pk2d statistical properties 
    - get_pk2d_model_statistics: compute the reference model statistical properties
    - get_pk2d_model_brute: compute the model in the brute force approach
    - get_pk2d_model_proj: compute the model in the projection case
    - get_radial_data: get the data map in the case of the radial component (i.e. the y map)
    - get_radial_noise_statistics: get the noise properties in the case of the radail component
    - get_radial_model: compute the model in the case of the radial component
    - get_p3d_to_p2d_from_window_function: derive the window function to go from Pk3d to Pk2d
    - _get_p2d_from_p3d_from_window_function_exact: derive Pk2d given Pk3d by integration over the window function
    
    """    

    #==================================================
    # Initialization
    #==================================================

    def __init__(self,
                 data,
                 model,
                 #
                 nuisance_ZL=0,
                 nuisance_Anoise=1,
                 #
                 kbin_min=0*u.arcsec**-1,
                 kbin_max=0.1*u.arcsec**-1,
                 kbin_Nbin=20,
                 kbin_scale='lin',
                 #
                 method_use_covmat=False,
                 method_parallel=False,
                 method_data_deconv=False,
                 method_w8=None,
                 #
                 mcmc_nwalkers=20,
                 mcmc_nsteps=500,
                 mcmc_burnin=100,
                 mcmc_reset=False,
                 mcmc_run=True,
                 mcmc_Nresamp=100,
                 #
                 silent=False,
                 output_dir='./pitszi_output',
    ):
        """
        Initialize the inference object. 
        All parameters can be changed on the fly.

        List of caveats in model projection/deprojection from Pk3d
        ----------------------------------------------------------
        Many subbtle Fourier sampling effects may alter the results. A deep investigation 
        of possible biases is recommanded using simulated data (windowing, binning, etc).
        Here is a non-exhaustive list of some identified effects:

        *** Prior binning/mask/etc ***
        - Small scales: the windowing by the profile P (\int P (1 + dP) dl) may generate some artifacts
        for very steep fluctuation spectra at high k. This is an issue in the simulation, not the model. 
        The power spectrum of the signal then present anisotropies in 2d. In practice, this should 
        be unsignificant because we do not expect such steep spectra and beam smoothing should kill 
        these scales anyway.
        - Large scales: the window function drops for k_z above a given cutoff, justifying the approximation
        that Pk2d \propto Pk3d (Churazov+2012, Eq 11). However, the approximation may break significantly 
        at low k, for spectra with low power on large scales since 
        \int W(kz) x P3d(sqrt(kx^2 + ky^2 + kz^2)) dkz --> P3d(sqrt(kx^2 + ky^2)) \int W(kz) dkz
        is no longer valid. Also note that the approximation holds better for flatter profiles since the window 
        function is sharper.
        - All scales: the input to recover spctra seems accurate within +/-5% depending on the exact grid choice
        - Profile truncation : any sharp discountinuity in P(r), e.g. due to a truncation above some limit, etc., 
        may introduce features in the power spectra

        *** From binning/mask/etc ***
        - Sharp mask/weights: very sharp mask or weighting may introduce issues.
        
        Parameters
        ----------
        - data (pitszi Class Data object): the data
        - model (pitszi Class Model object): the model
        
        - nuisance_ZL (float): map zero level, a nuisance parameter.
        - nuisance_Anoise (float): noise Pk amplitude, a nuisance parameter.

        - kbin_min (quantity): minimum k value for the 2D power spectrum (homogeneous to 1/angle or 1/kpc)
        - kbin_max (quantity): maximal k value for the 2D power spectrum (homogeneous to 1/angle or 1/kpc)
        - kbin_Nbin (int): number of bins in the 2d power spectrum
        - kbin_scale (str): bin spacing ('lin' or 'log')
      
        - method_use_covmat (bool): use covariance matrix in the fit
        - method_use_paral (bool): use parallelization
        - method_data_deconv (bool): use instrument response convolved or deconvolved 
        - method_w8 (2d array): weight map
      
        - mcmc_nwalkers (int): number of MCMC walkers
        - mcmc_nsteps (int): number of MCMC steps to run
        - mcmc_burnin (int): the burnin, number of point to remove from the chains
        - mcmc_reset (bool): True for restarting the MCMC even if a sampler exists
        - mcmc_run (bool): True to run the MCMC
        - mcmc_Nresamp (int): the number of Monte Carlo to resample the chains for results

        - silent (bool): set to False for printing information
        - output_dir (str): directory where outputs are saved
        
        """

        if not silent:
            title.show_inference()
        
        #----- Input data and model (deepcopy to avoid modifying the input when fitting)
        self.data  = copy.deepcopy(data)
        self.model = copy.deepcopy(model)

        #----- Nuisance parameters
        self.nuisance_ZL     = 0
        self.nuisance_Anoise = 1

        #----- Binning in k
        self.kbin_min   = kbin_min
        self.kbin_max   = kbin_max
        self.kbin_Nbin  = kbin_Nbin
        self.kbin_scale = kbin_scale # ['lin', 'log']
        
        #----- Analysis methodology
        self.method_use_covmat = method_use_covmat
        self.method_parallel = method_parallel
        if method_w8 == None:
            self.method_w8 = np.ones(data.image.shape)   # The weight applied to image = dy/y x w8
        else:
            self.method_w8 = method_w8
        self.method_data_deconv  = method_data_deconv    # Deconvolve the data from TF and beam prior Pk

        #----- MCMC parameters
        self.mcmc_nwalkers = mcmc_nwalkers
        self.mcmc_nsteps   = mcmc_nsteps
        self.mcmc_burnin   = mcmc_burnin
        self.mcmc_reset    = mcmc_reset
        self.mcmc_run      = mcmc_run
        self.mcmc_Nresamp  = mcmc_Nresamp

        #----- Admin
        self.silent     = silent
        self.output_dir = output_dir

        #========== Usefull hidden variable to be computed on the fly
        #----- Useful hiden parameters for radial profile
        self._ymap_invcov        = None

        #----- Useful hiden parameters for Pk fitting
        self._pk_setup_done      = False
        
        self._reso_arcsec        = None
        self._reso_kpc           = None
        self._kpc2arcsec         = None
                
        self._dy_image           = None
        self._ymap_sph1          = None
        self._ymap_sph2          = None
        
        self._k2d_norm           = None
        self._kedges_kpc         = None
        self._kedges_arcsec      = None
        self._kctr_kpc           = None
        self._kctr_arcsec        = None
        self._kcount             = None
        
        self._pk2d_data          = None
        self._pk2d_noise         = None
        self._pk2d_noise_rms     = None
        self._pk2d_noise_cov     = None
        self._pk2d_noise_invcov  = None
        self._pk2d_modref        = None
        self._pk2d_modref_rms    = None
        self._pk2d_totref_invcov = None

        self._conv_wf            = None
        self._conv_pk2d3d        = None
        
        self._Kmnmn              = None

        
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
    
    def save_inference(self):
        """
        Save the current inference object.
        
        Parameters
        ----------
            
        Outputs
        ----------
        The parameters are saved in the output directory

        """

        # Create the output directory if needed
        if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)

        # Save
        with open(self.output_dir+'/inference_parameters.pkl', 'wb') as pfile:
            #pickle.dump(self.__dict__, pfile, pickle.HIGHEST_PROTOCOL)
            dill.dump(self.__dict__, pfile)

        # Text file for user
        par = self.__dict__
        keys = list(par.keys())
        with open(self.output_dir+'/inference_parameters.txt', 'w') as txtfile:
            for k in range(len(keys)):
                txtfile.write('--- '+(keys[k])+'\n')
                txtfile.write('    '+str(par[keys[k]])+'\n')
                txtfile.write('    '+str(type(par[keys[k]]))+'\n')

                
    #==================================================
    # Load parameters
    #==================================================
    
    def load_inference(self, param_file):
        """
        Read the a given parameter file to re-initialize the inference object.
        
        Parameters
        ----------
        param_file (str): the parameter file to be read
            
        Outputs
        ----------
            
        """

        with open(param_file, 'rb') as pfile:
            par = dill.load(pfile)
            
        self.__dict__ = par
        
        
    #==================================================
    # Setup: preparation
    #==================================================
    
    def pk_setup(self, Nmc=1000):
        """
        This function defines the setup prior power spectrum inference.
        The setup is necessary for inference on the power spectrum:
        - Compute map resolution and k binning information
        - compute ymap residual to be used for Pk extraction
        - compute the smooth model
        - extract Pk data, reference model and noise properties
        - compute conversion from Pk 2d to Pk3d
        - compute the mode mixing matrix given the weights
        
        Parameters
        ----------
        - Nmc (int): the number of monte carlo to compute the model/noise statistics

        Outputs
        ----------
        Necessary product are computed and defined as hidden parameters

        """

        #---------- Info
        if not self.silent:
            print('----- Running the setup -----')
            if self._pk_setup_done :
                print('      - The Pk setup was already done, but it will be overwritten.')

        #---------- General
        self._reso_arcsec = self.model.get_map_reso(physical=False).to_value('arcsec')
        self._reso_kpc    = self.model.get_map_reso(physical=True).to_value('kpc')
        self._kpc2arcsec  = self._reso_arcsec / self._reso_kpc
        
        #---------- ymaps
        if not self.silent: print('    * Setup imaging')

        dy_image, model_ymap_sph1, model_ymap_sph2 = self.get_pk2d_data_image()
        
        self._ymap_sph1 = model_ymap_sph1
        self._ymap_sph2 = model_ymap_sph2
        self._dy_image  = dy_image

        #---------- Binning
        if not self.silent: print('    * Setup k binning')

        k2d_x, k2d_y, k2d_norm = self.get_k_grid(physical=True)
        kedges_kpc    = self.get_kedges(physical=True)
        kedges_arcsec = self.get_kedges(physical=False)
        
        self._k2d_norm      = k2d_norm.to_value('kpc-1')
        self._kedges_kpc    = kedges_kpc.to_value('kpc-1')
        self._kedges_arcsec = kedges_arcsec.to_value('arcsec-1')
        self._kctr_kpc      = 0.5 * (self._kedges_kpc[1:] + self._kedges_kpc[:-1])
        self._kctr_arcsec   = 0.5 * (self._kedges_arcsec[1:] + self._kedges_arcsec[:-1])
        self._kcount        = self.get_kbin_counts()
        if not self.silent:
            print('      - Counts in each k bin:', self._kcount)
            print('      - Minimal count in k bins:', np.amin(self._kcount))

        #---------- Pk
        if not self.silent: print('    * Setup Pk data, ref model and noise')

        _, noise_mean, noise_cov = self.get_pk2d_noise_statistics(physical=True, Nmc=Nmc)
        _, model_mean, model_cov = self.get_pk2d_model_statistics(physical=True, Nmc=Nmc)

        self._pk2d_data          = self.get_pk2d_data(physical=True)[1].to_value('kpc2')
        self._pk2d_noise         = noise_mean.to_value('kpc2')
        self._pk2d_noise_rms     = np.diag(noise_cov.to_value('kpc4'))**0.5
        self._pk2d_noise_cov     = noise_cov.to_value('kpc4')
        if self.method_use_covmat:
            self._pk2d_noise_invcov = np.linalg.inv(noise_cov.to_value('kpc4'))
        
        self._pk2d_modref        = model_mean.to_value('kpc2')
        self._pk2d_modref_rms    = np.diag(model_cov.to_value('kpc4'))**0.5
        self._pk2d_modref_cov    = model_cov.to_value('kpc4')

        if self.method_use_covmat:
            self._pk2d_totref_invcov = np.linalg.inv(noise_cov.to_value('kpc4') + model_cov.to_value('kpc4'))

        #---------- Convertion
        if not self.silent: print('    * Setup window function conversion')

        ymap_wf = self.get_p3d_to_p2d_from_window_function()
        
        self._conv_wf     = ymap_wf
        self._conv_pk2d3d = (np.sum(ymap_wf*self.method_w8**2)/np.sum(self.method_w8**2)).to_value('kpc-1')

        #---------- Bin-to-bin mixing
        if not self.silent: print('    * Setup bin-to-bin mixing')

        fft_w8 = np.fft.fftn(self.method_w8)
        
        self._Kmnmn = utils_pk.compute_Kmnmn(fft_w8)
        
        #---------- Update the setup
        self._pk_setup_done = True
        if not self.silent:
            print('----- The setup is done -----')
 
            
    #==================================================
    # Set the weight to inverse model
    #==================================================
    
    def set_method_w8(self,
                      roi_mask=None,
                      apply_data_mask=True,
                      apply_radial_model=False,
                      conv_radial_model_beam=True,
                      conv_radial_model_TF=False,
                      remove_GNFW_core=True,
                      smooth_FWHM=0*u.arcsec):
        """
        This function is a helper to set the weight map with the mask and 
        the smooth model. It can account for smoothing and remove the cluster 
        core for avoiding aliasing issues.
        
        Parameters
        ----------
        - roi_mask (2d array): a mask used to define the Region Of Interest
        - apply_data_mask (bool): application of the data mask in the weight
        - apply_radial_model (bool): multiply the wight with the radial model
        - conv_radial_model_beam (bool): apply beam smoothing to the radial model
        - conv_radial_model_TF (bool): apply the transfer function to the radial model
        - remove_GNFW_core (bool): if True, and if the model is a GNFW, it will 
        set the core parameter from the model to zero
        - smooth_FWHM (quantity): smoothing of weight map, homogeneous to arcsec

        Outputs
        ----------
        None, the weight are set to the requested map

        """
        
        w8map = np.ones(self.data.image.shape)

        #===== Apply masks
        if apply_data_mask:
            w8map *= self.data.mask 

        if roi_mask is not None:
            w8map *= roi_mask
            
        #===== Apply radial model
        if apply_radial_model:
            # Extract the SZ model map
            tmp_mod = copy.deepcopy(self.model)

            # Remove core if needed
            if tmp_mod.model_pressure_profile['name'] == 'GNFW':
                tmp_mod.model_pressure_profile['c'] = 0

            # Extract the model accounting for beam and TF
            if conv_radial_model_beam:
                the_beam = self.data.psf_fwhm
            else:
                the_beam = None

            if conv_radial_model_TF:
                the_tf   = self.data.transfer_function
            else:
                the_tf   = None
                
            radial_model = tmp_mod.get_sz_map(no_fluctuations=True,
                                              irfs_convolution_beam=the_beam,
                                              irfs_convolution_TF=the_tf)
            
            # Apply radial model
            w8map *= radial_model

        # Smoothing
        if smooth_FWHM>0:
            sigma2fwhm = 2 * np.sqrt(2*np.log(2))
            sigma = smooth_FWHM.to_value('deg')/sigma2fwhm/self.model.get_map_reso().to_value('deg')
            w8map = gaussian_filter(w8map, sigma=sigma)
        
        self.method_w8 = w8map
        
            
    #==================================================
    # Return the k grid information
    #==================================================
    
    def get_k_grid(self, physical=False):
        """
        Give the grid in kspace
        
        Parameters
        ----------
        - physical (bool): set to true to have 1/kpc units. Otherwise 1/arcsec.

        Outputs
        ----------
        - kx (2d array): the k grid along x
        - ky (2d array): the k grid along y
        - knorm (2d array): the grid of the k normalization

        """

        # Define the grid size
        Nx, Ny = self.data.image.shape

        # Get the resolution in the right unit (physical or angle)
        if physical:
            reso = self.model.get_map_reso(physical=True).to_value('kpc')
            unit = u.kpc**-1
        else:
            reso = self.model.get_map_reso(physical=False).to_value('arcsec')
            unit = u.arcsec**-1

        # Grid sampling in k space
        k_x = np.fft.fftfreq(Nx, reso)
        k_y = np.fft.fftfreq(Ny, reso)
        k2d_x, k2d_y = np.meshgrid(k_x, k_y, indexing='ij')
        k2d_norm = np.sqrt(k2d_x**2 + k2d_y**2)

        return k2d_x*unit, k2d_y*unit, k2d_norm*unit
    
    
    #==================================================
    # Define k edges from bining properties
    #==================================================
    
    def get_kedges(self, physical=False):
        """
        This function compute the edges of the bins in k space given the class 
        attribute: kbin_min, kbin_max, kbin_scale, kbin_Nbin
        
        Parameters
        ----------
        - physical (bool): if true, the edges are given in distance, else angle

        Outputs
        ----------
        - kedges (1d array): the edges of the bins 

        """
        
        # Get the kmin and kmax in the right unit
        try:
            kmin_sampling = self.kbin_min.to_value('arcsec-1')
            kmax_sampling = self.kbin_max.to_value('arcsec-1')
            unit = u.arcsec**-1
        except:
            kmin_sampling = self.kbin_min.to_value('kpc-1')
            kmax_sampling = self.kbin_max.to_value('kpc-1')
            unit = u.kpc**-1

        # Define the bining
        if self.kbin_scale == 'lin':
            kbins = np.linspace(kmin_sampling, kmax_sampling, self.kbin_Nbin+1)
        elif self.kbin_scale == 'log':
            kbins = np.logspace(np.log10(kmin_sampling), np.log10(kmax_sampling), self.kbin_Nbin+1)
        else:
            raise ValueError("Only lin or log scales are allowed. Here self.kbin_scale = "+self.kbin_scale)

        # Define the output unit
        if physical:
            if unit == u.arcsec**-1:
                kbins = kbins * (3600*180/np.pi) / self.model.D_ang.to_value('kpc') * u.kpc**-1
            elif unit == u.kpc**-1:
                kbins = kbins*u.kpc**-1
        else:
            if unit == u.arcsec**-1:
                kbins = kbins * u.arcsec**-1
            elif unit == u.kpc**-1:
                kbins = (kbins * self.model.D_ang.to_value('kpc')) / (3600*180/np.pi) * u.arcsec**-1

        return kbins


    #==================================================
    # Return the number of counts per bin
    #==================================================
    
    def get_kbin_counts(self):
        """
        Give the number of counts in each k bin, given the grid sampling and the 
        bin edges. The default bin edges are provided by get_kedges function
        
        Parameters
        ----------
        - kedges (quantity array): the bin edges in units of 1/angle or 1/distance
        
        Outputs
        ----------
        - kbin_counts (1d array): the counts in each bins 

        """

        # Get the k edges
        kedges = self.get_kedges(physical=True).to_value('kpc-1')

        # Grid sampling in k space
        _, _, k2d_norm = self.get_k_grid(physical=True)
        k2d_norm = k2d_norm.to_value('kpc-1')

        # Bining and counting
        kbin_counts, _, _ = stats.binned_statistic(k2d_norm.flatten(), k2d_norm.flatten(),
                                                   statistic='count', bins=kedges)

        return kbin_counts
    
    
    #==================================================
    # Compute the image data for Pk analysis
    #==================================================
    
    def get_pk2d_data_image(self):
        """
        This function compute the image used in the Pk analysis, 
        and also returns the smooth ymaps used in the ratio:

        dy_image = (y - model_ymap_sph1) / model_ymap_sph2
        
        Parameters
        ----------
        
        Outputs
        ----------
        - dy_image (2d array): image use for Pk analysis
        - model_ymap_sph1 (2d array): smooth model numerator
        - model_ymap_sph2 (2d array): smooth model denominator

        """
        
        #---------- Get the smooth model and the data image
        if self.method_data_deconv:
            img_y = utils_pk.deconv_transfer_function(self.data.image,
                                                      self.model.get_map_reso().to_value('arcsec'), 
                                                      self.data.psf_fwhm.to_value('arcsec'),
                                                      self.data.transfer_function, 
                                                      dec_TF_LS=True, dec_beam=True)
            model_ymap_sph1 = self.model.get_sz_map(no_fluctuations=True)
            model_ymap_sph2 = model_ymap_sph1
            
        else:
            img_y = self.data.image
            model_ymap_sph1 = self.model.get_sz_map(no_fluctuations=True,
                                                    irfs_convolution_beam=self.data.psf_fwhm,
                                                    irfs_convolution_TF=self.data.transfer_function)    
            model_ymap_sph2 = self.model.get_sz_map(no_fluctuations=True,
                                                    irfs_convolution_beam=self.data.psf_fwhm) 
        
        #---------- Compute the data to be used for Pk
        delta_y    = img_y - model_ymap_sph1
        dy_image = (delta_y - np.mean(delta_y)) / model_ymap_sph2 * self.method_w8
        
        return dy_image, model_ymap_sph1, model_ymap_sph2
    
    
    #==================================================
    # Compute the Pk2d from the data
    #==================================================
    
    def get_pk2d_data(self,
                      physical=False):
        """
        This function compute the data pk2d that is used for fitting
        
        Parameters
        ----------
        - physical (bool): set to true to have output in kpc units, else arcsec units

        Outputs
        ----------
        - k2d (1d array): the values of k in each bin
        - pk2d (1d array): the data Pk2d

        """

        #---------- Compute the data to be used for Pk
        dy_image, model_ymap_sph1, model_ymap_sph2 = self.get_pk2d_data_image() 
        
        #---------- Pk for the data
        kedges = self.get_kedges().to_value('arcsec-1')
        reso   = self.model.get_map_reso().to_value('arcsec')
        k2d, data_pk2d = utils_pk.extract_pk2d(dy_image, reso, kedges=kedges)

        #---------- Units
        if physical:
            kpc2arcsec = ((1*u.kpc/self.model.D_ang).to_value('')*u.rad).to_value('arcsec')
            k2d        = k2d * kpc2arcsec**1 *u.kpc**-1
            data_pk2d  = data_pk2d * kpc2arcsec**-2 * u.kpc**2
        else:
            k2d       = k2d * u.arcsec**-1
            data_pk2d = data_pk2d * u.arcsec**2

        #---------- return
        return k2d, data_pk2d

    
    #==================================================
    # Compute noise statistics
    #==================================================
    
    def get_pk2d_noise_statistics(self,
                                  physical=False,
                                  Nmc=None):
        """
        This function compute the noise properties associated
        with the data noise
        
        Parameters
        ----------
        - physical (bool): set to true to have output in kpc units, else arcsec units

        Outputs
        ----------
        - k2d (1d array): the values of k in each bin
        - noise_pk2d_ref (1d array): the noise mean
        - noise_pk2d_covmat (2d array): the noise covariance matrix

        """
        
        #----- Sanity check
        bin_counts = self.get_kbin_counts()
        if np.amin(bin_counts) == 0:
            raise ValueError('Some bins have zero counts. Please redefine the binning to avoid this issue.')
        
        #----- Useful info
        kedges = self.get_kedges().to_value('arcsec-1')
        reso   = self.model.get_map_reso().to_value('arcsec')

        #----- Model accounting/or not for beam and TF
        _, _, model_ymap_sph2 = self.get_pk2d_data_image()

        #----- Extract noise MC realization
        noise_ymap_mc = self.data.noise_mc
        Nmc0 = noise_ymap_mc.shape[0]
        if noise_ymap_mc is None:
            raise ValueError("Noise MC not available in data. Please use the class Data() to define it.")
        if Nmc0 < 100 and not self.silent:
            print('WARNING: the number of MC realizations to use is less than 100. This might not be enough.')

        # Redefine the number of Mc upon request
        if Nmc is not None:
            if Nmc0 < Nmc: # just use Nmc0 if the requested number of MC is larger than available
                Nmc = Nmc0
                print('WARNING: the number requested number of MC realizations is larger than what is available in Data().')
        else:
            Nmc = Nmc0

        #----- Compute Pk2d MC realization
        noise_pk2d_mc = np.zeros((Nmc, len(kedges)-1))
        for imc in range(Nmc):
            
            # Account for deconvolution choices
            if self.method_data_deconv:
                img_y = utils_pk.deconv_transfer_function(noise_ymap_mc[imc,:,:],
                                                          self.model.get_map_reso().to_value('arcsec'), 
                                                          self.data.psf_fwhm.to_value('arcsec'),
                                                          self.data.transfer_function, 
                                                          dec_TF_LS=True, dec_beam=True)
            else:
                img_y = noise_ymap_mc[imc,:,:]
                
            # Noise to Pk
            image_noise_mc = (img_y - np.mean(img_y))/model_ymap_sph2 * self.method_w8
            k2d, pk_mc =  utils_pk.extract_pk2d(image_noise_mc, reso, kedges=kedges)
            noise_pk2d_mc[imc,:] = pk_mc
            
        # Noise statistics
        noise_pk2d_mean = np.mean(noise_pk2d_mc, axis=0)
        noise_pk2d_rms  = np.std(noise_pk2d_mc, axis=0)

        #----- Compute covariance
        noise_pk2d_covmat = np.zeros((len(kedges)-1, len(kedges)-1))
        for imc in range(Nmc):
            noise_pk2d_covmat += np.matmul((noise_pk2d_mc[imc,:]-noise_pk2d_mean)[:,None],
                                           (noise_pk2d_mc[imc,:]-noise_pk2d_mean)[None,:])
        noise_pk2d_covmat /= Nmc

        #----- Sanity check again
        if np.sum(np.isnan(noise_pk2d_covmat)) > 0:
            if not self.silent:
                print('Some pixels in the covariance matrix are NaN.')
            raise ValueError('Issue with noise covariance matrix')

        #----- Units
        if physical:
            kpc2arcsec        = ((1*u.kpc/self.model.D_ang).to_value('')*u.rad).to_value('arcsec')
            k2d               = k2d * kpc2arcsec**1 * u.kpc**-1
            noise_pk2d_mean   = noise_pk2d_mean * kpc2arcsec**-2 * u.kpc**2
            noise_pk2d_covmat = noise_pk2d_covmat * kpc2arcsec**-4 * u.kpc**4
        else:
            k2d               = k2d * u.arcsec**-1
            noise_pk2d_mean   = noise_pk2d_mean * u.arcsec**2
            noise_pk2d_covmat = noise_pk2d_covmat * u.arcsec**4

        #---------- return
        return k2d, noise_pk2d_mean, noise_pk2d_covmat

    
    #==================================================
    # Compute model variance statistics
    #==================================================
    
    def get_pk2d_model_statistics(self,
                                  Nmc=1000,
                                  physical=False,
                                  seed=None):
        """
        This function compute the model variance properties associated
        with the reference input model.
        
        Parameters
        ----------
        - Nmc (int): number of monte carlo realization
        - physical (bool): set to true to have output in kpc units, else arcsec units
        - seed (int): fluctuation seed for reproducible results

        Outputs
        ----------
        - k2d (1d array): the values of k in each bin
        - model_pk2d_ref (1d array): the model mean
        - model_pk2d_covmat (2d array): the model covariance matrix

        """

        #----- Sanity check
        bin_counts = self.get_kbin_counts()
        if np.amin(bin_counts) == 0:
            raise ValueError('Some bins have zero counts. Please redefine the binning to avoid this issue.')
        
        #----- Useful info
        kedges = self.get_kedges().to_value('arcsec-1')
        reso   = self.model.get_map_reso().to_value('arcsec')

        #----- Model accounting/or not for beam and TF
        _, model_ymap_sph1, model_ymap_sph2 = self.get_pk2d_data_image()
        
        #----- Compute Pk2d MC realization
        model_pk2d_mc     = np.zeros((Nmc, len(kedges)-1))
        model_pk2d_covmat = np.zeros((len(kedges)-1, len(kedges)-1))
        
        for imc in range(Nmc):

            # Account or not for TF and beam
            if self.method_data_deconv:
                test_ymap = self.model.get_sz_map(seed=seed, no_fluctuations=False)
            else:
                test_ymap = self.model.get_sz_map(seed=seed, no_fluctuations=False,
                                                  irfs_convolution_beam=self.data.psf_fwhm,
                                                  irfs_convolution_TF=self.data.transfer_function)

            # Test image
            delta_y = test_ymap - model_ymap_sph1
            test_image = (delta_y - np.mean(delta_y))/model_ymap_sph2 * self.method_w8

            # Pk
            k2d, pk_mc =  utils_pk.extract_pk2d(test_image, reso, kedges=kedges)
            model_pk2d_mc[imc,:] = pk_mc

        # Model statistics
        model_pk2d_mean = np.mean(model_pk2d_mc, axis=0)
        model_pk2d_rms  = np.std(model_pk2d_mc, axis=0)

        #----- Compute covariance
        for imc in range(Nmc):
            model_pk2d_covmat += np.matmul((model_pk2d_mc[imc,:]-model_pk2d_mean)[:,None],
                                           (model_pk2d_mc[imc,:]-model_pk2d_mean)[None,:])
        model_pk2d_covmat /= Nmc

        #----- Sanity check again
        if np.sum(np.isnan(model_pk2d_covmat)) > 0:
            if not self.silent:
                print('Some pixels in the covariance matrix are NaN.')
                print('This can be that the number of bins is such that some bins are empty.')
            raise ValueError('Issue with noise covariance matrix')
        
        #----- Units
        if physical:
            kpc2arcsec        = ((1*u.kpc/self.model.D_ang).to_value('')*u.rad).to_value('arcsec')
            k2d               = k2d * kpc2arcsec**1 * u.kpc**-1
            model_pk2d_mean   = model_pk2d_mean * kpc2arcsec**-2 * u.kpc**2
            model_pk2d_covmat = model_pk2d_covmat * kpc2arcsec**-4 * u.kpc**4
        else:
            k2d               = k2d * u.arcsec**-1
            model_pk2d_mean   = model_pk2d_mean * u.arcsec**2
            model_pk2d_covmat = model_pk2d_covmat * u.arcsec**4

        #---------- return
        return k2d, model_pk2d_mean, model_pk2d_covmat


    #==================================================
    # Model for pk2d fit brute force
    #==================================================
    
    def get_pk2d_model_brute(self,
                             seed=None,
                             physical=False):
        """
        This function returns the model of pk for the brute force
        approach. This is the same as what is done to get the model variance properties.
        
        Parameters
        ----------
        - physical (bool): set to true to have output in kpc units, else arcsec units
        - seed (int): fluctuation seed for reproducible results

        Outputs
        ----------
        - k2d (1d np array): the k bin center
        - pk2d (1d np array): the pk model

        """

        #----- Check the setup, reequiered here
        if self._pk_setup_done == False:
            raise ValueError('This function requieres do run the setup first')
        
        #----- Compute the Pk for the cluster
        k2d, pk2d_modvar, _ = self.get_pk2d_model_statistics(physical=physical,
                                                             Nmc=1,
                                                             seed=seed)

        #----- Compute the final model
        pk_noise = self._pk2d_noise # baseline is kpc2
        if physical:
            pk_noise = pk_noise * u.kpc**2
        else:
            pk_noise = pk_noise * self._kpc2arcsec**2 * u.arcsec**2
        
        pk2d_tot = pk2d_modvar + self.nuisance_Anoise * pk_noise
        
        return k2d, pk2d_tot
    
    
    #==================================================
    # Model for pk2d fit for projection case
    #==================================================
    
    def get_pk2d_model_proj(self,
                            physical=False):
        """
        This function returns the model of pk for the 
        model projection approach
        
        Parameters
        ----------
        - physical (bool): set to true to have output in kpc units, else arcsec units

        Outputs
        ----------
        - k2d (1d np array): the k bin center
        - pk2d (1d np array): the pk model

        """
        
        #----- Check the setup, reequiered here
        if self._pk_setup_done == False:
            raise ValueError('This function requieres do run the setup first')
        
        #----- Compute the Pk for the cluster
        Nx, Ny   = self._k2d_norm.shape
        k2d_flat = self._k2d_norm.flatten()

        # Projection
        idx_sort = np.argsort(k2d_flat)
        revidx   = np.argsort(idx_sort)
        k3d_test = np.sort(k2d_flat)
        wok = (k3d_test > 0)
        k3d_test_clean = k3d_test[wok]
        _, pk3d_test_clean = self.model.get_pressure_fluctuation_spectrum(k3d_test_clean*u.kpc**-1)
        pk3d_test      = np.zeros(len(k3d_test))
        pk3d_test[wok] = pk3d_test_clean.to_value('kpc3')
        pk3d_test      = pk3d_test[revidx] # same shape as k2d_flat

        pk2d_flat = pk3d_test * self._conv_pk2d3d # Unit == kpc2, 2d grid

        # Beam and TF
        beam  = self.data.psf_fwhm.to_value('rad')*self.model.D_ang.to_value('kpc')
        TF_k  = self.data.transfer_function['k'].to_value('rad-1') / self.model.D_ang.to_value('kpc')
        TF_tf = self.data.transfer_function['TF']
        pk2d_flat = utils_pk.apply_pk_beam(k2d_flat, pk2d_flat, beam)
        pk2d_flat = utils_pk.apply_pk_transfer_function(k2d_flat, pk2d_flat, TF_k, TF_tf)
    
        # Apply Kmn (multiply_Kmnmn_bis, i.e. without loop, is slower)
        pk2d_K = np.abs(utils_pk.multiply_Kmnmn(np.abs(self._Kmnmn)**2,
                                                pk2d_flat.reshape(Nx, Ny))) / Nx / Ny

        # Bin
        pk2d_mod, _, _ = stats.binned_statistic(k2d_flat, pk2d_K.flatten(),
                                                statistic="mean", bins=self._kedges_kpc)
        
        # Unit
        if physical:
            pk2d_mod = pk2d_mod * u.kpc**2
            k2d      = self._kctr_kpc * u.kpc**-1
        else:
            pk2d_mod = pk2d_mod * self._kpc2arcsec**2 * u.arcsec**2
            k2d      = self._kctr_arcsec * u.arcsec**-1

        #----- Compute the final model
        pk_noise = self._pk2d_noise # baseline is kpc2
        if physical:
            pk_noise = pk_noise * u.kpc**2
        else:
            pk_noise = pk_noise * self._kpc2arcsec**2 * u.arcsec**2

        pk2d_tot = pk2d_mod + self.nuisance_Anoise * pk_noise
        
        return k2d, pk2d_tot
    
    
    #==================================================
    # Data for radial fit
    #==================================================
    
    def get_radial_data(self):
        """
        This function returns the data to be compared with the model in the case of radial
        fitting
        
        Parameters
        ----------

        Outputs
        ----------
        - image (2d np array): the data image

        """

        return self.data.image
    
    
    #==================================================
    # Noise for radial fit
    #==================================================
    
    def get_radial_noise_statistics(self):
        """
        This function returns the noise properties in the case of radial
        fitting
        
        Parameters
        ----------

        Outputs
        ----------
        - noise_rms (2d np array): the rms associated with the image
        - noise_covmat (Npix**2 np array): the noise covariance matrix

        """

        noise_rms = self.data.noise_rms
        if noise_rms is None:
            raise ValueError("Noise rms not available in data. Please use the class Data() to define it")

        if self.method_use_covmat:
            noise_covmat = self.data.noise_covmat
            if noise_covmat is None:
                raise ValueError("Noise covariance not available in data. Use the class Data() to define it.")
        else:
            noise_covmat = None    

        return noise_rms, noise_covmat
    
    
    #==================================================
    # Model for radial fit
    #==================================================
    
    def get_radial_model(self):
        """
        This function returns the model to be compared to the data in the case of radial
        fitting
        
        Parameters
        ----------

        Outputs
        ----------
        - model_image (2d np array): the model image

        """

        #----- Get the cluster model
        model_img = self.model.get_sz_map(seed=None, no_fluctuations=True, force_isotropy=False,
                                          irfs_convolution_beam=self.data.psf_fwhm,
                                          irfs_convolution_TF=self.data.transfer_function)
                
        #----- Compute the final model
        model_img = model_img + self.nuisance_ZL
        
        return model_img

        
    #==================================================
    # Window function
    #==================================================
    
    def get_p3d_to_p2d_from_window_function(self):
        """
        This function compute the window function associated with a pressure 
        profile model. The output window function is a map.
        See e.g., Romero et al. (2023) for details.
        
        Parameters
        ----------

        Outputs
        ----------
        - N_theta (2d np array): the Pk2D to Pk3D conversion coefficient map (real space)

        """

        #----- Get the grid
        Nx, Ny, Nz, proj_reso, proj_reso, los_reso = self.model.get_3dgrid()

        #----- Get the pressure profile model
        pressure3d_sph  = self.model.get_pressure_cube_profile()

        #----- Get the Compton model
        compton_sph = self.model.get_sz_map(no_fluctuations=True)
        compton3d = np.repeat(compton_sph[np.newaxis,:,:], pressure3d_sph.shape[0], axis=0)

        #----- Compute the window function in real space
        W3d = (cst.sigma_T / (cst.m_e * cst.c**2)* pressure3d_sph / compton3d).to_value('kpc-1')

        #----- Window function in Fourier space along kz
        W_ft = np.abs(np.fft.fft(W3d, axis=0))**2 * los_reso**2 # No dimension
        
        #----- Integrate along kz
        k_z = np.fft.fftfreq(Nz, los_reso)
        k_z_sort = np.fft.fftshift(k_z)
        W_ft_sort = np.fft.fftshift(W_ft, axes=0)
        
        N_theta = utils.trapz_loglog(W_ft_sort, k_z_sort, axis=0)*u.kpc**-1
    
        return N_theta
    
    
    #==================================================
    # Window function
    #==================================================
    
    def _get_p2d_from_p3d_from_window_function_exact(self):
        """
        This function computes P2d from P3d by integrating the window function and Pk3d.
        Warning, for large grids, this may take a lot of time! 
        Use it wisely.
        
        Parameters
        ----------

        Outputs
        ----------
        - k2d_norm (2d np array): the values of projected k
        - Pk2d     (2d np array): the 2D power spectrum in a grid, for each k2d_norm

        """

        #----- Get the grid
        Nx, Ny, Nz, proj_reso, proj_reso, los_reso = self.model.get_3dgrid()

        #----- Get the pressure profile model
        pressure3d_sph  = self.model.get_pressure_cube_profile()

        #----- Get the Compton model
        compton_sph = self.model.get_sz_map(no_fluctuations=True)
        compton3d = np.repeat(compton_sph[np.newaxis,:,:], pressure3d_sph.shape[0], axis=0)

        #----- Compute the window function in real space
        W3d = (cst.sigma_T / (cst.m_e * cst.c**2)* pressure3d_sph / compton3d).to_value('kpc-1')

        #----- Window function in Fourier space along kz
        W_ft = np.abs(np.fft.fft(W3d, axis=0))**2 * los_reso**2 # No dimension

        #----- Defines k   
        k_x = np.fft.fftfreq(Nx, proj_reso)
        k_y = np.fft.fftfreq(Ny, proj_reso)
        k_z = np.fft.fftfreq(Nz, los_reso)
        k3d_z, k3d_y, k3d_x = np.meshgrid(k_z, k_y, k_x, indexing='ij')
        k3d_norm = np.sqrt(k3d_x**2 + k3d_y**2 + k3d_z**2)
        k2d_y, k2d_x = np.meshgrid(k_y, k_x, indexing='ij')
        k2d_norm = np.sqrt(k2d_x**2 + k2d_y**2)
        
        #----- Compute Pk3d over sampled k
        k3d_norm_flat = k3d_norm.flatten()
        idx_sort = np.argsort(k3d_norm_flat)  # Index to rearange by sorting 
        revidx = np.argsort(idx_sort)         # Index to invert rearange by sorting
        k3d_norm_flat_sort = np.sort(k3d_norm_flat)
        _, P3d_flat_sort = self.model.get_pressure_fluctuation_spectrum(k3d_norm_flat_sort[1:]*u.kpc**-1)
        P3d_flat_sort = P3d_flat_sort.to_value('kpc3')
        P3d_flat_sort = np.append(np.array([0]), P3d_flat_sort) # Add k=0 back
        P3d_flat = P3d_flat_sort[revidx]       # Unsort
        P3d = P3d_flat.reshape(Nz,Ny,Nx)       # Reshape to k cube
        
        #----- Sort along kz       
        k_z_sort = np.fft.fftshift(k_z)
        W_ft_sort = np.fft.fftshift(W_ft, axes=0)
        P3d_kzsort = np.fft.fftshift(P3d, axes=0)

        #----- integrate along kz
        Pk2d = np.zeros((Ny, Nx)) # Shape kx, ky
        # Make a loop to avoid memory issue (otherwise 5d array kx, ky, x, y, k_z integrated along kz)
        #integrand = P3d_kzsort[:,:,np.newaxis,np.newaxis,:]*W_ft_sort[np.newaxis, np.newaxis, :,:,:] # kx, ky, x, y, k_z
        #Pk2d_xy = utils.trapz_loglog(integrand, k_z_sort, axis=4) #kx, ky, x, y
        #mask_kxky = self.data.mask[np.newaxis, np.newaxis, :,:]
        #Pk2d_exact = np.sum(Pk2d_xy * mask_kxky, axis=(2,3)) / np.sum(mask_kxky, axis=(2,3)) #kpc2
        for ik in range(Ny):
            for jk in range(Nx):
                integrand = P3d_kzsort[:,ik,jk,np.newaxis,np.newaxis]*W_ft_sort
                Pk2d_ikjk_xy = utils.trapz_loglog(integrand, k_z_sort, axis=0)
                Pk2d[ik,jk] = np.sum(Pk2d_ikjk_xy * self.data.mask) / np.sum(self.data.mask)

        return k2d_norm*u.kpc**-1, Pk2d*u.kpc**2
