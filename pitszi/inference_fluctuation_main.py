"""
This file contain the InferenceFluctuation class. 
It is dedicated to extract constraints on model fluctuation parameters.
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

from pitszi.inference_fluctuation_fitting import InferenceFluctuationFitting


#==================================================
# Inference class
#==================================================

class InferenceFluctuation(InferenceFluctuationFitting):
    """ Inference class
        This class infer the pressure fluctuation power spectrum properties 
        given input data (from class Data()) and model (from class Model())
        that are attributes.

    Attributes
    ----------
        - data1 (object from Class Data()): the data object
        - data2 (object from Class Data()): the secondary data object (used for cross spectra)
        - model (object from class Model()): the model object

        # Nuisance parameters
        - nuisance_Anoise (float): nuisance parameter -- the amplitude of the noise
        - nuisance_Abkg (float): nuisance parameter -- extra background amplitude
        - nuisance_bkg_mc1 (2d array): extra background MC realizations for data1
        - nuisance_bkg_mc2 (2d array): extra background MC realizations for data2

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
        - _k2d_norm_kpc (2d array): the norm of k in a grid in kpc
        - _k2d_norm_arcsec (2d array): the norm of k in a grid in arcsec
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
    - get_p3d_to_p2d_from_window_function: derive the window function to go from Pk3d to Pk2d
    - _get_p2d_from_p3d_from_window_function_exact: derive Pk2d given Pk3d by integration over the window function
    
    """    

    #==================================================
    # Initialization
    #==================================================

    def __init__(self,
                 data1,
                 model,
                 data2=None,
                 #
                 nuisance_Anoise=1,
                 nuisance_Abkg=0,
                 nuisance_bkg_mc1=None,
                 nuisance_bkg_mc2=None,
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
        All parameters can be changed on the fly, but changes might require to run the setup.

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
        - data1 (pitszi Class Data object): the data
        - model (pitszi Class Model object): the model
        - data2 (object from Class Data()): the secondary data object (used for cross spectra)

        - nuisance_Anoise (float): noise Pk amplitude, a nuisance parameter.
        - nuisance_Abkg (float): nuisance parameter -- extra background amplitude
        - nuisance_bkg_mc1 (2d array): extra background MC realizations for data1
        - nuisance_bkg_mc2 (2d array): extra background MC realizations for data2

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
            title.show_inference_fluctuation()
        
        #----- Input data and model (deepcopy to avoid modifying the input when fitting)
        self.data1  = copy.deepcopy(data1)
        self.model = copy.deepcopy(model)
        if data2 is not None:
            self.data2 = copy.deepcopy(data2)
            self._cross_spec = True        
        else:
            self.data2 = copy.deepcopy(data1)
            self._cross_spec = False        

        self._validate_data_and_model(silent)
        
        #----- Nuisance parameters
        if data2 is not None:
            self.nuisance_Anoise = 0 # As a reference no noise correlation in the two data
        else:
            self.nuisance_Anoise = 1 # The noise amplitude is one for a single auto spectrum

        # Extra background (e.g. CIB) realization and nuisance parameter as normalization
        if nuisance_bkg_mc1 is not None:
            self.nuisance_Abkg = 1
            self.nuisance_bkg_mc1 = nuisance_bkg_mc1
            if data2 is None:
                self.nuisance_bkg_mc2 = None # Cannot have a bkg for data2 if data2 is not used
            else:
                self.nuisance_bkg_mc2 = nuisance_bkg_mc2
        else:
            self.nuisance_Abkg = 0
            self.nuisance_bkg_mc1 = None
            self.nuisance_bkg_mc2 = None

        #----- Binning in k
        self.kbin_min   = kbin_min
        self.kbin_max   = kbin_max
        self.kbin_Nbin  = kbin_Nbin
        self.kbin_scale = kbin_scale # ['lin', 'log']
        
        #----- Analysis methodology
        self.method_use_covmat = method_use_covmat
        self.method_parallel   = method_parallel
        if method_w8 == None:
            self.method_w8 = np.ones(data1.image.shape)  # The weight applied to image = dy/y x w8
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
        self._pk_setup_done      = False # Tell if the setup was done
        
        self._reso_arcsec        = None  # map resolution in arcsec
        self._reso_kpc           = None  # map resollution in kpc
        self._kpc2arcsec         = None  # kpc to arcsec conversion
                
        self._dy_image1          = None  # fluctuation image data 1
        self._ymap_sphA1         = None  # smooth model subtracted to data image 1
        self._ymap_sphB1         = None  # smooth model in the denominator for data image 1
        self._dy_image2          = None  # fluctuation image data 1
        self._ymap_sphA2         = None  # smooth model subtracted to data image 2
        self._ymap_sphB2         = None  # smooth model in the denominator for data image 2
        
        self._k2d_norm_kpc       = None  # wavenumber norm in 1/kpc
        self._k2d_norm_arcsec    = None  # wavenumber norm in 1/arcsec
        self._kedges_kpc         = None  # wavenumber bin edges in 1/kpc
        self._kedges_arcsec      = None  # wavenumber bin edges in 1/arcsec
        self._kctr_kpc           = None  # wavenumber bin center in 1/kpc
        self._kctr_arcsec        = None  # wavenumber bin center in 1/arcsec
        self._kcount             = None  # number of counts in each k bin
        
        self._pk2d_data          = None  # data power spectrum
        self._pk2d_noise         = None  # noise power spectrum mean
        self._pk2d_noise_rms     = None  # noise power spectrum rms
        self._pk2d_noise_cov     = None  # noise power spectrum covariance matrix
        self._pk2d_noise_invcov  = None  # noise power spectrum inverse covariance matrix
        self._pk2d_modref        = None  # reference model power spectrum mean
        self._pk2d_modref_rms    = None  # reference model power spectrum rms
        self._pk2d_bkg           = None  # extra background power spectrum mean 
        self._pk2d_bkg_rms       = None  # extra background power spectrum rms
        self._pk2d_bkg_cov       = None  # extra background power spectrum covariance matrix
        self._pk2d_totref_invcov = None  # noise plus reference model power spectrum covariance matrix

        self._conv_wf            = None  # window function to go from 2D to 3D power spectrum
        self._conv_pk2d3d        = None  # conversion coefficient from 2d to 3d power spectrum
        
        self._Kmnmn              = None  # Mask mode to mode conversion matrix

        
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
    # Validate data and model
    #==================================================
    
    def _validate_data_and_model(self, silent):
        """
        Check that the data and the model are compatible in terms of sampling.
        
        Parameters
        ----------
        
        Outputs
        ----------
        
        """
        
        h1 = self.data1.header
        h2 = self.data2.header
        h3 = self.model.get_map_header()
        
        c0  = h1['NAXIS']  == h2['NAXIS']  == h3['NAXIS']
        c1  = h1['NAXIS1'] == h2['NAXIS1'] == h3['NAXIS1']
        c2  = h1['NAXIS2'] == h2['NAXIS2'] == h3['NAXIS2']
        c3  = h1['CRPIX1'] == h2['CRPIX1'] == h3['CRPIX1']
        c4  = h1['CRPIX2'] == h2['CRPIX2'] == h3['CRPIX2']
        c5  = h1['CRVAL1'] == h2['CRVAL1'] == h3['CRVAL1']
        c6  = h1['CRVAL2'] == h2['CRVAL2'] == h3['CRVAL2']
        c7  = h1['CDELT1'] == h2['CDELT1'] == h3['CDELT1']
        c8  = h1['CDELT2'] == h2['CDELT2'] == h3['CDELT2']
        c9  = h1['CTYPE1'] == h2['CTYPE1'] == h3['CTYPE1']
        c10 = h1['CTYPE2'] == h2['CTYPE2'] == h3['CTYPE2']
        
        if c0 and c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8 and c9 and c10:
            if not silent:
                print('----- Checking the inputs -----')
                print('      - Data1, data2, and model projection OK.')
        else:
            raise ValueError('The data1, data2 and model projection do not match.')

        
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
        
        images = self.get_pk2d_data_image()
        
        self._dy_image1  = images[0]
        self._ymap_sphA1 = images[1]
        self._ymap_sphB1 = images[2]
        self._dy_image2  = images[3]
        self._ymap_sphA2 = images[4]
        self._ymap_sphB2 = images[5]
        
        #---------- Binning
        if not self.silent: print('    * Setup k binning')

        k2d_x, k2d_y, k2d_norm = self.get_k_grid(physical=True)
        kedges_kpc    = self.get_kedges(physical=True)
        kedges_arcsec = self.get_kedges(physical=False)
        
        self._k2d_norm_kpc    = k2d_norm.to_value('kpc-1')
        self._k2d_norm_arcsec = k2d_norm.to_value('kpc-1') / self._kpc2arcsec
        self._kedges_kpc      = kedges_kpc.to_value('kpc-1')
        self._kedges_arcsec   = kedges_arcsec.to_value('arcsec-1')
        self._kctr_kpc        = 0.5 * (self._kedges_kpc[1:] + self._kedges_kpc[:-1])
        self._kctr_arcsec     = 0.5 * (self._kedges_arcsec[1:] + self._kedges_arcsec[:-1])
        self._kcount          = self.get_kbin_counts()
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
        
        self._pk2d_modref        = model_mean.to_value('kpc2')
        self._pk2d_modref_rms    = np.diag(model_cov.to_value('kpc4'))**0.5
        self._pk2d_modref_cov    = model_cov.to_value('kpc4')

        if self.nuisance_bkg_mc1 is not None:
            _, bkg_mean, bkg_cov = self.get_pk2d_bkg_statistics(physical=True)
            self._pk2d_bkg        = bkg_mean.to_value('kpc2')
            self._pk2d_bkg_rms    = np.diag(bkg_cov.to_value('kpc4'))**0.5
            self._pk2d_bkg_cov    = bkg_cov.to_value('kpc4')
        else:
            self._pk2d_bkg        = 0 * self._pk2d_data
            self._pk2d_bkg_rms    = 0 * self._pk2d_data
            self._pk2d_bkg_cov    = 0 * self._pk2d_noise_cov
            
        if self.method_use_covmat:
            self._pk2d_invcov        = np.linalg.inv(noise_cov.to_value('kpc4') + self._pk2d_bkg_cov)
            self._pk2d_invcov_totref = np.linalg.inv(noise_cov.to_value('kpc4')
                                                     + model_cov.to_value('kpc4')
                                                     + self._pk2d_bkg_cov)
        
        #---------- Convertion
        if not self.silent: print('    * Setup window function conversion')

        ymap_wf = self.get_p3d_to_p2d_from_window_function()
        
        self._conv_wf     = ymap_wf
        self._conv_pk2d3d = (np.nansum(ymap_wf*self.method_w8**2)/np.nansum(self.method_w8**2)).to_value('kpc-1')

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
                      smooth_FWHM=0*u.arcsec,
                      normalize=True):
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
        - normalize (bool): set to True to have a normalized weight map, i.e. sum(w) = Npix

        Outputs
        ----------
        None, the weight are set to the requested map

        """
        
        w8map = np.ones(self.data1.image.shape)

        #===== Apply masks
        # Mask associated with bad data
        if apply_data_mask:
            w8map *= (self.data1.mask * self.data2.mask)**0.5

        # Mask associated with the region of interest
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
                the_beam1 = self.data1.psf_fwhm
                the_beam2 = self.data2.psf_fwhm
            else:
                the_beam1 = None
                the_beam2 = None

            if conv_radial_model_TF:
                the_tf1   = self.data1.transfer_function
                the_tf2   = self.data2.transfer_function
            else:
                the_tf1   = None
                the_tf2   = None
                
            radial_model1 = tmp_mod.get_sz_map(no_fluctuations=True,
                                               irfs_convolution_beam=the_beam1,
                                               irfs_convolution_TF=the_tf1)
            radial_model2 = tmp_mod.get_sz_map(no_fluctuations=True,
                                               irfs_convolution_beam=the_beam2,
                                               irfs_convolution_TF=the_tf2)
            
            # Apply radial model
            w8map *= (radial_model1 * radial_model2)**0.5

        # Smoothing
        if smooth_FWHM>0:
            sigma2fwhm = 2 * np.sqrt(2*np.log(2))
            sigma = smooth_FWHM.to_value('deg')/sigma2fwhm/self.model.get_map_reso().to_value('deg')
            w8map = gaussian_filter(w8map, sigma=sigma)

        # Normalize
        if normalize:
            norm = np.sum(w8map) / len(w8map.flatten())
            w8map = w8map / norm
        
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
        Ny, Nx = self.data1.image.shape

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
        k2d_x, k2d_y = np.meshgrid(k_x, k_y, indexing='xy')
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

        dy_image = (y - model_ymap_sphA) / model_ymap_sphB
        
        Parameters
        ----------
        
        Outputs
        ----------
        - dy_image1 (2d array): image use for Pk analysis corresponding to data1
        - model_ymap_sphA1 (2d array): smooth model numerator data1
        - model_ymap_sphB1 (2d array): smooth model denominator data1
        - dy_image2 (2d array): image use for Pk analysis corresponding to data2
        - model_ymap_sphA2 (2d array): smooth model numerator data2
        - model_ymap_sphB2 (2d array): smooth model denominator data2
        """
        
        #---------- Get the smooth model and the data image
        if self.method_data_deconv:
            img_y1 = utils_pk.deconv_transfer_function(self.data1.image,
                                                       self.model.get_map_reso().to_value('arcsec'), 
                                                       self.data1.psf_fwhm.to_value('arcsec'),
                                                       self.data1.transfer_function, 
                                                       dec_TF_LS=True, dec_beam=True)
            img_y2 = utils_pk.deconv_transfer_function(self.data2.image,
                                                       self.model.get_map_reso().to_value('arcsec'), 
                                                       self.data2.psf_fwhm.to_value('arcsec'),
                                                       self.data2.transfer_function, 
                                                       dec_TF_LS=True, dec_beam=True)
            model_ymap_sphA1 = self.model.get_sz_map(no_fluctuations=True)
            model_ymap_sphA2 = model_ymap_sph1
            model_ymap_sphB1 = model_ymap_sph1
            model_ymap_sphB2 = model_ymap_sph1
            
        else:
            img_y1 = self.data1.image
            img_y2 = self.data2.image
            model_ymap_sphA1 = self.model.get_sz_map(no_fluctuations=True,
                                                     irfs_convolution_beam=self.data1.psf_fwhm,
                                                     irfs_convolution_TF=self.data1.transfer_function)    
            model_ymap_sphA2 = self.model.get_sz_map(no_fluctuations=True,
                                                     irfs_convolution_beam=self.data2.psf_fwhm,
                                                     irfs_convolution_TF=self.data2.transfer_function)
            
            model_ymap_sphB1 = self.model.get_sz_map(no_fluctuations=True,
                                                     irfs_convolution_beam=self.data1.psf_fwhm)
            model_ymap_sphB2 = self.model.get_sz_map(no_fluctuations=True,
                                                     irfs_convolution_beam=self.data2.psf_fwhm)            
            
        #---------- Compute the data to be used for Pk
        delta_y1    = img_y1 - model_ymap_sphA1
        delta_y2    = img_y2 - model_ymap_sphA2
        dy_image1 = (delta_y1 - np.mean(delta_y1)) / model_ymap_sphB1 * self.method_w8
        dy_image2 = (delta_y2 - np.mean(delta_y2)) / model_ymap_sphB2 * self.method_w8
        dy_image1[model_ymap_sphB1 <=0] = 0 # Ensure that bad pixels are set to zero, e.g. beyond R_trunc
        dy_image2[model_ymap_sphB2 <=0] = 0
        
        return dy_image1, model_ymap_sphA1, model_ymap_sphB1, dy_image2, model_ymap_sphA2, model_ymap_sphB2
    
    
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
        dy1, _, _, dy2, _, _ = self.get_pk2d_data_image() 
        
        #---------- Pk for the data
        kedges = self.get_kedges().to_value('arcsec-1')
        reso   = self.model.get_map_reso().to_value('arcsec')
        k2d, data_pk2d = utils_pk.extract_pk2d(dy1, reso, image2=dy2, kedges=kedges)

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
        _, _, model_ymap_sphB1, _, _, model_ymap_sphB2 = self.get_pk2d_data_image()

        #----- Extract noise MC realization
        noise_ymap_mc1 = self.data1.noise_mc
        noise_ymap_mc2 = self.data2.noise_mc
        Nmc01 = noise_ymap_mc1.shape[0]
        Nmc02 = noise_ymap_mc2.shape[0]
        if noise_ymap_mc1 is None or noise_ymap_mc2 is None:
            raise ValueError("Noise MC not available in data. Please use the class Data() to define it.")
        if (Nmc01 < 100 or Nmc02 < 100) and not self.silent:
            print('WARNING: the number of MC realizations to use is less than 100. This might not be enough.')

        # Redefine the number of MC upon request
        if Nmc is not None:
            if Nmc01 < Nmc or Nmc02 < Nmc: # just use Nmc0 if the requested number of MC is larger than available
                Nmc = np.amin(np.array([Nmc01, Nmc02]))
                print('WARNING: the requested number of MC realizations is larger than what is available in data')
        else:
            Nmc = np.amin(np.array([Nmc01, Nmc02]))

        #----- Compute Pk2d MC realization
        noise_pk2d_mc = np.zeros((Nmc, len(kedges)-1))
        for imc in range(Nmc):
            
            # Account for deconvolution choices
            if self.method_data_deconv:
                img_y1 = utils_pk.deconv_transfer_function(noise_ymap_mc1[imc,:,:],
                                                           self.model.get_map_reso().to_value('arcsec'), 
                                                           self.data1.psf_fwhm.to_value('arcsec'),
                                                           self.data1.transfer_function, 
                                                           dec_TF_LS=True, dec_beam=True)
                img_y2 = utils_pk.deconv_transfer_function(noise_ymap_mc2[imc,:,:],
                                                           self.model.get_map_reso().to_value('arcsec'), 
                                                           self.data2.psf_fwhm.to_value('arcsec'),
                                                           self.data2.transfer_function, 
                                                           dec_TF_LS=True, dec_beam=True)
            else:
                img_y1 = noise_ymap_mc1[imc,:,:]
                img_y2 = noise_ymap_mc2[imc,:,:]
                
            # Noise to Pk
            image_noise_mc1 = (img_y1 - np.mean(img_y1))/model_ymap_sphB1 * self.method_w8
            image_noise_mc2 = (img_y2 - np.mean(img_y2))/model_ymap_sphB2 * self.method_w8
            image_noise_mc1[model_ymap_sphB1 <= 0] = 0
            image_noise_mc2[model_ymap_sphB2 <= 0] = 0
            
            k2d, pk_mc =  utils_pk.extract_pk2d(image_noise_mc1, reso, image2=image_noise_mc2, kedges=kedges)
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
                                  physical=False):
        """
        This function compute the model variance properties associated
        with the reference input model.
        
        Parameters
        ----------
        - Nmc (int): number of monte carlo realization
        - physical (bool): set to true to have output in kpc units, else arcsec units

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
        _, model_ymap_sphA1, model_ymap_sphB1, _, model_ymap_sphA2, model_ymap_sphB2  = self.get_pk2d_data_image()
        
        #----- Compute Pk2d MC realization
        model_pk2d_mc     = np.zeros((Nmc, len(kedges)-1))
        model_pk2d_covmat = np.zeros((len(kedges)-1, len(kedges)-1))
        
        for imc in range(Nmc):

            # Account or not for TF and beam
            if self.method_data_deconv:
                test_ymap1 = self.model.get_sz_map(no_fluctuations=False)
                test_ymap2 = test_ymap1
            else:
                seed = int(np.random.uniform(0,1000000))
                test_ymap1 = self.model.get_sz_map(seed=seed, no_fluctuations=False,
                                                   irfs_convolution_beam=self.data1.psf_fwhm,
                                                   irfs_convolution_TF=self.data1.transfer_function)
                if self._cross_spec:
                    test_ymap2 = self.model.get_sz_map(seed=seed, no_fluctuations=False,
                                                       irfs_convolution_beam=self.data2.psf_fwhm,
                                                       irfs_convolution_TF=self.data2.transfer_function)
                else:
                    test_ymap2 = test_ymap1
                
            # Test image
            delta_y1 = test_ymap1 - model_ymap_sphA1
            delta_y2 = test_ymap2 - model_ymap_sphA2
            test_image1 = (delta_y1 - np.mean(delta_y1))/model_ymap_sphB1 * self.method_w8
            test_image2 = (delta_y2 - np.mean(delta_y2))/model_ymap_sphB2 * self.method_w8
            test_image1[model_ymap_sphB1 <= 0] = 0
            test_image2[model_ymap_sphB2 <= 0] = 0

            # Pk
            k2d, pk_mc =  utils_pk.extract_pk2d(test_image1, reso, image2=test_image2, kedges=kedges)
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
    # Compute extra background statistics
    #==================================================
    
    def get_pk2d_bkg_statistics(self, physical=False):
        """
        This function compute the extra background properties
        
        Parameters
        ----------
        - physical (bool): set to true to have output in kpc units, else arcsec units

        Outputs
        ----------
        - k2d (1d array): the values of k in each bin
        - bkg_pk2d_ref (1d array): the bkg mean
        - bkg_pk2d_covmat (2d array): the bkg covariance matrix

        """

        #----- Check
        if self.nuisance_bkg_mc1 is None:
            raise ValueError('No background MC provided. Cannot compute Bkg statistics.')
        
        #----- Sanity check
        bin_counts = self.get_kbin_counts()
        if np.amin(bin_counts) == 0:
            raise ValueError('Some bins have zero counts. Please redefine the binning to avoid this issue.')
        
        #----- Useful info
        kedges = self.get_kedges().to_value('arcsec-1')
        reso   = self.model.get_map_reso().to_value('arcsec')

        #----- Model accounting/or not for beam and TF
        _, _, model_ymap_sphB1, _, _, model_ymap_sphB2 = self.get_pk2d_data_image()

        #----- Extract bkg MC realization
        bkg_ymap_mc1 = self.nuisance_bkg_mc1
        if self.nuisance_bkg_mc2 is not None:
            bkg_ymap_mc2 = self.nuisance_bkg_mc2
        else:
            bkg_ymap_mc2 = self.nuisance_bkg_mc1

        Nmc01 = bkg_ymap_mc1.shape[0]
        Nmc02 = bkg_ymap_mc2.shape[0]
        if bkg_ymap_mc1 is None or bkg_ymap_mc2 is None:
            raise ValueError("Bkg MC not available.")
        if (Nmc01 < 100 or Nmc02 < 100) and not self.silent:
            print('WARNING: the number of MC realizations to use is less than 100. This might not be enough.')

        # Redefine the number of MC
        Nmc = np.amin(np.array([Nmc01, Nmc02]))

        #----- Compute Pk2d MC realization
        bkg_pk2d_mc = np.zeros((Nmc, len(kedges)-1))
        for imc in range(Nmc):
            
            # Account for deconvolution choices
            if self.method_data_deconv:
                img_y1 = utils_pk.deconv_transfer_function(bkg_ymap_mc1[imc,:,:],
                                                           self.model.get_map_reso().to_value('arcsec'), 
                                                           self.data1.psf_fwhm.to_value('arcsec'),
                                                           self.data1.transfer_function, 
                                                           dec_TF_LS=True, dec_beam=True)
                img_y2 = utils_pk.deconv_transfer_function(bkg_ymap_mc2[imc,:,:],
                                                           self.model.get_map_reso().to_value('arcsec'), 
                                                           self.data2.psf_fwhm.to_value('arcsec'),
                                                           self.data2.transfer_function, 
                                                           dec_TF_LS=True, dec_beam=True)
            else:
                img_y1 = bkg_ymap_mc1[imc,:,:]
                img_y2 = bkg_ymap_mc2[imc,:,:]
                
            # Noise to Pk
            image_bkg_mc1 = (img_y1 - np.mean(img_y1))/model_ymap_sphB1 * self.method_w8
            image_bkg_mc2 = (img_y2 - np.mean(img_y2))/model_ymap_sphB2 * self.method_w8
            image_bkg_mc1[model_ymap_sphB1 <= 0] = 0
            image_bkg_mc2[model_ymap_sphB2 <= 0] = 0
            
            k2d, pk_mc =  utils_pk.extract_pk2d(image_bkg_mc1, reso, image2=image_bkg_mc2, kedges=kedges)
            bkg_pk2d_mc[imc,:] = pk_mc
            
        # Noise statistics
        bkg_pk2d_mean = np.mean(bkg_pk2d_mc, axis=0)
        bkg_pk2d_rms  = np.std(bkg_pk2d_mc, axis=0)

        #----- Compute covariance
        bkg_pk2d_covmat = np.zeros((len(kedges)-1, len(kedges)-1))
        for imc in range(Nmc):
            bkg_pk2d_covmat += np.matmul((bkg_pk2d_mc[imc,:]-bkg_pk2d_mean)[:,None],
                                         (bkg_pk2d_mc[imc,:]-bkg_pk2d_mean)[None,:])
        bkg_pk2d_covmat /= Nmc

        #----- Sanity check again
        if np.sum(np.isnan(bkg_pk2d_covmat)) > 0:
            if not self.silent:
                print('Some pixels in the covariance matrix are NaN.')
            raise ValueError('Issue with noise covariance matrix')

        #----- Units
        if physical:
            kpc2arcsec        = ((1*u.kpc/self.model.D_ang).to_value('')*u.rad).to_value('arcsec')
            k2d               = k2d * kpc2arcsec**1 * u.kpc**-1
            bkg_pk2d_mean   = bkg_pk2d_mean * kpc2arcsec**-2 * u.kpc**2
            bkg_pk2d_covmat = bkg_pk2d_covmat * kpc2arcsec**-4 * u.kpc**4
        else:
            k2d               = k2d * u.arcsec**-1
            bkg_pk2d_mean   = bkg_pk2d_mean * u.arcsec**2
            bkg_pk2d_covmat = bkg_pk2d_covmat * u.arcsec**4

        #---------- return
        return k2d, bkg_pk2d_mean, bkg_pk2d_covmat

    
    #==================================================
    # Model for pk2d fit brute force
    #==================================================
    
    def get_pk2d_model_brute(self,
                             physical=False):
        """
        This function returns the model of pk for the brute force
        approach. This is the same as what is done to get the model variance properties.
        
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
        k2d, pk2d_modvar, _ = self.get_pk2d_model_statistics(physical=physical, Nmc=1)

        #----- Compute the final model
        pk_noise = self._pk2d_noise # baseline is kpc2
        pk_bkg   = self._pk2d_bkg     # baseline is kpc2
        if physical:
            pk_noise = pk_noise * u.kpc**2
            pk_bkg   = pk_bkg   * u.kpc**2
        else:
            pk_noise = pk_noise * self._kpc2arcsec**2 * u.arcsec**2
            pk_bkg = pk_bkg     * self._kpc2arcsec**2 * u.arcsec**2

        pk2d_tot = pk2d_modvar + self.nuisance_Anoise * pk_noise + self.nuisance_Abkg * pk_bkg

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
        Ny, Nx   = self._k2d_norm_kpc.shape
        k2d_flat = self._k2d_norm_kpc.flatten()

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
        beam1  = self.data1.psf_fwhm.to_value('rad')*self.model.D_ang.to_value('kpc')
        TF_k1  = self.data1.transfer_function['k'].to_value('rad-1') / self.model.D_ang.to_value('kpc')
        TF_tf1 = self.data1.transfer_function['TF']
        beam2  = self.data2.psf_fwhm.to_value('rad')*self.model.D_ang.to_value('kpc')
        TF_k2  = self.data2.transfer_function['k'].to_value('rad-1') / self.model.D_ang.to_value('kpc')
        TF_tf2 = self.data2.transfer_function['TF']
        pk2d_flat = utils_pk.apply_pk_beam(k2d_flat, pk2d_flat, beam1, beamFWHM2=beam2)
        pk2d_flat = utils_pk.apply_pk_transfer_function(k2d_flat, pk2d_flat, TF_k1, TF_tf1,
                                                        TF_k2=TF_k2, TF2=TF_tf2)
    
        # Apply Kmn (multiply_Kmnmn_bis, i.e. without loop, is slower)
        pk2d_K = np.abs(utils_pk.multiply_Kmnmn(np.abs(self._Kmnmn)**2,
                                                pk2d_flat.reshape(Ny, Nx))) / Nx / Ny

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
        pk_bkg   = self._pk2d_bkg     # baseline is kpc2
        if physical:
            pk_noise = pk_noise * u.kpc**2
            pk_bkg   = pk_bkg   * u.kpc**2
        else:
            pk_noise = pk_noise * self._kpc2arcsec**2 * u.arcsec**2
            pk_bkg   = pk_bkg   * self._kpc2arcsec**2 * u.arcsec**2
                    
        pk2d_tot = pk2d_mod + self.nuisance_Anoise * pk_noise + self.nuisance_Abkg * pk_bkg
        
        return k2d, pk2d_tot
    
            
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

        N_theta[np.isnan(N_theta.value)] = 0
        
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
        mask = (self.data1.mask * self.data2.mask)**0.5
        for ik in range(Ny):
            for jk in range(Nx):
                integrand = P3d_kzsort[:,ik,jk,np.newaxis,np.newaxis]*W_ft_sort
                Pk2d_ikjk_xy = utils.trapz_loglog(integrand, k_z_sort, axis=0)
                Pk2d[ik,jk] = np.sum(Pk2d_ikjk_xy * mask) / np.sum(mask)

        return k2d_norm*u.kpc**-1, Pk2d*u.kpc**2
