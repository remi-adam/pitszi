"""
This file contain the InferenceRadial class. 
It is dedicated to extract constraints on radial model parameters.
"""

#==================================================
# Requested imports
#==================================================

import os
import pprint
import copy
import pickle
import dill

from pitszi import title
from pitszi.inference_radial_fitting import InferenceRadialFitting


#==================================================
# Inference class
#==================================================

class InferenceRadial(InferenceRadialFitting):
    """ Inference class
        This class infer the radial profile given input
        data (from class Data()) and model (from class Model())
        that are attributes

    Attributes
    ----------
        - data (object from Class Data()): the data object
        - model (object from class Model()): the model object

        # Nuisance parameters
        self.nuisance_ZL (float): nuisance parameter -- the map zero level

        # Analysis methodology
        - method_use_covmat (bool): use covariance matrix in fit?
        - method_parallel (bool): use parallelization in fit

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
        
    Methods
    ----------
    - print_param: print the current inference parameters
    - save_inference: save the class object state
    - load_inference: load the class object
    - get_radial_data: get the data map in the case of the radial component (i.e. the y map)
    - get_radial_noise_statistics: get the noise properties in the case of the radail component
    - get_radial_model: compute the model in the case of the radial component
    
    """    

    #==================================================
    # Initialization
    #==================================================

    def __init__(self,
                 data,
                 model,
                 #
                 nuisance_ZL=0,
                 #
                 method_use_covmat=False,
                 method_parallel=False,
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

        Parameters
        ----------
        - data (pitszi Class Data object): the data
        - model (pitszi Class Model object): the model
        - nuisance_ZL (float): map zero level, a nuisance parameter.
        - method_use_covmat (bool): use covariance matrix in the fit
        - method_use_paral (bool): use parallelization      
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
            title.show_inference_radial()
        
        #----- Input data and model (deepcopy to avoid modifying the input when fitting)
        self.data  = copy.deepcopy(data)
        self.model = copy.deepcopy(model)

        #----- Nuisance parameters
        self.nuisance_ZL     = 0

        #----- Analysis methodology
        self.method_use_covmat = method_use_covmat
        self.method_parallel = method_parallel

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
        model_img = self.model.get_sz_map(no_fluctuations=True, force_isotropy=False,
                                          irfs_convolution_beam=self.data.psf_fwhm,
                                          irfs_convolution_TF=self.data.transfer_function)
                
        #----- Compute the final model
        model_img = model_img + self.nuisance_ZL
        
        return model_img

