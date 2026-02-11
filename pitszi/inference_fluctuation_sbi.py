"""
This file deals with the model fluctuation fitting using an SBI approach

"""

import numpy as np
import copy
import astropy.units as u
import sbi
from sbi import utils as utils
from sbi.inference import infer, SNPE
import torch
from joblib import dump, load
from tqdm import tqdm

from pitszi import utils_fitting
from pitszi import utils_pk
from pitszi import utils_plot


#==================================================
# InferenceFitting class
#==================================================

class InferenceFluctuationSBI(object):
    """ InferenceFluctuationSBI class
    This class serves as a parser to the main Inference class, to
    include the subclass InferenceFluctuationSBI in this other file.

    Attributes
    ----------
    The attributes are the same as the Inference class, see inference_fluctuation_main.py

    Methods
    ----------
    - sbi_fluctuation_simulator: fucntion that simulate a 2D power spectrum
    
    """


    #==================================================
    # SBI simulator
    #==================================================
    
    def sbi_fluctuation_simulator(self, param, parinfo):
        """
        This function generate a mock simulation given parameters
        
        Parameters
        ----------
        - param (np array): the value of the test parameters
        - parinfo_fluct (dictionary): the model parameters associated with 
        self.model.model_pressure_fluctuation to be fit as, e.g., 
        parinfo      = {'Norm':                     # --> Parameter key (mandatory)
                        {'guess':[0.5, 0.3],        # --> initial guess: center, uncertainty (mandatory)
                         'unit': None,              # --> unit (mandatory, None if unitless)
                         'limit':[0, np.inf],       # --> Allowed range, i.e. flat prior (optional)
                         'prior':[0.5, 0.1],        # --> Gaussian prior: mean, sigma (optional)
                         'sampling':'log',          # --> sampling of the parameter (log or lin). If log, guess/lim/prior should be in log
                        },
                        'slope':
                        {'limit':[-11/3-1, -11/3+1], 
                          'unit': None, 
                        },
                        'L_inj':
                        {'limit':[0, 10], 
                          'unit': u.Mpc, 
                        },
                       }  
        Other accepted parameters: 'Anoise'

        Outputs
        ----------
        - mock_pk2d (np.array): the mock power spectrum in 2D
        """

        #---------- Set the model parameters
        self.setpar_fluctuation(param, parinfo)
    
        #---------- Extract the new pk2d
        k2d, mock_pk2d = self.get_pk2d_model_brute(physical=True)
        
        #---------- Security check to get ride of NaN and Inf
        if not np.all(np.isfinite(mock_pk2d)):
            print(f"[Warning] simulator: NaN or Inf detected for parameters={parameters}. Correction applied.")
            import pdb
            pdb.set_trace()
            pk2d = np.nan_to_num(pk2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        return mock_pk2d

    
    #==================================================
    # SBI fluctuation
    #==================================================
    
    def run_sbi_fluctuation(self, parinfo,
                            filename_training=None,
                            show_fit_result=False,
                            set_bestfit=False,
                            extname='_Fluctuation',
                            true_pk3d=None,
                            true_param=None):
        """
        This function train the neural network
        
        Parameters
        ----------
        - parinfo_fluct (dictionary): the model parameters associated with 
        self.model.model_pressure_fluctuation to be fit as, e.g., 
        parinfo      = {'Norm':                     # --> Parameter key (mandatory)
                        {'guess':[0.5, 0.3],        # --> initial guess: center, uncertainty (mandatory)
                         'unit': None,              # --> unit (mandatory, None if unitless)
                         'limit':[0, np.inf],       # --> Allowed range, i.e. flat prior (optional)
                         'prior':[0.5, 0.1],        # --> Gaussian prior: mean, sigma (optional)
                         'sampling':'log',          # --> sampling of the parameter (log or lin). If log, guess/lim/prior should be in log
                        },
                        'slope':
                        {'limit':[-11/3-1, -11/3+1], 
                          'unit': None, 
                        },
                        'L_inj':
                        {'limit':[0, 10], 
                          'unit': u.Mpc, 
                        },
                       }  
        Other accepted parameters: 'Anoise'

        - filename_training (str): the file name of the training set to use.
        - show_fit_result (bool): set to true to produce plots for fitting results
        - extname (string): extra name to add after MCMC results file names
        - set_bestfit (bool): set the best fit to the model
        - true_pk3d (dict): pass a dictionary containing the spectrum to compare with
        in the form {'k':array in kpc-1, 'pk':array in kpc3}
        - true_param (list): the list of expected parameter value for the fit
       
        Outputs
        ----------
        - parlist (list): the list of the fit parameters
        - posterior (tbdsbi object): the posterior associated with model parameters of
        the smooth component
        """
        
        #========== Check the setup
        print('----- Checking the Pk setup -----')
        if self._pk_setup_done :
            print('      The setup was done.')
            print('      We can proceed, but make sure that it was done with the correct analysis framework.')
        else:
            print('      The setup was not done.')
            print('      Run pk_setup() with the correct analysis framework before proceeding.')

        #========== Copy the input model
        input_model = copy.deepcopy(self.model)

        #========== Check if the training was already recorded
        if filename_training is None:
            training_file = self.output_dir+'/pitszi_SBI'+extname+'_training.h5'
        else:
            training_file = filename_training
            
        sampler_exist = utils_fitting.check_sampler_exist(training_file, silent=False)

        #========== Wrapper to the simulator
        def simulator(theta):
            xs = []
            for t in tqdm(theta, desc="Simulating"):
                x = self.sbi_fluctuation_simulator(t, parinfo)
                xs.append(torch.as_tensor(x, dtype=torch.float32))
            return torch.stack(xs)

        #========== Fit parameter list and information
        par_list, _, _, par_min, par_max = self.defpar_fluctuation(parinfo)
        ndim = len(par_list)

        # Info
        if not self.silent:
            print('----- Fit parameters information -----')
            print('      - Fitted parameters:            ')
            print(par_list)
            print('      - Minimal value:                ')
            print(par_min)
            print('      - Maximal value:                ')
            print(par_max)
            print('      - Number of dimensions:         ')
            print(ndim)
            print('      - Parallel mode:                ')
            print(self.method_parallel)
            print('-----')

        #========== Define the prior
        prior = utils.BoxUniform(low=torch.tensor(par_min), high=torch.tensor(par_max))
        
        #========== Run the training
        if not sampler_exist or self.sbi_reset:
            inference = SNPE(prior=prior)
        else:
            inference = torch.load(training_file, weights_only=False)

        if self.sbi_run:
            print('      --- Runing new simulations...')
            theta = prior.sample((self.sbi_nsteps,))           # get new parameters from the prior
            x = simulator(theta)                               # get the new simulations
            inference = inference.append_simulations(theta, x) # append simulation to inference
            print('      --- Training the network...')
            inference.train(force_first_round_loss=True)       # train the network using new simulations
            torch.save(inference, training_file)               # save the inference for latter use
            
        #========== Build posterior at the end
        print('      --- Getting the posterior...')
        posterior = inference.build_posterior()

        #========== Apply the SBI fit
        observable = self.get_pk2d_data(physical=True)[1]
        samples = posterior.sample((self.sbi_Nresamp,), x=observable).numpy()

        #========== Show results
        if show_fit_result:
            self.get_sbi_chains_outputs_results(par_list, parinfo, sampler,
                                                truth=true_param,
                                                extname=extname)
            self.run_sbi_fluctuation_results(sampler, parinfo,
                                             true_pk3d=true_pk3d,
                                             extname=extname)
        
        #========== Compute the best-fit model and set it
        if set_bestfit:
            log_prob = posterior.log_prob(samples, observable)
            best_par = (samples[log_prob == log_prob.max()])[0]            
            self.setpar_fluctuation(best_par, parinfo)
        else:
            self.model = input_model

        return par_list, inference, posterior, samples
