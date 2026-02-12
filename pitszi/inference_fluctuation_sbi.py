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
    # Compute results for a given SBI sampling
    #==================================================
    
    def get_sbi_chains_outputs_results(self,
                                        parlist,
                                        parinfo,
                                        posterior,
                                        conf=68.0,
                                        truth=None,
                                        extname=''):
        """
        This function can be used to produce automated plots/files
        that give the results of the SBI samping
        
        Parameters
        ----------
        - parlist (list): the list of parameter names
        - posterior (sbi posterior): the posterior
        - conf (float): confidence limit in % used in results
        - truth (list): the list of expected parameter value for the fit
        - extname (string): extra name to add after MCMC in file name

        Outputs
        ----------
        Plots are produced

        """

        #---------- Get the chains and lnL
        observable = self.get_pk2d_data(physical=True)[1]
        samp = posterior.sample((100000,), x=observable).numpy()
        log_prob = posterior.log_prob(samp, observable).numpy()
        Nsample, Nparam = samp.shape
        
        #---------- Check the truth
        if truth is not None:
            if len(truth) != Nparam:
                raise ValueError("The 'truth' keyword should match the number of parameters")

        #---------- Add log if needed
        parlist_stat = []
        for ipar in range(Nparam):
            parlist_stat.append(parlist[ipar])
            if 'sampling' in parinfo[parlist[ipar]]:
                if parinfo[parlist[ipar]]['sampling']: parlist_stat[ipar]= 'log '+parlist_stat[ipar]
        
        #---------- Compute statistics for the chains
        utils_fitting.chains_statistics(samp[np.newaxis,:], log_prob[np.newaxis,:],
                                        parname=parlist_stat,
                                        conf=conf,
                                        outfile=self.output_dir+'/SBI'+extname+'_chain_statistics.txt')

        #---------- Produce 1D plots of the chains
        utils_plot.chains_1Dplots(samp[np.newaxis,:], parlist_stat, self.output_dir+'/SBI'+extname+'_chain_1d_plot.pdf')
        
        #---------- Produce 1D histogram of the chains
        namefiles = [self.output_dir+'/SBI'+extname+'_chain_hist_'+i+'.pdf' for i in parlist]
        utils_plot.chains_1Dhist(samp[np.newaxis,:], parlist_stat, namefiles,
                                 conf=conf, truth=truth)

        #---------- Produce 2D (corner) plots of the chains
        utils_plot.chains_2Dplots_corner(samp[np.newaxis,:],
                                         parlist_stat,
                                         self.output_dir+'/SBI'+extname+'_chain_2d_plot_corner.pdf',
                                         truth=truth)


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
            self.get_sbi_chains_outputs_results(par_list, parinfo, posterior,
                                                truth=true_param,
                                                extname=extname)
            self.run_sbi_fluctuation_results(posterior, parinfo,
                                             true_pk3d=true_pk3d,
                                             extname=extname)
        
        #========== Compute the best-fit model and set it
        if set_bestfit:
            samp = posterior.sample((100000,), x=observable).numpy()
            log_prob = posterior.log_prob(samp, observable)
            best_par = (samp[log_prob == log_prob.max()])[0]            
            self.setpar_fluctuation(best_par, parinfo)
        else:
            self.model = input_model

        return par_list, inference, posterior, samples









    
    #==================================================
    # Show the SBI fit results related to fluctuation
    #==================================================
    
    def run_sbi_fluctuation_results(self, posterior, parinfo,
                                    true_pk3d=None, extname='_Fluctuation'):
        """
        This function is used to show the results of the SBI fit
        regarding the fluctuation
            
        Parameters
        ----------
        - posterior (sbi object): the posterior object obtained within SBI
        - parinfo (dict): same as mcmc_fluctuation
        - true_pk3d (dict): pass a dictionary containing the spectrum to compare with
        in the form {'k':array in kpc-1, 'pk':array in kpc3}
        - extname (string): extra name to add after MCMC results file names
        
        Outputs
        ----------
        plots are produced

        """

        #========== Get noise MC 
        noise_mc1 = self.data1.noise_mc
        noise_mc2 = self.data2.noise_mc
        if noise_mc1.shape[0] >= self.sbi_Nresamp or noise_mc2.shape[0] >= self.sbi_Nresamp:
            Nmc = self.sbi_Nresamp
            noise_mc1 = noise_mc1[0:Nmc]
            noise_mc2 = noise_mc2[0:Nmc]
        else:
            Nmc = np.amin(np.array([noise_mc1.shape[0], noise_mc2.shape[0]]))
            if self.silent == False: print('WARNING: the number of noise MC is lower than requested')
            
        #========== Get the best-fit
        observable = self.get_pk2d_data(physical=True)[1]
        samp = posterior.sample((100000,), x=observable).numpy()
        log_prob = posterior.log_prob(samp, observable)
        best_par = (samp[log_prob == log_prob.max()])[0]            
        self.setpar_fluctuation(best_par, parinfo)        
        k3d, best_pk3d = self.model.get_pressure_fluctuation_spectrum(np.logspace(-4,-1,100)*u.kpc**-1)
        k2d, model_pk2d_ref, model_pk2d_covmat, _ = self.get_pk2d_model_statistics(physical=True,
                                                                                   Nmc=self.sbi_Nresamp)
        best_pk2d_noise = self.nuisance_Anoise * self._pk2d_noise
        best_pk2d_bkg   = self.nuisance_Abkg   * self._pk2d_bkg

        #========== Get the MC
        MC_pk3d = np.zeros((self.sbi_Nresamp, len(k3d)))
        MC_pk2d = np.zeros((self.sbi_Nresamp, len(k2d)))
        MC_pk2d_noise = np.zeros((self.sbi_Nresamp, len(k2d)))
        MC_pk2d_bkg = np.zeros((self.sbi_Nresamp, len(k2d)))
        
        MC_pars = posterior.sample((self.sbi_Nresamp,), x=observable).numpy()
        for imc in range(self.sbi_Nresamp):
            # Get MC model
            self.setpar_fluctuation(MC_pars[imc,:], parinfo)

            # Get MC noise
            MC_pk2d_noise[imc,:] = self.nuisance_Anoise*utils_pk.pk_data_augmentation(k2d,self._pk2d_noise_mc,Nsim=1,method='LogNormCov')[0]

            # Get MC bkg
            if self.nuisance_bkg_mc1 is not None and np.std(self._pk2d_bkg_mc) != 0:
                MC_pk2d_bkg[imc,:] = self.nuisance_Abkg*utils_pk.pk_data_augmentation(k2d,self._pk2d_bkg_mc,Nsim=1,method='LogNormCov')[0]
            
            # Get MC Pk2d
            MC_pk2d[imc,:] = self.get_pk2d_model_brute(physical=True, include_noise=False, include_bkg=False)[1].to_value('kpc2')

            # Get MC Pk3d
            MC_pk3d[imc,:] = self.model.get_pressure_fluctuation_spectrum(k3d)[1].to_value('kpc3')

        #========== Plot the fitted image data
        utils_plot.show_input_delta_ymap(self.output_dir+'/SBI'+extname+'_results_input_image1.pdf',
                                         self.data1.image,
                                         self._dy_image1,
                                         self._ymap_sphA1,
                                         self._ymap_sphB1,
                                         self.method_w8,
                                         self.data1.header,
                                         noise_mc1,
                                         mask=self.data1.mask,
                                         visu_smooth=10)
        utils_plot.show_input_delta_ymap(self.output_dir+'/SBI'+extname+'_results_input_image2.pdf',
                                         self.data2.image,
                                         self._dy_image2,
                                         self._ymap_sphA2,
                                         self._ymap_sphB2,
                                         self.method_w8,
                                         self.data2.header,
                                         noise_mc2,
                                         mask=self.data2.mask,
                                         visu_smooth=10)
        
        #========== Plot the covariance matrix
        if self.nuisance_bkg_mc1 is not None:
            covmat_bkg = self._pk2d_bkg_cov
        else:
            covmat_bkg = None
        utils_plot.show_fit_result_covariance(self.output_dir+'/SBI'+extname+'_results_covariance.pdf',
                                              self._pk2d_noise_cov,
                                              model_pk2d_covmat.to_value('kpc4'),
                                              covmat_model=self._pk2d_modref_cov,
                                              covmat_bkg=covmat_bkg)
        
        #========== Plot the Pk2d constraint
        utils_plot.show_fit_result_pk2d(self.output_dir+'/SBI'+extname+'_results_pk2d.pdf',
                                        self._kctr_kpc, self._pk2d_data,
                                        model_pk2d_ref.to_value('kpc2'),
                                        np.diag(model_pk2d_covmat.to_value('kpc4'))**0.5,
                                        self._pk2d_noise_rms, self._pk2d_bkg_rms,
                                        MC_pk2d,
                                        best_pk2d_noise, MC_pk2d_noise,
                                        best_pk2d_bkg, MC_pk2d_bkg)

        #========== Plot the Pk3d constraint
        utils_plot.show_fit_result_pk3d(self.output_dir+'/SBI'+extname+'_results_pk3d.pdf',
                                        k3d.to_value('kpc-1'), best_pk3d.to_value('kpc3'),
                                        MC_pk3d,
                                        true_pk3d=true_pk3d)
