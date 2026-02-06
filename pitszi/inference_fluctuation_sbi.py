"""
This file deals with the model fluctuation fitting using an SBI approach

"""

import numpy as np
import copy
import astropy.units as u

from pitszi import utils_fitting
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
    - get_sbi_chains_outputs_results: utility tools to get mcmc chain results
    - run_sbi_fluctuation: run the mcmc fit
    - run_sbi_fluctuation_results: produce the results given the mcmc fit
    """

    #==================================================
    # Compute results for a given MCMC sampling
    #==================================================
    
    def get_mcmc_chains_outputs_results(self,
                                        parlist,
                                        parinfo,
                                        sampler,
                                        conf=68.0,
                                        truth=None,
                                        extname=''):
        """
        This function can be used to produce automated plots/files
        that give the results of the MCMC samping
        
        Parameters
        ----------
        - parlist (list): the list of parameter names
        - sampler (emcee object): the sampler
        - conf (float): confidence limit in % used in results
        - truth (list): the list of expected parameter value for the fit
        - extname (string): extra name to add after MCMC in file name

        Outputs
        ----------
        Plots are produced

        """

        Nchains, Nsample, Nparam = sampler.chain.shape
        
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
                    
        #---------- Remove the burnin
        if self.mcmc_burnin <= Nsample:
            par_chains = sampler.chain[:,self.mcmc_burnin:,:]
            lnl_chains = sampler.lnprobability[:,self.mcmc_burnin:]
        else:
            par_chains = sampler.chain
            lnl_chains = sampler.lnprobability
            if not self.silent:
                print('The burnin could not be remove because it is larger than the chain size')
            
        #---------- Compute statistics for the chains
        utils_fitting.chains_statistics(par_chains, lnl_chains,
                                        parname=parlist_stat,
                                        conf=conf,
                                        outfile=self.output_dir+'/MCMC'+extname+'_chain_statistics.txt')

        #---------- Produce 1D plots of the chains
        utils_plot.chains_1Dplots(par_chains, parlist_stat, self.output_dir+'/MCMC'+extname+'_chain_1d_plot.pdf')
        
        #---------- Produce 1D histogram of the chains
        namefiles = [self.output_dir+'/MCMC'+extname+'_chain_hist_'+i+'.pdf' for i in parlist]
        utils_plot.chains_1Dhist(par_chains, parlist_stat, namefiles,
                                 conf=conf, truth=truth)

        #---------- Produce 2D (corner) plots of the chains
        utils_plot.chains_2Dplots_corner(par_chains,
                                         parlist_stat,
                                         self.output_dir+'/MCMC'+extname+'_chain_2d_plot_corner.pdf',
                                         truth=truth)

        
    #==================================================
    # Compute the Pk contraint via forward deprojection
    #==================================================
    
    def run_sbi_fluctuation(self, parinfo,
                            include_model_error=False,
                            filename_sampler=None,
                            show_fit_result=False,
                            set_bestfit=False,
                            extname='_Fluctuation',
                            true_pk3d=None,
                            true_param=None):
        """
        This function fits the 3d power spectrum
        using a forward modeling approach via deprojection
        
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

        - kind (str): projection or brute, for using 
        get_pk2d_model_proj or get_pk2d_model_brute to compute the model
        - include_model_error (bool): set to true to include intrinsic model uncertainty
        assuming the reference model
        - filename_sampler (str): the file name of the sampler to use.
        - show_fit_result (bool): set to true to produce plots for fitting results
        - extname (string): extra name to add after MCMC results file names
        - set_bestfit (bool): set the best fit to the model
        - true_pk3d (dict): pass a dictionary containing the spectrum to compare with
        in the form {'k':array in kpc-1, 'pk':array in kpc3}
        - true_param (list): the list of expected parameter value for the fit
        
        Outputs
        ----------
        - parlist (list): the list of the fit parameters
        - sampler (emcee object): the sampler associated with model parameters of
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

        if kind == 'brute' and include_model_error == False:
            print('      WARNING: brute method is dangerous without including model uncertainties .')
            
        #========== Copy the input model
        input_model = copy.deepcopy(self.model)
        
        #========== Check if the MCMC sampler was already recorded
        if filename_sampler is None:
            sampler_file = self.output_dir+'/pitszi_MCMC'+extname+'_sampler.h5'
        else:
            sampler_file = filename_sampler
            
        sampler_exist = utils_fitting.check_sampler_exist(sampler_file, silent=False)

        #========== Defines the fit parameters
        # Fit parameter list and information
        par_list, par0_value, par0_err, par_min, par_max = self.defpar_fluctuation(parinfo)
        ndim = len(par0_value)
        
        # Starting points of the chains
        pos = utils_fitting.emcee_starting_point(par0_value, par0_err, par_min, par_max, self.mcmc_nwalkers)
        if sampler_exist and (not self.mcmc_reset): pos = None

        # Parallel mode
        if self.method_parallel:
            mypool = ProcessPool()
        else:
            mypool = None
        
        # Info
        if not self.silent:
            print('----- Fit parameters information -----')
            print('      - Fitted parameters:            ')
            print(par_list)
            print('      - Starting point mean:          ')
            print(par0_value)
            print('      - Starting point dispersion :   ')
            print(par0_err)
            print('      - Minimal starting point:       ')
            print(par_min)
            print('      - Maximal starting point:       ')
            print(par_max)
            print('      - Number of dimensions:         ')
            print(ndim)
            print('      - Parallel mode:                ')
            print(self.method_parallel)
            print('      - Use covariance matrix?        ')
            print(self.method_use_covmat)
            print('-----')

        #========== Define the MCMC setup
        # Backend
        backend = utils_fitting.define_emcee_backend(sampler_file, sampler_exist,
                                                     self.mcmc_reset, self.mcmc_nwalkers, ndim, silent=False)

        # Moves
        if kind == 'projection': moves = emcee.moves.StretchMove(a=2.0)
        if kind == 'brute':      moves = emcee.moves.KDEMove()

        # Error statistics to pass
        if self._cross_spec:
            noise_ampli = 1
        else:
            noise_ampli = self.nuisance_Anoise

        if self.method_use_covmat:
            cov = noise_ampli**2*self._pk2d_noise_cov + self.nuisance_Abkg**2*self._pk2d_bkg_cov
            if include_model_error: cov += self._pk2d_modref_cov
            error_stat = np.linalg.inv(cov)
        else:
            error_stat =  noise_ampli**2 * self._pk2d_noise_rms**2          # noise rms
            error_stat += self.nuisance_Abkg**2 * self._pk2d_bkg_rms**2     # Add bkg
            if include_model_error: error_stat += self._pk2d_modref_rms**2  # Add model
            
        # sampler
        sampler = emcee.EnsembleSampler(self.mcmc_nwalkers, ndim,
                                        self.lnlike_fluctuation, 
                                        args=[parinfo, error_stat, kind], 
                                        pool=mypool,
                                        moves=moves,
                                        backend=backend)
        
        #========== Run the MCMC
        if not self.silent: print('----- MCMC sampling -----')
        if self.mcmc_run:
            if not self.silent: print('      - Runing '+str(self.mcmc_nsteps)+' MCMC steps')
            res = sampler.run_mcmc(pos, self.mcmc_nsteps, progress=True, store=True)
        else:
            if not self.silent: print('      - Not running, but restoring the existing sampler')
        
        #========== Show results
        if show_fit_result:
            self.get_mcmc_chains_outputs_results(par_list, parinfo, sampler,
                                                 truth=true_param,
                                                 extname=extname)
            self.run_mcmc_fluctuation_results(sampler, parinfo,
                                              true_pk3d=true_pk3d,
                                              extname=extname)

        #========== Compute the best-fit model and set it
        if set_bestfit:
            best_par = utils_fitting.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
            self.setpar_fluctuation(best_par, parinfo)
        else:
            self.model = input_model

        return par_list, sampler


    #==================================================
    # Show the fit results related to fluctuation
    #==================================================
    
    def run_mcmc_fluctuation_results(self, sampler, parinfo,
                                     true_pk3d=None, extname='_Fluctuation'):
        """
        This function is used to show the results of the MCMC
        regarding the fluctuation
            
        Parameters
        ----------
        - sampler (emcee object): the sampler obtained from mcmc_fluctuation
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
        if noise_mc1.shape[0] > self.mcmc_Nresamp or noise_mc2.shape[0] > self.mcmc_Nresamp:
            Nmc = self.mcmc_Nresamp
            noise_mc1 = noise_mc1[0:Nmc]
            noise_mc2 = noise_mc2[0:Nmc]
        else:
            Nmc = np.amin(np.array([noise_mc1.shape[0], noise_mc2.shape[0]]))
            if self.silent == False: print('WARNING: the number of noise MC is lower than requested')
            
        #========== Get the best-fit
        best_par = utils_fitting.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
        self.setpar_fluctuation(best_par, parinfo)
        k3d, best_pk3d = self.model.get_pressure_fluctuation_spectrum(np.logspace(-4,-1,100)*u.kpc**-1)
        k2d, model_pk2d_ref, model_pk2d_covmat = self.get_pk2d_model_statistics(physical=True,
                                                                                Nmc=self.mcmc_Nresamp)
        best_pk2d_noise = self.nuisance_Anoise * self._pk2d_noise
        best_pk2d_bkg   = self.nuisance_Abkg   * self._pk2d_bkg

        #========== Get the MC
        MC_pk3d = np.zeros((self.mcmc_Nresamp, len(k3d)))
        MC_pk2d = np.zeros((self.mcmc_Nresamp, len(k2d)))
        MC_pk2d_noise = np.zeros((self.mcmc_Nresamp, len(k2d)))
        MC_pk2d_bkg = np.zeros((self.mcmc_Nresamp, len(k2d)))
        
        MC_pars = utils_fitting.get_emcee_random_param(sampler, burnin=self.mcmc_burnin, Nmc=self.mcmc_Nresamp)
        for imc in range(self.mcmc_Nresamp):
            # Get MC model
            self.setpar_fluctuation(MC_pars[imc,:], parinfo)

            # Get MC noise
            MC_pk2d_noise[imc,:] = self.nuisance_Anoise * self._pk2d_noise

            # Get MC bkg
            MC_pk2d_bkg[imc,:] = self.nuisance_Abkg * self._pk2d_bkg
            
            # Get MC Pk2d
            corr = MC_pk2d_noise[imc,:] + MC_pk2d_bkg[imc,:]
            MC_pk2d[imc,:] = self.get_pk2d_model_proj(physical=True)[1].to_value('kpc2') - corr

            
            # Get MC Pk3d
            MC_pk3d[imc,:] = self.model.get_pressure_fluctuation_spectrum(k3d)[1].to_value('kpc3')

        #========== Plot the fitted image data
        utils_plot.show_input_delta_ymap(self.output_dir+'/MCMC'+extname+'_results_input_image1.pdf',
                                         self.data1.image,
                                         self._dy_image1,
                                         self._ymap_sphA1,
                                         self._ymap_sphB1,
                                         self.method_w8,
                                         self.data1.header,
                                         noise_mc1,
                                         mask=self.data1.mask,
                                         visu_smooth=10)
        utils_plot.show_input_delta_ymap(self.output_dir+'/MCMC'+extname+'_results_input_image2.pdf',
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
        utils_plot.show_fit_result_covariance(self.output_dir+'/MCMC'+extname+'_results_covariance.pdf',
                                              self._pk2d_noise_cov,
                                              model_pk2d_covmat.to_value('kpc4'),
                                              covmat_model=self._pk2d_modref_cov,
                                              covmat_bkg=covmat_bkg)
        
        #========== Plot the Pk2d constraint
        utils_plot.show_fit_result_pk2d(self.output_dir+'/MCMC'+extname+'_results_pk2d.pdf',
                                        self._kctr_kpc, self._pk2d_data,
                                        model_pk2d_ref.to_value('kpc2'),
                                        np.diag(model_pk2d_covmat.to_value('kpc4'))**0.5,
                                        self._pk2d_noise_rms, self._pk2d_bkg_rms,
                                        MC_pk2d,
                                        best_pk2d_noise, MC_pk2d_noise,
                                        best_pk2d_bkg, MC_pk2d_bkg)

        #========== Plot the Pk3d constraint
        utils_plot.show_fit_result_pk3d(self.output_dir+'/MCMC'+extname+'_results_pk3d.pdf',
                                        k3d.to_value('kpc-1'), best_pk3d.to_value('kpc3'),
                                        MC_pk3d,
                                        true_pk3d=true_pk3d)

    
