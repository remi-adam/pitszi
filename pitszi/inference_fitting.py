"""
This file deals with the model fitting

"""

import numpy

from pathos.multiprocessing import ProcessPool
from multiprocessing import Pool, cpu_count


#==================================================
# InferenceFitting class
#==================================================

class InferenceFitting(object):
    """ InferenceFitting class
    This class serves as a parser to the main Inference class, to
    include the subclass InferenceFitting in this other file.

    Attributes
    ----------
    The attributes are the same as the Inference class, see inference_main.py

    Methods
    ----------
    - 
    """

    #==================================================
    # Compute the smooth model via forward fitting
    #==================================================
    
    def fit_profile_forward(self,
                            parinfo_profile,
                            parinfo_center=None,
                            parinfo_ellipticity=None,
                            parinfo_ZL=None,
                            show_fit_result=False):
        """
        This function fits the data given the current model and the parameter information
        given by the user.
        
        Parameters
        ----------
        - parinfo_profile (dictionary): the model parameters associated with 
        self.model.model_pressure_profile to be fit as, e.g., 
        parinfo_prof = {'P_0':                   # --> Parameter key (mandatory)
                        {'guess':[0.01, 0.001],  # --> initial guess: center, uncertainty (mandatory)
                         'unit': u.keV*u.cm**-3, # --> unit (mandatory, None if unitless)
                         'limit':[0, np.inf],    # --> Allowed range, i.e. flat prior (optional)
                         'prior':[0.01, 0.001],  # --> Gaussian prior: mean, sigma (optional)
                        },
                        'r_p':
                        {'limit':[0, np.inf], 
                          'prior':[100, 1],
                          'unit': u.kpc, 
                        },
                       }    
        - parinfo_center (dictionary): same as parinfo_profile with accepted parameters
        'RA' and 'Dec'
        - parinfo_ellipticity (dictionary): same as parinfo_profile with accepted parameters
        'min_to_maj_axis_ratio' and 'angle'
        - parinfo_ZL (dictionary): same as parinfo_profile with accepted parameter 'ZL'
        - use_covmat (bool): if True, the noise covariance matrix is used instead 
        of the noise rms
        - show_fit_result (bool): show the best fit model and residual.

        Outputs
        ----------
        - parlist (list): the list of the fit parameters
        - sampler (emcee object): the sampler associated with model parameters of
        the smooth component

        """
        
        #========== Check if the MCMC sampler was already recorded
        sampler_file = self.output_dir+'/pitszi_MCMC_profile_sampler.h5'
        sampler_exist = utils_fitting.check_sampler_exist(sampler_file, silent=False)
        
        #========== Defines the fit parameters
        # Fit parameter list and information
        par_list,par0_value,par0_err,par_min,par_max=lnlike_defpar_profile(parinfo_profile,
                                                                           self.model,
                                                                           parinfo_ZL=parinfo_ZL,
                                                                           parinfo_center=parinfo_center,
                                                                           parinfo_ellipticity=parinfo_ellipticity)
        ndim = len(par0_value)
        
        # Starting points of the chains
        pos = utils_fitting.emcee_starting_point(par0_value, par0_err, par_min, par_max, self.mcmc_nwalkers)
        if sampler_exist and (not self.mcmc_reset): pos = None

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

        #========== Deal with how the noise should be accounted for
        if use_covmat:
            if self.data.noise_covmat is None:
                raise ValueError('Trying to use the noise covariance matrix, but this is undefined')
            noise_property = np.linalg.pinv(self.data.noise_covmat)
        else:
            if self.data.noise_rms is None:
                raise ValueError('Trying to use the noise rms, but this is undefined')
            noise_property = self.data.noise_rms

        #========== Define the MCMC setup
        backend = utils_fitting.define_emcee_backend(sampler_file, sampler_exist,
                                                     self.mcmc_reset, self.mcmc_nwalkers, ndim, silent=False)
        moves = emcee.moves.StretchMove(a=2.0)

        sampler = emcee.EnsembleSampler(self.mcmc_nwalkers, ndim,
                                        lnlike_profile_forward, 
                                        args=[parinfo_profile,
                                              #self.model,
                                              #self.data.image,
                                              noise_property,
                                              #self.data.mask, self.data.psf_fwhm, self.data.transfer_function,
                                              parinfo_center,
                                              parinfo_ellipticity,
                                              parinfo_ZL,
                                              use_covmat], 
                                        pool=ProcessPool(),
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
            self.get_mcmc_chains_outputs_results(par_list, sampler, extraname='_Pr')

            self.fit_profile_forward_results(sampler,
                                             parinfo_profile,
                                             parinfo_center=parinfo_center,
                                             parinfo_ellipticity=parinfo_ellipticity,
                                             parinfo_ZL=parinfo_ZL)
            
        #========== Compute the best-fit model and set it
        best_par = utils_fitting.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
        self.model, best_ZL = lnlike_setpar_profile(best_par, self.model,
                                                    parinfo_profile,
                                                    parinfo_center=parinfo_center,
                                                    parinfo_ellipticity=parinfo_ellipticity,
                                                    parinfo_ZL=parinfo_ZL)
                    
        return par_list, sampler
    
