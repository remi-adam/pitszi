"""
This file deals with the radial model fitting

"""

import numpy as np
import copy
import astropy.units as u
from astropy.coordinates import SkyCoord
from pathos.multiprocessing import ProcessPool
from multiprocessing import Pool, cpu_count
import emcee
from scipy.optimize import curve_fit
from minot.ClusterTools.map_tools import radial_profile_sb

from pitszi import utils_fitting
from pitszi import utils_plot


#==================================================
# InferenceFitting class
#==================================================

class InferenceRadialFitting(object):
    """ InferenceRadialFitting class
    This class serves as a parser to the main Inference class, to
    include the subclass InferenceRadialFitting in this other file.

    Attributes
    ----------
    The attributes are the same as the Inference class, see inference_radial_main.py

    Methods
    ----------
    - get_mcmc_chains_outputs_results: utility tools to get mcmc chain results
    - get_curvefit_outputs_results: utility tools to get curvefit results
    - defpar_profile: define the profile parameters
    - prior_profile: define the prior given the parameter
    - setpar_profile: set the model with the parameters
    - lnlike_profile: likelihood function definition
    - run_mcmc_profile: run the mcmc fit
    - run_mcmc_profile_results: produce the results given the mcmc fit
    - run_curvefit_profile: run the curvefit
    - run_curvefit_profile_results: produce the results given the curvefit

    """

    #==================================================
    # Compute results for a given MCMC sampling
    #==================================================
    
    def get_mcmc_chains_outputs_results(self,
                                       parlist,
                                       sampler,
                                       conf=68.0,
                                       truth=None,
                                       extraname=''):
        """
        This function can be used to produce automated plots/files
        that give the results of the MCMC samping
        
        Parameters
        ----------
        - parlist (list): the list of parameter names
        - sampler (emcee object): the sampler
        - conf (float): confidence limit in % used in results
        - truth (list): the list of expected parameter value for the fit
        - extraname (string): extra name to add after MCMC in file name

        Outputs
        ----------
        Plots are produced

        """

        Nchains, Nsample, Nparam = sampler.chain.shape
        
        #---------- Check the truth
        if truth is not None:
            if len(truth) != Nparam:
                raise ValueError("The 'truth' keyword should match the number of parameters")

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
                                        parname=parlist,
                                        conf=conf,
                                        outfile=self.output_dir+'/MCMC'+extraname+'_chain_statistics.txt')

        #---------- Produce 1D plots of the chains
        utils_plot.chains_1Dplots(par_chains, parlist, self.output_dir+'/MCMC'+extraname+'_chain_1d_plot.pdf')
        
        #---------- Produce 1D histogram of the chains
        namefiles = [self.output_dir+'/MCMC'+extraname+'_chain_hist_'+i+'.pdf' for i in parlist]
        utils_plot.chains_1Dhist(par_chains, parlist, namefiles,
                                 conf=conf, truth=truth)

        #---------- Produce 2D (corner) plots of the chains
        utils_plot.chains_2Dplots_corner(par_chains,
                                         parlist,
                                         self.output_dir+'/MCMC'+extraname+'_chain_2d_plot_corner.pdf',
                                         truth=truth)

        #utils_plot.chuains_2Dplots_sns(par_chains,
        #                              parlist,
        #                              self.output_dir+'/MCMC'+extraname+'_chain_2d_plot_sns.pdf',
        #                              truth=truth)
    

    #==================================================
    # Compute results for a given curvefit
    #==================================================
    
    def get_curvefit_outputs_results(self,
                                     parlist,
                                     parinfo,
                                     popt, pcov,
                                     conf=68.0,
                                     truth=None,
                                     extraname='',
                                     Nsample=10000):
        """
        This function can be used to produce automated plots/files
        that give the results of the curvefit fit
        
        Parameters
        ----------
        - parlist (list): the list of parameter names
        - popt (1d array): best fit parameter values
        - pcov (2d array): posterior covariance matrix
        - conf (float): confidence limit in % used in results
        - truth (list): the list of expected parameter value for the fit
        - extraname (string): extra name to add after MCMC in file name

        Outputs
        ----------
        Plots are produced

        """

        Nparam = len(parlist)
        Nchains = 1
        
        #---------- Check the truth
        if truth is not None:
            if len(truth) != Nparam:
                raise ValueError("The 'truth' keyword should match the number of parameters")

        #---------- Output best fit and errors
        print('   --- Parameter best-fit: ---')
        file = open(self.output_dir+'/CurveFit'+extraname+'_statistics_main.txt','w')
        for ipar in range(Nparam):
            bfval = str(popt[ipar])+' +/- '+str(pcov[ipar,ipar]**0.5)
            print('     param '+str(ipar)+' ('+parlist[ipar]+') = '+bfval)
            file.write('param '+str(ipar)+' ('+parlist[ipar]+') = '+bfval+'\n')
        file.close() 
            
        #---------- Mimic MCMC chains with multivariate Gaussian
        par_chains = np.zeros((Nsample, Nparam))
        isamp = 0
        ibad = 0
        while isamp < Nsample:
            param = np.random.multivariate_normal(popt, pcov)
            cond = np.isfinite(self.prior_profile(param, parinfo)) # make sure params are within limits
            if cond:
                par_chains[isamp,:] = param
                isamp += 1
            else:
                ibad += 1
            # Security in case issu in multivariate sampling
            if ibad == 5*Nsample:                
                if not self.silent:
                    print('WARNING: Cannot produce chains from multivariate sampling.')
                    print('         Tried '+str(ibad+isamp)+' times, failed '+str(ibad)+' times')
                    print('         This can be due to errors being much larger than the accepted limits.')
                    print('         Exit.')
                return
        
        # Add explicitely the best fit in the chain
        par_chains[-1,:] = popt

        # Compute the associated likelihood
        lnl_chains = np.zeros(Nsample)
        for i in range(Nsample):
            lnl_chains[i] = -0.5 * np.matmul((par_chains[i,:]-popt), np.matmul(pcov, (par_chains[i,:]-popt)))
        par_chains = par_chains[np.newaxis]
        lnl_chains = lnl_chains[np.newaxis]
            
        #---------- Compute statistics for the parameters
        utils_fitting.chains_statistics(par_chains, lnl_chains,
                                        parname=parlist,
                                        conf=conf,
                                        outfile=self.output_dir+'/CurveFit'+extraname+'_statistics.txt')
            
        #---------- Produce 1D plots of the chains
        utils_plot.chains_1Dplots(par_chains, parlist, self.output_dir+'/CurveFit'+extraname+'_1d_plot.pdf')
        
        #---------- Produce 1D histogram of the chains
        namefiles = [self.output_dir+'/CurveFit'+extraname+'_hist_'+i+'.pdf' for i in parlist]
        utils_plot.chains_1Dhist(par_chains, parlist, namefiles,
                                 conf=conf, truth=truth)

        #---------- Produce 2D (corner) plots of the chains
        utils_plot.chains_2Dplots_corner(par_chains,
                                         parlist,
                                         self.output_dir+'/CurveFit'+extraname+'_2d_plot_corner.pdf',
                                         truth=truth)

        #utils_plot.chains_2Dplots_sns(par_chains,
        #                              parlist,
        #                              self.output_dir+'/CurveFit'+extraname+'_2d_plot_sns.pdf',
        #                              truth=truth)
    
        
    #==================================================
    # Profile parameter definition
    #==================================================
        
    def defpar_profile(self, parinfo):
        """
        This function helps defining the parameter list, initial values and
        ranges for the profile fitting.
        We distinguish 4 kind of parameters:
        - profile parameters or M500
        - offset RA and Dec 
        - ellipticity min_to_maj_axis_ratio and angle
        - nuisance zero level ZL
            
        Parameters
        ----------
         - parinfo (dict): parameter information
    
        Outputs
        ----------
        - par_list (list): list of parameter names
        - par0_value (list): list of mean initial parameter value
        - par0_err (list): list of mean initial parameter uncertainty
        - par_min (list): list of minimal initial parameter value
        - par_max (list): list of maximal initial parameter value
        
        """
        
        parkeys = list(parinfo.keys())
        
        #========== Init the parameters
        par_list = []
        par0_value = []
        par0_err = []
        par_min = []
        par_max = []
        
        #========== Check the profile parameters
        # remove unwanted keys
        parinfo_prof = copy.deepcopy(parinfo)
        if 'RA'                    in parkeys: parinfo_prof.pop('RA')
        if 'Dec'                   in parkeys: parinfo_prof.pop('Dec')
        if 'min_to_maj_axis_ratio' in parkeys: parinfo_prof.pop('min_to_maj_axis_ratio')
        if 'angle'                 in parkeys: parinfo_prof.pop('angle')
        if 'ZL'                    in parkeys: parinfo_prof.pop('ZL')

        # Check that if M500 is a parameter, it is the only one
        Npar_prof = len(list(parinfo_prof.keys()))
        if 'M500' in list(parinfo_prof.keys()) and Npar_prof > 1:
            raise ValueError('M500 is a fit parameter, thus not other profile parameter is accepted.')    
            
        # Loop over the keys
        for ipar in range(Npar_prof):
            
            parkey = list(parinfo_prof.keys())[ipar]

            #----- Check that parameters are working
            try:
                if 'M500' == parkey:
                    bid = self.model.M500
                else:
                    bid = self.model.model_pressure_profile[parkey]
            except:
                raise ValueError('The parameter '+parkey+' is not in self.model.model_pressure_profile')    

            if 'M500' == parkey:
                try:
                    bid = parinfo_prof[parkey]['P_ref']
                except:
                    raise ValueError('M500 dict should contain "P_ref":{UPP, A10UPP, etc} passed to set_pressure_profile_universal_param')    

            if 'guess' not in parinfo_prof[parkey]:
                raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
            if 'unit' not in parinfo_prof[parkey]:
                raise ValueError('The unit key is mandatory. Use "unit":None if unitless')
    
            #----- Update the name list
            par_list.append(parkey)
    
            #----- Update the guess values
            par0_value.append(parinfo_prof[parkey]['guess'][0])
            par0_err.append(parinfo_prof[parkey]['guess'][1])
    
            #----- Update the limit values
            if 'limit' in parinfo_prof[parkey]:
                par_min.append(parinfo_prof[parkey]['limit'][0])
                par_max.append(parinfo_prof[parkey]['limit'][1])
            else:
                par_min.append(-np.inf)
                par_max.append(+np.inf)

        #========== Center parameters
        # Collect RA, Dec keys
        list_ctr_allowed = ['RA', 'Dec']
        parinfo_center = {key: parinfo[key] for key in list_ctr_allowed if key in parinfo.keys()}
        Npar_ctr = len(list(parinfo_center.keys()))
            
        # Loop over the keys
        for ipar in range(Npar_ctr):
            
            parkey = list(parinfo_center.keys())[ipar]
            
            #----- Check that the param is ok wrt the model
            if 'guess' not in parinfo_center[parkey]:
                raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
            if 'unit' not in parinfo_center[parkey]:
                raise ValueError('The unit key is mandatory. Use "unit":None if unitless')
            
            #----- Update the name list
            par_list.append(parkey)
            
            #----- Update the guess values
            par0_value.append(parinfo_center[parkey]['guess'][0])
            par0_err.append(parinfo_center[parkey]['guess'][1])
            
            #----- Update the limit values
            if 'limit' in parinfo_center[parkey]:
                par_min.append(parinfo_center[parkey]['limit'][0])
                par_max.append(parinfo_center[parkey]['limit'][1])
            else:
                par_min.append(-np.inf)
                par_max.append(+np.inf)
    
        #========== Ellipticity parameters
        list_ell_allowed = ['min_to_maj_axis_ratio', 'angle']
        parinfo_ellipticity = {key: parinfo[key] for key in list_ell_allowed if key in parinfo.keys()}
        Npar_ell = len(list(parinfo_ellipticity.keys()))

        #----- Check that both parameter are fitted
        if Npar_ell == 1:
            raise ValueError('Ellipticity parameter (min_to_maj_axis_ratio and angle) cannot be fitted separately')

        #----- Loop
        for ipar in range(Npar_ell):
    
            parkey = list(parinfo_ellipticity.keys())[ipar]
    
            #----- Check that the param is ok wrt the model
            if 'guess' not in parinfo_ellipticity[parkey]:
                raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty]')    
            if 'unit' not in parinfo_ellipticity[parkey]:
                raise ValueError('The unit key is mandatory. Use "unit":None if unitless')
    
            #----- Update the name list
            par_list.append(parkey)
    
            #----- Update the guess values
            par0_value.append(parinfo_ellipticity[parkey]['guess'][0])
            par0_err.append(parinfo_ellipticity[parkey]['guess'][1])
    
            #----- Update the limit values
            if 'limit' in parinfo_ellipticity[parkey]:
                par_min.append(parinfo_ellipticity[parkey]['limit'][0])
                par_max.append(parinfo_ellipticity[parkey]['limit'][1])
            else:
                if parkey == 'min_to_maj_axis_ratio':
                    par_min.append(0)
                    par_max.append(1)
                if parkey == 'angle':
                    par_min.append((-90*u.deg).to_value(parinfo_ellipticity[parkey]['unit']))
                    par_max.append((+90*u.deg).to_value(parinfo_ellipticity[parkey]['unit']))
    
        #========== Map offset parameter
        list_zl_allowed = ['ZL']
        parinfo_ZL = {key: parinfo[key] for key in list_zl_allowed if key in parinfo.keys()}
        Npar_zl = len(list(parinfo_ZL.keys()))
    
        for ipar in range(Npar_zl):
    
            parkey = list(parinfo_ZL.keys())[ipar]
    
            #----- Check that the param is ok wrt the model
            if 'guess' not in parinfo_ZL[parkey]:
                raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
            if 'unit' not in parinfo_ZL[parkey]:
                raise ValueError('The unit key is mandatory. Use "unit":None if unitless')
    
            #----- Update the name list
            par_list.append(parkey)
    
            #----- Update the guess values
            par0_value.append(parinfo_ZL[parkey]['guess'][0])
            par0_err.append(parinfo_ZL[parkey]['guess'][1])
    
            #----- Update the limit values
            if 'limit' in parinfo_ZL[parkey]:
                par_min.append(parinfo_ZL[parkey]['limit'][0])
                par_max.append(parinfo_ZL[parkey]['limit'][1])
            else:
                par_min.append(-np.inf)
                par_max.append(+np.inf)
            
        #========== Make it an np array
        par_list   = np.array(par_list)
        par0_value = np.array(par0_value)
        par0_err   = np.array(par0_err)
        par_min    = np.array(par_min)
        par_max    = np.array(par_max)
        
        return par_list, par0_value, par0_err, par_min, par_max

        
    #==================================================
    # Profile prior definition
    #==================================================
        
    def prior_profile(self, param, parinfo):
        """
        Compute the prior given the input parameter and some information
        about the parameters given by the user
        We distinguish 4 kind of parameters:
        - profile parameters
        - offset RA and Dec 
        - ellipticity min_to_maj_axis_ratio and angle
        - nuisance zero level ZL
        
         Parameters
         ----------
         - param (list): the parameter to apply to the model
         - parinfo (dict): profile parameter information
     
         Outputs
         ----------
         - prior (float): value of the prior

        """

        prior = 0
        idx_par = 0
        parkeys = list(parinfo.keys())

        #========== Check the profile parameters
        # remove unwanted keys
        parinfo_prof = copy.deepcopy(parinfo)
        if 'RA'                    in parkeys: parinfo_prof.pop('RA')
        if 'Dec'                   in parkeys: parinfo_prof.pop('Dec')
        if 'min_to_maj_axis_ratio' in parkeys: parinfo_prof.pop('min_to_maj_axis_ratio')
        if 'angle'                 in parkeys: parinfo_prof.pop('angle')
        if 'ZL'                    in parkeys: parinfo_prof.pop('ZL')
        
        # Loop over the keys
        parkeys_prof = list(parinfo_prof.keys())

        for ipar in range(len(parkeys_prof)):
            parkey = parkeys_prof[ipar]
        
            # Flat prior
            if 'limit' in parinfo_prof[parkey]:
                if param[idx_par] < parinfo_prof[parkey]['limit'][0]:
                    return -np.inf                
                if param[idx_par] > parinfo_prof[parkey]['limit'][1]:
                    return -np.inf
                
            # Gaussian prior
            if 'prior' in parinfo_prof[parkey]:
                expected = parinfo_prof[parkey]['prior'][0]
                sigma = parinfo_prof[parkey]['prior'][1]
                prior += -0.5*(param[idx_par] - expected)**2 / sigma**2
    
            # Increase param index
            idx_par += 1

        #========== Other parameters
        list_allowed  = ['RA', 'Dec', 'min_to_maj_axis_ratio', 'angle', 'ZL']
        parinfo_other = {key: parinfo[key] for key in list_allowed if key in parinfo.keys()}        
        parkeys_other = list(parinfo_other.keys())
        Npar_other    = len(parkeys_other)

        for ipar in range(Npar_other):
            parkey = parkeys_other[ipar]
            
            # Flat prior
            if 'limit' in parinfo_other[parkey]:
                if param[idx_par] < parinfo_other[parkey]['limit'][0]:
                    return -np.inf                
                if param[idx_par] > parinfo_other[parkey]['limit'][1]:
                    return -np.inf
                
            # Gaussian prior
            if 'prior' in parinfo_other[parkey]:
                expected = parinfo_other[parkey]['prior'][0]
                sigma = parinfo_other[parkey]['prior'][1]
                prior += -0.5*(param[idx_par] - expected)**2 / sigma**2
     
            # Increase param index
            idx_par += 1

        #========== Check on param numbers
        if idx_par != len(param):
            raise ValueError('Problem with the prior parameters')
        
        return prior
    
    
    #==================================================
    # Profile parameter settings
    #==================================================
        
    def setpar_profile(self, param, parinfo):
        """
        Set the model to the given profile parameters.
        We distinguish 4 kind of parameters:
        - profile parameters
        - offset RA and Dec 
        - ellipticity min_to_maj_axis_ratio and angle
        - nuisance zero level ZL        

        Parameters
        ----------
        - param (list): the parameter to apply to the model
        - parinfo (dict): see mcmc_profile

        Outputs
        ----------
        The model is updated with parameter settings

        """
        
        idx_par = 0
        parkeys = list(parinfo.keys())

        #========== Check the profile parameters
        # remove unwanted keys
        parinfo_prof = copy.deepcopy(parinfo)
        if 'RA'                    in parkeys: parinfo_prof.pop('RA')
        if 'Dec'                   in parkeys: parinfo_prof.pop('Dec')
        if 'min_to_maj_axis_ratio' in parkeys: parinfo_prof.pop('min_to_maj_axis_ratio')
        if 'angle'                 in parkeys: parinfo_prof.pop('angle')
        if 'ZL'                    in parkeys: parinfo_prof.pop('ZL')
        parkeys_prof = list(parinfo_prof.keys())
        
        # Loop Profile
        for ipar in range(len(parkeys_prof)):
            parkey = parkeys_prof[ipar]
            if parinfo_prof[parkey]['unit'] is not None:
                unit = parinfo_prof[parkey]['unit']
            else:
                unit = 1

            if parkey == 'M500':
                self.model.M500 = param[idx_par] * unit
                self.model.set_pressure_profile_universal_param(parinfo['M500']['P_ref'])
            else:
                self.model.model_pressure_profile[parkey] = param[idx_par] * unit
            idx_par += 1

        #========== Center
        if 'RA' in parkeys:
            RA = param[idx_par]
            unit1 = parinfo['RA']['unit']
            idx_par += 1
        else:
            RA = self.model.coord.icrs.ra.to_value('deg')
            unit1 = u.deg
            
        if 'Dec' in parkeys:
            Dec = param[idx_par]
            unit2 = parinfo['Dec']['unit']
            idx_par += 1
        else:
            Dec = self.model.coord.icrs.dec.to_value('deg')
            unit2 = u.deg

        self.model.coord = SkyCoord(RA*unit1, Dec*unit2, frame='icrs')
                
        #========== Ellipticity
        if 'min_to_maj_axis_ratio' in parkeys and 'angle' in parkeys:
            axis_ratio = param[idx_par]
            idx_par += 1
            
            angle = param[idx_par]
            angle_unit = parinfo['angle']['unit']
            idx_par += 1
            
            self.model.triaxiality = {'min_to_maj_axis_ratio':axis_ratio,
                                      'int_to_maj_axis_ratio':axis_ratio,
                                      'euler_angle1':0*u.deg,
                                      'euler_angle2':90*u.deg,
                                      'euler_angle3':angle*angle_unit}
        
        #========== Map zero level
        if 'ZL' in parkeys:
            self.nuisance_ZL = param[idx_par]
            idx_par += 1

        #========== Final check on parameter count
        if len(param) != idx_par:
            print('issue with the number of parameters')
            import pdb
            pdb.set_trace()

        
    #==================================================
    # lnL function for the profile fit
    #==================================================
    
    def lnlike_profile(self, param, parinfo):
        """
        This is the likelihood function used for the fit of the profile.
            
        Parameters
        ----------
        - param (np array): the value of the test parameters
        - parinfo (dict): see parinfo in mcmc_profile

        Outputs
        ----------
        - lnL (float): the value of the log likelihood
    
        """
        
        #========== Deal with flat and Gaussian priors
        prior = self.prior_profile(param, parinfo)
        if not np.isfinite(prior): return -np.inf
        
        #========== Change model parameters
        self.setpar_profile(param, parinfo)
        
        #========== Get the model
        model_img = self.get_radial_model()
        
        #========== Compute the likelihood
        if self.method_use_covmat:
            flat_resid = (self.data.mask * (model_img - self.data.image)).flatten()
            lnL = -0.5*np.matmul(flat_resid, np.matmul(self._ymap_invcov, flat_resid))
        else:
            lnL = -0.5*np.nansum(self.data.mask**2 * (model_img-self.data.image)**2 / self.data.noise_rms**2)
            
        lnL += prior
        
        #========== Check and return
        if np.isnan(lnL):
            return -np.inf
        else:
            return lnL        

        
    #==================================================
    # Compute the smooth model via forward fitting
    #==================================================
    
    def run_mcmc_profile(self, parinfo,
                         filename_sampler=None,
                         show_fit_result=False,
                         set_bestfit=False):
        """
        This function fits the data given the current model and the parameter information
        given by the user, using MCMC.
        
        Parameters
        ----------
        - parinfo (dictionary): the model parameters associated with 
        self.model.model_pressure_profile to be fit as, e.g., 
        parinfo =      {'P_0':                   # --> Parameter key (mandatory)
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
        Other accepted parameters: 'RA' and 'Dec', 'min_to_maj_axis_ratio' and 'angle', 'ZL'

        Instead of the profile parameters, the mass M500 is also accepted but the M500 disctionary 
        should also contain P_ref as the reference profile model, e.g. A10UPP, etc, to be passed to
        model.set_pressure_profile_universal_param : 'M500':{'guess':..., 'P_ref': 'A10UPP'}

        - filename_sampler (str): the file name of the sampler to use.
        - show_fit_result (bool): show the best fit model and residual.
        - set_bestfit (bool): set the best fit to the model

        Outputs
        ----------
        - parlist (list): the list of the fit parameters
        - sampler (emcee object): the sampler associated with model parameters of
        the smooth component

        """

        #========== Copy the input model
        input_model = copy.deepcopy(self.model)
        
        #========== Check if the MCMC sampler was already recorded
        if filename_sampler is None:
            sampler_file = self.output_dir+'/pitszi_MCMC_Profile_sampler.h5'
        else:
            sampler_file = filename_sampler
            
        sampler_exist = utils_fitting.check_sampler_exist(sampler_file, silent=False)
        
        #========== Defines the fit parameters
        # Fit parameter list and information
        par_list, par0_value, par0_err, par_min, par_max = self.defpar_profile(parinfo)
        ndim = len(par0_value)
        
        # Starting points of the chains
        pos = utils_fitting.emcee_starting_point(par0_value, par0_err, par_min, par_max, self.mcmc_nwalkers)
        if sampler_exist and (not self.mcmc_reset): pos = None

        # Parallel mode
        if self.method_parallel:
            mypool = ProcessPool()
        else:
            mypool = None

        # Get the inverse covariance matrix if needed
        if self.method_use_covmat:
            if not self.silent:
                print('Using the ymap covariance matrix may failmay be unstable. Starting to compute C^-1')
            self._ymap_invcov = np.linalg.inv(self.data.noise_covmat)
                
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
        backend = utils_fitting.define_emcee_backend(sampler_file, sampler_exist,
                                                     self.mcmc_reset, self.mcmc_nwalkers, ndim, silent=False)
        moves = emcee.moves.StretchMove(a=2.0)
        
        sampler = emcee.EnsembleSampler(self.mcmc_nwalkers, ndim,
                                        self.lnlike_profile, 
                                        args=[parinfo], 
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
            self.get_mcmc_chains_outputs_results(par_list, sampler, extraname='_Profile')
            self.run_mcmc_profile_results(sampler, parinfo)

        #========== Compute the best-fit model and set it
        if set_bestfit:
            best_par = utils_fitting.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
            self.setpar_profile(best_par, parinfo)
        else:
            self.model = input_model
            
        return par_list, sampler

    
    #==================================================
    # Show the fit results related to profile
    #==================================================
    
    def run_mcmc_profile_results(self, sampler, parinfo,
                                 visu_smooth=10*u.arcsec,
                                 binsize_prof=5*u.arcsec,
                                 true_pressure_profile=None,
                                 true_compton_profile=None):
        """
        This is function is used to show the results of the MCMC
        regarding the radial profile
            
        Parameters
        ----------
        - sampler (emcee object): the sampler obtained from mcmc_profile
        - parinfo (dict): same as mcmc_profile
        - visu_smooth (quantity): The extra smoothing FWHM for vidualization. Homogeneous to arcsec.
        - binsize_prof (quantity): The binsize for the y profile. Homogeneous to arcsec
        - true_pressure_profile (dict): pass a dictionary containing the profile to compare with
        in the form {'r':array in kpc, 'p':array in keV cm-3}
        - true_compton_profile (dict): pass a dictionary containing the profile to compare with
        in the form {'r':array in arcmin, 'y':array in [y]}
        
        Outputs
        ----------
        plots are produced

        """

        #========== Get noise MC 
        noise_mc = self.data.noise_mc
        if noise_mc.shape[0] > self.mcmc_Nresamp:
            Nmc = self.mcmc_Nresamp
            noise_mc = noise_mc[0:Nmc]
        else:
            Nmc = noise_mc.shape[0]
            if self.silent == False: print('WARNING: the number of noise MC is lower than requested')

        #========== rms for profile
        rms_prof = np.std(noise_mc, axis=0)
        mymask = self.data.mask**2
        mymask[self.data.mask == 0] = np.nan
        rms_prof = rms_prof/mymask
        rms_prof[~np.isfinite(rms_prof)] = np.nan
        
        #========== Get the best-fit
        best_par = utils_fitting.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
        self.setpar_profile(best_par, parinfo)
        best_model = copy.deepcopy(self.model)

        #========== Get the profile for the data, centered on best-fit center if fitted
        data_yprof_tmp = radial_profile_sb(self.data.image, 
                                           (best_model.coord.icrs.ra.to_value('deg'),
                                            best_model.coord.icrs.dec.to_value('deg')), 
                                           stddev=rms_prof, header=self.data.header, 
                                           binsize=binsize_prof.to_value('deg'))
        r2d = data_yprof_tmp[0]*60
        data_yprof = data_yprof_tmp[1]

        mc_y_data = np.zeros((Nmc, len(r2d)))
        for i in range(Nmc):
            mc_y_data[i,:] = radial_profile_sb(noise_mc[i,:,:], 
                                               (best_model.coord.icrs.ra.to_value('deg'),
                                                best_model.coord.icrs.dec.to_value('deg')), 
                                               stddev=rms_prof, header=self.data.header, 
                                               binsize=binsize_prof.to_value('deg'))[1]
        data_yprof_err = np.std(mc_y_data, axis=0)

        #========== Compute best fit observables
        self.setpar_profile(best_par, parinfo)
        best_ymap_sph = self.get_radial_model()

        best_y_profile = radial_profile_sb(best_ymap_sph, 
                                           (best_model.coord.icrs.ra.to_value('deg'),
                                            best_model.coord.icrs.dec.to_value('deg')), 
                                           stddev=best_ymap_sph*0+1, header=self.data.header, 
                                           binsize=binsize_prof.to_value('deg'))[1]

        r3d, best_pressure_profile = best_model.get_pressure_profile(np.logspace(1, 4, 100)*u.kpc)
        r3d = r3d.to_value('kpc')
        best_pressure_profile = best_pressure_profile.to_value('keV cm-3')
        
        #========== MC resampling
        MC_ymap_sph         = np.zeros((self.mcmc_Nresamp, best_ymap_sph.shape[0], best_ymap_sph.shape[1]))
        MC_y_profile        = np.zeros((self.mcmc_Nresamp, len(r2d)))
        MC_pressure_profile = np.zeros((self.mcmc_Nresamp, len(r3d)))
        
        MC_pars = utils_fitting.get_emcee_random_param(sampler, burnin=self.mcmc_burnin, Nmc=self.mcmc_Nresamp)
        for i in range(self.mcmc_Nresamp):
            # Get MC model
            self.setpar_profile(MC_pars[i,:], parinfo)
            MC_ymap_sph[i,:,:] = self.get_radial_model()
            
            # Get MC y profile
            MC_y_profile[i,:] = radial_profile_sb(MC_ymap_sph[i,:,:], 
                                                  (best_model.coord.icrs.ra.to_value('deg'),
                                                   best_model.coord.icrs.dec.to_value('deg')), 
                                                  stddev=self.data.image*0+1, header=self.data.header, 
                                                  binsize=binsize_prof.to_value('deg'))[1]
            # Get MC pressure profile
            MC_pressure_profile[i,:] = self.model.get_pressure_profile(r3d*u.kpc)[1].to_value('keV cm-3')

        #========== plots map
        utils_plot.show_fit_result_ymap(self.output_dir+'/MCMC_Profile_results_y_map.pdf',
                                        self.data.image,
                                        self.data.header,
                                        noise_mc,
                                        best_ymap_sph,
                                        mask=self.data.mask,
                                        visu_smooth=visu_smooth.to_value('arcsec'))
        
        #========== plots ymap profile
        utils_plot.show_fit_ycompton_profile(self.output_dir+'/MCMC_Profile_results_y_profile.pdf',
                                             r2d, data_yprof, data_yprof_err,
                                             best_y_profile, MC_y_profile,
                                             true_compton_profile=true_compton_profile)
        
        #========== plots pressure profile
        utils_plot.show_fit_result_pressure_profile(self.output_dir+'/MCMC_Profile_results_P_profile.pdf',
                                                    r3d, best_pressure_profile, MC_pressure_profile,
                                                    true_pressure_profile=true_pressure_profile)


    #==================================================
    # Compute the smooth model via forward curvefit
    #==================================================
    
    def run_curvefit_profile(self, parinfo,
                             show_fit_result=False,
                             set_bestfit=False,
                             maxfev=10000):
        """
        This function fits the data given the current model and the parameter information
        given by the user, using curvefit.
        
        Parameters
        ----------
        - parinfo (dictionary): the model parameters associated with 
        self.model.model_pressure_profile to be fit as, e.g., 
        parinfo =      {'P_0':                   # --> Parameter key (mandatory)
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
        Other accepted parameters: 'RA' and 'Dec', 'min_to_maj_axis_ratio' and 'angle', 'ZL'

        Instead of the profile parameters, the mass M500 is also accepted but the M500 disctionary 
        should also contain P_ref as the reference profile model, e.g. A10UPP, etc, to be passed to
        model.set_pressure_profile_universal_param : 'M500':{'guess':..., 'P_ref': 'A10UPP'}

        - show_fit_result (bool): show the best fit model and residual.
        - set_bestfit (bool): the the current model to the best-fit
        - maxfev (int): maxfev argument to pass to curvefit

        Outputs
        ----------
        - parlist (list): the list of the fit parameters
        - par_opt (list): the best fit parameters
        - par_cov (2d matrix): the posterior covariance matrix

        """

        #========== Copy the input model
        input_model = copy.deepcopy(self.model)
        
        #========== Defines the fit parameters
        par_list, par0_value, par0_err, par_min, par_max = self.defpar_profile(parinfo)

        #========== Define the sigma
        if self.method_use_covmat:
            sigma = self.data.noise_covmat
            if not self.silent:
                print('Using the ymap covariance matrix with curve_fit is a bit ambitious, it may fail.')
        else:
            sigma = self.data.noise_rms.flatten()

        #========== Defines the data
        ymap = (self.data.image*self.data.mask).flatten()
        
        #========== Fitting function
        def fitfunc(x, *pars):
            params = []
            for ip in range(len(pars)): params.append(pars[ip])
            self.setpar_profile(np.array(params), parinfo)
            model_img = self.get_radial_model()
            model_test = model_img * self.data.mask
            return model_test.flatten()

        #========== Run the fit
        par_opt, par_cov = curve_fit(fitfunc, 0,
                                     ymap,
                                     p0=par0_value,
                                     sigma=sigma,
                                     absolute_sigma=False,
                                     bounds=(par_min, par_max),
                                     maxfev=maxfev)

        #========== Show results
        if show_fit_result:
            self.get_curvefit_outputs_results(par_list, parinfo, par_opt, par_cov, extraname='_Profile')
            self.run_curvefit_profile_results(par_opt, par_cov, parinfo)
        
        #========== Compute the best-fit model and set it
        if set_bestfit:
            self.setpar_profile(par_opt, parinfo)
        else:
            self.model = input_model
        
        return par_list, par_opt, par_cov
    
    
    #==================================================
    # Show the fit results related to profile
    #==================================================
    
    def run_curvefit_profile_results(self, popt, pcov, parinfo,
                                     visu_smooth=10*u.arcsec,
                                     binsize_prof=5*u.arcsec,
                                     true_pressure_profile=None,
                                     true_compton_profile=None):
        """
        This is function is used to show the results of the MCMC
        regarding the radial profile
            
        Parameters
        ----------
        - popt (1d array): parameter best fit
        - pcov (2d array): parameter covariance matrix
        - parinfo (dict): same as mcmc_profile
        - visu_smooth (quantity): The extra smoothing FWHM for vidualization. Homogeneous to arcsec.
        - binsize_prof (quantity): The binsize for the y profile. Homogeneous to arcsec
        - true_pressure_profile (dict): pass a dictionary containing the profile to compare with
        in the form {'r':array in kpc, 'p':array in keV cm-3}
        - true_compton_profile (dict): pass a dictionary containing the profile to compare with
        in the form {'r':array in arcmin, 'y':array in [y]}
        
        Outputs
        ----------
        plots are produced

        """
        
        #========== Get noise MC 
        noise_mc = self.data.noise_mc
        if noise_mc.shape[0] > self.mcmc_Nresamp:
            Nmc = self.mcmc_Nresamp
            noise_mc = noise_mc[0:Nmc]
        else:
            Nmc = noise_mc.shape[0]
            if self.silent == False: print('WARNING: the number of noise MC is lower than requested')

        #========== rms for profile
        rms_prof = np.std(noise_mc, axis=0)
        mymask = self.data.mask**2
        mymask[self.data.mask == 0] = np.nan
        rms_prof = rms_prof/mymask
        rms_prof[~np.isfinite(rms_prof)] = np.nan
        
        #========== Get the best-fit
        self.setpar_profile(popt, parinfo)
        best_model = copy.deepcopy(self.model)

        #========== Get the profile for the data, centered on best-fit center if fitted
        data_yprof_tmp = radial_profile_sb(self.data.image, 
                                           (best_model.coord.icrs.ra.to_value('deg'),
                                            best_model.coord.icrs.dec.to_value('deg')), 
                                           stddev=rms_prof, header=self.data.header, 
                                           binsize=binsize_prof.to_value('deg'))
        r2d = data_yprof_tmp[0]*60
        data_yprof = data_yprof_tmp[1]

        mc_y_data = np.zeros((Nmc, len(r2d)))
        for i in range(Nmc):
            mc_y_data[i,:] = radial_profile_sb(noise_mc[i,:,:], 
                                               (best_model.coord.icrs.ra.to_value('deg'),
                                                best_model.coord.icrs.dec.to_value('deg')), 
                                               stddev=rms_prof, header=self.data.header, 
                                               binsize=binsize_prof.to_value('deg'))[1]
        data_yprof_err = np.std(mc_y_data, axis=0)
        
        #========== Compute best fit observables
        self.setpar_profile(popt, parinfo)
        best_ymap_sph = self.get_radial_model()

        best_y_profile = radial_profile_sb(best_ymap_sph, 
                                           (best_model.coord.icrs.ra.to_value('deg'),
                                            best_model.coord.icrs.dec.to_value('deg')), 
                                           stddev=best_ymap_sph*0+1, header=self.data.header, 
                                           binsize=binsize_prof.to_value('deg'))[1]

        r3d, best_pressure_profile = best_model.get_pressure_profile(np.logspace(1, 4, 100)*u.kpc)
        r3d = r3d.to_value('kpc')
        best_pressure_profile = best_pressure_profile.to_value('keV cm-3')
        
        #========== MC resampling
        MC_ymap_sph         = np.zeros((self.mcmc_Nresamp, best_ymap_sph.shape[0], best_ymap_sph.shape[1]))
        MC_y_profile        = np.zeros((self.mcmc_Nresamp, len(r2d)))
        MC_pressure_profile = np.zeros((self.mcmc_Nresamp, len(r3d)))
        
        MC_pars = np.zeros((self.mcmc_Nresamp, len(popt)))
        isamp = 0
        ibad = 0
        while isamp < self.mcmc_Nresamp:
            param = np.random.multivariate_normal(popt, pcov)
            cond = np.isfinite(self.prior_profile(param, parinfo)) # make sure params are within limits
            if cond:
                MC_pars[isamp,:] = param
                isamp += 1
            else:
                ibad += 1
            if ibad == 5*self.mcmc_Nresamp:
                if not self.silent:
                    print('WARNING: Cannot produce chains from multivariate sampling.')
                    print('         Tried '+str(ibad+isamp)+' times, failed '+str(ibad)+' times')
                    print('         This can be due to errors being much larger than the accepted limits.')
                    print('         Continue without uncertainties.')
                for isa in range(self.mcmc_Nresamp): MC_pars[isa,:] = popt
                isamp = self.mcmc_Nresamp
                
        for i in range(self.mcmc_Nresamp):
            # Get MC model
            self.setpar_profile(MC_pars[i,:], parinfo)
            MC_ymap_sph[i,:,:] = self.get_radial_model()
            
            # Get MC y profile
            MC_y_profile[i,:] = radial_profile_sb(MC_ymap_sph[i,:,:], 
                                                  (best_model.coord.icrs.ra.to_value('deg'),
                                                   best_model.coord.icrs.dec.to_value('deg')), 
                                                  stddev=self.data.image*0+1, header=self.data.header, 
                                                  binsize=binsize_prof.to_value('deg'))[1]
            # Get MC pressure profile
            MC_pressure_profile[i,:] = self.model.get_pressure_profile(r3d*u.kpc)[1].to_value('keV cm-3')
        
        #========== plots map
        utils_plot.show_fit_result_ymap(self.output_dir+'/CurveFit_Profile_results_y_map.pdf',
                                        self.data.image,
                                        self.data.header,
                                        noise_mc,
                                        best_ymap_sph,
                                        mask=self.data.mask,
                                        visu_smooth=visu_smooth.to_value('arcsec'))
        
        #========== plots ymap profile
        utils_plot.show_fit_ycompton_profile(self.output_dir+'/CurveFit_Profile_results_y_profile.pdf',
                                             r2d, data_yprof, data_yprof_err,
                                             best_y_profile, MC_y_profile,
                                             true_compton_profile=true_compton_profile)
        
        #========== plots pressure profile
        utils_plot.show_fit_result_pressure_profile(self.output_dir+'/CurveFit_Profile_results_P_profile.pdf',
                                                    r3d, best_pressure_profile, MC_pressure_profile,
                                                    true_pressure_profile=true_pressure_profile)
        
