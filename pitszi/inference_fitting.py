"""
This file deals with the model fitting

"""

import numpy as np
import copy
import astropy.units as u
from astropy.coordinates import SkyCoord
from pathos.multiprocessing import ProcessPool
from multiprocessing import Pool, cpu_count
import emcee
from minot.ClusterTools.map_tools import radial_profile_sb

from pitszi import utils_fitting
from pitszi import utils_plot


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

        # Produce 1D plots of the chains
        utils_plot.chains_1Dplots(par_chains, parlist, self.output_dir+'/MCMC'+extraname+'_chain_1d_plot.pdf')
        
        # Produce 1D histogram of the chains
        namefiles = [self.output_dir+'/MCMC'+extraname+'_chain_hist_'+i+'.pdf' for i in parlist]
        utils_plot.chains_1Dhist(par_chains, parlist, namefiles,
                                 conf=conf, truth=truth)

        # Produce 2D (corner) plots of the chains
        utils_plot.chains_2Dplots_corner(par_chains,
                                         parlist,
                                         self.output_dir+'/MCMC'+extraname+'_chain_2d_plot_corner.pdf',
                                         truth=truth)

        utils_plot.chains_2Dplots_sns(par_chains,
                                      parlist,
                                      self.output_dir+'/MCMC'+extraname+'_chain_2d_plot_sns.pdf',
                                      truth=truth)
    

    #==================================================
    # Profile parameter definition
    #==================================================
        
    def defpar_profile(self, parinfo):
        """
        This function helps defining the parameter list, initial values and
        ranges for the profile fitting.
        We distinguish 4 kind of parameters:
        - profile parameters
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

        # Loop over the keys
        Npar_prof = len(list(parinfo_prof.keys()))
        for ipar in range(Npar_prof):
            
            parkey = list(parinfo_prof.keys())[ipar]

            #----- Check that parameters are working
            try:
                bid = self.model.model_pressure_profile[parkey]
            except:
                raise ValueError('The parameter '+parkey+' is not in self.model.model_pressure_profile')    

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
            self.setup()
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
                             show_fit_result=False):
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
        - show_fit_result (bool): show the best fit model and residual.

        Outputs
        ----------
        - par_opt (list): the best fit parameters
        - par_cov (2d matrix): the posterior covariance matrix

        """


        



        















        
    #==================================================
    # Fluctuation parameter definition
    #==================================================
        
    def defpar_fluctuation(self, parinfo):
        """
        This function helps defining the parameter list, initial values and
        ranges for the fluctuation fitting.
        We distinguish two kind of parameters:
        - fluctuation parameters
        - noise parameters (noise amplitude only for now, but CMB, CIB could be added)
            
        Parameters
        ----------
        - parinfo_fluct (dict): see parinfo_fluct in get_pk3d_model_forward_fitting
    
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

        #========== Check the fluctuation parameters
        # remove unwanted keys
        parinfo_fluct = copy.deepcopy(parinfo)
        if 'Anoise' in parkeys: parinfo_fluct.pop('Anoise')
        
        # Loop over the keys
        Npar_fluct = len(list(parinfo_fluct.keys()))
        for ipar in range(Npar_fluct):
            
            parkey = list(parinfo_fluct.keys())[ipar]

            #----- Check that parameters are working
            try:
                bid = self.model.model_pressure_fluctuation[parkey]
            except:
                raise ValueError('The parameter '+parkey+' is not in self.model.model_pressure_fluctuation')    
            
            if 'guess' not in parinfo_fluct[parkey]:
                raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
            if 'unit' not in parinfo_fluct[parkey]:
                raise ValueError('The unit key is mandatory. Use "unit":None if unitless')
    
            #----- Update the name list
            par_list.append(parkey)
    
            #----- Update the guess values
            par0_value.append(parinfo_fluct[parkey]['guess'][0])
            par0_err.append(parinfo_fluct[parkey]['guess'][1])
    
            #----- Update the limit values
            if 'limit' in parinfo_fluct[parkey]:
                par_min.append(parinfo_fluct[parkey]['limit'][0])
                par_max.append(parinfo_fluct[parkey]['limit'][1])
            else:
                par_min.append(-np.inf)
                par_max.append(+np.inf)

        #========== Noise ampli
        list_nuisance_allowed = ['Anoise']
        parinfo_nuisance = {key: parinfo[key] for key in list_nuisance_allowed if key in parinfo.keys()}
        Npar_nuisance = len(list(parinfo_nuisance.keys()))
    
        for ipar in range(Npar_nuisance):
    
            parkey = list(parinfo_nuisance.keys())[ipar]
    
            #----- Check that the param is ok wrt the model
            if 'guess' not in parinfo_nuisance[parkey]:
                raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
            if 'unit' not in parinfo_nuisance[parkey]:
                raise ValueError('The unit key is mandatory. Use "unit":None if unitless')
    
            #----- Update the name list
            par_list.append(parkey)
    
            #----- Update the guess values
            par0_value.append(parinfo_nuisance[parkey]['guess'][0])
            par0_err.append(parinfo_nuisance[parkey]['guess'][1])
    
            #----- Update the limit values
            if 'limit' in parinfo_nuisance[parkey]:
                par_min.append(parinfo_nuisance[parkey]['limit'][0])
                par_max.append(parinfo_nuisance[parkey]['limit'][1])
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
    # Fluctuation prior definition
    #==================================================
        
    def prior_fluctuation(self, param, parinfo):
        """
        Compute the prior given the input parameter and some information
        about the parameters given by the user
        We distinguish two kind of parameters:
        - fluctuation parameters
        - noise parameters (noise amplitude only for now, but CMB, CIB could be added)
            
         Parameters
         ----------
         - param (list): the parameter to apply to the model
         - parinfo (dict): fluctuation parameter information
     
         Outputs
         ----------
         - prior (float): value of the prior

        """
        
        prior = 0
        idx_par = 0
        parkeys = list(parinfo.keys())

        #========== Check the fluctuation parameters
        # remove unwanted keys
        parinfo_fluct = copy.deepcopy(parinfo)
        if 'Anoise' in parkeys: parinfo_fluct.pop('Anoise')

        # Loop over the keys
        parkeys_fluct = list(parinfo_fluct.keys())

        for ipar in range(len(parkeys_fluct)):
            parkey = parkeys_fluct[ipar]
        
            # Flat prior
            if 'limit' in parinfo_fluct[parkey]:
                if param[idx_par] < parinfo_fluct[parkey]['limit'][0]:
                    return -np.inf                
                if param[idx_par] > parinfo_fluct[parkey]['limit'][1]:
                    return -np.inf
                
            # Gaussian prior
            if 'prior' in parinfo_fluct[parkey]:
                expected = parinfo_fluct[parkey]['prior'][0]
                sigma = parinfo_fluct[parkey]['prior'][1]
                prior += -0.5*(param[idx_par] - expected)**2 / sigma**2
    
            # Increase param index
            idx_par += 1

        #========== Other parameters
        list_allowed  = ['Anoise']
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
    # Set model with fluctuation parameter value
    #==================================================
        
    def setpar_fluctuation(self, param, parinfo):
        """
        Set the model to the given fluctuation parameters
        We distinguish two kind of parameters:
        - fluctuation parameters
        - noise parameters (noise amplitude only for now, but CMB, CIB could be added)
            
        Parameters
        ----------
        - param (list): the parameters to apply to the model
        - parinfo (dict): see mcmc_fluctuation
    
        Outputs
        ----------
        The model is updated
        
        """

        idx_par = 0
        parkeys = list(parinfo.keys())

        #========== Check the fluctuation parameters
        # remove unwanted keys
        parinfo_fluct = copy.deepcopy(parinfo)
        if 'Anoise' in parkeys: parinfo_fluct.pop('Anoise')
        parkeys_fluct = list(parinfo_fluct.keys())

        # Loop Profile
        for ipar in range(len(parkeys_fluct)):
            parkey = parkeys_fluct[ipar]
            if parinfo_fluct[parkey]['unit'] is not None:
                unit = parinfo_fluct[parkey]['unit']
            else:
                unit = 1
            self.model.model_pressure_fluctuation[parkey] = param[idx_par] * unit
            idx_par += 1

        #========== Noise amplitude
        if 'Anoise' in parkeys:
            self.nuisance_Anoise = param[idx_par]
            idx_par += 1

        #========== Final check on parameter count
        if len(param) != idx_par:
            print('issue with the number of parameters')
            import pdb
            pdb.set_trace()


    #==================================================
    # lnL function for the profile fit
    #==================================================
    
    def lnlike_fluctuation(self, param, parinfo, kind='projection'):
        """
        This is the likelihood function used for the fit of the fluctuation.
            
        Parameters
        ----------
        - param (np array): the value of the test parameters
        - parinfo (dict): see parinfo in mcmc_fluctuation
        - kind (str): projection or brute, for using 
        get_pk2d_model_proj or get_pk2d_model_brute to compute the model

        Outputs
        ----------
        - lnL (float): the value of the log likelihood
    
        """
        
        #========== Deal with flat and Gaussian priors
        prior = self.prior_fluctuation(param, parinfo)
        if not np.isfinite(prior): return -np.inf
        
        #========== Change model parameters
        self.setpar_fluctuation(param, parinfo)
        
        #========== Get the model
        if kind == 'brute':
            pk2d_test = self.get_pk2d_model_brute(physical=True, seed=None)[1].to_value('kpc2')
        elif kind == 'projection':
            pk2d_test = self.get_pk2d_model_proj(physical=True)[1].to_value('kpc2')
        else:
            raise ValueError('lnlike_fluctuation only accepts kind="brute" or kind="projection".')
            
        #========== Compute the likelihood
        # collect products
        residual_test = self._pk2d_data - pk2d_test
        
        if self.method_use_covmat:
            if kind == 'brute':      inverse_cov = self._pk2d_totref_invcov
            if kind == 'projection': inverse_cov = self._pk2d_noise_invcov
        else:
            if kind == 'brute':      variance = self._pk2d_noise_rms**2 + self._pk2d_modref_rms**2
            if kind == 'projection': variance = self._pk2d_noise_rms**2
        
        # compute lnL
        if self.method_use_covmat:
            lnL = -0.5*np.matmul(residual_test, np.matmul(inverse_cov, residual_test))
        else:
            lnL = -0.5*np.nansum(residual_test**2 / variance)
            
        lnL += prior
        
        #========== Check and return
        if np.isnan(lnL):
            return -np.inf
        else:
            return lnL        

        
    #==================================================
    # Compute the Pk contraint via forward deprojection
    #==================================================
    
    def run_mcmc_fluctuation(self, parinfo,
                             kind='projection',
                             filename_sampler=None,
                             show_fit_result=False,
                             set_bestfit=False):
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
        - filename_sampler (str): the file name of the sampler to use.
        - show_fit_result (bool): set to true to produce plots for fitting results
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
            sampler_file = self.output_dir+'/pitszi_MCMC_Fluctuation_'+kind+'_sampler.h5'
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
        backend = utils_fitting.define_emcee_backend(sampler_file, sampler_exist,
                                                     self.mcmc_reset, self.mcmc_nwalkers, ndim, silent=False)

        if kind == 'projection': moves = emcee.moves.StretchMove(a=2.0)
        if kind == 'brute':      moves = emcee.moves.KDEMove()

        sampler = emcee.EnsembleSampler(self.mcmc_nwalkers, ndim,
                                        self.lnlike_fluctuation, 
                                        args=[parinfo, kind], 
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
            self.get_mcmc_chains_outputs_results(par_list, sampler, extraname='_Fluctuation_'+kind)
            self.run_mcmc_fluctuation_results(sampler, parinfo, extraname='_'+kind)

        #========== Compute the best-fit model and set it
        if set_bestfit:
            best_par = utils_fitting.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
            self.setpar_fluctuation(best_par, parinfo)
            self.setup()
        else:
            self.model = input_model

        return par_list, sampler


    #==================================================
    # Show the fit results related to fluctuation
    #==================================================
    
    def run_mcmc_fluctuation_results(self, sampler, parinfo,
                                     true_pk3d=None, extraname=''):
        """
        This is function is used to show the results of the MCMC
        regarding the radial fluctuation
            
        Parameters
        ----------
        - sampler (emcee object): the sampler obtained from mcmc_profile
        - parinfo (dict): same as mcmc_profile
        - true_pk3d (dict): pass a dictionary containing the spectrum to compare with
        in the form {'k':array in kpc-1, 'pk':array in kpc3}
        
        Outputs
        ----------
        plots are produced

        """

        #========== Get the best-fit
        best_par = utils_fitting.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
        self.setpar_fluctuation(best_par, parinfo)
        k3d, best_pk3d = self.model.get_pressure_fluctuation_spectrum(np.logspace(-4,-1,100)*u.kpc**-1)

        k2d, model_pk2d_ref, model_pk2d_covmat = self.get_pk2d_model_statistics(physical=True)
        
        #========== Get the MC
        MC_pk3d = np.zeros((self.mcmc_Nresamp, len(k3d)))
        MC_pk2d = np.zeros((self.mcmc_Nresamp, len(k2d)))
        MC_pk2d_noise = np.zeros((self.mcmc_Nresamp, len(k2d)))
        
        MC_pars = utils_fitting.get_emcee_random_param(sampler, burnin=self.mcmc_burnin, Nmc=self.mcmc_Nresamp)
        for imc in range(self.mcmc_Nresamp):
            # Get MC model
            self.setpar_fluctuation(MC_pars[imc,:], parinfo)

            # Get MC noise
            MC_pk2d_noise[imc,:] = self.nuisance_Anoise * self._pk2d_noise

            # Get MC Pk2d
            MC_pk2d[imc,:] = self.get_pk2d_model_proj(physical=True)[1].to_value('kpc2') - MC_pk2d_noise[imc,:]
            
            # Get MC Pk3d
            MC_pk3d[imc,:] = self.model.get_pressure_fluctuation_spectrum(k3d)[1].to_value('kpc3')

        #========== Plot the fitted image data
        utils_plot.show_fit_result_delta_ymap(self.output_dir+'/MCMC_Fluctuation'+extraname+'_results_input_image.pdf',
                                              self.data.image,
                                              self._dy_image,
                                              self._ymap_sph1,
                                              self._ymap_sph2,
                                              self.method_w8,
                                              self.data.header)
        
        #========== Plot the covariance matrix
        utils_plot.show_fit_result_covariance(self.output_dir+'/MCMC_Fluctuation'+extraname+'_results_covariance.pdf',
                                              self._pk2d_noise_cov,
                                              model_pk2d_covmat,
                                              self._pk2d_modref_cov)
        
        #========== Plot the Pk2d constraint
        utils_plot.show_fit_result_pk2d(self.output_dir+'/MCMC_Fluctuation'+extraname+'_results_pk2d.pdf',
                                        self._kctr_kpc, self._pk2d_data,
                                        model_pk2d_ref.to_value('kpc2'), np.diag(model_pk2d_covmat.to_value('kpc4'))**0.5,
                                        self._pk2d_noise_rms,
                                        MC_pk2d, MC_pk2d_noise)
        
        #========== Plot the Pk3d constraint
        utils_plot.show_fit_result_pk3d(self.output_dir+'/MCMC_Fluctuation'+extraname+'_results_pk3d.pdf',
                                        k3d.to_value('kpc-1'), best_pk3d.to_value('kpc3'),
                                        MC_pk3d,
                                        true_pk3d=true_pk3d)
        

