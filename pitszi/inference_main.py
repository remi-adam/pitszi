"""
This file contain a subclass of the main_model.py module and ClusterModel class. It
is dedicated to the computing of mock observables.

"""

#==================================================
# Requested imports
#==================================================

import os
import numpy as np
import astropy.units as u
import astropy.constants as cst
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
import copy
import emcee
import corner

from pitszi import utils
from pitszi import plotlib

#==================================================
# Likelihood related functions
#==================================================

#========== Profile parameter definition
def defpar_lnlike_profile(fitpar_profile, model,
                          fitpar_center=None, fitpar_ellipticity=None, fitpar_ZL=None):
    """
    This function helps defining the parameter names and initial values for the profile fitting
        
    Parameters
    ----------
    - fitpar_profile (dict): see fitpar_profile in get_smooth_model_forward_fitting
    - model (class Model object): the model to be updated in the fit
    - fitpar_center (dict): see fitpar_center in get_smooth_model_forward_fitting
    - fitpar_ellipticity (dict): see fitpar_ellipticity in get_smooth_model_forward_fitting
    - fitpar_ZL (dict): see fitpar_ZL in get_smooth_model_forward_fitting

    Outputs
    ----------
    - par_list (list): list of parameter names
    - par0_value (list): list of mean initial parameter value
    - par0_err (list): list of mean initial parameter uncertainty
    - par_min (list): list of minimal initial parameter value
    - par_max (list): list of maximal initial parameter value

    """

    #========== Init the parameters
    par_list = []
    par0_value = []
    par0_err = []
    par_min = []
    par_max = []
    
    #========== Deal with profile parameters
    Npar_profile = len(list(fitpar_profile.keys()))
    for ipar in range(Npar_profile):
        
        parkey = list(fitpar_profile.keys())[ipar]

        #----- Check that the param is ok wrt the model
        try:
            bid = model.model_pressure_profile[parkey]
        except:
            raise ValueError('The parameter '+parkey+' is not in self.model')    
        if 'guess' not in fitpar_profile[parkey]:
            raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
        if 'unit' not in fitpar_profile[parkey]:
            raise ValueError('The unit key is mandatory. Use "unit":None if unitless')

        #----- Update the name list
        par_list.append(parkey)

        #----- Update the guess values
        par0_value.append(fitpar_profile[parkey]['guess'][0])
        par0_err.append(fitpar_profile[parkey]['guess'][1])

        #----- Update the limit values
        if 'limit' in fitpar_profile[parkey]:
            par_min.append(fitpar_profile[parkey]['limit'][0])
            par_max.append(fitpar_profile[parkey]['limit'][1])
        else:
            par_min.append(-np.inf)
            par_max.append(+np.inf)
            
    #========== Deal with center
    if fitpar_center is not None:
        Npar_ctr = len(list(fitpar_center.keys()))
        list_ctr_allowed = ['RA', 'Dec']

        # Check that we have first RA, then Dec
        if Npar_ctr == 2:
            if list(fitpar_center.keys())[0] != 'RA':
                raise ValueError('The center parameters should be RA, Dec, with this order')
        
        for ipar in range(Npar_ctr):

            parkey = list(fitpar_center.keys())[ipar]

            #----- Check that the param is ok wrt the model
            if parkey not in list_ctr_allowed:
                raise ValueError('fitpar_center allowed param are RA, Dec only, in that order')
            if 'guess' not in fitpar_center[parkey]:
                raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
            if 'unit' not in fitpar_center[parkey]:
                raise ValueError('The unit key is mandatory. Use "unit":None if unitless')

            #----- Update the name list
            par_list.append(parkey)

            #----- Update the guess values
            par0_value.append(fitpar_center[parkey]['guess'][0])
            par0_err.append(fitpar_center[parkey]['guess'][1])

            #----- Update the limit values
            if 'limit' in fitpar_center[parkey]:
                par_min.append(fitpar_center[parkey]['limit'][0])
                par_max.append(fitpar_center[parkey]['limit'][1])
            else:
                par_min.append(-np.inf)
                par_max.append(+np.inf)
                
    #========== Deal ellipticity
    if fitpar_ellipticity is not None:
        Npar_ell = len(list(fitpar_ellipticity.keys()))
        list_ell_allowed = ['min_to_maj_axis_ratio', 'angle']

        # Check that we have first fitpar_ellipticity, then angle
        if Npar_ell != 2:
            raise ValueError('The ellipticity parameters should be min_to_maj_axis_ratio and angle, the two being mandatory')
        if list(fitpar_ellipticity.keys())[0] != 'min_to_maj_axis_ratio':
            raise ValueError('The ellipticity parameters should be min_to_maj_axis_ratio, angle, with this order')
            
        for ipar in range(Npar_ell):

            parkey = list(fitpar_ellipticity.keys())[ipar]

            #----- Check that the param is ok wrt the model
            if parkey not in list_ell_allowed:
                raise ValueError('fitpar_ellipticity allowed param are min_to_maj_axis_ratio, angle only')
            if 'guess' not in fitpar_ellipticity[parkey]:
                raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
            if 'unit' not in fitpar_ellipticity[parkey]:
                raise ValueError('The unit key is mandatory. Use "unit":None if unitless')

            #----- Update the name list
            par_list.append(parkey)

            #----- Update the guess values
            par0_value.append(fitpar_ellipticity[parkey]['guess'][0])
            par0_err.append(fitpar_ellipticity[parkey]['guess'][1])

            #----- Update the limit values
            if 'limit' in fitpar_ellipticity[parkey]:
                par_min.append(fitpar_ellipticity[parkey]['limit'][0])
                par_max.append(fitpar_ellipticity[parkey]['limit'][1])
            else:
                if parkey == 'min_to_maj_axis_ratio':
                    par_min.append(0)
                    par_max.append(1)
                if parkey == 'angle':
                    par_min.append(-90)
                    par_max.append(+90)

    #========== Deal with offset
    if fitpar_ZL is not None:
        Npar_zl = len(list(fitpar_ZL.keys()))
        list_zl_allowed = ['ZL']
        
        for ipar in range(Npar_zl):

            parkey = list(fitpar_ZL.keys())[ipar]

            #----- Check that the param is ok wrt the model
            if parkey not in list_zl_allowed:
                raise ValueError('fitpar_ZL allowed param is ZL only')
            if 'guess' not in fitpar_ZL[parkey]:
                raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
            if 'unit' not in fitpar_ZL[parkey]:
                raise ValueError('The unit key is mandatory. Use "unit":None if unitless')

            #----- Update the name list
            par_list.append(parkey)

            #----- Update the guess values
            par0_value.append(fitpar_ZL[parkey]['guess'][0])
            par0_err.append(fitpar_ZL[parkey]['guess'][1])

            #----- Update the limit values
            if 'limit' in fitpar_ZL[parkey]:
                par_min.append(fitpar_ZL[parkey]['limit'][0])
                par_max.append(fitpar_ZL[parkey]['limit'][1])
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


#========== The the model to the best fit recovered profile
def set_bestfit_profile(model,
                        sampler,
                        burnin,
                        fitpar_profile,
                        fitpar_center=None,
                        fitpar_ellipticity=None):
    """
    Set the model to the best-fit profile parameters
        
    Parameters
    ----------
    - model (pitszi object): pitszi object from class Model
    - sampler (emcee object): the sampler
    - burnin (int): the number of mcmc points to burnin
    - fitpar_profile (dict): see get_smooth_model_forward_fitting
    - fitpar_center (dict): see get_smooth_model_forward_fitting
    - fitpar_ellipticity (dict): see get_smooth_model_forward_fitting

    Outputs
    ----------
    - model (float): the value of the likelihood

    """
    
    #---------- Extract best fit param
    param_chains = sampler.chain[:, burnin:, :]
    lnL_chains   = sampler.lnprobability[:, burnin:]
    par_flat = param_chains.reshape(param_chains.shape[0]*param_chains.shape[1], param_chains.shape[2])
    lnL_flat = lnL_chains.reshape(lnL_chains.shape[0]*lnL_chains.shape[1])
    wbest = np.where(lnL_flat == np.amax(lnL_flat))[0][0]
    par_best = par_flat[wbest]

    idx_par = 0

    #---------- Profile parameters
    parkeys_prof = list(fitpar_profile.keys())
    for ipar in range(len(parkeys_prof)):
        parkey = parkeys_prof[ipar]
        if fitpar_profile[parkey]['unit'] is not None:
            unit = fitpar_profile[parkey]['unit']
        else:
            unit = 1
        model.model_pressure_profile[parkey] = par_best[idx_par] * unit
        idx_par += 1

    #---------- Center parameters
    if fitpar_center is not None:
        if 'RA' in fitpar_center:
            RA = par_best[idx_par]
            unit1 = fitpar_center['RA']['unit']
            idx_par += 1
        else:
            RA = self.model.coord.icrs.ra.to_value('deg')
            unit1 = u.deg
        if 'Dec' in fitpar_center:
            Dec = par_best[idx_par]
            unit2 = fitpar_center['Dec']['unit']
            idx_par += 1
        else:
            Dec = self.model.coord.icrs.dec.to_value('deg')
            unit2 = u.deg

        model.coord = SkyCoord(RA*unit1, Dec*unit2, frame='icrs')
        
    #---------- Ellipticity parameters
    if fitpar_ellipticity is not None:
        axis_ratio = par_best[idx_par]
        idx_par += 1
        angle = par_best[idx_par]
        angle_unit = fitpar_ellipticity['angle']['unit']
        idx_par += 1
        model.triaxiality = {'min_to_maj_axis_ratio':axis_ratio,
                             'int_to_maj_axis_ratio':axis_ratio,
                             'euler_angle1':0*u.deg,
                             'euler_angle2':90*u.deg,
                             'euler_angle3':angle*angle_unit}

    return model


#========== lnL function for the profile file
global get_lnlike_profile # Must be global and here to avoid pickling errors
def get_lnlike_profile(param,
                       param_info_profile,
                       model,
                       data_img, data_noiseprop, data_mask,
                       data_psf, data_tf,
                       param_info_center=None, param_info_ellipticity=None, param_info_ZL=None,
                       use_covmat=False):
    """
    This is the likelihood function used for the fit of the profile.
        
    Parameters
    ----------
    - param (np array): the value of the test parameters
    - param_info_profile (dict): see fitpar_profile in get_smooth_model_forward_fitting
    - model (class Model object): the model to be updated
    - data_img (nd array): the image data
    - data_noiseprop (nd_array): the inverse covariance matrix to be used for the fit or the rms,
    depending on use_covmat.
    - data_mask (nd array): the image mask
    - data_psf (quantity): input to model_mock.get_sz_map
    - data_tf (dict): input to model_mock.get_sz_map
    - param_info_center (dict): see fitpar_center in get_smooth_model_forward_fitting
    - param_info_ellipticity (dict): see fitpar_ellipticity in get_smooth_model_forward_fitting
    - param_info_ZL (dict): see fitpar_ZL in get_smooth_model_forward_fitting
    - use_covmat (bool): if True, assumes that data_noiseprop is the inverse noise covariance matrix

    Outputs
    ----------
    - lnL (float): the value of the likelihood

    """
    
    Nparam = len(param)    

    #========== Deal with flat and Gaussian priors
    prior = 0
    idx_par = 0
    
    #---------- Profile parameters
    parkeys_prof = list(param_info_profile.keys())

    for ipar in range(len(parkeys_prof)):
        parkey = parkeys_prof[ipar]
        
        # Flat prior
        if 'limit' in param_info_profile[parkey]:
            if param[idx_par] < param_info_profile[parkey]['limit'][0]:
                return -np.inf                
            if param[idx_par] > param_info_profile[parkey]['limit'][1]:
                return -np.inf
            
        # Gaussian prior
        if 'prior' in param_info_profile[parkey]:
            expected = param_info_profile[parkey]['prior'][0]
            sigma = param_info_profile[parkey]['prior'][1]
            prior += -0.5*(param[idx_par] - expected)**2 / sigma**2

        # Increase param index
        idx_par += 1
        
    #---------- Center parameters
    if param_info_center is not None:
        parkeys_ctr = list(param_info_center.keys())
        
        for ipar in range(len(param_info_center)):
            parkey = parkeys_ctr[ipar]
            
            # Flat prior
            if 'limit' in param_info_center[parkey]:
                if param[idx_par] < param_info_center[parkey]['limit'][0]:
                    return -np.inf                
                if param[idx_par] > param_info_center[parkey]['limit'][1]:
                    return -np.inf
                
            # Gaussian prior
            if 'prior' in param_info_center[parkey]:
                expected = param_info_center[parkey]['prior'][0]
                sigma = param_info_center[parkey]['prior'][1]
                prior += -0.5*(param[idx_par] - expected)**2 / sigma**2

            # Increase param index
            idx_par += 1

    #---------- Ellipticity parameters
    if param_info_ellipticity is not None:
        parkeys_ell = list(param_info_ellipticity.keys())
        
        for ipar in range(len(param_info_ellipticity)):
            parkey = parkeys_ell[ipar]
            
            # Flat prior
            if 'limit' in param_info_ellipticity[parkey]:
                if param[idx_par] < param_info_ellipticity[parkey]['limit'][0]:
                    return -np.inf                
                if param[idx_par] > param_info_ellipticity[parkey]['limit'][1]:
                    return -np.inf
                
            # Gaussian prior
            if 'prior' in param_info_ellipticity[parkey]:
                expected = param_info_ellipticity[parkey]['prior'][0]
                sigma = param_info_ellipticity[parkey]['prior'][1]
                prior += -0.5*(param[idx_par] - expected)**2 / sigma**2

            # Increase param index
            idx_par += 1
        
    #---------- Offset parameters
    if param_info_ZL is not None:
        parkeys_zl = list(param_info_ZL.keys())
        
        for ipar in range(len(param_info_ZL)):
            parkey = parkeys_zl[ipar]
            
            # Flat prior
            if 'limit' in param_info_ZL[parkey]:
                if param[idx_par] < param_info_ZL[parkey]['limit'][0]:
                    return -np.inf                
                if param[idx_par] > param_info_ZL[parkey]['limit'][1]:
                    return -np.inf
                
            # Gaussian prior
            if 'prior' in param_info_ZL[parkey]:
                expected = param_info_ZL[parkey]['prior'][0]
                sigma = param_info_ZL[parkey]['prior'][1]
                prior += -0.5*(param[idx_par] - expected)**2 / sigma**2

            # Increase param index
            idx_par += 1

    #---------- Check on param numbers
    if idx_par != Nparam:
        raise ValueError('Problem with the prior parameters')
        
    #========== Change model parameters
    idx_par = 0

    #---------- Profile parameters
    parkeys_prof = list(param_info_profile.keys())
    for ipar in range(len(parkeys_prof)):
        parkey = parkeys_prof[ipar]
        if param_info_profile[parkey]['unit'] is not None:
            unit = param_info_profile[parkey]['unit']
        else:
            unit = 1
        model.model_pressure_profile[parkey] = param[idx_par] * unit
        idx_par += 1

    #---------- Center parameters
    if param_info_center is not None:
        
        if 'RA' in param_info_center:
            RA = param[idx_par]
            unit1 = param_info_center['RA']['unit']
            idx_par += 1
        else:
            RA = model.coord.icrs.ra.to_value('deg')
            unit1 = u.deg
        if 'Dec' in param_info_center:
            Dec = param[idx_par]
            unit2 = param_info_center['Dec']['unit']
            idx_par += 1
        else:
            Dec = model.coord.icrs.dec.to_value('deg')
            unit2 = u.deg

        model.coord = SkyCoord(RA*unit1, Dec*unit2, frame='icrs')
        
    #---------- Ellipticity parameters
    if param_info_ellipticity is not None:
        axis_ratio = param[idx_par]
        idx_par += 1
        angle = param[idx_par]
        angle_unit = param_info_ellipticity['angle']['unit']
        idx_par += 1
        model.triaxiality = {'min_to_maj_axis_ratio':axis_ratio, 'int_to_maj_axis_ratio':axis_ratio,
                             'euler_angle1':0*u.deg, 'euler_angle2':90*u.deg, 'euler_angle3':angle*angle_unit}
    
    #---------- Offset parameters
    if param_info_ZL is not None:
        zero_level = param[idx_par]
        idx_par += 1
    else:
        zero_level = 0
        
    #========== Get the model
    model_img = model.get_sz_map(seed=None, no_fluctuations=True, force_isotropy=False,
                                 irfs_convolution_beam=data_psf, irfs_convolution_TF=data_tf)
    model_img = model_img + zero_level
    
    #========== Compute the likelihood
    if use_covmat:
        flat_resid = (data_mask * (model_img - data_img)).flatten()
        lnL = -0.5*np.matmul(flat_resid, np.matmul(data_noiseprop, flat_resid))
    else:
        lnL = -0.5*np.nansum(data_mask**2 * (model_img - data_img)**2 / data_noiseprop**2)

    lnL += prior
    
    #========== Check and return
    if np.isnan(lnL):
        return -np.inf
    else:
        return lnL

    
#========== Profile parameter definition
def defpar_lnlike_fluct(fitpar_fluct, model,
                        fitpar_noise=None):
    """
    This function helps defining the parameter names and initial values for the fluctuation fitting
        
    Parameters
    ----------
    - fitpar_fluct (dict): see fitpar_fluct in get_pk3d_model_forward_fitting
    - model (class Model object): the model to be updated in the fit
    - fitpar_noise (dictionary): same as fitpar_fluct with accepted parameter 'Noise'

    Outputs
    ----------
    - par_list (list): list of parameter names
    - par0_value (list): list of mean initial parameter value
    - par0_err (list): list of mean initial parameter uncertainty
    - par_min (list): list of minimal initial parameter value
    - par_max (list): list of maximal initial parameter value

    """

    #========== Init the parameters
    par_list = []
    par0_value = []
    par0_err = []
    par_min = []
    par_max = []
    
    #========== Deal with fluctuation parameters
    Npar_fluct = len(list(fitpar_fluct.keys()))
    for ipar in range(Npar_fluct):
        
        parkey = list(fitpar_fluct.keys())[ipar]

        #----- Check that the param is ok wrt the model
        try:
            bid = model.model_pressure_fluctuation[parkey]
        except:
            raise ValueError('The parameter '+parkey+' is not in self.model')    
        if 'guess' not in fitpar_fluct[parkey]:
            raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
        if 'unit' not in fitpar_fluct[parkey]:
            raise ValueError('The unit key is mandatory. Use "unit":None if unitless')

        #----- Update the name list
        par_list.append(parkey)

        #----- Update the guess values
        par0_value.append(fitpar_fluct[parkey]['guess'][0])
        par0_err.append(fitpar_fluct[parkey]['guess'][1])

        #----- Update the limit values
        if 'limit' in fitpar_fluct[parkey]:
            par_min.append(fitpar_fluct[parkey]['limit'][0])
            par_max.append(fitpar_fluct[parkey]['limit'][1])
        else:
            par_min.append(-np.inf)
            par_max.append(+np.inf)

    #========== Deal with noise normalization
    if fitpar_noise is not None:
        Npar_noise = len(list(fitpar_noise.keys()))
        list_noise_allowed = ['noise']
        
        for ipar in range(Npar_noise):

            parkey = list(fitpar_noise.keys())[ipar]

            #----- Check that the param is ok wrt the model
            if parkey not in list_noise_allowed:
                raise ValueError('fitpar_noise allowed param is noise only')
            if 'guess' not in fitpar_noise[parkey]:
                raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
            if 'unit' not in fitpar_noise[parkey]:
                raise ValueError('The unit key is mandatory. Use "unit":None if unitless')

            #----- Update the name list
            par_list.append(parkey)

            #----- Update the guess values
            par0_value.append(fitpar_noise[parkey]['guess'][0])
            par0_err.append(fitpar_noise[parkey]['guess'][1])

            #----- Update the limit values
            if 'limit' in fitpar_noise[parkey]:
                par_min.append(fitpar_noise[parkey]['limit'][0])
                par_max.append(fitpar_noise[parkey]['limit'][1])
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


#========== lnL function for the profile file
global get_lnlike_fluctuation # Must be global and here to avoid pickling errors
def get_lnlike_fluctuation(param,
                           param_info_fluct,
                           method_fluctuation_image,
                           model,
                           model_pk2d_rmsref,
                           model_pk2d_ref,                           
                           data_pk2d,
                           noise_pk2d_rms,
                           noise_pk2d_mean,
                           model_ymap_sph,
                           model_ymap_sph_deconv,
                           reso_arcsec,
                           psf_fwhm,
                           transfer_function,
                           mask,
                           kbin_Nbin=30, 
                           kbin_min_arcsec=0.0,
                           kbin_max_arcsec=0.1,
                           kbin_scale='lin',
                           param_info_noise=None,
                           use_covmat=False,
                           covmat=None):
    """
    This is the likelihood function used for the forward fit of the fluctuation spectrum.
        
    Parameters
    ----------
    - param (np array): the value of the test parameters
    - param_info_fluct (dict): see fitpar_fluct in get_pk3d_model_forward_fitting
    - model (class Model object): the model to be updated
    - param_info_noise (dict): see fitpar_noise in get_pk3d_model_forward_fitting

    Outputs
    ----------
    - lnL (float): the value of the likelihood

    """

    Nparam = len(param)    
    
    #========== Deal with flat and Gaussian priors
    prior = 0
    idx_par = 0

    #---------- Fluctuation parameters
    parkeys_fluct = list(param_info_fluct.keys())

    for ipar in range(len(parkeys_fluct)):
        parkey = parkeys_fluct[ipar]
        
        # Flat prior
        if 'limit' in param_info_fluct[parkey]:
            if param[idx_par] < param_info_fluct[parkey]['limit'][0]:
                return -np.inf                
            if param[idx_par] > param_info_fluct[parkey]['limit'][1]:
                return -np.inf
            
        # Gaussian prior
        if 'prior' in param_info_fluct[parkey]:
            expected = param_info_fluct[parkey]['prior'][0]
            sigma = param_info_fluct[parkey]['prior'][1]
            prior += -0.5*(param[idx_par] - expected)**2 / sigma**2

        # Increase param index
        idx_par += 1

    #---------- Noise parameters
    if param_info_noise is not None:
        parkeys_noise = list(param_info_noise.keys())
        
        for ipar in range(len(param_info_noise)):
            parkey = parkeys_noise[ipar]
            
            # Flat prior
            if 'limit' in param_info_noise[parkey]:
                if param[idx_par] < param_info_noise[parkey]['limit'][0]:
                    return -np.inf                
                if param[idx_par] > param_info_noise[parkey]['limit'][1]:
                    return -np.inf
                
            # Gaussian prior
            if 'prior' in param_info_noise[parkey]:
                expected = param_info_noise[parkey]['prior'][0]
                sigma = param_info_noise[parkey]['prior'][1]
                prior += -0.5*(param[idx_par] - expected)**2 / sigma**2

            # Increase param index
            idx_par += 1

    #---------- Check on param numbers
    if idx_par != Nparam:
        raise ValueError('Problem with the prior parameters')

    #========== Change model parameters
    idx_par = 0

    #---------- Fliuctuation parameters
    parkeys_fluct = list(param_info_fluct.keys())
    for ipar in range(len(parkeys_fluct)):
        parkey = parkeys_fluct[ipar]
        if param_info_fluct[parkey]['unit'] is not None:
            unit = param_info_fluct[parkey]['unit']
        else:
            unit = 1
        model.model_pressure_fluctuation[parkey] = param[idx_par] * unit
        idx_par += 1
    
    #---------- Offset parameters
    if param_info_noise is not None:
        noise_ampli = param[idx_par]
        idx_par += 1
    else:
        noise_ampli = 1

    #========== Get the model
    #---------- Compute the ymap and fluctuation image
    test_ymap = model.get_sz_map(seed=None,
                                 no_fluctuations=False,
                                 irfs_convolution_beam=psf_fwhm,
                                 irfs_convolution_TF=transfer_function)

    if method_fluctuation_image == 'ratio':
        test_image = (test_ymap - model_ymap_sph)/model_ymap_sph_deconv
    elif method_fluctuation_image == 'subtract':
        test_image = test_ymap - model_ymap_sph
    else:
        raise ValueError('inference.method_fluctuation_image can be either "ratio" or "difference"')
    test_image *= mask

    #---------- Compute the test Pk
    test_k, test_pk =  utils.get_pk2d(test_image,
                                      reso_arcsec, 
                                      Nbin=kbin_Nbin, 
                                      kmin=kbin_min_arcsec,
                                      kmax=kbin_max_arcsec,
                                      scalebin=kbin_scale)
    
    model_pk2d = test_pk + noise_ampli*noise_pk2d_mean

    #========== Compute the likelihood
    if use_covmat:
        #covmat_model = covmat_model_ref
        #covariance = covmat_noise + covmat_model
        vec_test = model_pk2d-data_pk2d
        covariance_inverse = np.linalg.pinv(covmat)
        lnL = -0.5*np.matmul(vec_test, np.matmul(covariance_inverse, vec_test))
    else:
        # Rescale noise using Pk/Pk_rms is cte
        #model_pk2d_rms = model_pk2d_rmsref*test_pk/model_pk2d_ref
        model_pk2d_rms = model_pk2d_rmsref
        lnL = -0.5*np.nansum((model_pk2d - data_pk2d)**2 / (noise_pk2d_rms**2 + model_pk2d_rms**2))
        
    lnL += prior

    #========== Check and return
    if np.isnan(lnL):
        return -np.inf
    else:
        return lnL


#==================================================
# Mock class
#==================================================

class Inference():
    """ Inference class
        This class infer the profile and power spectrum properties 
        given input data (from class Data()) and model (from class Model())
        that are attributes

    Attributes
    ----------
    The attributes are the same as the Model class, see model_main.py
    
    Methods
    ----------
    - get_p3d_to_p2d_from_window_function
    - 
    -
    -
    -
    
    """    

    #==================================================
    # Initialization
    #==================================================

    def __init__(self,
                 data,
                 model,
    ):
        """
        Initialize the inference object.
        
        Parameters
        ----------
        - data (pitszi Class Data object): the data
        - model (pitszi Class Model object): the model
        
        """

        # Admin
        self.silent     = True
        self.output_dir = './pitszi_output'

        # Input data and model (deepcopy to avoid modifying the input when fitting)
        self.data  = copy.deepcopy(data)
        self.model = copy.deepcopy(model) 

        # Analysis methodology
        self.method_pk2d_extraction = 'naive'   # [naive, arevalo12]
        self.method_fluctuation_image = 'ratio' # [ratio, difference]

        # Binning in k
        self.kbin_min   = 0 * u.arcsec**-1
        self.kbin_max   = 1/10.0*u.arcsec**-1
        self.kbin_Nbin  = 30
        self.kbin_scale = 'lin' # ['lin', 'log']

        # MCMC parameters
        self.mcmc_nwalkers           = 100
        self.mcmc_nsteps             = 500
        self.mcmc_burnin             = 100
        self.mcmc_reset              = False
        self.mcmc_run                = True
        

    #==================================================
    # Compute the Pk contraint via forward fitting
    #==================================================
    
    def get_mcmc_output_results(self,
                                parlist,
                                sampler,
                                conf=68.0,
                                truth=None,
                                extraname=''):
        """
        This function brute force fits the 3d power spectrum
        using a forward modeling approach
        
        Parameters
        ----------
        - parlist (list): the list of parameter names
        - sampler (emcee object): the sampler
        - conf (float): confidence limit in % used in results
        - show (bool): show the results (or only output files)
        - extraname (string): extra name to add after MCMC in file name

        Outputs
        ----------
        Plots are produced

        """

        Nchains, Nsample, Nparam = sampler.chain.shape

        #---------- Remove the burnin
        if self.mcmc_burnin <= Nsample:
            par_chains = sampler.chain[:,self.mcmc_burnin:,:]
            lnl_chains = sampler.lnprobability[:,self.mcmc_burnin:]
        else:
            par_chains = sampler.chain
            lnl_chains = sampler.lnprobability
            print('The burnin could not be remove because it is larger than the chain size')
            
        #---------- Compute statistics for the chains
        utils.chains_statistics(par_chains, lnl_chains,
                                parname=parlist,
                                conf=conf,
                                outfile=self.output_dir+'/MCMC'+extraname+'_chain_statistics.txt')

        # Produce 1D plots of the chains
        plotlib.chains_1Dplots(par_chains, parlist, self.output_dir+'/MCMC'+extraname+'_chain_1d_plot.pdf')
        
        # Produce 1D histogram of the chains
        namefiles = [self.output_dir+'/MCMC'+extraname+'_chain_1d_hist_'+i+'.pdf' for i in parlist]
        plotlib.chains_1Dhist(par_chains, parlist, namefiles,
                              conf=conf, truth=truth)

        # Produce 2D (corner) plots of the chains
        plotlib.chains_2Dplots_corner(par_chains,
                                      parlist,
                                      self.output_dir+'/MCMC'+extraname+'_chain_2d_plot_corner.pdf',
                                      truth=truth)

        plotlib.chains_2Dplots_sns(par_chains,
                                   parlist,
                                   self.output_dir+'/MCMC'+extraname+'_chain_2d_plot_sns.pdf',
                                   truth=truth)

        
    #==================================================
    # Compute the smooth model via forward fitting
    #==================================================
    
    def get_smooth_model_forward_fitting(self,
                                         fitpar_profile,
                                         fitpar_center=None,
                                         fitpar_ellipticity=None,
                                         fitpar_ZL=None,
                                         use_covmat=False,
                                         set_best_fit=True,
                                         show_bestfit=False):
        """
        This function fits the data with a defined model
        
        Parameters
        ----------
        - fitpar_profile (dictionary): the model parameters associated with 
        self.model.model_pressure_profile to be fit as, e.g., 
        fitpar_prof = {'P_0':                    # --> Parameter key (mandatory)
                       {'guess':[0.01, 0.001],   # --> initial guess: center, uncertainty (mandatory)
                        'unit': u.keV*u.cm**-3,  # --> unit (mandatory, None if unitless)
                        'limit':[0, np.inf],     # --> Allowed range, i.e. flat prior (optional)
                        'prior':[0.01, 0.001],   # --> Gaussian prior: mean, sigma (optional)
                        },
                        'r_p':
                        {'limit':[0, np.inf], 
                          'prior':[100, 1],
                          'unit': u.kpc, 
                        },
                       }    
        - fitpar_center (dictionary): same as fitpar_profile with accepted parameters
        'RA' and 'Dec'
        - fitpar_ellipticity (dictionary): same as fitpar_profile with accepted parameters
        'min_to_maj_axis_ratio' and 'angle'
        - fitpar_ZL (dictionary): same as fitpar_profile with accepted parameter
        'ZL'
        - use_covmat (bool): if True, the inverse noise covariance matrix is used instead 
        of the noise rms
        - set_best_fit (bool): set the best fit model to the best fit parameters
        - show_best_fit (bool): show the best fit model and residual. If true, set_best_fit
        will automatically be true

        Outputs
        ----------
        - sampler (emcee object): the sampler associated with model parameters of
        the smooth component

        """
        
        #========== Check if the MCMC sampler was already recorded
        sampler_file = self.output_dir+'/pitszi_MCMC_profile_sampler.h5'
        sampler_exist = utils.check_sampler_exist(sampler_file, silent=False)
            
        #========== Defines the fit parameters
        # Fit parameter list and information
        par_list,par0_value,par0_err,par_min,par_max = defpar_lnlike_profile(fitpar_profile,
                                                                             self.model,
                                                                             fitpar_ZL=fitpar_ZL,
                                                                             fitpar_center=fitpar_center,
                                                                             fitpar_ellipticity=fitpar_ellipticity)
        ndim = len(par0_value)
        
        # Starting points of the chains
        pos = utils.emcee_starting_point(par0_value, par0_err, par_min, par_max, self.mcmc_nwalkers)
        if sampler_exist and (not self.mcmc_reset): pos = None
            
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
        backend = utils.define_emcee_backend(sampler_file, sampler_exist,
                                             self.mcmc_reset, self.mcmc_nwalkers, ndim, silent=False)
        moves = emcee.moves.StretchMove(a=2.0)
        sampler = emcee.EnsembleSampler(self.mcmc_nwalkers, ndim,
                                        get_lnlike_profile, 
                                        args=[fitpar_profile,
                                              self.model,
                                              self.data.image, noise_property,
                                              self.data.mask, self.data.psf_fwhm, self.data.transfer_function,
                                              fitpar_center,
                                              fitpar_ellipticity,
                                              fitpar_ZL,
                                              use_covmat], 
                                        pool=Pool(cpu_count()),
                                        moves=moves,
                                        backend=backend)
        
        #========== Run the MCMC
        print('----- MCMC sampling -----')
        if self.mcmc_run:
            print('      - Runing '+str(self.mcmc_nsteps)+' MCMC steps')
            res = sampler.run_mcmc(pos, self.mcmc_nsteps, progress=True, store=True)
        else:
            print('      - Not running, but restoring the existing sampler')

        #========== Set the best-fit model
        if show_bestfit: set_best_fit = True
        
        if set_best_fit:
            self.model = set_bestfit_profile(self.model, sampler, self.mcmc_burnin, fitpar_profile,
                                             fitpar_center=fitpar_center, fitpar_ellipticity=fitpar_ellipticity)
                
        #========== Show best-fit
        if show_bestfit:
            model_ymap_sph = self.model.get_sz_map(no_fluctuations=True,
                                                   irfs_convolution_beam=self.data.psf_fwhm,
                                                   irfs_convolution_TF=self.data.transfer_function)
            plotlib.show_best_fit_smoothmap_results(self.output_dir+'/MCMC_best_fit_profile.pdf',
                                                    self.data, model_ymap_sph,
                                                    visu_smooth=10)                
        
        return par_list, sampler

    
    #==================================================
    # Compute the Pk contraint via forward fitting
    #==================================================
    
    def get_pk3d_model_forward_fitting(self,
                                       fitpar_pk3d,
                                       fitpar_noise=None,
                                       Nmc_noise=1000,
                                       kbin_edges=None,
                                       set_best_fit=True):
        """
        This function brute force fits the 3d power spectrum
        using a forward modeling approach
        
        Parameters
        ----------
        - fitpar_pk (dictionary): the model parameters associated with 
        self.model.model_pressure_fluctuation to be fit as, e.g., 
        fitpar_pk = {'Norm':                     # --> Parameter key (mandatory)
                       {'guess':[0.5, 0.3],      # --> initial guess: center, uncertainty (mandatory)
                        'unit': None,            # --> unit (mandatory, None if unitless)
                        'limit':[0, np.inf],     # --> Allowed range, i.e. flat prior (optional)
                        'prior':[0.5, 0.1],      # --> Gaussian prior: mean, sigma (optional)
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
        
        Outputs
        ----------
        - sampler (emcee object): the sampler associated with model parameters of
        the smooth component

        """

        #========== Check if the MCMC sampler was already recorded
        sampler_file = self.output_dir+'/pitszi_MCMC_fluctuation_sampler.h5'
        sampler_exist = utils.check_sampler_exist(sampler_file, silent=False)
            
        #========== Defines the fit parameters
        # Fit parameter list and information
        par_list, par0_value, par0_err, par_min, par_max =  defpar_lnlike_fluct(fitpar_pk3d,
                                                                                self.model,
                                                                                fitpar_noise=fitpar_noise)
        ndim = len(par0_value)
        
        # Starting points of the chains
        pos = utils.emcee_starting_point(par0_value, par0_err, par_min, par_max, self.mcmc_nwalkers)
        if sampler_exist and (not self.mcmc_reset): pos = None

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

        '''
        #========== Deal with input images and Pk
        #---------- Input image
        model_ymap_sph_deconv = self.model.get_sz_map(no_fluctuations=True,
                                                      irfs_convolution_beam=self.data.psf_fwhm)    
        model_ymap_sph        = self.model.get_sz_map(no_fluctuations=True,
                                                      irfs_convolution_beam=self.data.psf_fwhm,
                                                      irfs_convolution_TF=self.data.transfer_function) 
        data_ymap = self.data.image

        if self.method_fluctuation_image == 'ratio':
            data_image = (data_ymap - model_ymap_sph)/model_ymap_sph_deconv
        elif self.method_fluctuation_image == 'subtract':
            data_image = data_ymap - model_ymap_sph
        else:
            raise ValueError('inference.method_fluctuation_image can be either "ratio" or "difference"')
        data_image *= self.data.mask

        #---------- Pk for the data
        k2d, data_pk2d = utils.get_pk2d(data_image,
                                        self.model.get_map_reso().to_value('arcsec'), 
                                        Nbin=self.kbin_Nbin, 
                                        kmin=self.kbin_min.to_value('arcsec-1'),
                                        kmax=self.kbin_max.to_value('arcsec-1'),
                                        scalebin=self.kbin_scale)
        
        #========== Deal with how the noise should be accounted for
        #---------- Get the model rms
        model_pk2d_mc     = np.zeros((Nmc_noise, len(k2d)))
        model_pk2d_covmat = np.zeros((len(k2d), len(k2d)))
        for imc in range(Nmc_noise):
            test_ymap = self.model.get_sz_map(seed=None,
                                              no_fluctuations=False,
                                              irfs_convolution_beam=self.data.psf_fwhm,
                                              irfs_convolution_TF=self.data.transfer_function)
            if self.method_fluctuation_image == 'ratio':
                test_image = (test_ymap - model_ymap_sph)/model_ymap_sph_deconv
            if self.method_fluctuation_image == 'subtract':
                test_image = test_ymap - model_ymap_sph
            test_image *= self.data.mask
            test_pk2d = utils.get_pk2d(test_image,
                                       self.model.get_map_reso().to_value('arcsec'), 
                                       Nbin=self.kbin_Nbin, 
                                       kmin=self.kbin_min.to_value('arcsec-1'),
                                       kmax=self.kbin_max.to_value('arcsec-1'),
                                       scalebin=self.kbin_scale)[1]
            model_pk2d_mc[imc,:] = test_pk2d
        model_pk2d_mean = np.mean(model_pk2d_mc, axis=0)
        model_pk2d_rms = np.std(model_pk2d_mc, axis=0)
        model_pk2d_ref = model_pk2d_mean
        for imc in range(Nmc_noise): model_pk2d_covmat += np.matmul((model_pk2d_mc[imc,:]-model_pk2d_mean)[:,None],
                                                                    (model_pk2d_mc[imc,:]-model_pk2d_mean)[None,:])
        model_pk2d_covmat /= Nmc_noise

        #---------- Get the noise bias and rms
        noise_ymap_mc = self.data.get_noise_monte_carlo_from_model(center=None,
                                                                   seed=None,
                                                                   Nmc=Nmc_noise)
        noise_pk2d_mc     = np.zeros((Nmc_noise, len(k2d)))
        noise_pk2d_covmat = np.zeros((len(k2d), len(k2d)))
        for imc in range(Nmc_noise):
            if self.method_fluctuation_image == 'ratio':
                image_noise_mc = (noise_ymap_mc[imc,:,:])/model_ymap_sph_deconv
            if self.method_fluctuation_image == 'subtract':
                image_noise_mc = noise_ymap_mc[imc,:,:]
            image_noise_mc *= self.data.mask
            test_pk2d = utils.get_pk2d(image_noise_mc,
                                       self.model.get_map_reso().to_value('arcsec'), 
                                       Nbin=self.kbin_Nbin, 
                                       kmin=self.kbin_min.to_value('arcsec-1'),
                                       kmax=self.kbin_max.to_value('arcsec-1'),
                                       scalebin=self.kbin_scale)[1]
            noise_pk2d_mc[imc,:] = test_pk2d
        noise_pk2d_mean = np.mean(noise_pk2d_mc, axis=0)
        noise_pk2d_rms = np.std(noise_pk2d_mc, axis=0)
        for imc in range(Nmc_noise): noise_pk2d_covmat += np.matmul((noise_pk2d_mc[imc,:]-noise_pk2d_mean)[:,None],
                                                                    (noise_pk2d_mc[imc,:]-noise_pk2d_mean)[None,:])
        noise_pk2d_covmat /= Nmc_noise

        #----- Check that all bins are filled
        if np.sum(np.isnan(noise_pk2d_covmat)) > 0 or np.sum(np.isnan(model_pk2d_covmat))>0:
            print('The number of bins is such that some bins are empty, leading to nan in the data.')
            print('You may pass the bin edges directly for specific bining strategies.')
            raise ValueError('Issue with binning')
        
        '''

        #========== Define the MCMC setup
        backend = utils.define_emcee_backend(sampler_file, sampler_exist,
                                             self.mcmc_reset, self.mcmc_nwalkers, ndim, silent=False)
        moves = emcee.moves.KDEMove()
        sampler = emcee.EnsembleSampler(self.mcmc_nwalkers, ndim,
                                        get_lnlike_fluctuation,
                                        args=[fitpar_pk3d,
                                              self.method_fluctuation_image,
                                              self.model,
                                              model_pk2d_rms,
                                              model_pk2d_ref,
                                              data_pk2d,
                                              noise_pk2d_rms,
                                              noise_pk2d_mean,
                                              model_ymap_sph,
                                              model_ymap_sph_deconv,
                                              self.model.get_map_reso().to_value('arcsec'),
                                              self.data.psf_fwhm,
                                              self.data.transfer_function,
                                              self.data.mask,
                                              self.kbin_Nbin, 
                                              self.kbin_min.to_value('arcsec-1'),
                                              self.kbin_max.to_value('arcsec-1'),
                                              self.kbin_scale,
                                              fitpar_noise,
                                              True,
                                              noise_pk2d_covmat+model_pk2d_covmat,
                                        ],
                                        pool=Pool(cpu_count()),
                                        moves=moves,
                                        backend=backend)
        
        #========== Run the MCMC
        print('----- MCMC sampling -----')
        if self.mcmc_run:
            print('      - Runing '+str(self.mcmc_nsteps)+' MCMC steps')
            res = sampler.run_mcmc(pos, self.mcmc_nsteps, progress=True, store=True)
        else:
            print('      - Not running, but restoring the existing sampler')

        #========== Set the best-fit model
        if set_best_fit:
            param_chains = sampler.chain[:, self.mcmc_burnin:, :]
            lnL_chains   = sampler.lnprobability[:, self.mcmc_burnin:]
            par_flat = param_chains.reshape(param_chains.shape[0]*param_chains.shape[1], param_chains.shape[2])
            lnL_flat = lnL_chains.reshape(lnL_chains.shape[0]*lnL_chains.shape[1])
            wbest = np.where(lnL_flat == np.amax(lnL_flat))[0][0]
            par_best = par_flat[wbest]
            
            idx_par = 0

            #---------- Fliuctuation parameters
            parkeys_fluct = list(fitpar_pk3d.keys())
            for ipar in range(len(parkeys_fluct)):
                parkey = parkeys_fluct[ipar]
                if fitpar_pk3d[parkey]['unit'] is not None:
                    unit = fitpar_pk3d[parkey]['unit']
                else:
                    unit = 1
                self.model.model_pressure_fluctuation[parkey] = par_best[idx_par] * unit
                idx_par += 1

        return par_list, sampler


    #==================================================
    # Extract Pk 3D from deprojection
    #==================================================
    
    def get_pk3d_deprojection(self):


        return

    
    #==================================================
    # Window function
    #==================================================
    
    def get_p3d_to_p2d_from_window_function(self):
        """
        This function compute the window function associated with a pressure 
        profile model. The output window function is a map in each pixel.
        See Romero et al. (2023) for details.
        
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
        compton3d = np.repeat(compton_sph[:,:,np.newaxis], pressure3d_sph.shape[2], axis=2)

        #----- Compute the window function in real space
        W3d = (cst.sigma_T / (cst.m_e * cst.c**2)* pressure3d_sph / compton3d).to_value('kpc-1')

        #----- Window function in Fourier space along kz
        W_ft = np.abs(np.fft.fft(W3d, axis=2))**2 * los_reso**2 # No dimension
        
        #----- Integrate along kz
        k_z = np.fft.fftfreq(Nz, los_reso)
        k_z_sort = np.fft.fftshift(k_z)
        W_ft_sort = np.fft.fftshift(W_ft, axes=2)
        
        N_theta = utils.trapz_loglog(W_ft_sort, k_z_sort, axis=2)*u.kpc**-1
    
        return N_theta





    
    #==================================================
    # Extract the 3D power spectrum
    #==================================================
    
    def extract_p3d(self, image, proj_reso,
                    kctr=None,
                    Nbin=100, kmin=None, kmax=None, scalebin='lin', kedges=None,
                    unbias=False,
                    mask=None,
                    statistic='mean',
                    method='Naive',
                    BeamDeconv=None, TFDeconv=None):
        """
        Extract the power spectrum of the map
        
        Parameters
        ----------
        - image (2d np.array): the image from which to extract the power spectrum
        - proj_reso (float): the resolution of the image (e.g. kpc, arcsec). The output k
        will be in a unit inverse to proj_reso
        - kmin/kmax (float): min and max k used to defined the k array. Same unit as 1/proj_reso
        - kscale (str): 'lin' or 'log'/ The scale use to define the k array.
        - kbins (1d np.array): the edge of the bins used to define the k array.
        - method (str): Method to extract the power spectrum. Possible options are
        'Naive', 'Arevalo12'
        - BeamDeconv (quantity): beam FWHM to deconvolve, either arcsec or kpc
        - TFDeconv (dict): transfer fucntion to deconvolve, dictionary containing 
          'k_arcsec' or 'k_kpc' and 'TF'

        Outputs
        ----------
        - k3d (1d np array): the array of k at the center of the bins
        - Pk3d (1d np array): the power spectrum in the bins

        """

        #----- Compute the conversion from Pk2d to Pk3d
        conv2d = self.get_p3d_to_p2d_from_window_function()
        if mask is None:
            conv = np.mean(conv2d)
        else:
            conv = np.sum(conv2d*mask) / np.sum(mask)
            
        #----- Compute the Pk2D        
        if method == 'Naive':
            
            if mask is not None:
                image = image*mask
                
            k2d, pk2d  = utils.get_pk2d(image, proj_reso,
                                        Nbin=Nbin, kmin=kmin, kmax=kmax, scalebin=scalebin, kedges=kedges,
                                        statistic=statistic)
            if mask is not None:
                pk2d *= mask.size/np.sum(mask) # mean fsky correction
            
        elif method == 'Arevalo12':
            k2d, pk2d  = utils.get_pk2d_arevalo(image, proj_reso,
                                                kctr=kctr,
                                                Nbin=Nbin, scalebin=scalebin, kmin=kmin, kmax=kmax, kedges=kedges,
                                                mask=mask,
                                                unbias=unbias)
            
        else:
            raise ValueError('method can only be Naive or Arevalo12')    

        #----- Beam deconvolution
        if BeamDeconv is not None:
            pk2d = deconv_beam_pk(k2d, pk2d, BeamDeconv)

        #----- TF deconvolution
        if TFDeconv is not None:
            pk2d = deconv_pk_transfer_function(k2d, pk2d, TFDeconv['k'], TFDeconv['TF'])

        #----- Compute the Pk3D
        k3d  = k2d*u.kpc**-1
        Pk3d = pk2d*u.kpc**2 / conv

        return k3d, Pk3d


    #==================================================
    # Generate a mock power spectrum
    #==================================================
    
    def get_mock_pk2d(self,
                      seed=None,
                      ConvbeamFWHM=0*u.arcsec,
                      ConvTF={'k_arcsec':np.linspace(0,1,1000)*u.arcsec, 'TF':np.linspace(0,1,1000)*0+1},
                      method_fluct='ratio',
                      mask=None,
                      method_pk='Naive',
                      kctr=None,
                      Nbin=100, kmin=None, kmax=None, scalebin='lin', kedges=None,
                      unbias=False,
                      statistic='mean',
                      DeconvBeamFWHM=None,
                      DeconvTF=None,

    ):
        """
        Generate a mock power spectrum given the model, and instrument response 
        function arguments
        
        Parameters
        ----------
        - seed (bool): set to a number for reproducible fluctuations

        Outputs
        ----------
        - k2d (1d np array): the array of k at the center of the bins
        - Pk2d (1d np array): the power spectrum in the bins

        """

        #----- Compute the model map
        compton_true_fluct = self.get_sz_map(seed=seed)
        compton_true_spher = self.get_sz_map(no_fluctuations=True)

        #----- Apply instrumental effects
        map_reso_arcsec = self.get_map_header()['CDELT2']*3600
        map_reso_kpc    = (map_reso_arcsec/3600*np.pi/180) * self._D_ang.to_value('kpc')

        # Apply IRF to map with fluctuation
        compton_mock_fluct = utils.apply_transfer_function(compton_true_fluct, map_reso_arcsec, 
                                                           ConvbeamFWHM.to_value('arcsec'), ConvTF, 
                                                           apps_TF_LS=True, apps_beam=True)
        # Apply IRF to map without fluctuation
        compton_mock_spher = utils.apply_transfer_function(compton_true_spher, map_reso_arcsec, 
                                                           ConvbeamFWHM.to_value('arcsec'), ConvTF, 
                                                           apps_TF_LS=True, apps_beam=True)
        
        # Apply beam only to map without fluctuation
        compton_mockB_spher = utils.apply_transfer_function(compton_true_spher, map_reso_arcsec, 
                                                            ConvbeamFWHM.to_value('arcsec'), None, 
                                                            apps_TF_LS=False, apps_beam=True)
        
        #----- Derive the fluctuation map
        if method_fluct is 'difference':
            fluctuation = compton_mock_fluct - compton_mock_spher

        elif method_fluct is 'ratio':
            fluctuation = (compton_mock_fluct - compton_mock_spher) / compton_mockB_spher

        else:
            raise ValueError('Only "difference" and "ratio" are possible methods')
        
        #----- Apply mask
        if mask is None:
            mask = fluctuation*0+1
        image = fluctuation * mask

        #----- Extract Pk
        if method_pk == 'Naive':
            k2d, pk2d  = utils.get_pk2d(image, map_reso_kpc,
                                        Nbin=Nbin, kmin=kmin, kmax=kmax, scalebin=scalebin, kedges=kedges,
                                        statistic=statistic)
            if mask is not None:
                pk2d *= mask.size/np.sum(mask) # mean fsky correction
            
        elif method_pk == 'Arevalo12':
            k2d, pk2d  = utils.get_pk2d_arevalo(image, map_reso_kpc,
                                                kctr=kctr,
                                                Nbin=Nbin, scalebin=scalebin, kmin=kmin, kmax=kmax, kedges=kedges,
                                                mask=mask,
                                                unbias=unbias)
            
        else:
            raise ValueError('method can only be Naive or Arevalo12')    

        #----- Beam deconvolution
        if DeconvBeamFWHM is not None:
            pk2d = deconv_beam_pk(k2d, pk2d, DeconvBeamFWHM)

        #----- TF deconvolution
        if DeconvTF is not None:
            pk2d = deconv_pk_transfer_function(k2d, pk2d, DeconvTF['k'], DeconvTF['TF'])

        return k2d*u.kpc**-1, pk2d*u.kpc**2

