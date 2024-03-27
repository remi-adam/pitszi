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
from scipy.ndimage import gaussian_filter
import scipy.stats as stats
from multiprocessing import Pool, cpu_count
import pprint
import copy
import emcee
import corner

from minot.ClusterTools.map_tools import radial_profile_sb

from pitszi import utils
from pitszi import utils_pk
from pitszi import utils_fitting
from pitszi import utils_plot


import time

#==================================================
# Likelihood: Profile parameter definition
#==================================================

def lnlike_defpar_profile(parinfo_profile,
                          model,
                          parinfo_center=None,
                          parinfo_ellipticity=None,
                          parinfo_ZL=None):
    """
    This function helps defining the parameter list, initial values and
    ranges for the profile fitting. 

    The order of the parameters is set here:
    i) first the profile parameters, 
    ii) then RA and/or Dec in this order if center is used, 
    iii) then min_to_maj_axis_ratio and/or angle in this order if ellipticity is used, 
    iv) and ZL if zero level is used.
        
    Parameters
    ----------
    - parinfo_profile (dict): see parinfo_profile in fit_profile_forward
    - model (class Model object): the model to be updated in the fit
    - parinfo_center (dict): see parinfo_center in fit_profile_forward
    - parinfo_ellipticity (dict): see parinfo_ellipticity in fit_profile_forward
    - parinfo_ZL (dict): see parinfo_ZL in fit_profile_forward

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
    Npar_profile = len(list(parinfo_profile.keys()))
    for ipar in range(Npar_profile):
        
        parkey = list(parinfo_profile.keys())[ipar]

        #----- Check that the param is ok wrt the model
        try:
            bid = model.model_pressure_profile[parkey]
        except:
            raise ValueError('The parameter '+parkey+' is not in self.model')    
        if 'guess' not in parinfo_profile[parkey]:
            raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
        if 'unit' not in parinfo_profile[parkey]:
            raise ValueError('The unit key is mandatory. Use "unit":None if unitless')

        #----- Update the name list
        par_list.append(parkey)

        #----- Update the guess values
        par0_value.append(parinfo_profile[parkey]['guess'][0])
        par0_err.append(parinfo_profile[parkey]['guess'][1])

        #----- Update the limit values
        if 'limit' in parinfo_profile[parkey]:
            par_min.append(parinfo_profile[parkey]['limit'][0])
            par_max.append(parinfo_profile[parkey]['limit'][1])
        else:
            par_min.append(-np.inf)
            par_max.append(+np.inf)
            
    #========== Deal with center
    if parinfo_center is not None:
        Npar_ctr = len(list(parinfo_center.keys()))
        list_ctr_allowed = ['RA', 'Dec']

        # Check that we have first RA, then Dec
        if Npar_ctr == 2:
            if list(parinfo_center.keys())[0] != 'RA':
                raise ValueError('The center parameters should be RA, Dec, with this order')
        
        for ipar in range(Npar_ctr):

            parkey = list(parinfo_center.keys())[ipar]

            #----- Check that the param is ok wrt the model
            if parkey not in list_ctr_allowed:
                raise ValueError('parinfo_center allowed param are RA, Dec only, in that order')
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
                
    #========== Deal ellipticity
    if parinfo_ellipticity is not None:
        Npar_ell = len(list(parinfo_ellipticity.keys()))
        list_ell_allowed = ['min_to_maj_axis_ratio', 'angle']

        # Check that we have first parinfo_ellipticity, then angle
        if Npar_ell != 2:
            raise ValueError('The ellipticity parameters should be min_to_maj_axis_ratio and angle, the two being mandatory')
        if list(parinfo_ellipticity.keys())[0] != 'min_to_maj_axis_ratio':
            raise ValueError('The ellipticity parameters should be min_to_maj_axis_ratio, angle, with this order')
            
        for ipar in range(Npar_ell):

            parkey = list(parinfo_ellipticity.keys())[ipar]

            #----- Check that the param is ok wrt the model
            if parkey not in list_ell_allowed:
                raise ValueError('parinfo_ellipticity allowed param are min_to_maj_axis_ratio, angle only')
            if 'guess' not in parinfo_ellipticity[parkey]:
                raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
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
                    par_min.append(-90)
                    par_max.append(+90)

    #========== Deal with offset
    if parinfo_ZL is not None:
        Npar_zl = len(list(parinfo_ZL.keys()))
        list_zl_allowed = ['ZL']
        
        for ipar in range(Npar_zl):

            parkey = list(parinfo_ZL.keys())[ipar]

            #----- Check that the param is ok wrt the model
            if parkey not in list_zl_allowed:
                raise ValueError('parinfo_ZL allowed param is ZL only')
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
# Likelihood: Fluctuation parameter definition
#==================================================
    
def lnlike_defpar_fluct(parinfo_fluct,
                        model,
                        parinfo_noise=None):
    """
    This function helps defining the parameter list, initial values and
    ranges for the fluctuation fitting.
        
    Parameters
    ----------
    - parinfo_fluct (dict): see parinfo_fluct in get_pk3d_model_forward_fitting
    - model (class Model object): the model to be updated in the fit
    - parinfo_noise (dictionary): same as parinfo_fluct with accepted parameter 'ampli'

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
    Npar_fluct = len(list(parinfo_fluct.keys()))
    for ipar in range(Npar_fluct):
        
        parkey = list(parinfo_fluct.keys())[ipar]

        #----- Check that the param is ok wrt the model
        try:
            bid = model.model_pressure_fluctuation[parkey]
        except:
            raise ValueError('The parameter '+parkey+' is not in self.model')    
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

    #========== Deal with noise normalization
    if parinfo_noise is not None:
        Npar_noise = len(list(parinfo_noise.keys()))
        list_noise_allowed = ['ampli']
        
        for ipar in range(Npar_noise):

            parkey = list(parinfo_noise.keys())[ipar]

            #----- Check that the param is ok wrt the model
            if parkey not in list_noise_allowed:
                raise ValueError('parinfo_noise allowed param is "ampli" only')
            if 'guess' not in parinfo_noise[parkey]:
                raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
            if 'unit' not in parinfo_noise[parkey]:
                raise ValueError('The unit key is mandatory. Use "unit":None if unitless')

            #----- Update the name list
            par_list.append(parkey)

            #----- Update the guess values
            par0_value.append(parinfo_noise[parkey]['guess'][0])
            par0_err.append(parinfo_noise[parkey]['guess'][1])

            #----- Update the limit values
            if 'limit' in parinfo_noise[parkey]:
                par_min.append(parinfo_noise[parkey]['limit'][0])
                par_max.append(parinfo_noise[parkey]['limit'][1])
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
# Likelihood: compute the prior for the profile
#==================================================

def lnlike_prior_profile(param,
                         parinfo_profile,
                         parinfo_center=None,
                         parinfo_ellipticity=None,
                         parinfo_ZL=None):
    """
    Compute the prior given the input parameter and some information
    about the parameters given by the user
        
    Parameters
    ----------
    - param (list): the parameter to apply to the model
    - parinfo_profile (dict): see fit_profile_forward
    - parinfo_center (dict): see fit_profile_forward
    - parinfo_ellipticity (dict): see fit_profile_forward
    - parinfo_ZL (dict): see fit_profile_forward

    Outputs
    ----------
    - prior (float): value of the prior

    """

    prior = 0
    idx_par = 0
    
    #---------- Profile parameters
    parkeys_prof = list(parinfo_profile.keys())

    for ipar in range(len(parkeys_prof)):
        parkey = parkeys_prof[ipar]
        
        # Flat prior
        if 'limit' in parinfo_profile[parkey]:
            if param[idx_par] < parinfo_profile[parkey]['limit'][0]:
                return -np.inf                
            if param[idx_par] > parinfo_profile[parkey]['limit'][1]:
                return -np.inf
            
        # Gaussian prior
        if 'prior' in parinfo_profile[parkey]:
            expected = parinfo_profile[parkey]['prior'][0]
            sigma = parinfo_profile[parkey]['prior'][1]
            prior += -0.5*(param[idx_par] - expected)**2 / sigma**2

        # Increase param index
        idx_par += 1
        
    #---------- Center parameters
    if parinfo_center is not None:
        parkeys_ctr = list(parinfo_center.keys())
        
        for ipar in range(len(parinfo_center)):
            parkey = parkeys_ctr[ipar]
            
            # Flat prior
            if 'limit' in parinfo_center[parkey]:
                if param[idx_par] < parinfo_center[parkey]['limit'][0]:
                    return -np.inf                
                if param[idx_par] > parinfo_center[parkey]['limit'][1]:
                    return -np.inf
                
            # Gaussian prior
            if 'prior' in parinfo_center[parkey]:
                expected = parinfo_center[parkey]['prior'][0]
                sigma = parinfo_center[parkey]['prior'][1]
                prior += -0.5*(param[idx_par] - expected)**2 / sigma**2

            # Increase param index
            idx_par += 1

    #---------- Ellipticity parameters
    if parinfo_ellipticity is not None:
        parkeys_ell = list(parinfo_ellipticity.keys())
        
        for ipar in range(len(parinfo_ellipticity)):
            parkey = parkeys_ell[ipar]
            
            # Flat prior
            if 'limit' in parinfo_ellipticity[parkey]:
                if param[idx_par] < parinfo_ellipticity[parkey]['limit'][0]:
                    return -np.inf                
                if param[idx_par] > parinfo_ellipticity[parkey]['limit'][1]:
                    return -np.inf
                
            # Gaussian prior
            if 'prior' in parinfo_ellipticity[parkey]:
                expected = parinfo_ellipticity[parkey]['prior'][0]
                sigma = parinfo_ellipticity[parkey]['prior'][1]
                prior += -0.5*(param[idx_par] - expected)**2 / sigma**2

            # Increase param index
            idx_par += 1
        
    #---------- Offset parameters
    if parinfo_ZL is not None:
        parkeys_zl = list(parinfo_ZL.keys())
        
        for ipar in range(len(parinfo_ZL)):
            parkey = parkeys_zl[ipar]
            
            # Flat prior
            if 'limit' in parinfo_ZL[parkey]:
                if param[idx_par] < parinfo_ZL[parkey]['limit'][0]:
                    return -np.inf                
                if param[idx_par] > parinfo_ZL[parkey]['limit'][1]:
                    return -np.inf
                
            # Gaussian prior
            if 'prior' in parinfo_ZL[parkey]:
                expected = parinfo_ZL[parkey]['prior'][0]
                sigma = parinfo_ZL[parkey]['prior'][1]
                prior += -0.5*(param[idx_par] - expected)**2 / sigma**2

            # Increase param index
            idx_par += 1

    #---------- Check on param numbers
    if idx_par != len(param):
        raise ValueError('Problem with the prior parameters')
        
    return prior


#==================================================
# Likelihood: compute the prior for the fluctuation
#==================================================

def lnlike_prior_fluct(param,
                       parinfo_fluct,
                       parinfo_noise=None):
    """
    Compute the prior given the input parameter and some information
    about the parameters given by the user
       
    Parameters
    ----------
    - param (list): the parameter to apply to the model
    - parinfo_fluct (dict): see get_pk3d_model_forward_fitting
    - parinfo_noise (dict): see get_pk3d_model_forward_fitting

    Outputs
    ----------
    - prior (float): value of the prior

    """

    prior = 0
    idx_par = 0

    #---------- Fluctuation parameters
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

    #---------- Noise parameters
    if parinfo_noise is not None:
        parkeys_noise = list(parinfo_noise.keys())
        
        for ipar in range(len(parinfo_noise)):
            parkey = parkeys_noise[ipar]
            
            # Flat prior
            if 'limit' in parinfo_noise[parkey]:
                if param[idx_par] < parinfo_noise[parkey]['limit'][0]:
                    return -np.inf                
                if param[idx_par] > parinfo_noise[parkey]['limit'][1]:
                    return -np.inf
                
            # Gaussian prior
            if 'prior' in parinfo_noise[parkey]:
                expected = parinfo_noise[parkey]['prior'][0]
                sigma = parinfo_noise[parkey]['prior'][1]
                prior += -0.5*(param[idx_par] - expected)**2 / sigma**2

            # Increase param index
            idx_par += 1

    #---------- Check on param numbers
    if idx_par != len(param):
        raise ValueError('Problem with the prior parameters')

    return prior


#==================================================
# Likelihood: set model with profile parameter value
#==================================================

def lnlike_setpar_profile(param,
                          model,
                          parinfo_profile,
                          parinfo_center=None,
                          parinfo_ellipticity=None,
                          parinfo_ZL=None):
    """
    Set the model to the given profile parameters.
        
    Parameters
    ----------
    - param (list): the parameter to apply to the model
    - model (pitszi object): pitszi object from class Model
    - parinfo_profile (dict): see fit_profile_forward
    - parinfo_center (dict): see fit_profile_forward
    - parinfo_ellipticity (dict): see fit_profile_forward
    - parinfo_ZL (dict): see fit_profile_forward

    Outputs
    ----------
    - model (float): the model
    - ZL (float): nuisance parameter for the zero level

    """
    
    idx_par = 0

    #---------- Profile parameters
    parkeys_prof = list(parinfo_profile.keys())
    for ipar in range(len(parkeys_prof)):
        parkey = parkeys_prof[ipar]
        if parinfo_profile[parkey]['unit'] is not None:
            unit = parinfo_profile[parkey]['unit']
        else:
            unit = 1
        model.model_pressure_profile[parkey] = param[idx_par] * unit
        idx_par += 1

    #---------- Center parameters
    if parinfo_center is not None:
        if 'RA' in parinfo_center:
            RA = param[idx_par]
            unit1 = parinfo_center['RA']['unit']
            idx_par += 1
        else:
            RA = self.model.coord.icrs.ra.to_value('deg')
            unit1 = u.deg
        if 'Dec' in parinfo_center:
            Dec = param[idx_par]
            unit2 = parinfo_center['Dec']['unit']
            idx_par += 1
        else:
            Dec = self.model.coord.icrs.dec.to_value('deg')
            unit2 = u.deg

        model.coord = SkyCoord(RA*unit1, Dec*unit2, frame='icrs')
        
    #---------- Ellipticity parameters
    if parinfo_ellipticity is not None:
        axis_ratio = param[idx_par]
        idx_par += 1
        angle = param[idx_par]
        angle_unit = parinfo_ellipticity['angle']['unit']
        idx_par += 1
        model.triaxiality = {'min_to_maj_axis_ratio':axis_ratio,
                             'int_to_maj_axis_ratio':axis_ratio,
                             'euler_angle1':0*u.deg,
                             'euler_angle2':90*u.deg,
                             'euler_angle3':angle*angle_unit}

    #---------- Zero level parameters
    if parinfo_ZL is not None:
        ZL = param[idx_par]
        idx_par += 1
    else:
        ZL = 0

    #---------- Final check
    if len(param) != idx_par:
        print('issue with the number of parameters')
        import pdb
        pdb.set_trace()

    return model, ZL


#==================================================
# Likelihood: set model with fluctuation parameter value
#==================================================
    
def lnlike_setpar_fluct(param,
                        model,
                        parinfo_fluct,
                        parinfo_noise=None):
    """
    Set the model to the given fluctuation parameters
        
    Parameters
    ----------
    - param (list): the parameters to apply to the model
    - model (pitszi object): pitszi object from class Model
    - parinfo_fluct (dict): see get_pk3d_model_forward_fitting
    - parinfo_noise (dict): see get_pk3d_model_forward_fitting

    Outputs
    ----------
    - model (float): the model
    - noise_ampli (float): the nuisance parameter, noise amplitude

    """
    
    idx_par = 0

    #---------- Fluctuation parameters
    parkeys_fluct = list(parinfo_fluct.keys())
    for ipar in range(len(parkeys_fluct)):
        parkey = parkeys_fluct[ipar]
        if parinfo_fluct[parkey]['unit'] is not None:
            unit = parinfo_fluct[parkey]['unit']
        else:
            unit = 1
        model.model_pressure_fluctuation[parkey] = param[idx_par] * unit
        idx_par += 1


    #---------- Noise parameters
    if parinfo_noise is not None:
        noise_ampli = param[idx_par]
        idx_par += 1
    else:
        noise_ampli = 1

    #---------- Final check
    if len(param) != idx_par:
        print('issue with the number of parameters')
        import pdb
        pdb.set_trace()

    return model, noise_ampli


#==================================================
# lnL function for the profile file
#==================================================
# Must be global and here to avoid pickling errors
global lnlike_profile_forward
def lnlike_profile_forward(param,
                           parinfo_profile,
                           model,
                           data_img, data_noiseprop, data_mask,
                           data_psf, data_tf,
                           parinfo_center=None, parinfo_ellipticity=None, parinfo_ZL=None,
                           use_covmat=False):
    """
    This is the likelihood function used for the fit of the profile.
        
    Parameters
    ----------
    - param (np array): the value of the test parameters
    - parinfo_profile (dict): see parinfo_profile in fit_profile_forward
    - model (class Model object): the model to be updated
    - data_img (nd array): the image data
    - data_noiseprop (nd_array): the inverse covariance matrix to be used for the fit or the rms,
    depending on use_covmat.
    - data_mask (nd array): the image mask
    - data_psf (quantity): input to model_mock.get_sz_map
    - data_tf (dict): input to model_mock.get_sz_map
    - parinfo_center (dict): see parinfo_center in fit_profile_forward
    - parinfo_ellipticity (dict): see parinfo_ellipticity in fit_profile_forward
    - parinfo_ZL (dict): see parinfo_ZL in fit_profile_forward
    - use_covmat (bool): if True, assumes that data_noiseprop is the inverse noise covariance matrix

    Outputs
    ----------
    - lnL (float): the value of the log likelihood

    """
    
    #========== Deal with flat and Gaussian priors
    prior = lnlike_prior_profile(param, parinfo_profile,parinfo_center=parinfo_center,
                                 parinfo_ellipticity=parinfo_ellipticity,parinfo_ZL=parinfo_ZL)
    if not np.isfinite(prior): return -np.inf
    
    #========== Change model parameters
    model, zero_level = lnlike_setpar_profile(param, model, parinfo_profile,
                                              parinfo_center=parinfo_center,
                                              parinfo_ellipticity=parinfo_ellipticity,
                                              parinfo_ZL=parinfo_ZL)
    
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


#==================================================
# lnL function for the profile file
#==================================================
# Must be global and here to avoid pickling errors
global lnlike_fluct_brute
def lnlike_fluct_brute(param,
                       parinfo_fluct,
                       model,
                       model_ymap_sph1,
                       model_ymap_sph2,
                       w8map,
                       model_pk2d_covmat_ref,
                       model_pk2d_ref,
                       data_pk2d,
                       noise_pk2d_covmat,
                       noise_pk2d_mean,
                       reso_arcsec,
                       kedges_arcsec,
                       psf_fwhm,
                       transfer_function,
                       parinfo_noise=None,
                       use_covmat=False,
                       scale_model_variance=False):
    """
    This is the likelihood function used for the brute force forward 
    fit of the fluctuation spectrum.
        
    Parameters
    ----------
    - param (np array): the value of the test parameters
    - parinfo_fluct (dict): see parinfo_fluct in get_pk3d_model_forward_fitting
    - w8map(2d np array): the weight map that multiply the image before Pk extraction
    - model (class Model object): the model to be updated
    - model_ymap_sph (2d array): best-fit smooth model
    - model_ymap_sph_deconv (2d array): best-fit smooth model deconvolved from TF
    - model_pk2d_covmat_ref (2d array): model covariance matrix
    - data_pk2d (1d array): the Pk extracted from the data
    - noise_pk2d_covmat (2d array): the noise covariance matrix
    - noise_pk2d_mean (1d array): the mean noise bias expected
    - reso_arcsec (float): the resolution in arcsec
    - kedges_arcsec (1d array): the edges of the k bins in arcsec-1
    - psf_fwhm (float): the psf FWHM
    - transfer_function (dict): the transfer function
    - parinfo_noise (dict): see parinfo_noise in get_pk3d_model_forward_fitting
    - use_covmat (bool): set to true to use the covariance matrix
    - scale_model_variance (bool): set to true to rescale the model variance according to the given model

    Outputs
    ----------
    - lnL (float): the value of the log likelihood

    """

    #========== Deal with flat and Gaussian priors
    prior = lnlike_prior_fluct(param, parinfo_fluct,parinfo_noise=parinfo_noise)
    if not np.isfinite(prior): return -np.inf
    
    #========== Change model parameters
    model, noise_ampli = lnlike_setpar_fluct(param, model, parinfo_fluct,
                                             parinfo_noise=parinfo_noise)
    
    #========== Get the model
    #---------- Compute the ymap and fluctuation image
    test_ymap = model.get_sz_map(seed=None, no_fluctuations=False,
                                 irfs_convolution_beam=psf_fwhm, irfs_convolution_TF=transfer_function)
    delta_y = test_ymap - model_ymap_sph1
    test_image = (delta_y - np.mean(delta_y))/model_ymap_sph2 * w8map

    #---------- Compute the test Pk
    test_k, test_pk =  utils_pk.extract_pk2d(test_image, reso_arcsec, kedges=kedges_arcsec)
    model_pk2d = test_pk + noise_ampli*noise_pk2d_mean

    #========== Compute the likelihood
    if use_covmat: # Using the covariance matrix
        if scale_model_variance: # Here we assume that the correlation matrix does not depend on the model
            correlation_ref = utils.correlation_from_covariance(model_pk2d_covmat_ref)
            std_scaled = np.diag(model_pk2d_covmat_ref)**0.5 * (test_pk/model_pk2d_ref)
            covariance_scaled = utils.covariance_from_correlation(correlation_ref, std_scaled)
            covariance_inverse = np.linalg.inv(noise_pk2d_covmat + covariance_scaled)
        else:
            covariance_inverse = np.linalg.inv(noise_pk2d_covmat + model_pk2d_covmat_ref)
            
        residual_test = model_pk2d - data_pk2d
        lnL = -0.5*np.matmul(residual_test, np.matmul(covariance_inverse, residual_test))
        
    else: # Using the rms only
        if scale_model_variance:
            variance = np.diag(noise_pk2d_covmat) + np.diag(model_pk2d_covmat_ref)*(test_pk/model_pk2d_ref)**2
        else:
            variance = np.diag(noise_pk2d_covmat) + np.diag(model_pk2d_covmat_ref)
        lnL = -0.5*np.nansum((model_pk2d - data_pk2d)**2 / variance)
        
    lnL += prior

    #========== Check and return
    if np.isnan(lnL):
        return -np.inf
    else:
        return lnL
    

#==================================================
# Compute the Pk2d observable for forward deprojection
#==================================================

def lnlike_model_fluct_deproj_forward(model,
                                      k2d,
                                      conv_2d3d,
                                      Kmnmn,
                                      beam_FWHM,
                                      TF,
                                      kedges):
    """
    This function compute the Pk2d model to be compared with the 
    data in the case of forward deprojection
        
    Parameters
    ----------
    - model (pitszi Model class): the model object
    - k2d (2d array): the k2d norm grid associated with underlying FFT sampling
    - conv_2d3d (float): the Pk3d to Pk2d conversion in kpc-1
    - Kmnmn (complex array): the mode mixing matrix
    - beam_FWHM (quantity): the beam FWHM homogeneous to arcsec
    - TF (dict): the transfer function with k (homogenous to arcsec-1) and TF keys 
    - kedges (1d array): the bin of the k array

    Outputs
    ----------
    - pk2d_K_bin (np array): 

    """

    #---------- Useful variable
    Nx, Ny = k2d.shape
    k2d_flat = k2d.flatten()
    
    #---------- Extract P3d on the k2d grid avoiding k=0
    idx_sort = np.argsort(k2d_flat)
    revidx   = np.argsort(idx_sort)
    k3d_test = np.sort(k2d_flat)
    wok = (k3d_test > 0)
    k3d_test_clean = k3d_test[wok]
    _, pk3d_test_clean = model.get_pressure_fluctuation_spectrum(k3d_test_clean*u.kpc**-1)
    pk3d_test      = np.zeros(len(k3d_test))
    pk3d_test[wok] = pk3d_test_clean.to_value('kpc3')
    pk3d_test      = pk3d_test[revidx] # same shape as k2d_flat
 
    #---------- Convert Pk3d to Pk2d
    pk2d_flat = pk3d_test * conv_2d3d            # Unit == kpc2, 2d grid

    #---------- Apply beam and transfer function
    pk2d_flat = utils_pk.apply_pk_beam(k2d_flat, pk2d_flat, 
                                       beam_FWHM.to_value('rad')*model.D_ang.to_value('kpc'))
    pk2d_flat = utils_pk.apply_pk_transfer_function(k2d_flat, pk2d_flat, 
                                                    TF['k'].to_value('rad-1')/model.D_ang.to_value('kpc'),
                                                    TF['TF'])
    
    #---------- Apply Kmnmn
    pk2d_K = np.abs(utils_pk.multiply_Kmnmn(np.abs(Kmnmn)**2, pk2d_flat.reshape(Nx, Ny))) / Nx / Ny

    #---------- Bin
    pk2d_K_bin, _, _ = stats.binned_statistic(k2d_flat, pk2d_K.flatten(), 
                                              statistic="mean", bins=kedges)
    
    return pk2d_K_bin

    
#==================================================
# lnL function for the profile file
#==================================================
# Must be global and here to avoid pickling errors
global lnlike_fluct_deproj_forward
def lnlike_fluct_deproj_forward(param,
                                parinfo_fluct,
                                model,
                                conv_2d3d,
                                Kmnmn,
                                k2d_norm,
                                data_pk2d,
                                noise_pk2d_info,
                                noise_pk2d_mean,
                                kedges_kpc,
                                psf_fwhm,
                                transfer_function,
                                parinfo_noise=None,
                                use_covmat=False):
    """
    This is the likelihood function used for the deprojection forward 
    fit of the fluctuation spectrum.
    
    Parameters
    ----------
    - param (np array): the value of the test parameters
    - parinfo_fluct (dict): see parinfo_fluct in get_pk3d_model_forward_fitting
    - model (class Model object): the model to be updated
    - conv_2d3d (float): the Pk3d to Pk2d conversion in kpc-1
    - Kmnmn (complex array): the mode mixing matrix
    - k2d_norm (2d array): the k2d norm grid associated with underlying FFT sampling
    - data_pk2d (1d array): the Pk extracted from the data
    - noise_pk2d_info (2d array): the noise information. Either noise rms, or noise 
    inverse covariance matrix depending if use_covmat is set to True
    - noise_pk2d_mean (1d array): the mean noise bias expected
    - kedges_kpc (1d array): the edges of the k bins in kpc-1
    - psf_fwhm (quantity): the psf FWHM
    - transfer_function (dict): the transfer function
    - parinfo_noise (dict): see parinfo_noise in get_pk3d_model_forward_fitting
    - use_covmat (bool): set to true to use the covariance matrix

    Outputs
    ----------
    - lnL (float): the value of the log likelihood

    """
    
    #========== Deal with flat and Gaussian priors
    prior = lnlike_prior_fluct(param, parinfo_fluct,parinfo_noise=parinfo_noise)
    if not np.isfinite(prior): return -np.inf
    
    #========== Change model parameters
    model, noise_ampli = lnlike_setpar_fluct(param, model, parinfo_fluct,
                                             parinfo_noise=parinfo_noise)
    
    #========== Get the model
    #---------- Pressure fluctuation power spectrum
    test_pk = lnlike_model_fluct_deproj_forward(model, k2d_norm, conv_2d3d, Kmnmn,
                                                psf_fwhm, transfer_function, kedges_kpc)

    #---------- Pk model plus noise
    model_pk2d = test_pk + noise_ampli*noise_pk2d_mean
    
    #========== Compute the likelihood
    if use_covmat: # Using the covariance matrix            
        residual_test = model_pk2d - data_pk2d
        lnL = -0.5*np.matmul(residual_test, np.matmul(noise_pk2d_info, residual_test))
        
    else: # Using the rms only
        lnL = -0.5*np.nansum((model_pk2d - data_pk2d)**2 / noise_pk2d_info)
        
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
                 #
                 silent=False,
                 output_dir='./pitszi_output',
                 #
                 method_noise_covmat='model',
                 method_data_deconv=False,
                 method_w8=None,
                 #
                 kbin_min=0*u.arcsec**-1,
                 kbin_max=0.1*u.arcsec**-1,
                 kbin_Nbin=20,
                 kbin_scale='lin',
                 #
                 mcmc_nwalkers=100,
                 mcmc_nsteps=500,
                 mcmc_burnin=100,
                 mcmc_reset=False,
                 mcmc_run=True,
                 mcmc_Nresamp=100):
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
        - silent (bool): set to False for printing information
        - output_dir (str): directory where outputs are saved
        - method_noise_covmat (str): method to compute noise MC realization 
        (implemented: 'model' or 'covariance')
        - kbin_min (quantity): minimum k value for the 2D power spectrum (homogeneous to 1/angle or 1/kpc)
        - kbin_max (quantity): maximal k value for the 2D power spectrum (homogeneous to 1/angle or 1/kpc)
        - kbin_Nbin (int): number of bins in the 2d power spectrum
        - kbin_scale (str): bin spacing ('lin' or 'log')
        - mcmc_nwalkers (int): number of MCMC walkers
        - mcmc_nsteps (int): number of MCMC steps to run
        - mcmc_burnin (int): the burnin, number of point to remove from the chains
        - mcmc_reset (bool): True for restarting the MCMC even if a sampler exists
        - mcmc_run (bool): True to run the MCMC
        - mcmc_Nresamp (int): the number of Monte Carlo to resample the chains for results

        """

        # Admin
        self.silent     = silent
        self.output_dir = output_dir

        # Input data and model (deepcopy to avoid modifying the input when fitting)
        self.data  = copy.deepcopy(data)
        self.model = copy.deepcopy(model)

        # Analysis methodology
        self.method_noise_covmat = method_noise_covmat   # ['covariance', 'model']
        if method_w8 == None:
            self.method_w8 = np.ones(data.image.shape)   # The weight applied to image = dy/y x w8
        else:
            self.method_w8 = method_w8
        self.method_data_deconv  = method_data_deconv    # Deconvolve the data from TF and beam prior Pk

        # Binning in k
        self.kbin_min   = kbin_min
        self.kbin_max   = kbin_max
        self.kbin_Nbin  = kbin_Nbin
        self.kbin_scale = kbin_scale # ['lin', 'log']

        # MCMC parameters
        self.mcmc_nwalkers = mcmc_nwalkers
        self.mcmc_nsteps   = mcmc_nsteps
        self.mcmc_burnin   = mcmc_burnin
        self.mcmc_reset    = mcmc_reset
        self.mcmc_run      = mcmc_run
        self.mcmc_Nresamp  = mcmc_Nresamp
        

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
    # Set the weight to inverse model
    #==================================================
    
    def set_method_w8(self,
                      apply_mask=True,
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
        apply_mask (bool): application of the data mask in the weight
        apply_radial_model (bool): multiply the wight with the radial model
        conv_radial_model_beam (bool): apply beam smoothing to the radial model
        conv_radial_model_TF (bool): apply the transfer function to the radial model
        - remove_GNFW_core (bool): if True, and if the model is a GNFW, it will 
        set the core parameter from the model to zero
        - smooth_FWHM (quantity): smoothing of weight map, homogeneous to arcsec

        Outputs
        ----------
        None, the weight are set to the requested map

        """
        
        w8map = np.ones(self.data.image.shape)

        #===== Apply mask
        if apply_mask:
            w8map *= self.data.mask 

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
            w8map = gaussian_filter(w8map,
                                    sigma=smooth_FWHM.to_value('deg')/sigma2fwhm/self.model.get_map_reso().to_value('deg'))
        
        self.method_w8 = w8map

        
    #==================================================
    # Define k edges from bining properties
    #==================================================
    
    def get_kedges(self, physical=False):
        """
        This function compute the edges of the bins in k space.
        
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
    
    def get_kbin_counts(self, kedges=None):
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

        # Get the k edges if not provided
        if kedges is None:
            kedges = self.get_kedges()

        # Define the grid size
        Nx, Ny = self.data.image.shape

        # Get the resolution in the right unit (physical or angle)
        try:
            kedges = kedges.to_value('arcsec-1')
            reso = self.model.get_map_reso(physical=False).to_value('arcsec')
        except:
            kedges = kedges.to_value('kpc-1')
            reso = self.model.get_map_reso(physical=True).to_value('kpc')

        # Grid sampling in k space
        k_x = np.fft.fftfreq(Nx, reso)
        k_y = np.fft.fftfreq(Ny, reso)
        k2d_x, k2d_y = np.meshgrid(k_x, k_y, indexing='ij')
        k2d_norm = np.sqrt(k2d_x**2 + k2d_y**2)

        # Bining and counting
        kbin_counts, _, _ = stats.binned_statistic(k2d_norm.flatten(), k2d_norm.flatten(),
                                                   statistic='count', bins=kedges)

        # Information
        if not self.silent:
            print('----- Minimal bin counts:', np.amin(kbin_counts))
            print('----- Counts in each k bin:', kbin_counts)

        return kbin_counts

    
    #==================================================
    # Return the number of counts per bin
    #==================================================
    
    def get_k_grid(self, physical=False):
        """
        Give the grid in kspace
        
        Parameters
        ----------
        - physical (bool): set to true to have 1/kpc units. Otherwise 1/arcsec.

        Outputs
        ----------
        - kx, ky (2d array): the k grid along x and y
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
    # Window function
    #==================================================
    
    def get_p2d_from_p3d_from_window_function_exact(self):
        """
        This function computes P2d from P3d by integrating the window function and Pk3d.
        Warning, for large grids, this may take a lot of time! Use it wisely.
        
        Parameters
        ----------

        Outputs
        ----------
        - k2d_norm (2d np array): the values of projected k
        - Pk2d (2d np array): the 2D power spectrum for each k2d_norm

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

        #----- Defines k   
        k_x = np.fft.fftfreq(Nx, proj_reso)
        k_y = np.fft.fftfreq(Ny, proj_reso)
        k_z = np.fft.fftfreq(Nz, los_reso)
        k3d_x, k3d_y, k3d_z = np.meshgrid(k_x, k_y, k_z, indexing='ij')
        k3d_norm = np.sqrt(k3d_x**2 + k3d_y**2 + k3d_z**2)
        k2d_x, k2d_y = np.meshgrid(k_x, k_y, indexing='ij')
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
        P3d = P3d_flat.reshape(Nx,Ny,Nz)       # Reshape to k cube
        
        #----- Sort along kz       
        k_z_sort = np.fft.fftshift(k_z)
        W_ft_sort = np.fft.fftshift(W_ft, axes=2)
        P3d_kzsort = np.fft.fftshift(P3d, axes=2)

        #----- integrate along kz
        Pk2d = np.zeros((Nx, Ny)) # Shape kx, ky
        # Make a loop to avoid memory issue (otherwise 5d array kx, ky, x, y, k_z integrated along kz)
        #integrand = P3d_kzsort[:,:,np.newaxis,np.newaxis,:]*W_ft_sort[np.newaxis, np.newaxis, :,:,:] # kx, ky, x, y, k_z
        #Pk2d_xy = utils.trapz_loglog(integrand, k_z_sort, axis=4) #kx, ky, x, y
        #mask_kxky = self.data.mask[np.newaxis, np.newaxis, :,:]
        #Pk2d_exact = np.sum(Pk2d_xy * mask_kxky, axis=(2,3)) / np.sum(mask_kxky, axis=(2,3)) #kpc2
        for ik in range(Nx):
            for jk in range(Ny):
                integrand = P3d_kzsort[ik,jk,np.newaxis,np.newaxis,:]*W_ft_sort
                Pk2d_ikjk_xy = utils.trapz_loglog(integrand, k_z_sort, axis=2)
                Pk2d[ik,jk] = np.sum(Pk2d_ikjk_xy * self.data.mask) / np.sum(self.data.mask)

        return k2d_norm*u.kpc**-1, Pk2d*u.kpc**2


    #==================================================
    # Compute noise statistics
    #==================================================
    
    def get_pk2d_noise_statistics(self,
                                  kbin_edges=None,
                                  Nmc=1000,
                                  physical=False):
        """
        This function compute the noise properties associated
        with the data noise
        
        Parameters
        ----------
        - kbin_edges (1d array, quantity): array of k edges that can be used instead of default one
        - Nmc (int): number of monte carlo realization
        - physical (bool): set to true to have output in kpc units, else arcsec units

        Outputs
        ----------
        - k2d (1d array): the values of k in each bin
        - noise_pk2d_ref (1d array): the noise mean
        - noise_pk2d_covmat (2d array): the noise covariance matrix

        """
        
        #----- Info to user
        if not self.silent:
            print('----- Computing Pk2d noise covariance -----')

        #----- Sanity check
        bin_counts = self.get_kbin_counts(kedges=kbin_edges)
        if np.amin(bin_counts) == 0:
            raise ValueError('Some bins have zero counts. This will cause issues. Please redefine the binning to avoid this')
        
        #----- Useful info
        if kbin_edges is None:
            kedges = self.get_kedges().to_value('arcsec-1')
        else:
            try:
                kedges = kbin_edges.to_value('arcsec-1')
            except:
                try:
                    kedges = (kbin_edges.to_value('kpc-1')*self.model.D_ang.to_value('kpc')*u.rad**-1).to_value('arcsec-1')
                except:
                    raise ValueError('kbin_edges should be given homogeneous to kpc-1 or arcsec-1')
            
        reso = self.model.get_map_reso().to_value('arcsec')

        # Model accounting/or not for beam and TF
        if self.method_data_deconv:
            model_ymap_sph2 = self.model.get_sz_map(no_fluctuations=True)
        else:
            model_ymap_sph2 = self.model.get_sz_map(no_fluctuations=True,
                                                    irfs_convolution_beam=self.data.psf_fwhm)

        #----- Extract noise MC realization
        if self.method_noise_covmat == 'model':
            noise_ymap_mc = self.data.get_noise_monte_carlo_from_model(center=None, seed=None, Nmc=Nmc)
            if noise_ymap_mc is None:
                raise ValueError("Noise MC could not be computed using method_noise_covmat='model'")

        elif self.method_noise_covmat == 'covariance':
            noise_ymap_mc = self.data.get_noise_monte_carlo_from_covariance(seed=None, Nmc=Nmc)
            if noise_ymap_mc is None:
                raise ValueError("Noise MC could not be computed using method_noise_covmat='covariance'")
        else:
            raise ValueError("Noise MC can only be computed using method_noise_covmat='model'/'covariance'")
                        
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
            
        noise_pk2d_mean = np.mean(noise_pk2d_mc, axis=0)
        noise_pk2d_rms  = np.std(noise_pk2d_mc, axis=0)

        #----- Compute covariance
        noise_pk2d_covmat = np.zeros((len(kedges)-1, len(kedges)-1))
        for imc in range(Nmc):
            noise_pk2d_covmat += np.matmul((noise_pk2d_mc[imc,:]-noise_pk2d_mean)[:,None],
                                           (noise_pk2d_mc[imc,:]-noise_pk2d_mean)[None,:])
        noise_pk2d_covmat /= Nmc

        #----- Sanity check
        if np.sum(np.isnan(noise_pk2d_covmat)) > 0:
            if not self.silent:
                print('Some pixels in the covariance matrix are NaN.')
            raise ValueError('Issue with noise covariance matrix')

        #----- Units
        if physical:
            kpc2arcsec = ((1*u.kpc/self.model.D_ang).to_value('')*u.rad).to_value('arcsec')
            k2d /= kpc2arcsec**-1
            noise_pk2d_mean /= kpc2arcsec**2
            noise_pk2d_covmat /= kpc2arcsec**4
            return k2d*u.kpc**-1, noise_pk2d_mean*u.kpc*2, noise_pk2d_covmat*u.kpc**4
        else:
            return k2d*u.arcsec**-1, noise_pk2d_mean*u.arcsec**2, noise_pk2d_covmat*u.arcsec**4


    #==================================================
    # Compute model variance statistics
    #==================================================
    
    def get_pk2d_modelvar_statistics(self,
                                     kbin_edges=None,
                                     Nmc=1000,
                                     physical=False):
        """
        This function compute the model variance properties associated
        with the reference input model.
        
        Parameters
        ----------
        - kbin_edges (1d array quantity): array of k edges that can be used instead of default one
        - Nmc (int): number of monte carlo realization
        - physical (bool): set to true to have output in kpc units, else arcsec units

        Outputs
        ----------
        - k2d (1d array): the values of k in each bin
        - model_pk2d_ref (1d array): the model mean
        - model_pk2d_covmat (2d array): the model covariance matrix

        """

        #----- Info to user
        if not self.silent:
            print('----- Computing Pk2d model covariance -----')

        #----- Sanity check
        bin_counts = self.get_kbin_counts(kedges=kbin_edges)
        if np.amin(bin_counts) == 0:
            raise ValueError('Some bins have zero counts. This will cause issues. Please redefine the binning to avoid this')
        
        #----- Useful info
        if kbin_edges is None:
            kedges = self.get_kedges().to_value('arcsec-1')
        else:
            try:
                kedges = kbin_edges.to_value('arcsec-1')
            except:
                try:
                    kedges = (kbin_edges.to_value('kpc-1')*self.model.D_ang.to_value('kpc')*u.rad**-1).to_value('arcsec-1')
                except:
                    raise ValueError('kbin_edges should be given homogeneous to kpc-1 or arcsec-1')
            
        reso = self.model.get_map_reso().to_value('arcsec')

        # Model accounting/or not for beam and TF
        if self.method_data_deconv:
            model_ymap_sph1 = self.model.get_sz_map(no_fluctuations=True)
            model_ymap_sph2 = model_ymap_sph1
        else:
            model_ymap_sph1 = self.model.get_sz_map(no_fluctuations=True,
                                                    irfs_convolution_beam=self.data.psf_fwhm,
                                                    irfs_convolution_TF=self.data.transfer_function)
            model_ymap_sph2 = self.model.get_sz_map(no_fluctuations=True,
                                                    irfs_convolution_beam=self.data.psf_fwhm)
            
        #----- Compute Pk2d MC realization
        model_pk2d_mc     = np.zeros((Nmc, len(kedges)-1))
        model_pk2d_covmat = np.zeros((len(kedges)-1, len(kedges)-1))
        
        for imc in range(Nmc):

            # Account or not for TF and beam
            if self.method_data_deconv:
                test_ymap = self.model.get_sz_map(seed=None, no_fluctuations=False)
            else:
                test_ymap = self.model.get_sz_map(seed=None, no_fluctuations=False,
                                                  irfs_convolution_beam=self.data.psf_fwhm,
                                                  irfs_convolution_TF=self.data.transfer_function)

            # Test image
            delta_y = test_ymap - model_ymap_sph1
            test_image = (delta_y - np.mean(delta_y))/model_ymap_sph2 * self.method_w8

            # Pk
            k2d, pk_mc =  utils_pk.extract_pk2d(test_image, reso, kedges=kedges)
            model_pk2d_mc[imc,:] = pk_mc
            
        model_pk2d_mean = np.mean(model_pk2d_mc, axis=0)
        model_pk2d_rms = np.std(model_pk2d_mc, axis=0)

        #----- Compute covariance
        for imc in range(Nmc):
            model_pk2d_covmat += np.matmul((model_pk2d_mc[imc,:]-model_pk2d_mean)[:,None],
                                           (model_pk2d_mc[imc,:]-model_pk2d_mean)[None,:])
        model_pk2d_covmat /= Nmc

        #----- Sanity check
        if np.sum(np.isnan(model_pk2d_covmat)) > 0:
            if not self.silent:
                print('Some pixels in the covariance matrix are NaN.')
                print('This can be that the number of bins is such that some bins are empty.')
            raise ValueError('Issue with noise covariance matrix')

        #----- Units
        if physical:
            kpc2arcsec = ((1*u.kpc/self.model.D_ang).to_value('')*u.rad).to_value('arcsec')
            k2d /= kpc2arcsec**-1
            model_pk2d_mean /= kpc2arcsec**2
            model_pk2d_covmat /= kpc2arcsec**4
            return k2d*u.kpc**-1, model_pk2d_mean*u.kpc*2, model_pk2d_covmat*u.kpc**4
        else:
            return k2d*u.arcsec**-1, model_pk2d_mean*u.arcsec**2, model_pk2d_covmat*u.arcsec**4


    #==================================================
    # Compute the Pk2d from the data
    #==================================================
    
    def get_pk2d_data(self,
                      kbin_edges=None,
                      output_ymodel=False,
                      physical=False):
        """
        This function compute the data pk2d that is used for fitting
        
        Parameters
        ----------
        - kbin_edges (1d array quantity): array of k edges that can be used instead of default one
        - output_ymodel (bool): if true, ymap used for model are also output
        - physical (bool): set to true to have output in kpc units, else arcsec units

        Outputs
        ----------
        - k2d (1d array): the values of k in each bin
        - pk2d (1d array): the data Pk2d
        - model_ymap_sph (2d array): the ymap radial model convolved with IRF
        - model_ymap_sph_deconv (2d array): the ymap radial model deconvolve from TF

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
        delta_y = img_y - model_ymap_sph1
        data_image = (delta_y - np.mean(delta_y))/model_ymap_sph2 * self.method_w8
        
        #---------- Pk for the data
        if kbin_edges is None:
            kedges = self.get_kedges().to_value('arcsec-1')
        else:
            try:
                kedges = kbin_edges.to_value('arcsec-1')
            except:
                try:
                    kedges = (kbin_edges.to_value('kpc-1')*self.model.D_ang.to_value('kpc')*u.rad**-1).to_value('arcsec-1')
                except:
                    raise ValueError('kbin_edges should be given homogeneous to kpc-1 or arcsec-1')
            
        reso = self.model.get_map_reso().to_value('arcsec')
        k2d, data_pk2d = utils_pk.extract_pk2d(data_image, reso, kedges=kedges)

        #---------- Units
        if physical:
            kpc2arcsec = ((1*u.kpc/self.model.D_ang).to_value('')*u.rad).to_value('arcsec')
            k2d *= kpc2arcsec**1 *u.kpc**-1
            model_pk2d_mean *= kpc2arcsec**-2 * u.kpc**2
            model_pk2d_covmat *= kpc2arcsec**-4 *u.kpc**4
        else:
            k2d *= u.arcsec**-1
            model_pk2d_mean *= u.arcsec**2
            model_pk2d_covmat *= u.arcsec**4

        #---------- return
        if output_ymodel:
            return k2d, data_pk2d, data_image, model_ymap_sph1, model_ymap_sph2
        else:
            return k2d, data_pk2d
        
        
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
    # Compute the smooth model via forward fitting
    #==================================================
    
    def fit_profile_forward(self,
                            parinfo_profile,
                            parinfo_center=None,
                            parinfo_ellipticity=None,
                            parinfo_ZL=None,
                            use_covmat=False,
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
        - show_fit_result (bool): show the best fit model and residual. If true, set_best_fit
        will automatically be true

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
                                              self.model,
                                              self.data.image, noise_property,
                                              self.data.mask, self.data.psf_fwhm, self.data.transfer_function,
                                              parinfo_center,
                                              parinfo_ellipticity,
                                              parinfo_ZL,
                                              use_covmat], 
                                        pool=Pool(cpu_count()),
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


    #==================================================
    # Show the fit results related to profile
    #==================================================
    
    def fit_profile_forward_results(self,
                                    sampler,
                                    parinfo_profile,
                                    parinfo_center=None,
                                    parinfo_ellipticity=None,
                                    parinfo_ZL=None,
                                    visu_smooth=10*u.arcsec,
                                    binsize_prof=5*u.arcsec,
                                    Nmc=1000,
                                    true_pressure_profile=None,
                                    true_compton_profile=None):
        """
        This is function is used to show the results of the MCMC
        regarding the radial profile
            
        Parameters
        ----------
        - sampler (emcee object): the sampler obtained from fit_profile_forward
        - parinfo_profile (dict): same as fit_profile_forward
        - parinfo_center (dict): same as fit_profile_forward
        - parinfo_ellipticity (dict): same as fit_profile_forward
        - parinfo_ZL (dict): same as fit_profile_forward
        - visu_smooth (quantity): The extra smoothing FWHM for vidualization. Homogeneous to arcsec.
        - binsize_prof (quantity): The binsize for the y profile. Homogeneous to arcsec
        - Nmc (int): the number of Monte CArlo realization used to propagate data errors
        - true_pressure_profile (dict): pass a dictionary containing the profile to compare with
        in the form {'r':array in kpc, 'p':array in keV cm-3}
        - true_y_profile (dict): pass a dictionary containing the profile to compare with
        in the form {'r':array in arcmin, 'y':array in [y]}
        
        Outputs
        ----------
        plots are produced

        """

        #========== Get noise MC 
        if self.method_noise_covmat == 'model':
            noise_mc = self.data.get_noise_monte_carlo_from_model(Nmc=Nmc)
        elif self.method_noise_covmat == 'covariance':
            noise_mc = self.data.get_noise_monte_carlo_from_model(Nmc=Nmc)
        else:
            raise ValueError("Noise MC can only be computed using method_noise_covmat='model'/'covariance'")

        #========== rms for profile
        rms_prof = np.std(noise_mc, axis=0)
        rms_prof = rms_prof/self.data.mask**2
        rms_prof[~np.isfinite(rms_prof)] = np.nan
        
        #========== Get the best-fit
        best_par = utils_fitting.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
        best_model, best_ZL = lnlike_setpar_profile(best_par, self.model,
                                                    parinfo_profile,
                                                    parinfo_center=parinfo_center,
                                                    parinfo_ellipticity=parinfo_ellipticity,
                                                    parinfo_ZL=parinfo_ZL)

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
        best_ymap_sph = best_ZL + best_model.get_sz_map(no_fluctuations=True,
                                                        irfs_convolution_beam=self.data.psf_fwhm,
                                                        irfs_convolution_TF=self.data.transfer_function)

        best_y_profile = radial_profile_sb(best_ymap_sph, 
                                           (best_model.coord.icrs.ra.to_value('deg'),
                                            best_model.coord.icrs.dec.to_value('deg')), 
                                           stddev=best_ymap_sph*0+1, header=self.data.header, 
                                           binsize=binsize_prof.to_value('deg'))[1]

        
        r3d, best_pressure_profile = best_model.get_pressure_profile()
        r3d = r3d.to_value('kpc')
        best_pressure_profile = best_pressure_profile.to_value('keV cm-3')
        
        #========== MC resampling
        MC_ymap_sph         = np.zeros((self.mcmc_Nresamp, best_ymap_sph.shape[0], best_ymap_sph.shape[1]))
        MC_y_profile        = np.zeros((self.mcmc_Nresamp, len(r2d)))
        MC_pressure_profile = np.zeros((self.mcmc_Nresamp, len(r3d)))
        
        MC_pars = utils_fitting.get_emcee_random_param(sampler, burnin=self.mcmc_burnin, Nmc=self.mcmc_Nresamp)
        for i in range(self.mcmc_Nresamp):
            # Get MC model
            mc_model, mc_ZL = lnlike_setpar_profile(MC_pars[i,:], self.model, parinfo_profile,
                                                    parinfo_center=parinfo_center,
                                                    parinfo_ellipticity=parinfo_ellipticity,
                                                    parinfo_ZL=parinfo_ZL)
            # Get MC ymap
            MC_ymap_sph[i,:,:] = mc_ZL + mc_model.get_sz_map(no_fluctuations=True,
                                                             irfs_convolution_beam=self.data.psf_fwhm,
                                                             irfs_convolution_TF=self.data.transfer_function)
            # Get MC y profile
            MC_y_profile[i,:] = radial_profile_sb(MC_ymap_sph[i,:,:], 
                                                  (best_model.coord.icrs.ra.to_value('deg'),
                                                   best_model.coord.icrs.dec.to_value('deg')), 
                                                  stddev=best_ymap_sph*0+1, header=self.data.header, 
                                                  binsize=binsize_prof.to_value('deg'))[1]
            # Get MC pressure profile
            MC_pressure_profile[i,:] = mc_model.get_pressure_profile()[1].to_value('keV cm-3')

        #========== plots map
        utils_plot.show_fit_result_ymap(self.output_dir+'/MCMC_radial_results_y_map.pdf',
                                        self.data.image,
                                        self.data.header,
                                        noise_mc,
                                        best_ymap_sph,
                                        mask=self.data.mask,
                                        visu_smooth=visu_smooth.to_value('arcsec'))
        
        #========== plots ymap profile
        utils_plot.show_fit_ycompton_profile(self.output_dir+'/MCMC_radial_results_y_profile.pdf',
                                             r2d, data_yprof, data_yprof_err,
                                             best_y_profile, MC_y_profile,
                                             true_compton_profile=true_compton_profile)
        
        #========== plots pressure profile
        utils_plot.show_fit_result_pressure_profile(self.output_dir+'/MCMC_radial_results_P_profile.pdf',
                                                    r3d, best_pressure_profile, MC_pressure_profile,
                                                    true_pressure_profile=true_pressure_profile)
        
    
    #==================================================
    # Compute the Pk contraint via brute force fitting
    #==================================================
    
    def fit_fluct_brute(self,
                        parinfo_fluct,
                        parinfo_noise=None,
                        Nmc_noise=1000,
                        kbin_edges=None,
                        use_covmat=False,
                        scale_model_variance=False,
                        show_fit_result=False):
        """
        This function brute force fits the 3d power spectrum
        using a forward modeling approach
        
        Parameters
        ----------
        - parinfo_fluct (dictionary): the model parameters associated with 
        self.model.model_pressure_fluctuation to be fit as, e.g., 
        parinfo_fluct = {'Norm':                     # --> Parameter key (mandatory)
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
        - parinfo_noise (dictionary): same as parinfo_fluct but for the noise, i.e. parameter 'ampli'
        - Nmc_noise (int): the number of MC realization used for computing uncertainty (rms/covariance)
        - kbin_edges (1d array, quantity): array of k edges that can be used instead of default one, for 
        very specific bining
        - use_covmat (bool): set to true to use the covariance matrix or false for the rms only
        - scale_model_variance (bool): set to true to rescale the model variance according to the given model
        - show_fit_result (bool): set to true to produce plots for fitting results
        
        Outputs
        ----------
        - parlist (list): the list of the fit parameters
        - sampler (emcee object): the sampler associated with model parameters of
        the smooth component

        """

        #========== Check if the MCMC sampler was already recorded
        sampler_file = self.output_dir+'/pitszi_MCMC_fluctuation_sampler.h5'
        sampler_exist = utils_fitting.check_sampler_exist(sampler_file, silent=False)
            
        #========== Defines the fit parameters
        # Fit parameter list and information
        par_list, par0_value, par0_err, par_min, par_max = lnlike_defpar_fluct(parinfo_fluct, self.model,
                                                                               parinfo_noise=parinfo_noise)
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

        #========== Binning info    
        if kbin_edges is None:
            kedges = self.get_kedges().to_value('arcsec-1')
        else:
            kedges = kbin_edges.to_value('arcsec-1')
            
        reso = self.model.get_map_reso().to_value('arcsec')
        
        #========== Deal with input images and Pk        
        k2d, data_pk2d, dy_over_y, model_ymap_sph1, model_ymap_sph2 = self.get_pk2d_data(kbin_edges=kbin_edges,
                                                                                         output_ymodel=True)
        
        #========== Deal with how the noise should be accounted for
        _, noise_pk2d_ref, noise_pk2d_covmat = self.get_pk2d_noise_statistics(kbin_edges=kbin_edges, Nmc=Nmc_noise)
        _, model_pk2d_ref, model_pk2d_covmat = self.get_pk2d_modelvar_statistics(kbin_edges=kbin_edges, Nmc=Nmc_noise)
        
        #========== Define the MCMC setup
        backend = utils_fitting.define_emcee_backend(sampler_file, sampler_exist,
                                                     self.mcmc_reset, self.mcmc_nwalkers, ndim, silent=False)
        moves = emcee.moves.KDEMove()
        sampler = emcee.EnsembleSampler(self.mcmc_nwalkers, ndim,
                                        lnlike_fluct_brute,
                                        args=[parinfo_fluct,
                                              self.model,
                                              model_ymap_sph1,
                                              model_ymap_sph2,
                                              self.method_w8,
                                              model_pk2d_covmat,
                                              model_pk2d_ref,
                                              data_pk2d,
                                              noise_pk2d_covmat,
                                              noise_pk2d_ref,
                                              reso, kedges,
                                              self.data.psf_fwhm,
                                              self.data.transfer_function,
                                              parinfo_noise,
                                              use_covmat,
                                              scale_model_variance],
                                        pool=Pool(cpu_count()),
                                        moves=moves,
                                        backend=backend)
        
        #========== Run the MCMC
        if not self.silent: print('----- MCMC sampling -----')
        if self.mcmc_run:
            if not self.silent: print('      - Runing '+str(self.mcmc_nsteps)+' MCMC steps')
            model_copy = copy.deepcopy(self.model)
            res = sampler.run_mcmc(pos, self.mcmc_nsteps, progress=True, store=True)
            self.model = model_copy
        else:
            if not self.silent: print('      - Not running, but restoring the existing sampler')
            
        #========== Show the fit results
        if show_fit_result:
            self.get_mcmc_chains_outputs_results(par_list, sampler, extraname='_Pk_brute')
            
            self.fit_fluct_brute_results(sampler,
                                         parinfo_fluct,
                                         parinfo_noise=parinfo_noise,
                                         Nmc=Nmc_noise)

            
        #========== Make sure the model is set to the best fit
        best_par = utils_fitting.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
        self.model, best_noise = lnlike_setpar_fluct(best_par, self.model,
                                                     parinfo_fluct,
                                                     parinfo_noise=parinfo_noise)
        
        return par_list, sampler
    
    
    #==================================================
    # Show the fit results related to fluctuation
    #==================================================
    
    def fit_fluct_brute_results(self,
                                sampler,
                                parinfo_fluct,
                                parinfo_noise=None,
                                kbin_edges=None,
                                Nmc=1000,
                                true_pk3d=None):
        """
        This is function is used to show the results of the MCMC
        regarding the fluctuations
            
        Parameters
        ----------
        - sampler (emcee object): the sampler obtained from fit_profile_forward
        - parinfo_fluct (dictionary): the model parameters, see  fit_fluct_brute
        - parinfo_noise (dictionary): the noise parameters, see  fit_fluct_brute
        - kbin_edges (1d array, quantity): array of k edges that can be used instead of default one, for 
        very specific bining
        - Nmc (int): number of MC sampling
        - true_pk3d (dict): pass a dictionary containing the Pk3d to compare with
        in the form {'k':array in kpc-1, 'pk':array in kpc3}

        Outputs
        ----------
        plots are produced

        """
        
        #========== Binning info    
        if kbin_edges is None:
            kedges = self.get_kedges().to_value('arcsec-1')
        else:
            kedges = kbin_edges.to_value('arcsec-1')
            
        reso = self.model.get_map_reso().to_value('arcsec')

        #========== Recover data and error
        #---------- Deal with input images and Pk        
        k2d, data_pk2d, data_image, model_ymap_sph1, model_ymap_sph2 = self.get_pk2d_data(kbin_edges=kbin_edges,
                                                                                         output_ymodel=True)
        
        #---------- Deal with how the noise should be accounted for
        _, noise_pk2d_ref, noise_pk2d_covmat = self.get_pk2d_noise_statistics(kbin_edges=kbin_edges, Nmc=Nmc)
        _, model_pk2d_ref, model_pk2d_covmat = self.get_pk2d_modelvar_statistics(kbin_edges=kbin_edges, Nmc=Nmc)
              
        #========== Get the best-fit
        best_par = utils_fitting.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
        best_model, best_Anoise = lnlike_setpar_fluct(best_par, self.model,
                                                      parinfo_fluct,
                                                      parinfo_noise=parinfo_noise)
        
        #========== Compute best fit observables
        k3d, best_pk3d = best_model.get_pressure_fluctuation_spectrum()
        k3d = k3d.to_value('kpc-1')
        best_pk3d = best_pk3d.to_value('kpc3')
        
        #========== MC resampling
        MC_pk3d = np.zeros((self.mcmc_Nresamp, len(k3d)))
        MC_pk2d = np.zeros((self.mcmc_Nresamp, len(k2d)))
        MC_pk2d_noise = np.zeros((self.mcmc_Nresamp, len(k2d)))
        
        MC_pars = utils_fitting.get_emcee_random_param(sampler, burnin=self.mcmc_burnin, Nmc=self.mcmc_Nresamp)
        for imc in range(self.mcmc_Nresamp):
            # Get MC model
            mc_model, mc_Anoise = lnlike_setpar_fluct(MC_pars[imc,:], self.model, parinfo_fluct,
                                                      parinfo_noise=parinfo_noise)

            # Get MC noise
            MC_pk2d_noise[imc,:] = mc_Anoise * noise_pk2d_ref
            
            # Get MC Pk2d
            mc_ymap = mc_model.get_sz_map(seed=None, no_fluctuations=False,
                                          irfs_convolution_beam=self.data.psf_fwhm,
                                          irfs_convolution_TF=self.data.transfer_function)
            delta_y = mc_ymap - model_ymap_sph1
            mc_image = (delta_y - np.mean(delta_y))/model_ymap_sph2 * self.method_w8
            MC_pk2d[imc,:] = utils_pk.extract_pk2d(mc_image, reso, kedges=kedges)[1]
            
            # Get MC Pk3d
            MC_pk3d[imc,:] = mc_model.get_pressure_fluctuation_spectrum()[1].to_value('kpc3')

        #========== Plot the fitted image data
        utils_plot.show_fit_result_delta_ymap(self.output_dir+'/MCMC_fluctuation_results_input_image.pdf',
                                              self.data.image,
                                              data_image,
                                              model_ymap_sph1,
                                              model_ymap_sph2,
                                              self.method_w8,
                                              self.data.header)
        
        #========== Plot the covariance matrix
        utils_plot.show_fit_result_covariance(self.output_dir+'/MCMC_fluctuation_results_covariance.pdf',
                                              noise_pk2d_covmat,
                                              model_pk2d_covmat)
        
        #========== Plot the Pk2d constraint
        utils_plot.show_fit_result_pk2d(self.output_dir+'/MCMC_fluctuation_results_pk2d.pdf',
                                        k2d, data_pk2d,
                                        model_pk2d_ref,
                                        np.diag(model_pk2d_covmat)**0.5, np.diag(noise_pk2d_covmat)**0.5,
                                        MC_pk2d, MC_pk2d_noise)

            
        #========== Plot the Pk3d constraint
        utils_plot.show_fit_result_pk3d(self.output_dir+'/MCMC_fluctuation_results_pk3d.pdf',
                                        k3d, best_pk3d, MC_pk3d, true_pk3d=true_pk3d)



    #==================================================
    # Compute the Pk prediction for forward deprojection
    #==================================================
    
    def modelpred_fluct_deproj_forward(self, physical=True):
        """
        This function uses the current inference and model objects state
        to extract the model prediction for the forward deprojection model.
        
        Parameters
        ----------
        - physical (bool): set to true to have physical unit, otherwise angles.
        
        Outputs
        ----------
        - k_out (quantity array): bin center in kpc-1 or arcsec-1
        - pk_out (quantity array): Pk in the binin kpc2 or arcsec2

        """

        # k information
        _, _, k2d = self.get_k_grid(physical=True)
        k2d = k2d.to_value('kpc-1')
        kedges = self.get_kedges(physical=True).to_value('kpc-1')

        # Compute Kmnmn
        fft_w8 = np.fft.fftn(self.method_w8)
        Kmnmn = utils_pk.compute_Kmnmn(fft_w8)

        # Compute the conversion factor
        ymap_wf = self.get_p3d_to_p2d_from_window_function()
        conv = np.sum(ymap_wf*self.method_w8**2) / np.sum(self.method_w8**2)
        conv = conv.to_value('kpc-1')

        # Pk prediction
        pk_pred = lnlike_model_fluct_deproj_forward(self.model,
                                                    k2d,
                                                    conv,
                                                    Kmnmn,
                                                    self.data.psf_fwhm,
                                                    self.data.transfer_function,
                                                    kedges)
        
        # Bin center
        k_pred = 0.5 * (kedges[1:] + kedges[:-1])

        # Unit changes
        if physical:
            k_out = k_pred*u.kpc**-1
            pk_out = pk_pred*u.kpc**2
        else:
            k_out = ((k_pred*u.kpc**-1 * self.model.D_ang).to_value('')*u.rad**-1).to('arcsec-1')
            pk_out = ((pk_pred*u.kpc**2 / self.model.D_ang**2).to_value('')*u.rad**2).to('arcsec2')

        return k_out, pk_out
        
        
    #==================================================
    # Compute the Pk contraint via forward deprojection
    #==================================================
    
    def fit_fluct_deproj_forward(self,
                                 parinfo_fluct,
                                 parinfo_noise=None,
                                 Nmc_noise=1000,
                                 kbin_edges=None,
                                 use_covmat=False,
                                 show_fit_result=False):
        """
        This function fits the 3d power spectrum
        using a forward modeling approach via deprojection
        
        Parameters
        ----------
        - parinfo_fluct (dictionary): the model parameters associated with 
        self.model.model_pressure_fluctuation to be fit as, e.g., 
        parinfo_fluct = {'Norm':                     # --> Parameter key (mandatory)
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
        - parinfo_noise (dictionary): same as parinfo_fluct but for the noise, i.e. parameter 'ampli'
        - Nmc_noise (int): the number of MC realization used for computing uncertainty (rms/covariance)
        - kbin_edges (1d array, quantity): array of k edges that can be used instead of default one, for 
        very specific bining
        - use_covmat (bool): set to true to use the covariance matrix or false for the rms only
        - show_fit_result (bool): set to true to produce plots for fitting results
        
        Outputs
        ----------
        - parlist (list): the list of the fit parameters
        - sampler (emcee object): the sampler associated with model parameters of
        the smooth component

        """

        #========== Check if the MCMC sampler was already recorded
        sampler_file = self.output_dir+'/pitszi_MCMC_fluctuation_sampler.h5'
        sampler_exist = utils_fitting.check_sampler_exist(sampler_file, silent=False)
            
        #========== Defines the fit parameters
        # Fit parameter list and information
        par_list, par0_value, par0_err, par_min, par_max = lnlike_defpar_fluct(parinfo_fluct, self.model,
                                                                               parinfo_noise=parinfo_noise)
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

        #========== Binning info    
        if kbin_edges is None:
            kedges = self.get_kedges(physical=True).to_value('kpc-1')
        else:
            kedges = kbin_edges.to_value('kpc-1')
            
        k2d_norm = self.get_k_grid(physical=True)[2].to_value('kpc-1')
        
        #========== Deal with input images and Pk        
        k2d, data_pk2d  = self.get_pk2d_data(kbin_edges=kbin_edges)
        
        #========== Deal with how the noise should be accounted for
        _, noise_pk2d_ref, noise_pk2d_covmat = self.get_pk2d_noise_statistics(kbin_edges=kbin_edges, Nmc=Nmc_noise)
        
        if use_covmat:
            noise_pk2d_info = np.linalg.inv(noise_pk2d_covmat)
        else:
            noise_pk2d_info = np.diag(noise_pk2d_covmat)**0.5

        #========== Compute model products
        #---------- Window function and 2D-3D conversion
        ymap_wf = self.get_p3d_to_p2d_from_window_function()
        conv = np.sum(ymap_wf*self.method_w8**2) / np.sum(self.method_w8**2)

        #---------- Kmnmn
        fft_w8 = np.fft.fftn(self.method_w8)
        Kmnmn = utils_pk.compute_Kmnmn(fft_w8)

        #========== Define the MCMC setup
        backend = utils_fitting.define_emcee_backend(sampler_file, sampler_exist,
                                                     self.mcmc_reset, self.mcmc_nwalkers, ndim, silent=False)
        from pathos import multiprocessing as mpp
        sampler = emcee.EnsembleSampler(self.mcmc_nwalkers, ndim,
                                        lnlike_fluct_deproj_forward,
                                        args=[parinfo_fluct,
                                              self.model,
                                              conv.to_value('kpc-1'),
                                              Kmnmn,
                                              k2d_norm,
                                              data_pk2d,
                                              noise_pk2d_info,
                                              noise_pk2d_ref,
                                              kedges,
                                              self.data.psf_fwhm,
                                              self.data.transfer_function,
                                              parinfo_noise,
                                              use_covmat],
                                        pool=mpp.Pool(mpp.cpu_count()),
                                        #pool=Pool(cpu_count()),
                                        backend=backend)
        
        #========== Run the MCMC
        if not self.silent: print('----- MCMC sampling -----')
        if self.mcmc_run:
            if not self.silent: print('      - Runing '+str(self.mcmc_nsteps)+' MCMC steps')
            model_copy = copy.deepcopy(self.model)
            res = sampler.run_mcmc(pos, self.mcmc_nsteps, progress=True, store=True)
            self.model = model_copy
        else:
            if not self.silent: print('      - Not running, but restoring the existing sampler')
            
        #========== Show the fit results
        if show_fit_result:
            self.get_mcmc_chains_outputs_results(par_list, sampler, extraname='_Pk_brute')
            
            self.fit_fluct_brute_results(sampler,
                                         parinfo_fluct,
                                         parinfo_noise=parinfo_noise,
                                         Nmc=Nmc_noise)

            
        #========== Make sure the model is set to the best fit
        best_par = utils_fitting.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
        self.model, best_noise = lnlike_setpar_fluct(best_par, self.model,
                                                     parinfo_fluct,
                                                     parinfo_noise=parinfo_noise)
        
        return par_list, sampler
    
    
















        
    '''
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
                
            k2d, pk2d  = utils_pk.extract_pk2d(image, proj_reso,
                                        Nbin=Nbin, kmin=kmin, kmax=kmax, scalebin=scalebin, kedges=kedges,
                                        statistic=statistic)
            if mask is not None:
                pk2d *= mask.size/np.sum(mask) # mean fsky correction
            
        elif method == 'Arevalo12':
            k2d, pk2d  = utils_pk.extract_pk2d_arevalo(image, proj_reso,
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
        compton_mock_fluct = utils_pk.apply_transfer_function(compton_true_fluct, map_reso_arcsec, 
                                                           ConvbeamFWHM.to_value('arcsec'), ConvTF, 
                                                           apps_TF_LS=True, apps_beam=True)
        # Apply IRF to map without fluctuation
        compton_mock_spher = utils_pk.apply_transfer_function(compton_true_spher, map_reso_arcsec, 
                                                           ConvbeamFWHM.to_value('arcsec'), ConvTF, 
                                                           apps_TF_LS=True, apps_beam=True)
        
        # Apply beam only to map without fluctuation
        compton_mockB_spher = utils_pk.apply_transfer_function(compton_true_spher, map_reso_arcsec, 
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
            k2d, pk2d  = utils_pk.extract_pk2d(image, map_reso_kpc,
                                        Nbin=Nbin, kmin=kmin, kmax=kmax, scalebin=scalebin, kedges=kedges,
                                        statistic=statistic)
            if mask is not None:
                pk2d *= mask.size/np.sum(mask) # mean fsky correction
            
        elif method_pk == 'Arevalo12':
            k2d, pk2d  = utils_pk.extract_pk2d_arevalo(image, map_reso_kpc,
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
    '''
