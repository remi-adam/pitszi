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
    - parinfo_profile (dict): see parinfo_profile in get_smooth_model_forward_fitting
    - model (class Model object): the model to be updated in the fit
    - parinfo_center (dict): see parinfo_center in get_smooth_model_forward_fitting
    - parinfo_ellipticity (dict): see parinfo_ellipticity in get_smooth_model_forward_fitting
    - parinfo_ZL (dict): see parinfo_ZL in get_smooth_model_forward_fitting

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
    
def lnlike_defpar_fluct(parinfo_pk3d,
                        model,
                        parinfo_noise=None):
    """
    This function helps defining the parameter list, initial values and
    ranges for the fluctuation fitting.
        
    Parameters
    ----------
    - parinfo_pk3d (dict): see parinfo_pk3d in get_pk3d_model_forward_fitting
    - model (class Model object): the model to be updated in the fit
    - parinfo_noise (dictionary): same as parinfo_pk3d with accepted parameter 'Noise'

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
    Npar_fluct = len(list(parinfo_pk3d.keys()))
    for ipar in range(Npar_fluct):
        
        parkey = list(parinfo_pk3d.keys())[ipar]

        #----- Check that the param is ok wrt the model
        try:
            bid = model.model_pressure_fluctuation[parkey]
        except:
            raise ValueError('The parameter '+parkey+' is not in self.model')    
        if 'guess' not in parinfo_pk3d[parkey]:
            raise ValueError('The guess key is mandatory for starting the chains, as "guess":[guess_value, guess_uncertainty] ')    
        if 'unit' not in parinfo_pk3d[parkey]:
            raise ValueError('The unit key is mandatory. Use "unit":None if unitless')

        #----- Update the name list
        par_list.append(parkey)

        #----- Update the guess values
        par0_value.append(parinfo_pk3d[parkey]['guess'][0])
        par0_err.append(parinfo_pk3d[parkey]['guess'][1])

        #----- Update the limit values
        if 'limit' in parinfo_pk3d[parkey]:
            par_min.append(parinfo_pk3d[parkey]['limit'][0])
            par_max.append(parinfo_pk3d[parkey]['limit'][1])
        else:
            par_min.append(-np.inf)
            par_max.append(+np.inf)

    #========== Deal with noise normalization
    if parinfo_noise is not None:
        Npar_noise = len(list(parinfo_noise.keys()))
        list_noise_allowed = ['noise']
        
        for ipar in range(Npar_noise):

            parkey = list(parinfo_noise.keys())[ipar]

            #----- Check that the param is ok wrt the model
            if parkey not in list_noise_allowed:
                raise ValueError('parinfo_noise allowed param is noise only')
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
    - parinfo_profile (dict): see get_smooth_model_forward_fitting
    - parinfo_center (dict): see get_smooth_model_forward_fitting
    - parinfo_ellipticity (dict): see get_smooth_model_forward_fitting
    - parinfo_ZL (dict): see get_smooth_model_forward_fitting

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
                       parinfo_pk3d,
                       parinfo_noise=None):
    """
    Compute the prior given the input parameter and some information
    about the parameters given by the user
       
    Parameters
    ----------
    - param (list): the parameter to apply to the model
    - parinfo_pk3d (dict): see get_pk3d_model_forward_fitting
    - parinfo_noise (dict): see get_pk3d_model_forward_fitting

    Outputs
    ----------
    - prior (float): value of the prior

    """

    prior = 0
    idx_par = 0

    #---------- Fluctuation parameters
    parkeys_fluct = list(parinfo_pk3d.keys())

    for ipar in range(len(parkeys_fluct)):
        parkey = parkeys_fluct[ipar]
        
        # Flat prior
        if 'limit' in parinfo_pk3d[parkey]:
            if param[idx_par] < parinfo_pk3d[parkey]['limit'][0]:
                return -np.inf                
            if param[idx_par] > parinfo_pk3d[parkey]['limit'][1]:
                return -np.inf
            
        # Gaussian prior
        if 'prior' in parinfo_pk3d[parkey]:
            expected = parinfo_pk3d[parkey]['prior'][0]
            sigma = parinfo_pk3d[parkey]['prior'][1]
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
    - parinfo_profile (dict): see get_smooth_model_forward_fitting
    - parinfo_center (dict): see get_smooth_model_forward_fitting
    - parinfo_ellipticity (dict): see get_smooth_model_forward_fitting
    - parinfo_ZL (dict): see get_smooth_model_forward_fitting

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
                        parinfo_pk3d,
                        parinfo_noise=None):
    """
    Set the model to the given fluctuation parameters
        
    Parameters
    ----------
    - param (list): the parameters to apply to the model
    - model (pitszi object): pitszi object from class Model
    - parinfo_pk3d (dict): see get_pk3d_model_forward_fitting
    - parinfo_noise (dict): see get_pk3d_model_forward_fitting

    Outputs
    ----------
    - model (float): the model
    - noise_ampli (float): the nuisance parameter, noise amplitude

    """
    
    idx_par = 0

    #---------- Fluctuation parameters
    parkeys_fluct = list(parinfo_pk3d.keys())
    for ipar in range(len(parkeys_fluct)):
        parkey = parkeys_fluct[ipar]
        if parinfo_pk3d[parkey]['unit'] is not None:
            unit = parinfo_pk3d[parkey]['unit']
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
global lnlike_profile
def lnlike_profile(param,
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
    - parinfo_profile (dict): see parinfo_profile in get_smooth_model_forward_fitting
    - model (class Model object): the model to be updated
    - data_img (nd array): the image data
    - data_noiseprop (nd_array): the inverse covariance matrix to be used for the fit or the rms,
    depending on use_covmat.
    - data_mask (nd array): the image mask
    - data_psf (quantity): input to model_mock.get_sz_map
    - data_tf (dict): input to model_mock.get_sz_map
    - parinfo_center (dict): see parinfo_center in get_smooth_model_forward_fitting
    - parinfo_ellipticity (dict): see parinfo_ellipticity in get_smooth_model_forward_fitting
    - parinfo_ZL (dict): see parinfo_ZL in get_smooth_model_forward_fitting
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
global lnlike_fluct
def lnlike_fluct(param,
                 parinfo_pk3d,
                 method_fluctuation_image,
                 model,
                 model_ymap_sph,
                 model_ymap_sph_deconv,
                 model_pk2d_covmat_ref,
                 data_pk2d,
                 noise_pk2d_covmat,
                 noise_pk2d_mean,
                 mask,
                 reso_arcsec,
                 psf_fwhm,
                 transfer_function,
                 kedges=None,
                 parinfo_noise=None,
                 use_covmat=False):
    """
    This is the likelihood function used for the forward fit of the fluctuation spectrum.
        
    Parameters
    ----------
    - param (np array): the value of the test parameters
    - parinfo_pk3d (dict): see parinfo_pk3d in get_pk3d_model_forward_fitting
    - method_fluctuation_image (str): the method used to compute the fluctuation image (ratio, subtract)
    - model (class Model object): the model to be updated
    - model_ymap_sph (2d array): best-fit smooth model
    - model_ymap_sph_deconv (2d array): best-fit smooth model deconvolved from TF
    - model_pk2d_covmat_ref (2d array): model covariance matrix
    - data_pk2d (1d array): the Pk extracted from the data
    - noise_pk2d_covmat (2d array): the noise covariance matrix
    - noise_pk2d_mean (1d array): the mean noise bias expected
    - mask (2d array): the mask associated with the data
    - reso_arcsec (float): the resolution in arcsec
    - psf_fwhm (float): the psf FWHM
    - transfer_function (dict): the transfer function
    - kedges (1d array): the edges of the k bins
    - parinfo_noise (dict): see parinfo_noise in get_pk3d_model_forward_fitting
    - use_covmat (bool): set to true to use the covariance matrix
    
    Outputs
    ----------
    - lnL (float): the value of the log likelihood

    """

    #========== Deal with flat and Gaussian priors
    prior = lnlike_prior_fluct(param, parinfo_pk3d,parinfo_noise=parinfo_noise)
    if not np.isfinite(prior): return -np.inf
    
    #========== Change model parameters
    model, noise_ampli = lnlike_setpar_fluct(param, model, parinfo_pk3d,
                                             parinfo_noise=parinfo_noise)
    
    #========== Get the model
    #---------- Compute the ymap and fluctuation image
    test_ymap = model.get_sz_map(seed=None, no_fluctuations=False,
                                 irfs_convolution_beam=psf_fwhm, irfs_convolution_TF=transfer_function)

    if method_fluctuation_image == 'ratio':
        test_image = (test_ymap - model_ymap_sph)/model_ymap_sph_deconv
    elif method_fluctuation_image == 'subtract':
        test_image = test_ymap - model_ymap_sph
    else:
        raise ValueError('inference.method_fluctuation_image can be either "ratio" or "difference"')
    test_image *= mask

    #---------- Compute the test Pk
    test_k, test_pk =  utils.get_pk2d(test_image, reso_arcsec, kedges=kedges)
    model_pk2d = test_pk + noise_ampli*noise_pk2d_mean

    #========== Compute the likelihood
    if use_covmat: # Using the covariance matrix
        covariance_inverse = np.linalg.pinv(noise_pk2d_covmat + model_pk2d_covmat_ref)
        residual_test = model_pk2d - data_pk2d
        lnL = -0.5*np.matmul(residual_test, np.matmul(covariance_inverse, residual_test))
        
    else: # Using the rms only
        variance = np.diag(noise_pk2d_covmat) + np.diag(model_pk2d_covmat_ref)
        lnL = -0.5*np.nansum((model_pk2d - data_pk2d)**2 / variance)
        
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
        self.silent     = False
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
        self.kbin_Nbin  = 20
        self.kbin_scale = 'lin' # ['lin', 'log']

        # MCMC parameters
        self.mcmc_nwalkers = 100
        self.mcmc_nsteps   = 500
        self.mcmc_burnin   = 100
        self.mcmc_reset    = False
        self.mcmc_run      = True

        
    #==================================================
    # Define k edges from bining properties
    #==================================================
    
    def get_kedges(self):
        """
        This function compute the edges of the bins in k space
        
        Parameters
        ----------
        
        Outputs
        ----------
        - kedges (1d array): the edges of the bins 

        """

        kmin_sampling = self.kbin_min.to_value('arcsec-1')
        kmax_sampling = self.kbin_max.to_value('arcsec-1')
        
        if self.kbin_scale == 'lin':
            kbins = np.linspace(kmin_sampling, kmax_sampling, self.kbin_Nbin+1)
        elif self.kbin_scale == 'log':
            kbins = np.logspace(np.log10(kmin_sampling), np.log10(kmax_sampling), self.kbin_Nbin+1)
        else:
            raise ValueError("Only lin or log scales are allowed. Here self.kbin_scale="+self.kbin_scale)

        return kbins*u.arcsec**-1


    #==================================================
    # Compute noise properties
    #==================================================
    
    def get_pk2d_noise_properties(self,
                                  kbin_edges=None,
                                  Nmc=1000,
                                  method_mc='model'):
        """
        This function compute the noise properties associated
        with the data noise
        
        Parameters
        ----------
        - kbin_edges (1d array): array of k edges that can be used instead of default one
        - Nmc (int): number of monte carlo realization
        - method_mc (str): the method to compute MC realization (model or covariance)

        Outputs
        ----------
        - noise_pk2d_ref (): the noise mean
        - noise_pk2d_rms (): the noise rms
        - noise_pk2d_covmat (): the noise covariance matrix

        """
        
        # Info to user
        if not self.silent:
            print('----- Computing Pk2d noise covariance -----')
        
        # Useful info
        if kbin_edges is None:
            kedges = self.get_kedges().to_value('arcsec-1')
        else:
            kedges = kbin_edges
        reso = self.model.get_map_reso().to_value('arcsec')

        model_ymap_sph_deconv = self.model.get_sz_map(no_fluctuations=True,
                                                      irfs_convolution_beam=self.data.psf_fwhm)    
        model_ymap_sph        = self.model.get_sz_map(no_fluctuations=True,
                                                      irfs_convolution_beam=self.data.psf_fwhm,
                                                      irfs_convolution_TF=self.data.transfer_function) 
        
        # Extract noise MC realization
        if method_mc == 'model':
            noise_ymap_mc = self.data.get_noise_monte_carlo_from_model(center=None, seed=None, Nmc=Nmc)
        elif kind_mc == 'covariance':
            noise_ymap_mc = self.data.get_noise_monte_carlo_from_covariance(seed=None, Nmc=Nmc)
        else:
            raise ValueError("The noise MC can only be computed using kind_mc = 'model' or kind_mc = 'covariance'")
                        
        # Compute Pk2d MC realization
        noise_pk2d_mc = np.zeros((Nmc, len(kedges)-1))
        for imc in range(Nmc):
            
            if self.method_fluctuation_image == 'ratio':
                image_noise_mc = (noise_ymap_mc[imc,:,:])/model_ymap_sph_deconv
            if self.method_fluctuation_image == 'subtract':
                image_noise_mc = noise_ymap_mc[imc,:,:]
                
            image_noise_mc *= self.data.mask
                        
            noise_pk2d_mc[imc,:] = utils.get_pk2d(image_noise_mc, reso, kedges=kedges)[1]
            
        noise_pk2d_mean = np.mean(noise_pk2d_mc, axis=0)
        noise_pk2d_rms  = np.std(noise_pk2d_mc, axis=0)

        # Compute covariance
        noise_pk2d_covmat = np.zeros((len(kedges)-1, len(kedges)-1))
        for imc in range(Nmc):
            noise_pk2d_covmat += np.matmul((noise_pk2d_mc[imc,:]-noise_pk2d_mean)[:,None],
                                           (noise_pk2d_mc[imc,:]-noise_pk2d_mean)[None,:])
        noise_pk2d_covmat /= Nmc

        # Sanity check
        if np.sum(np.isnan(noise_pk2d_covmat)) > 0:
            if not self.silent:
                print('The number of bins is such that some bins are empty, leading to nan in the data.')
            raise ValueError('Issue with binning')

        return noise_pk2d_mean, noise_pk2d_rms, noise_pk2d_covmat


    #==================================================
    # Compute model variance properties
    #==================================================
    
    def get_pk2d_modelvar_properties(self,
                                     kbin_edges=None,
                                     Nmc=1000):
        """
        This function compute the model variance properties associated
        with the reference input model.
        
        Parameters
        ----------
        - kbin_edges (1d array): array of k edges that can be used instead of default one
        - Nmc (int): number of monte carlo realization

        Outputs
        ----------
        - model_pk2d_ref (): the model mean
        - model_pk2d_rms (): the model rms
        - model_pk2d_covmat (): the model covariance matrix

        """

        # Info to user
        if not self.silent:
            print('----- Computing Pk2d model covariance -----')
        
        # Useful info
        if kbin_edges is None:
            kedges = self.get_kedges().to_value('arcsec-1')
        else:
            kedges = kbin_edges
        reso = self.model.get_map_reso().to_value('arcsec')

        model_ymap_sph_deconv = self.model.get_sz_map(no_fluctuations=True,
                                                      irfs_convolution_beam=self.data.psf_fwhm)    
        model_ymap_sph        = self.model.get_sz_map(no_fluctuations=True,
                                                      irfs_convolution_beam=self.data.psf_fwhm,
                                                      irfs_convolution_TF=self.data.transfer_function) 

        # Compute Pk2d MC realization
        model_pk2d_mc     = np.zeros((Nmc, len(kedges)-1))
        model_pk2d_covmat = np.zeros((len(kedges)-1, len(kedges)-1))
        
        for imc in range(Nmc):
            
            test_ymap = self.model.get_sz_map(seed=None,
                                              no_fluctuations=False,
                                              irfs_convolution_beam=self.data.psf_fwhm,
                                              irfs_convolution_TF=self.data.transfer_function)
            
            if self.method_fluctuation_image == 'ratio':
                test_image = (test_ymap - model_ymap_sph)/model_ymap_sph_deconv
            if self.method_fluctuation_image == 'subtract':
                test_image = test_ymap - model_ymap_sph
                
            test_image *= self.data.mask
            
            model_pk2d_mc[imc,:] = utils.get_pk2d(test_image, reso, kedges=kedges)[1]
            
        model_pk2d_mean = np.mean(model_pk2d_mc, axis=0)
        model_pk2d_rms = np.std(model_pk2d_mc, axis=0)

        # Compute covariance
        for imc in range(Nmc):
            model_pk2d_covmat += np.matmul((model_pk2d_mc[imc,:]-model_pk2d_mean)[:,None],
                                           (model_pk2d_mc[imc,:]-model_pk2d_mean)[None,:])
        model_pk2d_covmat /= Nmc

        # Sanity check
        if np.sum(np.isnan(model_pk2d_covmat)) > 0:
            if not self.silent:
                print('The number of bins is such that some bins are empty, leading to nan in the data.')
            raise ValueError('Issue with binning')

        return model_pk2d_mean, model_pk2d_rms, model_pk2d_covmat
    
    
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
                raise ValueError("The 'truth' parameter should match the number of parameters")

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
                                         parinfo_profile,
                                         parinfo_center=None,
                                         parinfo_ellipticity=None,
                                         parinfo_ZL=None,
                                         use_covmat=False,
                                         set_best_fit=True,
                                         show_fit_result=False):
        """
        This function fits the data with a defined model
        
        Parameters
        ----------
        - parinfo_profile (dictionary): the model parameters associated with 
        self.model.model_pressure_profile to be fit as, e.g., 
        parinfo_prof = {'P_0':                    # --> Parameter key (mandatory)
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
        - parinfo_center (dictionary): same as parinfo_profile with accepted parameters
        'RA' and 'Dec'
        - parinfo_ellipticity (dictionary): same as parinfo_profile with accepted parameters
        'min_to_maj_axis_ratio' and 'angle'
        - parinfo_ZL (dictionary): same as parinfo_profile with accepted parameter
        'ZL'
        - use_covmat (bool): if True, the inverse noise covariance matrix is used instead 
        of the noise rms
        - set_best_fit (bool): set the best fit model to the best fit parameters
        - show_fit_result (bool): show the best fit model and residual. If true, set_best_fit
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
        par_list,par0_value,par0_err,par_min,par_max=lnlike_defpar_profile(parinfo_profile,
                                                                           self.model,
                                                                           parinfo_ZL=parinfo_ZL,
                                                                           parinfo_center=parinfo_center,
                                                                           parinfo_ellipticity=parinfo_ellipticity)
        ndim = len(par0_value)
        
        # Starting points of the chains
        pos = utils.emcee_starting_point(par0_value, par0_err, par_min, par_max, self.mcmc_nwalkers)
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
        backend = utils.define_emcee_backend(sampler_file, sampler_exist,
                                             self.mcmc_reset, self.mcmc_nwalkers, ndim, silent=False)
        moves = emcee.moves.StretchMove(a=2.0)
        sampler = emcee.EnsembleSampler(self.mcmc_nwalkers, ndim,
                                        lnlike_profile, 
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

        #========== Set the best-fit model
        if set_best_fit or show_fit_result:
            best_par = utils.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
            best_model, best_ZL = lnlike_setpar_profile(best_par, self.model,
                                                        parinfo_profile,
                                                        parinfo_center=parinfo_center,
                                                        parinfo_ellipticity=parinfo_ellipticity,
                                                        parinfo_ZL=parinfo_ZL)
        
        #========== Show best-fit
        if show_fit_result:
            
            # Get noise MC 
            if self.data.noise_covmat is not None:
                noise_mc = self.data.get_noise_monte_carlo_from_model(Nmc=1000)
            elif (self.data.noise_model_pk_center is not None) and (self.data.noise_model_radial is not None):
                noise_mc = self.data.get_noise_monte_carlo_from_model(Nmc=1000)
            else:
                raise ValueError('Noise MC could not be computed when showing results')

            # Compute best fit map
            best_ymap_sph = best_ZL + best_model.get_sz_map(no_fluctuations=True,
                                                            irfs_convolution_beam=self.data.psf_fwhm,
                                                            irfs_convolution_TF=self.data.transfer_function)

            # MC resampling
            MC_ymap_sph = np.zeros((100, best_ymap_sph.shape[0], best_ymap_sph.shape[1]))
            MC_pars = utils.get_emcee_random_param(sampler, burnin=self.mcmc_burnin, Nmc=100)
            for i in range(100):
                mc_model, mc_ZL = lnlike_setpar_profile(MC_pars[i,:], self.model, parinfo_profile,
                                                        parinfo_center=parinfo_center,
                                                        parinfo_ellipticity=parinfo_ellipticity,
                                                        parinfo_ZL=parinfo_ZL)
                MC_ymap_sph[i,:,:] = mc_ZL + mc_model.get_sz_map(no_fluctuations=True,
                                                                 irfs_convolution_beam=self.data.psf_fwhm,
                                                                 irfs_convolution_TF=self.data.transfer_function)
                                   
            # plots profile
            plotlib.show_best_fit_radial_profile(self.output_dir+'/MCMC_radial_best_profile.pdf',
                                                 self.data.image,
                                                 self.data.header,
                                                 noise_mc,
                                                 self.model,
                                                 best_ymap_sph,
                                                 MC_ymap_sph,
                                                 mask=self.data.mask,
                                                 binsize=5)
            
            # plots map
            plotlib.show_best_fit_radial_map(self.output_dir+'/MCMC_radial_best_map.pdf',
                                             self.data.image,
                                             self.data.header,
                                             noise_mc,
                                             best_ymap_sph,
                                             mask=self.data.mask,
                                             visu_smooth=10)

            # Make sure the model is the best
            self.model = best_model
            
        return par_list, sampler
    
    
    #==================================================
    # Compute the Pk contraint via forward fitting
    #==================================================
    
    def get_pk3d_model_forward_fitting(self,
                                       parinfo_pk3d,
                                       parinfo_noise=None,
                                       Nmc_noise=1000,
                                       method_mc='model',
                                       kbin_edges=None,
                                       use_covmat=False,
                                       set_best_fit=True,
                                       show_fit_result=False):
        """
        This function brute force fits the 3d power spectrum
        using a forward modeling approach
        
        Parameters
        ----------
        - parinfo_pk (dictionary): the model parameters associated with 
        self.model.model_pressure_fluctuation to be fit as, e.g., 
        parinfo_pk = {'Norm':                     # --> Parameter key (mandatory)
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
        par_list, par0_value, par0_err, par_min, par_max = lnlike_defpar_fluct(parinfo_pk3d, self.model,
                                                                               parinfo_noise=parinfo_noise)
        ndim = len(par0_value)
        
        # Starting points of the chains
        pos = utils.emcee_starting_point(par0_value, par0_err, par_min, par_max, self.mcmc_nwalkers)
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
        
        #========== Deal with input images and Pk        
        #---------- Get the smooth model and the data image
        model_ymap_sph_deconv = self.model.get_sz_map(no_fluctuations=True,
                                                      irfs_convolution_beam=self.data.psf_fwhm)    
        model_ymap_sph        = self.model.get_sz_map(no_fluctuations=True,
                                                      irfs_convolution_beam=self.data.psf_fwhm,
                                                      irfs_convolution_TF=self.data.transfer_function) 
        if self.method_fluctuation_image == 'ratio':
            data_image = (self.data.image - model_ymap_sph)/model_ymap_sph_deconv
        elif self.method_fluctuation_image == 'subtract':
            data_image = self.data.image - model_ymap_sph
        else:
            raise ValueError('inference.method_fluctuation_image can be either "ratio" or "difference"')
        data_image *= self.data.mask

        #---------- Pk for the data
        if kbin_edges is None:
            kedges = self.get_kedges().to_value('arcsec-1')
        else:
            kedges = kbin_edges
        reso = self.model.get_map_reso().to_value('arcsec')
        k2d, data_pk2d = utils.get_pk2d(data_image, reso, kedges=kedges)
        
        #========== Deal with how the noise should be accounted for
        noise_pk2d_ref, noise_pk2d_rms, noise_pk2d_covmat = self.get_pk2d_noise_properties(kedges, Nmc_noise,
                                                                                           method_mc=method_mc)
        model_pk2d_ref, model_pk2d_rms, model_pk2d_covmat = self.get_pk2d_modelvar_properties(kedges, Nmc_noise)
        
        #========== Define the MCMC setup
        backend = utils.define_emcee_backend(sampler_file, sampler_exist,
                                             self.mcmc_reset, self.mcmc_nwalkers, ndim, silent=False)
        moves = emcee.moves.KDEMove()
        sampler = emcee.EnsembleSampler(self.mcmc_nwalkers, ndim,
                                        lnlike_fluct,
                                        args=[parinfo_pk3d,
                                              self.method_fluctuation_image,
                                              self.model,
                                              model_ymap_sph,
                                              model_ymap_sph_deconv,
                                              model_pk2d_covmat,
                                              data_pk2d,
                                              noise_pk2d_covmat,
                                              noise_pk2d_ref,
                                              self.data.mask,
                                              reso,
                                              self.data.psf_fwhm,
                                              self.data.transfer_function,
                                              kedges,
                                              parinfo_noise,
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

        #========== Set the best-fit model
        if set_best_fit:
            best_par = utils.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
            best_model, best_noise = lnlike_setpar_fluct(best_par, self.model,
                                                         parinfo_pk3d,
                                                         parinfo_noise=parinfo_noise)
            
        #========== Set the best-fit model
        if show_fit_result:
            
            k3d, pk3d_best = best_model.get_pressure_fluctuation_spectrum()

            MC_pars = utils.get_emcee_random_param(sampler, burnin=self.mcmc_burnin, Nmc=100)
            pk3d_mc = np.zeros((100, len(k3d)))
            for imc in range(100):
                mc_model, mc_Anoise = lnlike_setpar_fluct(MC_pars[imc,:], self.model, parinfo_pk3d,
                                                          parinfo_noise=parinfo_noise)                
                pk3d_mc[imc,:] = mc_model.get_pressure_fluctuation_spectrum()[1]

            # Plot the Pk3d constraint
            plotlib.show_pk3d_contraint(self.output_dir+'/MCMC_fluctuation_constraint_pk3d.pdf',
                                        k3d, pk3d_best, pk3d_mc)

            # Plot the Pk2d constraint
            #plotlib.show_pk2d_contraint()
            
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

