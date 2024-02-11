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
import pprint
import copy
import emcee
import corner

from minot.ClusterTools.map_tools import radial_profile_sb

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
    - parinfo_profile (dict): see parinfo_profile in get_radial_model_forward_fitting
    - model (class Model object): the model to be updated in the fit
    - parinfo_center (dict): see parinfo_center in get_radial_model_forward_fitting
    - parinfo_ellipticity (dict): see parinfo_ellipticity in get_radial_model_forward_fitting
    - parinfo_ZL (dict): see parinfo_ZL in get_radial_model_forward_fitting

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
    - parinfo_profile (dict): see get_radial_model_forward_fitting
    - parinfo_center (dict): see get_radial_model_forward_fitting
    - parinfo_ellipticity (dict): see get_radial_model_forward_fitting
    - parinfo_ZL (dict): see get_radial_model_forward_fitting

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
    - parinfo_profile (dict): see get_radial_model_forward_fitting
    - parinfo_center (dict): see get_radial_model_forward_fitting
    - parinfo_ellipticity (dict): see get_radial_model_forward_fitting
    - parinfo_ZL (dict): see get_radial_model_forward_fitting

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
    - parinfo_profile (dict): see parinfo_profile in get_radial_model_forward_fitting
    - model (class Model object): the model to be updated
    - data_img (nd array): the image data
    - data_noiseprop (nd_array): the inverse covariance matrix to be used for the fit or the rms,
    depending on use_covmat.
    - data_mask (nd array): the image mask
    - data_psf (quantity): input to model_mock.get_sz_map
    - data_tf (dict): input to model_mock.get_sz_map
    - parinfo_center (dict): see parinfo_center in get_radial_model_forward_fitting
    - parinfo_ellipticity (dict): see parinfo_ellipticity in get_radial_model_forward_fitting
    - parinfo_ZL (dict): see parinfo_ZL in get_radial_model_forward_fitting
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
                 parinfo_fluct,
                 method_fluctuation_image,
                 model,
                 model_ymap_sph,
                 model_ymap_sph_deconv,
                 model_pk2d_covmat_ref,
                 model_pk2d_ref,
                 data_pk2d,
                 noise_pk2d_covmat,
                 noise_pk2d_mean,
                 mask,
                 reso_arcsec,
                 psf_fwhm,
                 transfer_function,
                 kedges=None,
                 parinfo_noise=None,
                 use_covmat=False,
                 scale_model_variance=False):
    """
    This is the likelihood function used for the forward fit of the fluctuation spectrum.
        
    Parameters
    ----------
    - param (np array): the value of the test parameters
    - parinfo_fluct (dict): see parinfo_fluct in get_pk3d_model_forward_fitting
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
                 silent=False,
                 output_dir='./pitszi_output',
                 method_pk2d_extraction='naive',
                 method_fluctuation_image='ratio',
                 method_noise_covmat='model',
                 kbin_min=0*u.arcsec**-1,
                 kbin_max=0.1*u.arcsec**-1,
                 kbin_Nbin=20,
                 kbin_scale='lin',
                 mcmc_nwalkers=100,
                 mcmc_nsteps=500,
                 mcmc_burnin=100,
                 mcmc_reset=False,
                 mcmc_run=True,
                 mcmc_Nresamp=100):
        """
        Initialize the inference object. 
        All parameters can be changed on the fly.
        
        Parameters
        ----------
        - data (pitszi Class Data object): the data
        - model (pitszi Class Model object): the model
        - silent (bool): set to False for printing information
        - output_dir (str): directory where outputs are saved
        - method_pk2d_extraction (str): Pk2d extraction method 
        (implemented: 'naive' or 'arevalo12')
        - method_fluctuation_image (str): fluctuation calculation method 
        (implemented: 'ratio' or 'subtract')
        - method_noise_covmat (str): method to compute noise MC realization 
        (implemented: 'model' or 'covariance')
        - kbin_min (quantity): minimum k value for the 2D power spectrum (homogeneous to angle)
        - kbin_max (quantity): maximal k value for the 2D power spectrum (homogeneous to angle)
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
        self.method_pk2d_extraction   = method_pk2d_extraction   # [naive, arevalo12]
        self.method_fluctuation_image = method_fluctuation_image # [ratio, difference]
        self.method_noise_covmat      = method_noise_covmat      # [covariance, model]

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
    # Compute noise statistics
    #==================================================
    
    def get_pk2d_noise_statistics(self,
                                  kbin_edges=None,
                                  Nmc=1000):
        """
        This function compute the noise properties associated
        with the data noise
        
        Parameters
        ----------
        - kbin_edges (1d array, quantity): array of k edges that can be used instead of default one
        - Nmc (int): number of monte carlo realization

        Outputs
        ----------
        - noise_pk2d_ref (1d array): the noise mean
        - noise_pk2d_covmat (2d array): the noise covariance matrix

        """
        
        #----- Info to user
        if not self.silent:
            print('----- Computing Pk2d noise covariance -----')
        
        #----- Useful info
        if kbin_edges is None:
            kedges = self.get_kedges().to_value('arcsec-1')
        else:
            kedges = kbin_edges.to_value('arcsec-1')
            
        reso = self.model.get_map_reso().to_value('arcsec')

        model_ymap_sph_deconv = self.model.get_sz_map(no_fluctuations=True,
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
            
            if self.method_fluctuation_image == 'ratio':
                image_noise_mc = (noise_ymap_mc[imc,:,:])/model_ymap_sph_deconv
            if self.method_fluctuation_image == 'subtract':
                image_noise_mc = noise_ymap_mc[imc,:,:]
                
            image_noise_mc *= self.data.mask

            k2d, pk_mc =  utils.get_pk2d(image_noise_mc, reso, kedges=kedges)
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
                print('This can be that the number of bins is such that some bins are empty.')
            raise ValueError('Issue with noise covariance matrix')

        return k2d, noise_pk2d_mean, noise_pk2d_covmat


    #==================================================
    # Compute model variance statistics
    #==================================================
    
    def get_pk2d_modelvar_statistics(self,
                                     kbin_edges=None,
                                     Nmc=1000):
        """
        This function compute the model variance properties associated
        with the reference input model.
        
        Parameters
        ----------
        - kbin_edges (1d array quantity): array of k edges that can be used instead of default one
        - Nmc (int): number of monte carlo realization

        Outputs
        ----------
        - model_pk2d_ref (1d array): the model mean
        - model_pk2d_covmat (2d array): the model covariance matrix

        """

        #----- Info to user
        if not self.silent:
            print('----- Computing Pk2d model covariance -----')
        
        #----- Useful info
        if kbin_edges is None:
            kedges = self.get_kedges().to_value('arcsec-1')
        else:
            kedges = kbin_edges.to_value('arcsec-1')
            
        reso = self.model.get_map_reso().to_value('arcsec')

        model_ymap_sph_deconv = self.model.get_sz_map(no_fluctuations=True,
                                                      irfs_convolution_beam=self.data.psf_fwhm)    
        model_ymap_sph        = self.model.get_sz_map(no_fluctuations=True,
                                                      irfs_convolution_beam=self.data.psf_fwhm,
                                                      irfs_convolution_TF=self.data.transfer_function) 

        #----- Compute Pk2d MC realization
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

            k2d, pk_mc =  utils.get_pk2d(test_image, reso, kedges=kedges)
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
                print('This cna be that the number of bins is such that some bins are empty.')
            raise ValueError('Issue with noise covariance matrix')

        return k2d, model_pk2d_mean, model_pk2d_covmat
    
    
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
        utils.chains_statistics(par_chains, lnl_chains,
                                parname=parlist,
                                conf=conf,
                                outfile=self.output_dir+'/MCMC'+extraname+'_chain_statistics.txt')

        # Produce 1D plots of the chains
        plotlib.chains_1Dplots(par_chains, parlist, self.output_dir+'/MCMC'+extraname+'_chain_1d_plot.pdf')
        
        # Produce 1D histogram of the chains
        namefiles = [self.output_dir+'/MCMC'+extraname+'_chain_hist_'+i+'.pdf' for i in parlist]
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
    
    def get_radial_model_forward_fitting(self,
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

        #========== Show results
        if show_fit_result: self.get_radial_model_forward_fitting_results(sampler,
                                                                          parinfo_profile,
                                                                          parinfo_center=parinfo_center,
                                                                          parinfo_ellipticity=parinfo_ellipticity,
                                                                          parinfo_ZL=parinfo_ZL)
            
        #========== Compute the best-fit model and set it
        best_par = utils.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
        self.model, best_ZL = lnlike_setpar_profile(best_par, self.model,
                                                    parinfo_profile,
                                                    parinfo_center=parinfo_center,
                                                    parinfo_ellipticity=parinfo_ellipticity,
                                                    parinfo_ZL=parinfo_ZL)
                    
        return par_list, sampler


    #==================================================
    # Show the fit results related to profile
    #==================================================
    
    def get_radial_model_forward_fitting_results(self,
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
        - sampler (emcee object): the sampler obtained from get_radial_model_forward_fitting
        - parinfo_profile (dict): same as get_radial_model_forward_fitting
        - parinfo_center (dict): same as get_radial_model_forward_fitting
        - parinfo_ellipticity (dict): same as get_radial_model_forward_fitting
        - parinfo_ZL (dict): same as get_radial_model_forward_fitting
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
        best_par = utils.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
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
        
        MC_pars = utils.get_emcee_random_param(sampler, burnin=self.mcmc_burnin, Nmc=self.mcmc_Nresamp)
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
        plotlib.show_fit_result_ymap(self.output_dir+'/MCMC_radial_results_y_map.pdf',
                                     self.data.image,
                                     self.data.header,
                                     noise_mc,
                                     best_ymap_sph,
                                     mask=self.data.mask,
                                     visu_smooth=visu_smooth.to_value('arcsec'))
        
        #========== plots ymap profile
        plotlib.show_fit_ycompton_profile(self.output_dir+'/MCMC_radial_results_y_profile.pdf',
                                          r2d, data_yprof, data_yprof_err,
                                          best_y_profile, MC_y_profile,
                                          true_compton_profile=true_compton_profile)
        
        #========== plots pressure profile
        plotlib.show_fit_result_pressure_profile(self.output_dir+'/MCMC_radial_results_P_profile.pdf',
                                                 r3d, best_pressure_profile, MC_pressure_profile,
                                                 true_pressure_profile=true_pressure_profile)
    
    
    #==================================================
    # Compute the Pk contraint via forward fitting
    #==================================================
    
    def get_fluct_model_forward_fitting(self,
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
        sampler_exist = utils.check_sampler_exist(sampler_file, silent=False)
            
        #========== Defines the fit parameters
        # Fit parameter list and information
        par_list, par0_value, par0_err, par_min, par_max = lnlike_defpar_fluct(parinfo_fluct, self.model,
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
            kedges = kbin_edges.to_value('arcsec-1')
            
        reso = self.model.get_map_reso().to_value('arcsec')
        k2d, data_pk2d = utils.get_pk2d(data_image, reso, kedges=kedges)
        
        #========== Deal with how the noise should be accounted for
        bid, noise_pk2d_ref, noise_pk2d_covmat = self.get_pk2d_noise_statistics(kedges*u.arcsec**-1, Nmc_noise)
        bid, model_pk2d_ref, model_pk2d_covmat = self.get_pk2d_modelvar_statistics(kedges*u.arcsec**-1, Nmc_noise)
        
        #========== Define the MCMC setup
        backend = utils.define_emcee_backend(sampler_file, sampler_exist,
                                             self.mcmc_reset, self.mcmc_nwalkers, ndim, silent=False)
        moves = emcee.moves.KDEMove()
        sampler = emcee.EnsembleSampler(self.mcmc_nwalkers, ndim,
                                        lnlike_fluct,
                                        args=[parinfo_fluct,
                                              self.method_fluctuation_image,
                                              self.model,
                                              model_ymap_sph,
                                              model_ymap_sph_deconv,
                                              model_pk2d_covmat,
                                              model_pk2d_ref,
                                              data_pk2d,
                                              noise_pk2d_covmat,
                                              noise_pk2d_ref,
                                              self.data.mask,
                                              reso,
                                              self.data.psf_fwhm,
                                              self.data.transfer_function,
                                              kedges,
                                              parinfo_noise,
                                              use_covmat,
                                              scale_model_variance],
                                        #pool=Pool(cpu_count()),
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
            
        #========== Set the best-fit model
        if show_fit_result: self.get_fluct_model_forward_fitting_results(sampler,
                                                                         parinfo_fluct,
                                                                         parinfo_noise=parinfo_noise,
                                                                         Nmc=Nmc_noise)
        
        #========== Make sure the model is set to the best fit
        best_par = utils.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
        self.model, best_noise = lnlike_setpar_fluct(best_par, self.model,
                                                     parinfo_fluct,
                                                     parinfo_noise=parinfo_noise)
        
        return par_list, sampler


    #==================================================
    # Show the fit results related to fluctuation
    #==================================================
    
    def get_fluct_model_forward_fitting_results(self,
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
        - sampler (emcee object): the sampler obtained from get_radial_model_forward_fitting

        Outputs
        ----------
        plots are produced

        """

        #========== Recover data and error
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
            kedges = kbin_edges.to_value('arcsec-1')
            
        reso = self.model.get_map_reso().to_value('arcsec')
        k2d, data_pk2d = utils.get_pk2d(data_image, reso, kedges=kedges)

        bid, noise_pk2d_ref, noise_pk2d_covmat = self.get_pk2d_noise_statistics(kedges*u.arcsec**-1, Nmc)
        bid, model_pk2d_ref, model_pk2d_covmat = self.get_pk2d_modelvar_statistics(kedges*u.arcsec**-1, Nmc)
                         
        #========== Get the best-fit
        best_par = utils.get_emcee_bestfit_param(sampler, self.mcmc_burnin)
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
        
        MC_pars = utils.get_emcee_random_param(sampler, burnin=self.mcmc_burnin, Nmc=self.mcmc_Nresamp)
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
            if self.method_fluctuation_image == 'ratio':
                mc_image = (mc_ymap - model_ymap_sph)/model_ymap_sph_deconv
            elif self.method_fluctuation_image == 'subtract':
                mc_image = mc_ymap - model_ymap_sph
            mc_image *= self.data.mask            
            MC_pk2d[imc,:] = utils.get_pk2d(mc_image, reso, kedges=kedges)[1]
            
            # Get MC Pk3d
            MC_pk3d[imc,:] = mc_model.get_pressure_fluctuation_spectrum()[1].to_value('kpc3')
        
        #========== Plot the Pk3d constraint
        plotlib.show_fit_result_pk3d(self.output_dir+'/MCMC_fluctuation_results_pk3d.pdf',
                                    k3d, best_pk3d, MC_pk3d, true_pk3d=true_pk3d)

        #========== Plot the Pk2d constraint
        plotlib.show_fit_result_pk2d(self.output_dir+'/MCMC_fluctuation_results_pk2d.pdf',
                                     k2d, data_pk2d,
                                     model_pk2d_ref,
                                     np.diag(model_pk2d_covmat)**0.5, np.diag(noise_pk2d_covmat)**0.5,
                                     MC_pk2d, MC_pk2d_noise)
        
        
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
    '''
