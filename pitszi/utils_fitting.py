"""
This file contains a library of functions related to fitting and MCMC

"""

import os
import numpy as np
import emcee


#==================================================
# Define the starting point of MCMC with emcee
#==================================================

def emcee_starting_point(guess_value, guess_error, par_min, par_max, nwalkers):
    """
    Sample the parameter space for emcee MCMC starting point
    from a uniform distribution in the parameter space.
        
    Parameters
    ----------
    - guess_value (Nparam array): guess value of the parameters
    - guess_error (float): flat dispersion allowed within this range for unborned parameters
    - parmin (Nparam array): min value of the parameters (i.e. sharp prior)
    - parmax (Nparam array): max value of the parameters (i.e. sharp prior)
    - nwalkers (int): number of walkers

    Output
    ------
    start: the starting point of the chains in the parameter space

    """

    ndim = len(guess_value)

    # First range using guess + dispersion
    vmin = guess_value - guess_error 
    vmax = guess_value + guess_error

    # Check that the parameters are in the prior range
    wup = vmin < np.array(par_min)
    wlo = vmax > np.array(par_max)
    vmin[wup] = np.array(par_min)[wup]
    vmax[wlo] = np.array(par_max)[wlo]

    # Get parameters
    start = [np.random.uniform(low=vmin, high=vmax) for i in range(nwalkers)]
    
    return start


#==================================================
# Check if the sampler exists
#==================================================

def check_sampler_exist(sampler_file, silent=False):
    """
    Check if the sampler already exists
        
    Parameters
    ----------
    - sampler_file (str): the filename of the sampler
    - silent (bool): to show or not info

    Output
    ------
    sampler_exist (bool): true if sampler exist, false otherwise

    """

    sampler_exist = os.path.exists(sampler_file)
    
    if not silent:
        if sampler_exist:
            print('----- Existing sampler:')
            print('      '+sampler_file)
        else:
            print('----- No existing sampler found')

    return sampler_exist


#==================================================
# Define backend
#==================================================

def define_emcee_backend(sampler_file, sampler_exist, mcmc_reset, nwalkers, ndim, silent=False):
    """
    Define the backend used for emcee
        
    Parameters
    ----------
    - sampler_file (str): the filename of the sampler
    - silent (bool): to show or not info

    Output
    ------
    backend (emcee object): the backend to be used for mcmc run

    """

    backend = emcee.backends.HDFBackend(sampler_file)
        
    if not silent: print('----- Does the sampler already exist? -----')
    if sampler_exist:
        if mcmc_reset:
            if not silent: print('      - Yes, but reset the MCMC even though the sampler already exists')
            backend.reset(nwalkers, ndim)
        else:
            if not silent: print('      - Yes, use the existing MCMC sampler')
            if not silent: print("      - Initial size: {0}".format(backend.iteration))
    else:
        print('      - No, start from scratch')
        backend.reset(nwalkers, ndim)
        
    return backend


#==================================================
# Extract best-fit parameters
#==================================================

def get_emcee_bestfit_param(sampler,
                            burnin=0):
    """
    Extract the best fit parameters from the sampler
        
    Parameters
    ----------
    - sampler (emcee sampler): the sampler
    - burnin (int): remove the first samples of the chains

    Output
    ------
    par_best (list): the best_fit parameters

    """

    param_chains = sampler.chain[:, burnin:, :]
    lnL_chains   = sampler.lnprobability[:, burnin:]
    par_flat = param_chains.reshape(param_chains.shape[0]*param_chains.shape[1], param_chains.shape[2])
    lnL_flat = lnL_chains.reshape(lnL_chains.shape[0]*lnL_chains.shape[1])
    wbest = np.where(lnL_flat == np.amax(lnL_flat))[0][0]
    par_best = par_flat[wbest]

    return par_best


#==================================================
# Extract random parameters from the chains
#==================================================

def get_emcee_random_param(sampler,
                           burnin=0,
                           Nmc=100):
    """
    Extract Nmc random set of parameter from the chain
        
    Parameters
    ----------
    - sampler (emcee sampler): the sampler
    - burnin (int): remove the first samples of the chains
    - Nmc (int): number of parameter set to extract

    Output
    ------
    param (list): the parameters grid (Nmc x Nparam)

    """

    param_chains = sampler.chain[:, burnin:, :]
    par_flat = param_chains.reshape(param_chains.shape[0]*param_chains.shape[1], param_chains.shape[2])
    loc = np.random.randint(0, par_flat.shape[0]-1, size=Nmc)
    pars = par_flat[loc,:]
    
    return pars


#==================================================
# Compute chain statistics
#==================================================

def chains_statistics(param_chains,
                      lnL_chains,
                      parname=None,
                      conf=68.0,
                      show=True,
                      outfile=None):
    """
    Get the statistics of the chains, such as maximum likelihood,
    parameters errors, etc.
        
    Parameters
    ----------
    - param_chains (np array): parameters as Nchain x Npar x Nsample
    - lnl_chains (np array): log likelihood values corresponding to the chains
    - parname (list): list of parameter names
    - conf (float): confidence interval in %
    - show (bool): show or not the values
    - outfile (str): full path to file to write results

    Output
    ------
    - par_best (float): best-fit parameter
    - par_percentile (list of float): median, lower bound at CL, upper bound at CL
    
    """

    if show: print('   --- Parameter sampling: chain statistics ---')
    
    if outfile is not None:
        file = open(outfile,'w')
        
    Npar = len(param_chains[0,0,:])

    wbest = (lnL_chains == np.amax(lnL_chains))
    par_best       = np.zeros(Npar)
    par_percentile = np.zeros((3, Npar))
    for ipar in range(Npar):
        # Maximum likelihood
        par_best[ipar]          = param_chains[:,:,ipar][wbest][0]

        # Median and xx % CL
        perc = np.percentile(param_chains[:,:,ipar].flatten(),
                             [(100-conf)/2.0, 50, 100 - (100-conf)/2.0])
        par_percentile[:, ipar] = perc
        if show:
            if parname is not None:
                parnamei = parname[ipar]
            else:
                parnamei = 'no name'

            q = np.diff(perc)
            txt = "{0}_{{-{1}}}^{{{2}}}"
            txt = txt.format(perc[1], q[0], q[1])
            
            medval = str(perc[1])+' -'+str(perc[1]-perc[0])+' +'+str(perc[2]-perc[1])
            bfval = str(par_best[ipar])+' -'+str(par_best[ipar]-perc[0])+' +'+str(perc[2]-par_best[ipar])

            print('param '+str(ipar)+' ('+parnamei+'): ')
            print('   median   = '+medval)
            print('   best-fit = '+bfval)
            print('   '+parnamei+' = '+txt)

            if outfile is not None:
                file.write('param '+str(ipar)+' ('+parnamei+'): '+'\n')
                file.write('  median = '+medval+'\n')
                file.write('  best   = '+bfval+'\n')
                file.write('   '+parnamei+' = '+txt+'\n')

    if outfile is not None:
        file.close() 
            
    return par_best, par_percentile

