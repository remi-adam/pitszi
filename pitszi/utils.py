"""
This file contains utilities for various making calculations used in pitszi

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, fourier_gaussian
import scipy.stats as stats
from scipy.special import gamma
from astropy.coordinates import SkyCoord
import astropy.units as u

from minot.ClusterTools import map_tools

#==================================================
# Check array
#==================================================

def check_qarray(qarr, unit=None):
    """
    Make sure quantity array are arrays

    Parameters
    ----------
    - qarr (quantity): array or float, homogeneous to some unit

    Outputs
    ----------
    - qarr (quantity): quantity array

    """

    if unit is None:
        if type(qarr) == float or type(qarr) == np.float64:
            qarr = np.array([qarr])
            
    else:
        try:
            test = qarr.to(unit)
        except:
            raise TypeError("Unvalid unit for qarr")

        if type(qarr.to_value()) == float or type(qarr.to_value()) == np.float64:
            qarr = np.array([qarr.to_value()]) * qarr.unit

    return qarr


#==================================================
# Def array based on point per decade, min and max
#==================================================

def sampling_array(xmin, xmax, NptPd=10, unit=False):
    """
    Make an array with a given number of point per decade
    from xmin to xmax

    Parameters
    ----------
    - xmin (quantity): min value of array
    - xmax (quantity): max value of array
    - NptPd (int): the number of point per decade

    Outputs
    ----------
    - array (quantity): the array

    """

    if unit:
        my_unit = xmin.unit
        array = np.logspace(np.log10(xmin.to_value(my_unit)),
                            np.log10(xmax.to_value(my_unit)),
                            int(NptPd*(np.log10(xmax.to_value(my_unit)/xmin.to_value(my_unit)))))*my_unit
    else:
        array = np.logspace(np.log10(xmin), np.log10(xmax), int(NptPd*(np.log10(xmax/xmin))))

    return array


#==================================================
# Integration loglog space with trapezoidale rule
#==================================================

def trapz_loglog(y, x, axis=-1, intervals=False):
    """
    Integrate along the given axis using the composite trapezoidal rule in
    loglog space. Integrate y(x) along given axis in loglog space. y can be a function
    with multiple dimension. This follows the script in the Naima package.
    
    Parameters
    ----------
    - y (array_like): Input array to integrate.
    - x (array_like):  optional. Independent variable to integrate over.
    - axis (int): Specify the axis.
    - intervals (bool): Return array of shape x not the total integral, default: False
    
    Returns
    -------
    - trapz (float): Definite integral as approximated by trapezoidal rule in loglog space.
    """
    
    log10 = np.log10
    
    #----- Check for units
    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.0
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.0

    y = np.asanyarray(y)
    x = np.asanyarray(x)

    #----- Define the slices
    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)
    slice1, slice2 = tuple(slice1), tuple(slice2)

    #----- arrays with uncertainties contain objects, remove tiny elements
    if y.dtype == "O":
        from uncertainties.unumpy import log10
        # uncertainties.unumpy.log10 can't deal with tiny values see
        # https://github.com/gammapy/gammapy/issues/687, so we filter out the values
        # here. As the values are so small it doesn't affect the final result.
        # the sqrt is taken to create a margin, because of the later division
        # y[slice2] / y[slice1]
        valid = y > np.sqrt(np.finfo(float).tiny)
        x, y = x[valid], y[valid]

    #----- reshaping x
    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)
        
    #-----
    with np.errstate(invalid="ignore", divide="ignore"):
        # Compute the power law indices in each integration bin
        b = log10(y[slice2] / y[slice1]) / log10(x[slice2] / x[slice1])
        
        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use normal
        # powerlaw integration
        trapzs = np.where(np.abs(b + 1.0) > 1e-10,
                          (y[slice1] * (x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1]))
                          / (b + 1),
                          x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]))
        
    tozero = (y[slice1] == 0.0) + (y[slice2] == 0.0) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.0
    
    if intervals:
        return trapzs * x_unit * y_unit
    
    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit
    
    return ret


#==================================================
# Define the maximum k for having isotropy in 3D
#==================================================

def kmax_isotropic(Nx, Ny, Nz, proj_reso, los_reso):
    """
    Compute the maximum value of k so that we have isotropic sampling.
    I.e., all k values beyond min (kmax_x, kmax_y, kmax_z) are not 
    isotropic.
    
    Parameters
    ----------
    - Nx, Ny, Nz (int): the number of pixel along x, y, z
    - proj_reso (float, kpc): the resolution along the projected dirtection
    - los_reso (float, kpc): the resolution along the line-of-sight direction
    
    Outputs
    ----------
    - kmax (float): the isotropic kmax in kpc^-1
    """

    # check if even or odd and get kmax along each dimension
    if (Nx % 2) == 0:
        kmax_x = 1/(2*proj_reso)
    else:
        kmax_x = (Nx-1)/(2*Nx*proj_reso)

    if (Ny % 2) == 0: 
        kmax_y = 1/(2*proj_reso)
    else:
        kmax_y = (Ny-1)/(2*Ny*proj_reso)

    if (Nz % 2) == 0: 
        kmax_z = 1/(2*los_reso)
    else:
        kmax_z = (Nz-1)/(2*Nz*los_reso)
        
    # Take the min of the kmax along each axis as kmax isotropic
    kmax_iso = np.amin([kmax_x, kmax_y, kmax_z])
        
    return kmax_iso


#==================================================
# Power spectrum attenuation of a Gaussian function
#==================================================

def gaussian_pk(k, FWHM):
    """
    Compute the Gaussian power spectrum
    
    Parameters
    ----------
    - k (np array): array of k, same unit as 1/FWHM
    - FWHM (float): Gaussian FWHM, same unit as 1/k

    Outputs
    ----------
    - G_k (np array): the power spectrum of a gaussian
    """
    
    sigma2fwhm = 2 * np.sqrt(2*np.log(2))

    sigma = FWHM / sigma2fwhm
    k_0 = 1 / (np.sqrt(2) * np.pi * sigma)
    G_k = np.exp(-k**2 / k_0**2)
    
    return G_k


#==================================================
# Apply NIKA-like transfer function
#==================================================
def apply_transfer_function(image, reso, beamFWHM, TF, apps_TF_LS=True, apps_beam=True):
    """
    Convolve SZ image with instrumental transfer function, decomposed in two: beam and processing.
    The beam is applied in Fourier to be better coherent for Pk estimates.
    
    Parameters
    ----------
    - image (np array): input raw image
    - reso (float): map resolution in arcsec
    - beamFWHM (float): the map beam FWHM in units of arcsec
    - TF (dict): dictionary with 'k_arcsec' and 'TF' as keys, i.e.
    two np arrays containing the k in arcsec^-1 and the transfer function
    - apps_TF_LS (bool): set to true to apply the large scale TF filtering
    - apps_beam (bool): set to true to apply beam smoothing
        
    Outputs
    ----------
    - map_filt (2d np array): the convolved map
    """

    FT_map = np.fft.fft2(image)

    # Beam smoothing
    if apps_beam:
        sigma2fwhm = 2 * np.sqrt(2*np.log(2))
        FT_map_sm = fourier_gaussian(FT_map, sigma=beamFWHM/sigma2fwhm/reso)
    else:
        FT_map_sm = FT_map*1 + 0
    
    # TF filtering
    if apps_TF_LS:
        Nx = image.shape[0]
        Ny = image.shape[1]
        k_x = np.fft.fftfreq(Nx, reso)
        k_y = np.fft.fftfreq(Ny, reso)
        k2d_x, k2d_y= np.meshgrid(k_x, k_y, indexing='ij')
        k2d_norm = np.sqrt(k2d_x**2 + k2d_y**2)
        k2d_norm_flat = k2d_norm.flatten()

        # interpolate by putting 1 outside the definition of the TF, i.e. no filtering, e.g. very small scale
        itpl = interp1d(TF['k_arcsec'], TF['TF'], bounds_error=False, fill_value=1)
        filtering_flat = itpl(k2d_norm_flat)
        filtering = np.reshape(filtering_flat, k2d_norm.shape)

        map_filt = np.real(np.fft.ifft2(FT_map_sm * filtering))
    else:
        map_filt = np.real(np.fft.ifft2(FT_map_sm))
        
    return map_filt


#==================================================
# Apply NIKA-like transfer function
#==================================================
def deconv_transfer_function(image, reso, TF):
    """
    Deconvolve SZ image from instrumental transfer function.
    
    Parameters
    ----------
    - image (np array): input raw image
    - reso (float): map resolution in arcsec
    - TF (dict): dictionary with 'k_arcsec' and 'TF' as keys, i.e.
    two np arrays containing the k in arcsec^-1 and the transfer function
        
    Outputs
    ----------
    - map_deconv (2d np array): the deconvolved map
    """

    # FFT the map
    FT_map = np.fft.fft2(image)

    # Get the k arrays
    Nx = image.shape[0]
    Ny = image.shape[1]
    k_x = np.fft.fftfreq(Nx, reso)
    k_y = np.fft.fftfreq(Ny, reso)
    k2d_x, k2d_y= np.meshgrid(k_x, k_y, indexing='ij')
    k2d_norm = np.sqrt(k2d_x**2 + k2d_y**2)
    k2d_norm_flat = k2d_norm.flatten()

    # interpolate by putting 1 outside the definition of the TF, i.e. no filtering, e.g. very small scale
    itpl = interp1d(TF['k_arcsec'], TF['TF'], bounds_error=False, fill_value=1)
    filtering_flat = itpl(k2d_norm_flat)
    filtering = np.reshape(filtering_flat, k2d_norm.shape)
    
    # Check undefined filtering
    condition = (filtering == 0)
    filtering[condition] = 1

    # Deconvolution in Fourier
    FT_map_deconv = FT_map / filtering

    # Set the zero level depending on the TF at k=0
    FT_map_deconv[condition] = 0 # keep the zero level untouched if 0 in TF

    map_deconv = np.real(np.fft.ifft2(FT_map_deconv))

    return map_deconv


#==================================================
# Beam Power spectrum deconvolution
#==================================================

def deconv_pk_beam(k, pk, beamFWHM):
    """
    This function corrects the power spectrum from beam 
    attenuation.
    
    Parameters
    ----------
    - k (nd array): k array in inverse scale unit
    - pk (nd array): power spectrum array
    - beamFWHM (quantity): the beam FWHM homogeneous to the inverse of k

    Outputs
    ----------
    - pk_deconv (np array): the deconvolved pk

    """
    
    Beam_k = utils.gaussian_pk(k, beamFWHM)
    pk_deconv = pk/Beam_k**2

    return pk_deconv


#==================================================
# Transfer function power spectrum deconvolution
#==================================================

def deconv_pk_transfer_function(k, pk, TF_k, TF):
    """
    This function corrects the power spectrum from transfer 
    function attenuation.
    
    Parameters
    ----------
    - k (nd array): k array in the same unit as in the TF_k
    - pk (nd array): power spectrum array
    - TF_k (nd array): k associated with the transfer function (same unit as k)
    - TF (nd array): filtering (same lenght as TF_k)

    Outputs
    ----------
    - pk_deconv (np array): the deconvolved pk

    """

    itpl = interp1d(TF_k, TF, fill_value=1)
    TF_kpc = itpl(k)
    
    pk_deconv = pk/TF_kpc**2
    
    return pk_deconv


#==================================================
# Define the starting point of MCMC with emcee
#==================================================

def emcee_starting_point(guess, disp, par_min, par_max, nwalkers):
    """
    Sample the parameter space for emcee MCMC starting point
    from a uniform distribution in the parameter space.
        
    Parameters
    ----------
    - guess (Nparam array): guess value of the parameters
    - disp (float): dispersion allowed for unborned parameters
    - parmin (Nparam array): min value of the parameters
    - parmax (Nparam array): max value of the parameters
    - nwalkers (int): number of walkers

    Output
    ------
    start: the starting point of the chains in the parameter space

    """

    ndim = len(guess)

    # First range using guess + dispersion
    vmin = guess - guess*disp 
    vmax = guess + guess*disp

    # If parameters are 0, uses born
    w0 = np.where(guess < 1e-4)[0]
    for i in range(len(w0)):
        if np.array(par_min)[w0[i]] != np.inf and np.array(par_min)[w0[i]] != -np.inf:
            vmin[w0[i]] = np.array(par_min)[w0[i]]
            vmax[w0[i]] = np.array(par_max)[w0[i]]
        else:
            vmin[w0[i]] = -1.0
            vmax[w0[i]] = +1.0
            print('Warning: some starting point parameters are difficult to estimate')
            print('because the guess parameter is 0 and the par_min/max are infinity')

    # Check that the parameters are in the prior range
    wup = vmin < np.array(par_min)
    wlo = vmax > np.array(par_max)
    vmin[wup] = np.array(par_min)[wup]
    vmax[wlo] = np.array(par_max)[wlo]

    # Get parameters
    start = [np.random.uniform(low=vmin, high=vmax) for i in range(nwalkers)]
    
    return start


#==================================================
# Measure the 3D power spectrum naively
#==================================================

def get_pk3d(cube, proj_reso, los_reso,
             Nbin=100, scalebin='lin', kmin=None, kmax=None, kedges=None,
             statistic='mean',
             apply_volume=False):
    """
    Measure the power spectrum in 3 dimensions in k bins.
    The unit of the k array is the inverse of the resolution
    The unit of Pk is that of the resolution element cubed.
    
    Parameters
    ----------
    - cube (np array): 3d data cube as nd array
    - proj_reso (float): the resolution along the projected direction
    - los_reso (float) the resolution along the line of sight
    - Nbin (int): the number of bin for output Pk
    - scalebin (str): lin or log, the way the Pk is binned along k
    - kmin/max (float): the min and max k to use in defining the bins
    - kedges (1d np array): directly provide the bin edges in a Nbin+1 array (in this case,
    - statistic (str or function): the statistics to be used in binned_statistic
    - apply_volume (bool): set True to multiply each bin by 4 pi delta_k^3 volume
        
    Outputs
    ----------
    - kvals (np array): values of k binned
    - Pk_bins (np array): values of amplitude binned
    """
    
    # Get the number of pixels
    Nx, Ny, Nz = cube.shape

    # Define the k_i and k_norm
    k_x = np.fft.fftfreq(Nx, proj_reso)
    k_y = np.fft.fftfreq(Ny, proj_reso)
    k_z = np.fft.fftfreq(Nz, los_reso)
    k3d_x, k3d_y, k3d_z = np.meshgrid(k_x, k_y, k_z, indexing='ij')
    k3d_norm = np.sqrt(k3d_x**2 + k3d_y**2 + k3d_z**2)
    
    # Compute the Pk cube
    fourier_cube = np.fft.fftn(cube)
    fourier_pk = np.abs(fourier_cube)**2

    # Get the flattened k, Pk
    knrm = k3d_norm.flatten()
    fourier_pk = fourier_pk.flatten()

    # Define the bins
    if kmin is None:
        kmin_sampling = np.amin(k3d_norm[k3d_norm > 0])
    else:
        kmin_sampling = kmin
    if kmax is None:
        kmax_sampling = np.amax(k3d_norm)
    else:
        kmax_sampling = kmax        
        
    if scalebin is 'lin':
        kbins = np.linspace(kmin_sampling, kmax_sampling, Nbin+1)
    elif scalebin is 'log':
        kbins = np.logspace(np.log10(kmin_sampling), np.log10(kmax_sampling), Nbin+1)
    else:
        raise ValueError("Only lin or log scales are allowed")

    if kedges is None:
        kbins = kbins
    else:
        kbins = kedges
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    # Bin the Pk
    Pk_bins, _, _ = stats.binned_statistic(knrm, fourier_pk,statistic=statistic, bins=kbins)
    Pk_bins *= (proj_reso*proj_reso*los_reso) / (Nx*Ny*Nz)
    
    # Apply volume if needed
    if apply_volume: 
        Pk_bins *= 4*np.pi * (kbins[1:]**3 - kbins[:-1]**3)
    
    return kvals, Pk_bins


#==================================================
# Measure the 3D power spectrum naively
#==================================================

def get_pk2d(image, proj_reso,
             Nbin=100, scalebin='lin', kmin=None, kmax=None, kedges=None,
             statistic='mean',
             apply_volume=False):
    """
    Measure the power spectrum in 2 dimensions in k bins.
    The unit of the k array is the inverse of the resolution
    The unit of Pk is that of the resolution element squared.
    
    Parameters
    ----------
    - image (np array): 2d data cube as nd array
    - proj_reso (float): the resolution along the projected direction (any unit, e.g. kpc, arcsec)
    - Nbin (int): the number of bin for output Pk
    - scalebin (str): lin or log, the way the Pk is binned along k
    - kmin/max (float): the min and max k to use in defining the bins
    - kedges (1d np array): directly provide the bin edges in a Nbin+1 array (in this case,
      Nbin, kmin/kmax and scalebin are ignored)
    - statistic (str or function): the statistics to be used in binned_statistic
    - apply_volume (bool): set True to multiply each bin by 2 pi delta_k^2 volume
        
    Outputs
    ----------
    - kvals (np array): values of k binned
    - Pk_bins (np array): values of amplitude binned
    """
    
    # Get the number of pixels
    Nx, Ny = image.shape

    # Define the k_i and k_norm
    k_x = np.fft.fftfreq(Nx, proj_reso)
    k_y = np.fft.fftfreq(Ny, proj_reso)
    k2d_x, k2d_y = np.meshgrid(k_x, k_y, indexing='ij')
    k2d_norm = np.sqrt(k2d_x**2 + k2d_y**2)
    
    # Compute the Pk amplitide
    fourier_img = np.fft.fftn(image)
    fourier_pk = np.abs(fourier_img)**2 # img unit squared

    # Get the flattened k, Pk
    knrm = k2d_norm.flatten()
    fourier_pk = fourier_pk.flatten()
    
    # Define the bins
    if kmin is None:
        kmin_sampling = np.amin(k2d_norm[k2d_norm > 0])
    else:
        kmin_sampling = kmin
    if kmax is None:
        kmax_sampling = np.amax(k2d_norm)
    else:
        kmax_sampling = kmax
        
    if scalebin is 'lin':
        kbins = np.linspace(kmin_sampling, kmax_sampling, Nbin+1)
    elif scalebin is 'log':
        kbins = np.logspace(np.log10(kmin_sampling), np.log10(kmax_sampling), Nbin+1)
    else:
        raise ValueError("Only lin or log scales are allowed")

    if kedges is None:
        kbins = kbins
    else:
        kbins = kedges
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    # Bin the Pk
    Pk_bins, _, _ = stats.binned_statistic(knrm, fourier_pk, statistic=statistic, bins=kbins)
    Pk_bins *= (proj_reso*proj_reso) / (Nx*Ny) # img unit squared x reso unit squared
    
    # Apply volume if needed
    if apply_volume: 
        Pk_bins *= 2*np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    return kvals, Pk_bins


#==================================================
# Measure the 3D power spectrum naively
#==================================================

def get_pk2d_arevalo(image, proj_reso,
                     kctr=None,
                     Nbin=100, scalebin='lin', kmin=None, kmax=None, kedges=None,
                     epsilon=1e-3, mask=None, unbias=False):
    """
    Implement the method of Arevalo et al. (2012) to compute the power spectrum.
    Warning, the method is biased for steep spectra, which is the case 
    near the beam cutoff. The unbiais keyword attempt to unbias the spectrum
    from itself.

    The unit of output k is the same as the inverse of the resolution, and that of Pk 
    is the same as resolution^2.
    
    See also Romero et al. (2023) for the bias estimate due to the PSF.

    Parameters
    ----------
    - image (np array): 2d data cube as nd array
    - proj_reso (float): the resolution along the projected direction (any unit, e.g. kpc, arcsec)
    - kctr (nd array): the k values at which to compute Pk. If given, Nbin, scalebin, kmin/max,
      and kedges are irrelevant.
    - Nbin (int): the number of bin for output Pk
    - scalebin (str): lin or log, the way the Pk is binned along k
    - kmin/max (float): the min and max k to use in defining the bins
    - kedges (1d np array): directly provide the bin edges in a Nbin+1 array (in this case,
      Nbin, kmin/kmax and scalebin are ignored)
    - epsilon (float): parameter of the sigma calculation
    - mask (nd array): same size as image, but with 0 or 1 to mask bad pixels
    - unbias (bool): set to true to unbias the spectrum given the estimated power law bias
    using the first guess measured spectrum as the input

    Outputs
    ----------
    - k2d (np array): values of k2d
    - Pk (np array): values of spectrum at k2d
    """

    #----- Possibility to define the bins in different ways        
    if kctr is None:    # The user does not gives the central k values
        # First need the k grid
        k_x = np.fft.fftfreq(image.shape[0], proj_reso)
        k_y = np.fft.fftfreq(image.shape[1], proj_reso)
        k2d_x, k2d_y = np.meshgrid(k_x, k_y, indexing='ij')
        k2d_norm = np.sqrt(k2d_x**2 + k2d_y**2)

        # Define the kmin/max from the grid or the user
        if kmin is None:
            kmin_sampling = np.amin(k2d_norm[k2d_norm > 0])
        else:
            kmin_sampling = kmin
        if kmax is None:
            kmax_sampling = np.amax(k2d_norm)
        else:
            kmax_sampling = kmax

        # get the bins
        if scalebin is 'lin':
            kbins = np.linspace(kmin_sampling, kmax_sampling, Nbin+1)
        elif scalebin is 'log':
            kbins = np.logspace(np.log10(kmin_sampling), np.log10(kmax_sampling), Nbin+1)
        else:
            raise ValueError("Only lin or log scales are allowed")

        # Defines the bin accounting for possible user defined bins
        if kedges is None:
            kbins = kbins
        else:
            kbins = kedges

        # Get the central values
        k2d = 0.5 * (kbins[1:] + kbins[:-1])

    else:           # The user directly gives the central k values
        k2d = kctr

    #----- Make sure the image is masked
    if mask is not None:
        image = image*mask
        
    #----- Compute the spectrum
    sigma2fwhm = 2 * np.sqrt(2*np.log(2))
    sigma_list = 1 / (k2d * np.sqrt(2 * np.pi**2))

    Pk=[]
    for sigma in sigma_list:
        k_i = 1 / (sigma * np.sqrt(2 * np.pi**2))
        sigma1 = sigma / np.sqrt(1+epsilon)
        sigma2 = sigma * np.sqrt(1+epsilon)
        img_conv1 = apply_transfer_function(image, proj_reso, sigma2fwhm*sigma1, 0, apps_TF_LS=False)
        img_conv2 = apply_transfer_function(image, proj_reso, sigma2fwhm*sigma2, 0, apps_TF_LS=False)

        if mask is not None:
            mask_conv1 = apply_transfer_function(mask, proj_reso, sigma2fwhm*sigma1, 0, apps_TF_LS=False)
            mask_conv2 = apply_transfer_function(mask, proj_reso, sigma2fwhm*sigma2, 0, apps_TF_LS=False)
            wbad1 = (mask_conv1 == 0) 
            wbad2 = (mask_conv2 == 0)
            mask_conv1[wbad1] = np.nan
            mask_conv2[wbad2] = np.nan
            img_conv = (img_conv1/mask_conv1 - img_conv2/mask_conv2) * mask
            img_conv[wbad1] = 0
            img_conv[wbad2] = 0
            var = np.sum(img_conv**2) / img_conv.size * (mask.size / np.sum(mask))
        else:
            img_conv = (img_conv1 - img_conv2)
            var = np.sum(img_conv**2) / img_conv.size
            
        spec_i = var / epsilon**2 / np.pi / k_i**2
        Pk.append(spec_i)
        
    Pk = np.array(Pk)

    #----- Attempt to unbias the spectrum
    if unbias:
        der = -np.gradient(np.log10(Pk), np.log10(k2d))
        bias = 2**(der/2) * gamma(3 - der/2) / gamma(3)
        Pk = Pk/bias

    return k2d, Pk
