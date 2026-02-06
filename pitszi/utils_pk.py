"""
This file contains a library of functions related to power spectra estimates

"""

import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, fourier_gaussian
import scipy.stats as stats
from scipy.special import gamma
import astropy.units as u
from sklearn.neighbors import NearestNeighbors


#==================================================
# 2D pixel window function
#==================================================

def pixel_wf_mn(kx2d, ky2d, reso):
    """
    Compute the pixel window function for each k in 2D
    
    Parameters
    ----------
    - kx2d (2d np array): array of k along x (inverse unit of reso)
    - ky2d (2d np array): array of k along y (inverse unit of reso)
    - reso (float): the pixel size in inverse unit of k

    Outputs
    ----------
    - pix_mn (np array): the pixel window function for each k
    """

    term1 = np.sinc(kx2d*reso)
    term2 = np.sinc(ky2d*reso)
    pix_mn = term1 * term2
    
    return pix_mn


#==================================================
# 2D beam window function
#==================================================

def beam_wf_mn(kx2d, ky2d, FWHM):
    """
    Compute the beam window function for each k in 2D
    
    Parameters
    ----------
    - kx2d (2d np array): array of k along x (inverse unit of FWHM)
    - ky2d (2d np array): array of k along y (inverse unit of FWHM)
    - FWHM (float): the beam FWHM (same unit as 1/k)

    Outputs
    ----------
    - beam_mn (np array): the beam window function for each k in 2D
    """

    sigma2fwhm = 2 * np.sqrt(2*np.log(2))
    
    sigma = FWHM / sigma2fwhm
    k_0 = 1 / (np.sqrt(2) * np.pi * sigma)
    G_k = np.exp(-(kx2d**2 + ky2d**2) / k_0**2)
    
    return G_k


#==================================================
# 1D Power spectrum attenuation of a Gaussian beam
#==================================================

def beam_wf_pk(k, FWHM):
    """
    Compute the Gaussian beam window function power spectrum 
    (same as beam_wf_mn but with norm k as input, in 1D)
    
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
    - TF (dict): dictionary with 'k' (unit homogeneous to 1/arcsec) 
    and 'TF' as keys, i.e. two np arrays containing the k and the 
    transfer function
    - apps_TF_LS (bool): set to true to apply the large scale TF filtering
    - apps_beam (bool): set to true to apply beam smoothing
        
    Outputs
    ----------
    - map_filt (2d np array): the convolved map
    """
    
    FT_map = np.fft.fft2(image)

    # Beam smoothing
    if apps_beam and beamFWHM>0:
        sigma2fwhm = 2 * np.sqrt(2*np.log(2))
        FT_map_sm = fourier_gaussian(FT_map, sigma=beamFWHM/sigma2fwhm/reso)
    else:
        FT_map_sm = FT_map*1 + 0
    
    # TF filtering
    if apps_TF_LS:
        Ny, Nx = image.shape
        k_x = np.fft.fftfreq(Nx, reso)
        k_y = np.fft.fftfreq(Ny, reso)
        k2d_x, k2d_y = np.meshgrid(k_x, k_y, indexing='xy')
        k2d_norm = np.sqrt(k2d_x**2 + k2d_y**2)
        k2d_norm_flat = k2d_norm.flatten()

        # interpolate by putting 1 outside the definition of the TF, i.e. no filtering, e.g. very small scale
        itpl = interp1d(TF['k'].to_value('arcsec-1'), TF['TF'], bounds_error=False, fill_value=(0,1))
        filtering_flat = itpl(k2d_norm_flat)
        filtering = np.reshape(filtering_flat, k2d_norm.shape)

        map_filt = np.real(np.fft.ifft2(FT_map_sm * filtering))
    else:
        map_filt = np.real(np.fft.ifft2(FT_map_sm))
        
    return map_filt


#==================================================
# Deconv NIKA-like transfer function
#==================================================

def deconv_transfer_function(image, reso, beamFWHM, TF, dec_TF_LS=True, dec_beam=True):
    """
    Deconvolve SZ image from instrumental transfer function.
    
    Parameters
    ----------
    - image (np array): input raw image
    - reso (float): map resolution in arcsec
    - beamFWHM (float): the map beam FWHM in units of arcsec
    - TF (dict): dictionary with 'k' (unit homogeneous to 1/arcsec) 
    and 'TF' as keys, i.e. two np arrays containing the k and the 
    transfer function
    - dec_TF_LS (bool): set to true to deconvol the large scale TF filtering
    - dec_beam (bool): set to true to deconvol beam smoothing
        
    Outputs
    ----------
    - map_deconv (2d np array): the deconvolved map
    """

    # FFT the map
    FT_map = np.fft.fft2(image)

    # Get the k arrays
    Ny, Nx = image.shape
    k_x = np.fft.fftfreq(Nx, reso)
    k_y = np.fft.fftfreq(Ny, reso)
    k2d_x, k2d_y= np.meshgrid(k_x, k_y, indexing='xy')
    k2d_norm = np.sqrt(k2d_x**2 + k2d_y**2)
    k2d_norm_flat = k2d_norm.flatten()

    #===== Beam smoothing
    if dec_beam:
        beam_mn = beam_wf_mn(k2d_x, k2d_y, beamFWHM)
        FT_map_B = FT_map / beam_mn

    else:
        FT_map_B = FT_map*1+0
            
    #===== TF filtering
    if dec_TF_LS:
        # interpolate by putting 1 outside the definition of the TF, i.e. no filtering, e.g. very small scale
        itpl = interp1d(TF['k'].to_value('arcsec-1'), TF['TF'], bounds_error=False, fill_value=(0,1))
        filtering_flat = itpl(k2d_norm_flat)
        filtering = np.reshape(filtering_flat, k2d_norm.shape)
    
        # Check undefined filtering
        condition = (filtering == 0)
        filtering[condition] = 1

        # Deconvolution in Fourier
        FT_map_TB = FT_map_B / filtering
        
        # Set the zero level depending on the TF at k=0
        FT_map_TB[condition] = 0 # keep the zero level untouched if 0 in TF

    else:
        FT_map_TB = FT_map_B*1+0

    #===== Back to real space
    map_deconv = np.real(np.fft.ifft2(FT_map_TB))

    return map_deconv


#==================================================
# Beam Power spectrum convolution
#==================================================

def apply_pk_beam(k, pk, beamFWHM1, beamFWHM2=None):
    """
    This function apply the power spectrum from beam 
    attenuation.
    
    Parameters
    ----------
    - k (nd array): k array in inverse scale unit
    - pk (nd array): power spectrum array
    - beamFWHM1 (quantity): the beam FWHM homogeneous to the inverse of k
    - beamFWHM2 (quantity): the second beam FWHM in case pk is a cross between 2 maps with different beam

    Outputs
    ----------
    - pk_conv (np array): the convolved pk

    """

    # Beam 1
    if beamFWHM1 > 0:
        Beam_k1 = beam_wf_pk(k, beamFWHM1)
    else:
        Beam_k1 = 1

    # Beam 2
    if beamFWHM2 != None:
        if beamFWHM2 > 0:
            Beam_k2 = beam_wf_pk(k, beamFWHM2)
        else:
            Beam_k2 = 1
    else:
        Beam_k2 = Beam_k1

    # Application
    pk_conv = pk*Beam_k1*Beam_k2

    return pk_conv


#==================================================
# Beam Power spectrum deconvolution
#==================================================

def deconv_pk_beam(k, pk, beamFWHM1, beamFWHM2=None):
    """
    This function corrects the power spectrum from beam 
    attenuation.
    
    Parameters
    ----------
    - k (nd array): k array in inverse scale unit
    - pk (nd array): power spectrum array
    - beamFWHM1 (quantity): the beam FWHM homogeneous to the inverse of k
    - beamFWHM2 (quantity): the second beam FWHM in case pk is a cross between 2 maps with different beam

    Outputs
    ----------
    - pk_deconv (np array): the deconvolved pk

    """

    # Beam 1
    if beamFWHM1 > 0:
        Beam_k1 = beam_wf_pk(k, beamFWHM1)
    else:
        Beam_k1 = 1
        
    # Beam 2
    if beamFWHM2 != None:
        if beamFWHM2 > 0:
            Beam_k2 = beam_wf_pk(k, beamFWHM2)
        else:
            Beam_k2 = 1
    else:
        Beam_k2 = Beam_k1

    # Application
    pk_deconv = pk/Beam_k1/Beam_k2

    return pk_deconv


#==================================================
# Transfer function power spectrum convolution
#==================================================

def apply_pk_transfer_function(k, pk, TF_k1, TF1,
                               TF_k2=None, TF2=None):
    """
    This function apply the power spectrum from transfer 
    function attenuation.
    
    Parameters
    ----------
    - k (nd array): k array in the same unit as in the TF_k
    - pk (nd array): power spectrum array
    - TF_k1 (nd array): k associated with the transfer function (same unit as k)
    - TF1 (nd array): filtering (same lenght as TF_k)
    - TF_k2 (nd array): same as TF_k1 in case of a second map with different TF
    - TF2 (nd array): same as TF1 in case of a second map with different TF

    Outputs
    ----------
    - pk_conv (np array): the convolved pk

    """

    # TF1
    itpl1 = interp1d(TF_k1, TF1, bounds_error=False, fill_value=(0,1))
    TF_kpc1 = itpl1(k)

    # TF2
    if TF_k2 is not None and TF2 is not None:
        itpl2 = interp1d(TF_k2, TF2, bounds_error=False, fill_value=(0,1))
        TF_kpc2 = itpl2(k)
    else:
        TF_kpc2 = TF_kpc1
    
    # Application
    pk_conv = pk*TF_kpc1*TF_kpc2
    
    return pk_conv


#==================================================
# Transfer function power spectrum deconvolution
#==================================================

def deconv_pk_transfer_function(k, pk, TF_k1, TF1,
                                TF_k2=None, TF2=None):
    """
    This function corrects the power spectrum from transfer 
    function attenuation.
    
    Parameters
    ----------
    - k (nd array): k array in the same unit as in the TF_k
    - pk (nd array): power spectrum array
    - TF_k1 (nd array): k associated with the transfer function (same unit as k)
    - TF1 (nd array): filtering (same lenght as TF_k)
    - TF_k2 (nd array): same as TF_k1 in case of a second map with different TF
    - TF2 (nd array): same as TF1 in case of a second map with different TF

    Outputs
    ----------
    - pk_deconv (np array): the deconvolved pk

    """

    # TF1
    itpl1 = interp1d(TF_k1, TF1, bounds_error=False, fill_value=(0,1))
    TF_kpc1 = itpl1(k)

    # TF2
    if TF_k2 is not None and TF2 is not None:
        itpl2 = interp1d(TF_k2, TF2, bounds_error=False, fill_value=(0,1))
        TF_kpc2 = itpl2(k)
    else:
        TF_kpc2 = TF_kpc1
    
    # Application
    pk_deconv = pk/TF_kpc1/TF_kpc2
    
    return pk_deconv


#==================================================
# Compute the matrix K used in Mbb calculations
#==================================================

def compute_Kmnmn(W):
    """
    Compute the matrix K_m,n,mp,np from Ponthieu et al. (2011)
    See Eq 10 and Appendix A.
    This gives accounts for masking/weighting effects in Fourier.
    Could certainly be optimized.

    Parameters
    ----------
    - W (complex 2d array): the FFT of the wight map

    Outputs
    ----------
    - K (np 4D array): the matrix K

    """
    
    Ny, Nx = W.shape
    K = np.zeros((Ny, Nx, Ny, Nx), dtype=complex)  # m, n, m1, n1
    
    for n1 in range(Nx):
        for m1 in range(Ny):
            for n in range(Nx):
                for m in range(Ny):
                    if m1 <= m and n1 <= n:
                        K[m, n, m1, n1] = W[m - m1, n - n1,]
                    elif m1 <= m and n1 > n:
                        K[m, n, m1, n1] = W[m - m1, Nx + n - n1]
                    elif m1 > m and n1 <= n:
                        K[m, n,  m1, n1] = W[Ny + m - m1, n - n1]
                    elif m1 > m and n1 > n:
                        K[m, n, m1, n1] = W[Ny + m - m1, Nx + n - n1]
    
    return K


#==================================================
# Compute the bining operators R,Q from POKER
#==================================================

def compute_RQ(k2d, kedge, beta=0):
    """
    Compute the binning operators R and Q from Ponthieu et al. (2011)
    See Eq 11,12

    Parameters
    ----------
    - k2d (complex 2d array): the norm of k on the 2d grid
    - kedge (1D array): the edges of the bins
    - beta (float): bin k^beta Pk

    Outputs
    ----------
    - R (np array): the bining operator
    - Q (np array): the inverse bining operator

    """
       
    Ny, Nx = k2d.shape
    Nb = len(kedge)-1
    R = np.zeros((Nb, Ny, Nx))
    Q = np.zeros((Nb, Ny, Nx))
    
    for b in range(Nb):
        wbin = (k2d >= kedge[b]) * (k2d < kedge[b+1])
        sigma = np.sum(wbin) # Number of kmn falling in bin
        for n in range(Nx):
            for m in range(Ny):
                cond = (kedge[b] <= k2d[m,n]) and (kedge[b+1] > k2d[m,n])
                if cond:
                    if beta == 0:
                        R[b, m,n] = 1.0 / sigma 
                        Q[b, m,n] = 1.0
                    else:
                        R[b, m,n] = k2d[m,n]**beta / sigma
                        Q[b, m,n] = 1/k2d[m,n]**beta
                else:
                    R[b, m,n] = 0.0
                    Q[b, m,n] = 0.0
                    
    return R, Q


#==================================================
# Compute the bin-to-bin mixing matrix
#==================================================

def compute_Mbb(R, K, Q):
    """
    Compute the bin-to-bin mixing matrix Mbb from Ponthieu et al. (2011)
    See Eq 16

    Parameters
    ----------
    - R (nd array): the bining operator
    - K (nd array): the K_m,n,mp,np array
    - Q (nd array): the inverse bining operator

    Outputs
    ----------
    - Mbb (2d array): the bin to bin mixing matrix

    """
    
    Nb, Ny, Nx = R.shape
    Mbb = np.zeros((Nb, Nb))
    for ib in range(Nb):
        for jb in range(Nb):            
                sum_KQ  = np.sum(np.abs(K)**2 * (Q[jb, :,:])[np.newaxis, np.newaxis, :, :], axis=(2,3))
                sum_RKQ = np.sum(R[ib, :,:] * sum_KQ)
                Mbb[ib, jb] = sum_RKQ
                
    return Mbb/(Nx*Ny)**2


#==================================================
# Apply K matrix to 2d power spectra
#==================================================

def multiply_Kmnmn(K, T):
    """
    Application of K onto 2D power spectra T.
    Eq. A.5 from Ponthieu+2011

    Parameters
    ----------
    - K (complex 2d array): output of 'compute_Kmnmn'
    - T (2d array): the input model for the power spectrum on a 2d grid

    Outputs
    ----------
    - KxT (np 2D array): the power spectrum in 2D convolved with K

    """

    Ny, Nx = T.shape
    KxT = np.zeros((Ny, Nx), dtype=complex)  # m,n
    
    for n in range(Nx):
        for m in range(Ny):
            KxT[m, n] = np.sum(K[m, n, :, :] * T) 
        
    return KxT / Ny/Nx


def multiply_Kmnmn_bis(K, T):
    """
    Same as multiply_Kmnmn but avoid loop.
    Application of K onto 2D power spectra T.
    Eq. A.5 from Ponthieu+2011

    Parameters
    ----------
    - K (complex 2d array): output of 'compute_Kmnmn'
    - T (2d array): the input model for the power spectrum on a 2d grid

    Outputs
    ----------
    - KxT (np 2D array): the power spectrum in 2D convolved with K

    """

    Ny, Nx = T.shape
    KxT = np.zeros((Ny, Nx), dtype=complex)  # m,n
    T_reshaped = np.tile(T, (Ny, Nx, 1, 1))
    KxT = np.sum(K * T_reshaped, axis=(2, 3))
        
    return KxT / Nx/Ny


#==================================================
# Measure the 3D power spectrum naively
#==================================================

def convert_pkln_to_pkgauss(Pk_r, reso_x, reso_y, reso_z):
    """
    Let Pk_r be the power spectrum of the simulated lognormal field r
    and Pk_s be the power spectrum of the gaussian field s that is
    used to generate the lognormal field as
    field_ln = exp(s)
    This function convert Pk_r to the Pk_s
    Note that the field mean is zero here and should be defined a 
    posteriori in real space (see e.g. get_pressure_cube_fluctuation in 
    model_mock.py).
    See M. Greiner and T. A. Enßlin, A&A 574, A86 (2015)
        
    Parameters
    ----------
    - Pk_r (3d np array): lognormal field powerspectrum
    - reso_x (float): the resolution along the x axis
    - reso_y (float): the resolution along the y axis
    - reso_z (float): the resolution along the z axis

    Outputs
    ----------
    - Pk_s (np array): Pk followed by the gaussian field 
    """
    
    # Get sampling info
    Nz, Ny, Nx = Pk_r.shape
    Vx = reso_x * reso_y * reso_z
    Vk = 1 / (Nx*Ny*Nz*Vx)

    k_x = np.fft.fftfreq(Nx, reso_x) # 1/kpc
    k_y = np.fft.fftfreq(Ny, reso_y)
    k_z = np.fft.fftfreq(Nz, reso_z)
    k3d_z, k3d_y, k3d_x = np.meshgrid(k_z, k_y, k_x, indexing='ij')
    knorm = (k3d_z**2 + k3d_y**2 + k3d_x**2)**0.5

    # Apply the convertion
    I    = np.abs(np.fft.fftn(Pk_r)*Vk + 1)    
    Pk_s = np.abs(np.fft.ifftn(np.log(I))*Vx*Nx*Ny*Nz)
    Pk_s[knorm == 0] = 0
    
    return Pk_s


#==================================================
# Decompose an image into periodic + smooth
#==================================================

def periodic_smooth_decomp(I):
    """
    Decompose the image into a periodic plus smooth 
    component. This is based on:
    Periodic Plus Smooth Image Decomposition
    Moisan, L. J Math Imaging Vis (2011) 39: 161.
    doi.org/10.1007/s10851-010-0227-1
    Original code from https://github.com/jacobkimmel/ps_decomp/blob/master/psd.py
    
    Parameters
    ----------
    - I (np array): 2d data image

    Outputs
    ----------
    - p (np array): periodic component of the image
    - s_f (np array): smooth component
    """
    # Get the image zeroed expect for the outermost rows and cols
    def u2v(u):
        v = np.zeros(u.shape, dtype=np.float64)
        v[0, :] = np.subtract(u[-1, :], u[0,  :], dtype=np.float64)
        v[-1,:] = np.subtract(u[0,  :], u[-1, :], dtype=np.float64)
        v[:,  0] += np.subtract(u[:, -1], u[:,  0], dtype=np.float64)
        v[:, -1] += np.subtract(u[:,  0], u[:, -1], dtype=np.float64)
        return v
    # Computes the maximally smooth component
    def v2s(v_hat):
        M, N = v_hat.shape
        q = np.arange(M).reshape(M, 1).astype(v_hat.dtype)
        r = np.arange(N).reshape(1, N).astype(v_hat.dtype)
        den = (2*np.cos( np.divide((2*np.pi*q), M) ) \
             + 2*np.cos( np.divide((2*np.pi*r), N) ) - 4)
        s = np.divide(v_hat, den, out=np.zeros_like(v_hat), where=den!=0)
        s[0, 0] = 0
        return s
    
    u = I.astype(np.float64)
    v = u2v(u)
    v_fft = np.fft.fftn(v)
    s = v2s(v_fft)
    s_i = np.fft.ifftn(s)
    s_f = np.real(s_i)
    p = u - s_f
    return p, s_f


#==================================================
# Measure the 3D power spectrum naively
#==================================================

def extract_pk3d(cube, proj_reso, los_reso,
                 cube2=None,
                 Nbin=100, scalebin='lin',
                 kmin=None, kmax=None, kedges=None,
                 statistic='mean'):
    """
    Measure the power spectrum in 3 dimensions in k bins.
    The unit of the k array is the inverse of the resolution
    The unit of Pk is that of the resolution element cubed.
    
    Parameters
    ----------
    - cube (np array): 3d data cube as nd array
    - proj_reso (float): the resolution along the projected direction
    - los_reso (float) the resolution along the line of sight
    - cube2 (np array): same as cube. Use to do cross spectra
    - Nbin (int): the number of bin for output Pk. If Nbin is None, then no 
    binning is done
    - scalebin (str): lin or log, the way the Pk is binned along k
    - kmin/max (float): the min and max k to use in defining the bins
    - kedges (1d np array): directly provide the bin edges in a Nbin+1 array (in this case,
    - statistic (str or function): the statistics to be used in binned_statistic
        
    Outputs
    ----------
    - kvals (np array): values of k binned
    - Pk_bins (np array): values of amplitude binned
    """
    
    # Get the number of pixels
    Nz, Ny, Nx = cube.shape

    # Define the k_i and k_norm
    k_x = np.fft.fftfreq(Nx, proj_reso)
    k_y = np.fft.fftfreq(Ny, proj_reso)
    k_z = np.fft.fftfreq(Nz, los_reso)
    k3d_z, k3d_y, k3d_x = np.meshgrid(k_z, k_y, k_x, indexing='ij')
    k3d_norm = np.sqrt(k3d_x**2 + k3d_y**2 + k3d_z**2)
    
    # Compute the Pk cube
    if cube2 is None:
        fourier_cube = np.fft.fftn(cube)
        fourier_pk   = np.abs(fourier_cube)**2 # img unit squared
    else:
        fourier_c1 = np.fft.fftn(cube)
        fourier_c2 = np.fft.fftn(cube2)
        fourier_pk = np.real(fourier_c1*np.conjugate(fourier_c2)+np.conjugate(fourier_c1)*fourier_c2)/2

    # Get the flattened k, Pk
    knrm = k3d_norm.flatten()
    fourier_pk = fourier_pk.flatten()

    #----- Case of binning
    if Nbin is not None:
        # Define the bins
        if kedges is None:
            if kmin is None:
                kmin_sampling = np.amin(k3d_norm[k3d_norm > 0])
            else:
                kmin_sampling = kmin
                
            if kmax is None:
                kmax_sampling = np.amax(k3d_norm)
            else:
                kmax_sampling = kmax
                
            if scalebin == 'lin':
                kbins = np.linspace(kmin_sampling, kmax_sampling, Nbin+1)
            elif scalebin == 'log':
                kbins = np.logspace(np.log10(kmin_sampling), np.log10(kmax_sampling), Nbin+1)
            else:
                raise ValueError("Only lin or log scales are allowed. Here scalebin="+scalebin)
            
        else:
            kbins = kedges
            
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        
        # Bin the Pk
        Pk_bins, _, _ = stats.binned_statistic(knrm, fourier_pk,statistic=statistic, bins=kbins)
        Pk_bins *= (proj_reso*proj_reso*los_reso) / (Nx*Ny*Nz)

    #----- Case unbinned
    else:
        kvals = knrm
        Pk_bins = fourier_pk * (proj_reso*proj_reso*los_reso) / (Nx*Ny*Nz)

    return kvals, Pk_bins


#==================================================
# Measure the 3D power spectrum naively
#==================================================

def extract_pk2d(image, proj_reso,
                 image2=None,
                 Nbin=100, scalebin='lin', kmin=None, kmax=None, kedges=None,
                 statistic='mean'):
    """
    Measure the power spectrum in 2 dimensions in k bins.
    The unit of the k array is the inverse of the resolution
    The unit of Pk is that of the resolution element squared.
    
    Parameters
    ----------
    - image (np array): 2d data image as nd array
    - proj_reso (float): the resolution along the projected direction (any unit, e.g. kpc, arcsec)
    - image2 (np array): same as image. Use to do cross spectra
    - Nbin (int): the number of bin for output Pk. If Nbin is None, then no 
    binning is done
    - scalebin (str): lin or log, the way the Pk is binned along k
    - kmin/max (float): the min and max k to use in defining the bins
    - kedges (1d np array): directly provide the bin edges in a Nbin+1 array (in this case,
      Nbin, kmin/kmax and scalebin are ignored)
    - statistic (str or function): the statistics to be used in binned_statistic
        
    Outputs
    ----------
    - kvals (np array): values of k binned
    - Pk_bins (np array): values of amplitude binned
    """
    
    # Get the number of pixels
    Ny, Nx = image.shape

    # Define the k_i and k_norm
    k_x = np.fft.fftfreq(Nx, proj_reso)
    k_y = np.fft.fftfreq(Ny, proj_reso)
    k2d_x, k2d_y = np.meshgrid(k_x, k_y, indexing='xy')
    k2d_norm = np.sqrt(k2d_x**2 + k2d_y**2)
    
    # Compute the Pk amplitide
    if image2 is None:
        fourier_img = np.fft.fftn(image)
        fourier_pk = np.abs(fourier_img)**2 # img unit squared
    else:
        fourier_img1 = np.fft.fftn(image)
        fourier_img2 = np.fft.fftn(image2)
        fourier_pk = np.real(fourier_img1*np.conjugate(fourier_img2)+np.conjugate(fourier_img1)*fourier_img2)/2

    # Get the flattened k, Pk
    knrm = k2d_norm.flatten()
    fourier_pk = fourier_pk.flatten()

    #----- Case of binning
    if Nbin is not None:
        # Define the bins
        if kedges is None:
            if kmin is None:
                kmin_sampling = np.amin(k2d_norm[k2d_norm > 0])
            else:
                kmin_sampling = kmin
                
            if kmax is None:
                kmax_sampling = np.amax(k2d_norm)
            else:
                kmax_sampling = kmax
                
            if scalebin == 'lin':
                kbins = np.linspace(kmin_sampling, kmax_sampling, Nbin+1)
            elif scalebin == 'log':
                kbins = np.logspace(np.log10(kmin_sampling), np.log10(kmax_sampling), Nbin+1)
            else:
                raise ValueError("Only lin or log scales are allowed. Here scalebin="+scalebin)
            
        else:
            kbins = kedges
            
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        
        # Bin the Pk
        Pk_bins, _, _ = stats.binned_statistic(knrm, fourier_pk, statistic=statistic, bins=kbins)
        Pk_bins *= (proj_reso*proj_reso) / (Nx*Ny) # img unit squared x reso unit squared

    #----- Case unbinned
    else:
        kvals = knrm
        Pk_bins = fourier_pk * (proj_reso*proj_reso) / (Nx*Ny)

    return kvals, Pk_bins


#==================================================
# Measure the 3D power spectrum naively
#==================================================

def extract_pk2d_arevalo(image, proj_reso,
                         kctr=None,
                         Nbin=100, scalebin='lin', kmin=None, kmax=None, kedges=None,
                         epsilon=1e-3, mask=None,
                         unbias_apply=False, unbias_slope=None, unbias_beamFWHM=0):
    """
    Implement the method of Arevalo et al. (2012) to compute the power spectrum.
    Warning, the method is biased for steep spectra, which is the case 
    near the beam cutoff. The unbiais keyword attempt to unbias the spectrum
    from itself.

    The unit of output k is the same as the inverse of the resolution, and that of Pk 
    is the same as resolution^2.
    
    See also Romero et al. (2023) and its erratum for the bias estimate due to the PSF.

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
    - unbias_apply (bool): set to true to unbias the spectrum given the estimated power law bias
    using the first guess measured spectrum as the input
    - unbias_slope (float or array with lenght Nbin): the slope of the underlying true spectrum 
    needed for debiasing
    - unbias_beamFWHM (float): the Gaussian beam FWHM (same unit as proj_reso)

    Outputs
    ----------
    - k2d (np array): values of k2d
    - Pk (np array): values of spectrum at k2d
    """

    #----- Possibility to define the bins in different ways        
    if kctr is None:    # The user does not gives the central k values
        # First need the k grid
        k_x = np.fft.fftfreq(image.shape[1], proj_reso)
        k_y = np.fft.fftfreq(image.shape[0], proj_reso)
        k2d_x, k2d_y = np.meshgrid(k_x, k_y, indexing='xy')
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
        if scalebin == 'lin':
            kbins = np.linspace(kmin_sampling, kmax_sampling, Nbin+1)
        elif scalebin == 'log':
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
    if unbias_apply:
        # Compute the slope from the spectrum itself if not provided (i.e. do what you can)
        if unbias_slope is None:
            der = -np.gradient(np.log10(Pk), np.log10(k2d))
        else:
            der = unbias_slope
        biasPL = 2**(der/2) * gamma(3 - der/2) / gamma(3)
        
        # Compute the beam smoothing bias correction
        if unbias_beamFWHM > 0:
            sigma_i = unbias_beamFWHM/sigma2fwhm
            k_i = 1 / (np.sqrt(2) * np.pi * sigma_i)
            x_i = k_i / k2d
            biasB = (1 + 1/x_i**2)**(-3+der/2)
        else:
            biasB = 1
        
        Pk = Pk/(biasPL*biasB)

    return k2d, Pk


#==================================================
# Increase the number of Pk in a sample
#==================================================

def pk_data_augmentation(k, pks,
                         Nsim=10000,
                         method='LogNormCov',
                         n_nearest=10):
    """
    Starting from a set of spectra, this code perform data augmentation
    using different methods.
    (Developed by A. Zarka)
    
    Parameters
    ----------
    - k (np array): 1d array
    - pks (np array): 2d array of shape (Nspec, Nkbins)
    - Nsim (int): the number of new spectra to be generated
    - method (str): 1) 'LogNormCov' for log-norm P(k) distributions
                     plus covariance between bins
                    2) 'NearNeighborsItpl' for the nearest
                     neighbors interpolation
    - n_nearest (int): number of nearest neighbors for each point
    
    Outputs
    ----------
    - Pknew (np array): 2d array of shape (Nsim, Nkbins)
    """

    # Define the new Pk
    Pknew = np.zeros((Nsim, len(k)))

    # In the case of log normal assumption + bin-to-bin covariance
    if method == 'LogNormCov':
        log_list = np.log(pks)               # log of the spectra
        mu_log   = np.mean(log_list, axis=0)      # mean of the spectra
        cov_log  = np.cov(log_list, rowvar=False) # Covariance matrix

        for i in range (Nsim):
            sample_log = np.random.multivariate_normal(mu_log, cov_log)
            Pknew[i,:] = np.exp(sample_log)

    # In the case of nearest neighbors interpolation
    elif method == "NearNeighborsItpl":
        # Find the k nearest neighbors for each point
        nn = NearestNeighbors(n_neighbors=n_nearest+1).fit(pks)  # k+1 to include the point itself
        distances, indices = nn.kneighbors(pks)

        for j, sample in enumerate(pks):
            for i in range (Nsim):

                # Select a neighbor randomly from the k nearest ones
                neighbor_idx = np.random.choice(indices[j][1:])  # indices[1:] to exclude the point itself
                neighbor = pks[neighbor_idx]
            
                # Generate an interpolation coefficient between 0 and 1
                alpha = np.random.uniform(0, 1)

                # Compute interpolated point
                new_sample = sample + alpha * (neighbor - sample)  # Linear interpolation

                # Fill the Pk
                Pknew[i,:] = new_sample
                
    # No other solution so far
    else:
        raise ValueError('Only "LogNormCov" and "" methods are available.')

    return Pknew
