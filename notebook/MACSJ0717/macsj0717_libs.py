# This file probides useful functions for the analysis of MACS J0717.5+3745 pressure fluctuations

import os
import numpy as np
from astropy.io import fits
import astropy.units as u
from reproject import reproject_interp
from scipy.interpolate import interp1d
from minot.ClusterTools import map_tools


#============================================================
# Extract the transfer function
#============================================================

def get_TF():
    
    hdul = fits.open('/Users/adam/Project/Notes-Papier-Conf/2016_12_Edge_Search/Save/Products/TransferFunction150GHz_MACSJ0717.fits')
    data_TF = hdul[1].data
    TF_i = data_TF['TF'][0]
    TF_i[np.isnan(TF_i)] = 1
    TF = {'k':data_TF['WAVE_NUMBER_ARCSEC'][0]*u.arcsec**-1/2**0.5, 'TF':TF_i}
    
    return TF

#============================================================
# Compute the conversion coefficient from Compton to Jy/beam
#============================================================

def compton2jy(header):
    
    # Conversion coefficients from Adam et al. (2017), Table 1
    Tarr = np.array([1     , 5     ,10     ,     15,     20,     25])
    Carr = np.array([-11.63, -11.34, -11.00, -10.71, -10.38, -10.17])

    # X-ray image from Adam et al. (2017)
    hdulX = fits.open('/Users/adam/Project/Notes-Papier-Conf/2015_10_MACSJ0717_Paper/IDL/Save/Xray/tmap_macsj0717_xmm.fits')[0]
    Tx = hdulX.data
    hx = hdulX.header
    hx['RADESYS'] = 'ICRS' # correct issue with header
    hx['LONPOLE'] = 180.0  # correct issue with header

    # Coefficient map
    itpl = interp1d(Tarr, Carr, bounds_error=False, fill_value=(-11.63, -10.17))
    Cmap_flat = itpl(Tx.flatten())
    Cmap = Cmap_flat.reshape(Tx.shape)

    # reproject to given header
    conversion, fp = reproject_interp((Cmap, hx), header)
    conversion[np.isnan(conversion)] = -11.63
    
    return conversion


#============================================================
# Compute CIB simulations
#============================================================

def simu_cib(RA0, Dec0, header, beam_FWHM,
             Nsim=1000, Scut_high=100, Scut_low=0.0):

    sigma2fwhm = 2 * np.sqrt(2*np.log(2))

    ramap, decmap = map_tools.get_radec_map(header)

    src = np.zeros((Nsim, header['NAXIS1'], header['NAXIS2']))
    
    path     = '/Users/adam/Project/NIKA/Software/Processing/Labtools/RA/pitszi/sides-public-release-main/cats/CIB_cat_sim/'
    ignored  = {".DS_Store"}
    catfiles = [x for x in os.listdir(path) if x not in ignored]
    np.random.shuffle(catfiles)
    
    for imc in range(Nsim):
        if Nsim>1 and imc % 5 == 0: print(imc, '/', Nsim)
        hdul = fits.open(path+catfiles[imc])
        cat_imc = hdul[1].data
        hdul.close()
        w1 = cat_imc['SNIKA2000']*1e3 > Scut_low
        w2 = cat_imc['SNIKA2000']*1e3 < Scut_high
        cat_imc = cat_imc[w1*w2]
        
        for isrc in range(len(cat_imc)):
            dist_map = map_tools.greatcircle(ramap, decmap, RA0+cat_imc['ra'][isrc], Dec0+cat_imc['dec'][isrc])
            flux = cat_imc['SNIKA2000'][isrc]*1e3
            if flux < Scut_high:
                src[imc,:,:] += flux*np.exp(-dist_map**2/2/(beam_FWHM.to_value('deg')/sigma2fwhm)**2)
                
    return src
