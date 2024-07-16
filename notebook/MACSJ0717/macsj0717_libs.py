# This file probides useful functions for the analysis of MACS J0717.5+3745 pressure fluctuations

import os
import sys
sys.path.insert(0,'/Users/adam/Project/NIKA/Software/Processing/Labtools/RA/pitszi/')
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from reproject import reproject_interp
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from minot.ClusterTools import map_tools
import pitszi

sigma2fwhm = 2 * np.sqrt(2*np.log(2))

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
# Extract the data
#============================================================

def extract_data(FoV, reso, ps_mask_lim=0.1, show=False, clean_ksz=False):

    # define header
    hdul = fits.open('/Users/adam/Project/Notes-Papier-Conf/2016_12_Edge_Search/Save/Products/MAP150GHz_MACSJ0717.fits')
    cl_head = map_tools.define_std_header(hdul[0].header['CRVAL1'], hdul[0].header['CRVAL2'],
                                      FoV.to_value('deg'), FoV.to_value('deg'), reso.to_value('deg'))
    hdul.close()
    
    # kSZ image
    if clean_ksz:
        hdul1 = fits.open('/Users/adam/Project/Notes-Papier-Conf/2016_12_Edge_Search/Save/Products/MAP150GHz_MACSJ0717.fits')
        hdul2 = fits.open('/Users/adam/Project/Notes-Papier-Conf/2016_12_Edge_Search/Save/Products/MAP150GHz_MACSJ0717kSZ.fits')
        img1  = hdul1[0].data
        img2  = hdul2[0].data
        kszcorr, _ = reproject_interp((img1-img2, hdul1[0].header), cl_head)
        kszcorr[np.isnan(kszcorr)] = 0
        hdul1.close()
        hdul2.close()

    # Data image
    hdul = fits.open('/Users/adam/Project/Notes-Papier-Conf/2016_12_Edge_Search/Save/Products/MAP150GHz_MACSJ0717.fits')
    cl_head = map_tools.define_std_header(hdul[0].header['CRVAL1'], hdul[0].header['CRVAL2'],
                                      FoV.to_value('deg'), FoV.to_value('deg'), reso.to_value('deg'))
    img_ini  = hdul[0].data
    img_ini, _ = reproject_interp((img_ini, hdul[0].header), cl_head)
    img_ini[np.isnan(img_ini)] = 0
    if clean_ksz: img_ini -= kszcorr
    hdul.close()

    # Point sources
    hdul = fits.open('/Users/adam/Project/Notes-Papier-Conf/2016_12_Edge_Search/Save/Products/PointSourceModel150GHz_MACSJ0717.fits')
    img_ps = hdul[0].data
    img_ps, _ = reproject_interp((img_ps, hdul[0].header), cl_head)
    img_ps[np.isnan(img_ps)] = 0
    cl_img = img_ini - img_ps
    hdul.close()

    # Jackknife
    hdul = fits.open('/Users/adam/Project/Notes-Papier-Conf/2015_10_MACSJ0717_Paper/IDL/Save/NIKA_JK_map_2mm.fits')
    img_jk = hdul[0].data*1e3
    img_jk, _ = reproject_interp((img_jk, hdul[0].header), cl_head)
    img_jk[np.isnan(img_jk)] = 0
    cl_jk = img_jk
    hdul.close()

    # Half maps
    cl_img1 = cl_img - cl_jk
    cl_img2 = cl_img + cl_jk
    if clean_ksz: cl_img1 -= kszcorr
    if clean_ksz: cl_img2 -= kszcorr

    # Noise
    hdul = fits.open('/Users/adam/Project/Notes-Papier-Conf/2016_12_Edge_Search/Save/Products/NoiseMC150GHz_MACSJ0717.fits')
    noise = hdul[1].data    
    noise =  np.swapaxes(np.swapaxes(noise, 2,0), 1,2)
    noise_rep = np.zeros((len(noise[:,0,0]), img_ini.shape[0], img_ini.shape[1]))
    for imc in range(len(noise[:,0,0])):
        repro, _ = reproject_interp((noise[imc,:,:], hdul[0].header), cl_head)
        repro[np.isnan(repro)] = 0
        noise_rep[imc,:,:] = repro
    cl_noise = noise_rep
    cl_rms = np.std(noise_rep, axis=0)
    cl_rms[cl_rms == 0] = np.amax(cl_rms)*1e3
    cl_rms = gaussian_filter(cl_rms,sigma=10/sigma2fwhm/cl_head['CDELT2']/3600)
    hdul.close()

    # Point source mask and noise cut
    cl_mask = img_ps*0 + 1
    cl_mask[img_ps > ps_mask_lim] = 0
    cl_mask[cl_rms > np.amin(cl_rms)*2] = 0

    # Jy/beam to Compton conversion
    y2jy = compton2jy(cl_head)*1e3
    cl_img   = cl_img/y2jy
    cl_img1  = cl_img1/y2jy
    cl_img2  = cl_img2/y2jy
    cl_jk    = cl_jk/y2jy
    cl_ps   = img_ps/y2jy
    cl_rms   = cl_rms/np.abs(y2jy)
    cl_noise = cl_noise/y2jy

    # Show the inputs
    if show == True:
        fig = plt.figure(0, figsize=(15, 7))
        ax = plt.subplot(2, 4, 1, projection=WCS(cl_head))
        plt.imshow(cl_img*1e5, origin='lower')
        plt.colorbar()
        plt.title(r'$10^5$ y-map')
        
        ax = plt.subplot(2, 4, 2, projection=WCS(cl_head))
        plt.imshow(cl_rms*1e5, origin='lower')
        plt.colorbar()
        plt.title(r'$10^5$ y-map rms')
        
        ax = plt.subplot(2, 4, 3, projection=WCS(cl_head))
        plt.imshow(cl_img/cl_rms, origin='lower')
        plt.colorbar()
        plt.title(r'S/N')
        
        ax = plt.subplot(2, 4, 4, projection=WCS(cl_head))
        plt.imshow(cl_ps*y2jy, origin='lower')
        plt.colorbar()
        plt.title('PS model (mJy/beam)')
        
        ax = plt.subplot(2, 4, 5, projection=WCS(cl_head))
        plt.imshow(cl_mask, origin='lower')
        plt.colorbar()
        plt.title('Mask')
        
        ax = plt.subplot(2, 4, 6, projection=WCS(cl_head))
        plt.imshow(cl_jk*1e5, origin='lower')
        plt.colorbar()
        plt.title('$10^5$ JK')
        
        ax = plt.subplot(2, 4, 7, projection=WCS(cl_head))
        plt.imshow(cl_img1/cl_rms/2**0.5, origin='lower')
        plt.colorbar()
        plt.title('S/N y-map1')
        
        ax = plt.subplot(2, 4, 8, projection=WCS(cl_head))
        plt.imshow(cl_img2/cl_rms/2**0.5, origin='lower')
        plt.colorbar()
        plt.title('S/N y-map2')
        
    return cl_head, y2jy, cl_img, cl_img1, cl_img2, cl_jk, cl_ps, cl_rms, cl_noise, cl_mask


#============================================================
# Defines the radial model
#============================================================

def def_fitparprof(RM):
    
    cl_coord = SkyCoord(109.3806*u.deg, 37.7583*u.deg, frame='icrs')
    RA  = cl_coord.ra
    Dec = cl_coord.dec

    if RM == 1:
        fitpar_prof = {
            'M500':{'guess':[10,1], 'unit':1e14*u.Msun, 'limit':[1, 100], 'P_ref':'A10MD'},
            'ZL':{'guess':[0,1e-5],'unit':None},
        }
    if RM == 2: 
        fitpar_prof = {
            'M500':{'guess':[10,1], 'unit':1e14*u.Msun, 'limit':[1, 100], 'P_ref':'A10MD'},
            'RA': {'guess':[RA.to_value('deg'), 0.5/60], 'unit': u.deg, 
                   'limit':[RA.to_value('deg')-0.5/60, RA.to_value('deg')+0.5/60]},
            'Dec': {'guess':[Dec.to_value('deg'), 0.5/60], 'unit': u.deg, 
                    'limit':[Dec.to_value('deg')-0.5/60, Dec.to_value('deg')+0.5/60]},
            'ZL':{'guess':[0,1e-5],'unit':None},
        }    
    if RM == 3: 
        fitpar_prof = {
            'M500':{'guess':[10,1], 'unit':1e14*u.Msun, 'limit':[1, 100], 'P_ref':'A10MD'},
            'RA': {'guess':[RA.to_value('deg'), 0.5/60], 'unit': u.deg, 
                   'limit':[RA.to_value('deg')-0.5/60, RA.to_value('deg')+0.5/60]},
            'Dec': {'guess':[Dec.to_value('deg'), 0.5/60], 'unit': u.deg, 
                    'limit':[Dec.to_value('deg')-0.5/60, Dec.to_value('deg')+0.5/60]},
            'min_to_maj_axis_ratio':{'guess':[0.5,0.1], 'unit':None, 'limit':[0,1]}, 
            'angle':{'guess':[20,10], 'unit':u.deg, 'limit':[-90,90]},
            'ZL':{'guess':[0,1e-5],'unit':None},
        }
    if RM == 4: 
        fitpar_prof = {
            'P_0': {'guess':[0.02, 0.001], 'unit': u.keV*u.cm**-3, 'limit':[0, np.inf]},
            'r_p': {'guess':[1000, 1000], 'unit': u.kpc, 'limit':[0, np.inf]},
            'RA': {'guess':[RA.to_value('deg'), 0.5/60], 'unit': u.deg, 
                   'limit':[RA.to_value('deg')-0.5/60, RA.to_value('deg')+0.5/60]},
            'Dec': {'guess':[Dec.to_value('deg'), 0.5/60], 'unit': u.deg, 
                    'limit':[Dec.to_value('deg')-0.5/60, Dec.to_value('deg')+0.5/60]},
            'min_to_maj_axis_ratio':{'guess':[0.5,0.1], 'unit':None, 'limit':[0,1]}, 
            'angle':{'guess':[20,10], 'unit':u.deg, 'limit':[-90,90]},
            'ZL':{'guess':[0,1e-5],'unit':None},
        }
    if RM == 5: 
        fitpar_prof = {
            'P_0': {'guess':[0.02, 0.001], 'unit': u.keV*u.cm**-3, 'limit':[0, np.inf]},
            'r_p': {'guess':[1000, 1000], 'unit': u.kpc, 'limit':[0, np.inf]},
            'a': {'guess':[1, 0.5], 'unit': None, 'limit':[0, 10]},
            'b': {'guess':[5, 0.5], 'unit': None, 'limit':[2, 8]},
            'c': {'guess':[0.5, 0.5], 'unit': None, 'limit':[0, 2]},
            'RA': {'guess':[RA.to_value('deg'), 0.5/60], 'unit': u.deg, 
                   'limit':[RA.to_value('deg')-0.5/60, RA.to_value('deg')+0.5/60]},
            'Dec': {'guess':[Dec.to_value('deg'), 0.5/60], 'unit': u.deg, 
                    'limit':[Dec.to_value('deg')-0.5/60, Dec.to_value('deg')+0.5/60]},
            'min_to_maj_axis_ratio':{'guess':[0.5,0.1], 'unit':None, 'limit':[0,1]}, 
            'angle':{'guess':[20,10], 'unit':u.deg, 'limit':[-90,90]},
            'ZL':{'guess':[0,1e-5],'unit':None},
        }
        
    return fitpar_prof


#============================================================
# Compute CIB simulations
#============================================================

def simu_cib(RA0, Dec0, header, beam_FWHM, TF,
             Nsim=1000, Scut_high=100, Scut_low=0.0):

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
        
        src_imc = np.zeros((header['NAXIS1'], header['NAXIS2']))
        for isrc in range(len(cat_imc)):
            dist_map = map_tools.greatcircle(ramap, decmap, RA0+cat_imc['ra'][isrc], Dec0+cat_imc['dec'][isrc])
            flux = cat_imc['SNIKA2000'][isrc]*1e3
            src_imc += flux*np.exp(-dist_map**2/2/(beam_FWHM.to_value('deg')/sigma2fwhm)**2)
                
        img_conv = pitszi.utils_pk.apply_transfer_function(src_imc, header['CDELT2']*3600, 
                                                           beam_FWHM.to_value('arcsec'), TF)
        src[imc,:,:] = img_conv    
                
    return src


#============================================================
# Compute low-M halos simulations
#============================================================

def simu_lowmc(RA0, Dec0, header, beam_FWHM, TF, Nsim=100):

    sigma2fwhm = 2 * np.sqrt(2*np.log(2))

    ramap, decmap = map_tools.get_radec_map(header)

    src = np.zeros((Nsim, header['NAXIS1'], header['NAXIS2']))
    
    path     = '/Users/adam/Project/NIKA/Software/Processing/Labtools/RA/pitszi/COSMOS_sim/'
    ignored  = {".DS_Store"}
    catfiles = [x for x in os.listdir(path) if x not in ignored]
    np.random.shuffle(catfiles)
    
    for imc in range(Nsim):
        if Nsim>1 and imc % 5 == 0: print(imc, '/', Nsim)
        hdul = fits.open(path+catfiles[imc])
        lmh_data = hdul[5].data /(-11.9e3)
        lmh_head = hdul[5].header
        hdul.close()
        lmh_head['CRVAL1'] = RA0
        lmh_head['CRVAL2'] = Dec0
        lmh_rep, _ = reproject_interp((lmh_data, lmh_head), header)
        
        lmh_rep_conv = pitszi.utils_pk.apply_transfer_function(lmh_rep, header['CDELT2']*3600, 
                                                               beam_FWHM.to_value('arcsec'), TF)
        
        src[imc,:,:] = lmh_rep_conv
        
    return src


#============================================================
# Defines the data
#============================================================

def def_data(cl_img, cl_head, cl_noise, cl_mask, beam_FWHM, TF, outdir, Nsim):
    
    cl_data = pitszi.Data(cl_img, cl_head, 
                          psf_fwhm = beam_FWHM,
                          transfer_function=TF,
                          silent=True, output_dir=outdir)

    #----- Define the noise properties
    cl_data.noise_mc = cl_noise
    cl_data.set_noise_model_from_mc()
    cl_data.noise_mc = cl_data.get_noise_monte_carlo_from_model(Nmc=Nsim)
    cl_data.noise_rms = cl_data.get_noise_rms_from_model(Nmc=Nsim)
    cl_data.noise_rms[cl_data.noise_rms == 0] = np.amax(cl_data.noise_rms)*1e3
    cl_data.noise_rms = gaussian_filter(cl_data.noise_rms,sigma=10/sigma2fwhm/cl_head['CDELT2']/3600)
    
    #----- Define the mask
    cl_data.mask = cl_mask

    return cl_data


#============================================================
# Defines the ROI
#============================================================

def def_roi(cl_head, model, mask_theta=2*u.arcmin, show=False):
    
    ramap, decmap = map_tools.get_radec_map(cl_head)
    dist_map = map_tools.greatcircle(ramap, decmap, 
                                     model.coord.ra.to_value('deg'), 
                                     model.coord.dec.to_value('deg'))
    roi = dist_map * 0 + 1
    roi[dist_map > mask_theta.to_value('deg')] = 0 

    if show==True:
        fig = plt.figure(0, figsize=(12, 5))
        ax = plt.subplot(1, 1, 1, projection=WCS(cl_head))
        plt.imshow(roi, origin='lower')
        plt.colorbar()
        plt.title('ROI')
        
    return roi

