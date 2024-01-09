"""
This file contain a subclass of the main_model.py module and ClusterModel class. It
is dedicated to the computing of mock observables.

"""

#==================================================
# Requested imports
#==================================================

import numpy as np
import astropy.units as u
import astropy.constants as cst
from scipy.interpolate import interp1d


from pitszi import utils


#==================================================
# Mock class
#==================================================

class ModelInference(object):
    """ ModelInference class
        This class serves as a parser to the main Model class, 
        it is used to infer constraints on the model given data.

    Attributes
    ----------
    The attributes are the same as the Model class, see model_main.py

    Methods
    ----------
    - p3d_to_p2d_from_window_function
    - 
    -
    -
    -
    
    """    

    #==================================================
    # Window function
    #==================================================
    
    def p3d_to_p2d_from_window_function(self):
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
        Nx, Ny, Nz, proj_reso, proj_reso, los_reso = self.define_3dgrid()

        #----- Get the pressure profile model
        pressure3d_sph  = self.pressure_cube_profile()

        #----- Get the Compton model
        compton_sph = self.get_sz_map(no_fluctuations=True)
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
    # Beam Power spectrum deconvolution
    #==================================================
    
    def beam_pk_deconv(self, k, pk, beamFWHM):
        """
        This function corrects the power spectrum from beam 
        attenuation.
        
        Parameters
        ----------
        - k (nd array): k array in kpc
        - pk (nd array): power spectrum array
        - beamFWHM (quantity): the beam FWHM homogeneous to kpc or arcsec

        Outputs
        ----------
        - pk_deconv (np array): the deconvolved pk

        """

        #----- Test the inputs
        try:
            beam_kpc = beamFWHM.to_value('kpc')
        except:
            try:
                beam_kpc = beamFWHM.to_value('radian')*self._D_ang.to_value('kpc')
            except:
                raise ValueError('beamFWHM is the beam FWHM, a quantity homogeneous to kpc or arcsec')    

        try:
            k_kpc = k.to_value('kpc-1')
        except:
            try:
                k_kpc = k.to_value('radian-1')/self._D_ang.to_value('kpc')
            except:
                raise ValueError('k is the wavenumber, a quantity homogeneous to 1/kpc or 1/arcsec')    

        #----- Apply the deconvolution
        Beam_k = utils.gaussian_pk(k_kpc, beam_kpc)
        pk_deconv = pk/Beam_k**2

        return pk_deconv

    
    #==================================================
    # Transfer function power spectrum deconvolution
    #==================================================
    
    def transfer_function_pk_deconv(self, k, pk, TF):
        """
        This function corrects the power spectrum from transfer 
        function attenuation.
        
        Parameters
        ----------
        - k (nd array): k array in kpc
        - pk (nd array): power spectrum array
        - TF (dictionary): dictionary containing 'k_arcsec' or 'k_kpc' and 'TF'

        Outputs
        ----------
        - pk_deconv (np array): the deconvolved pk

        """
        
        #----- Test the input
        try:
            k_tf = (TF['k_arcsec']*3600*180/np.pi)/self._D_ang.to_value('kpc')
            tf   = TF['TF']
        except:
            try:
                k_tf = TF['k_kpc']
                tf   = TF['TF']
            except:
                raise ValueError('TF is a disctionary that contains "k_arcsec" or "k_kpc" and "TF"')    

        try:
            k_kpc = k.to_value('kpc-1')
        except:
            try:
                k_kpc = k.to_value('radian-1')/self._D_ang.to_value('kpc')
            except:
                raise ValueError('k is the wavenumber, a quantity homogeneous to 1/kpc or 1/arcsec')    

        #----- Apply the deconvolution
        itpl = interp1d(k_tf, tf, fill_value=1)
        TF_kpc = itpl(k_kpc)
        
        pk_deconv = pk/TF_kpc**2

        return pk_deconv
    
    
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
        conv2d = self.p3d_to_p2d_from_window_function()
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
            pk2d = self.beam_pk_deconv(k2d, pk2d, BeamDeconv)

        #----- TF deconvolution
        if TFDeconv is not None:
            pk2d = self.transfer_function_pk_deconv(k2d, pk2d, TFDeconv)

        #----- Compute the Pk3D
        k3d  = k2d*u.kpc**-1
        Pk3d = pk2d*u.kpc**2 / conv

        return k3d, Pk3d


    #==================================================
    # Generate a mock power spectrum
    #==================================================
    
    def make_mock_pk2d(self,
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
            pk2d = self.beam_pk_deconv(k2d, pk2d, DeconvBeamFWHM)

        #----- TF deconvolution
        if DeconvTF is not None:
            pk2d = self.transfer_function_pk_deconv(k2d, pk2d, DeconvTF)

        return k2d*u.kpc**-1, pk2d*u.kpc**2

