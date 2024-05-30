"""
This file contain a subclass of the main_model.py module and ClusterModel class. It
is dedicated to the computing of mock observables.

"""

#==================================================
# Requested imports
#==================================================

import numpy as np
from scipy.interpolate import interp1d
from astropy.coordinates import SkyOffsetFrame, SkyCoord
import astropy.units as u
import astropy.constants as cst
from scipy.spatial.transform import Rotation

from pitszi import utils
from pitszi import utils_pk


#==================================================
# Mock class
#==================================================

class ModelMock(object):
    """ ModelMock class
    This class serves as a parser to the main Model class, to
    include the subclass ModelMock in this other file.

    Attributes
    ----------
    The attributes are the same as the Model class, see model_main.py

    Methods
    ----------
    - get_pressure_profile
    - get_pressure_fluctuation_spectrum
    - get_pressure_cube_profile
    - get_pressure_cube_fluctuation
    - get_sz_map
    
    """
    
    #==================================================
    # Get the electron pressure profile
    #==================================================

    def get_pressure_profile(self,
                             radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the thermal electron pressure profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - p_r (quantity): the electron pressure profile in unit of keV cm-3

        """
        
        # In case the input is not an array
        radius = utils.check_qarray(radius, unit='kpc')

        # get profile
        p_r = self._get_generic_profile(radius, self._model_pressure_profile)
        p_r[radius > self._R_truncation] *= 0

        return radius, p_r.to('keV cm-3')


    #==================================================
    # Get the electron pressure fluctuation spectrum
    #==================================================

    def get_pressure_fluctuation_spectrum(self,
                                          kvec=np.logspace(-1,2,1000)*u.Mpc**-1,
                                          kmin_norm=None,
                                          kmax_norm=None,
                                          Npt_norm=10000):
        """
        Get the thermal electron pressure fluctuation spectrum (delta P / P).
        
        Parameters
        ----------
        - kvec (quantity) : the physical wavenumber in units homogeneous to kpc-1, as a 1d array
        - kmin/max_norm (quantity): the wavenumber range used for notmalization
        - Npt_norm (int): number of point to generagte the spectrum used for normalization

        Outputs
        ----------
        - kvec (quantity): the physical wavenumber in units homogeneous to kpc-1, as a 1d array
        - P3d_k (quantity): the 3D power spectrum in unit homogeneous to kpc^3

        """

        # In case the input is not an array
        kvec = utils.check_qarray(kvec, unit='kpc-1')
        
        # compute the model
        '''
        This part could go, as for the profile, in a generic fluctuation library and be read fom there
        This can be done when the library includes more functionnal forms
        '''
        if self._model_pressure_fluctuation['name'] == 'CutoffPowerLaw':

            if kmin_norm is None:
                kmin = 1/(4*self._model_pressure_fluctuation['Linj'].to_value('kpc'))
            else:
                kmin = kmin_norm.to_value('kpc')
            if kmax_norm is None:
                kmax = 4/self._model_pressure_fluctuation['Ldis'].to_value('kpc')
            else:
                kmax = kmax_norm.to_value('kpc')

            # K array for normalization
            kvec_norm = np.logspace(np.log10(kmin),np.log10(kmax), Npt_norm) # kpc

            # First extract the normalization givenintegrating within some k range
            cut_low  = np.exp(-(1/(kvec_norm*self._model_pressure_fluctuation['Linj'].to_value('kpc'))**2))
            cut_high = np.exp(-(kvec_norm*self._model_pressure_fluctuation['Ldis'].to_value('kpc'))**2)
            pl = kvec_norm**self._model_pressure_fluctuation['slope']
            f_k = pl * cut_high * cut_low
            Normalization = utils.trapz_loglog(4*np.pi*kvec_norm**2 * f_k, kvec_norm)

            # Then compute Pk at requested scales accounting for the normalization
            k = kvec.to_value('kpc-1')
            A = self._model_pressure_fluctuation['Norm']
            cut_low  = np.exp(-(1/(k*self._model_pressure_fluctuation['Linj'].to_value('kpc'))**2))
            cut_high = np.exp(-(k*self._model_pressure_fluctuation['Ldis'].to_value('kpc'))**2)
            pl = k**self._model_pressure_fluctuation['slope']
            P3d_k = A**2 * pl*cut_high*cut_low / Normalization # adu * u1 / (u1*k**3) = kpc^3
        else:
            raise ValueError("No other model implemented yet")
       
        return kvec, P3d_k*u.kpc**3
    
    
    #==================================================
    # Pressure profile to grid
    #==================================================
    
    def get_pressure_cube_profile(self):
        """
        Fill the grid with the pressure profile.
        
        
        Parameters
        ----------
        
        Outputs
        ----------
        p_grid (ndarray quantity) : the output pressure grid with keV / cm3 unit attached
        
        """

        #----- Get the grid properties
        Nx, Ny, Nz, proj_reso, proj_reso, los_reso = self.get_3dgrid()

        #----- Get the offset of the cluster
        map_center = self.get_map_center()
        offset = (self._coord).transform_to(SkyOffsetFrame(origin=map_center))
        dRA    = -offset.data.lon.to_value('radian')
        dDec   = offset.data.lat.to_value('radian')
        dx     = dRA*self._D_ang.to_value('kpc')   # Offset map along x
        dy     = dDec*self._D_ang.to_value('kpc')  # Offset map along y
        
        #----- Compute the pixel center coordinates
        ctr_xpix_vec = np.linspace(-Nx*proj_reso/2, Nx*proj_reso/2, Nx) - dx
        ctr_ypix_vec = np.linspace(-Ny*proj_reso/2, Ny*proj_reso/2, Ny) - dy
        ctr_zpix_vec = np.linspace(-Nz*los_reso/2,  Nz*los_reso/2,  Nz)

        coord_z, coord_y, coord_x = np.meshgrid(ctr_zpix_vec, ctr_ypix_vec, ctr_xpix_vec,
                                                indexing='ij')

        #----- Apply ellipticity
        angle_1 = self._triaxiality['euler_angle1'].to_value('deg')
        angle_2 = self._triaxiality['euler_angle2'].to_value('deg')
        angle_3 = self._triaxiality['euler_angle3'].to_value('deg')

        rot = Rotation.from_euler('ZXZ', [angle_1, angle_2, angle_3], degrees=True)
        coordinates = np.array([coord_x.flatten(), coord_y.flatten(), coord_z.flatten()]).T
        rotated_coordinates = rot.apply(coordinates)

        rot_coord_x = rotated_coordinates[:,0].reshape(coord_x.shape) / self._triaxiality['min_to_maj_axis_ratio']
        rot_coord_y = rotated_coordinates[:,1].reshape(coord_y.shape) / self._triaxiality['int_to_maj_axis_ratio']
        rot_coord_z = rotated_coordinates[:,2].reshape(coord_z.shape)
    
        #----- Compute the radius grid and flatten
        rad_grid = np.sqrt((rot_coord_x)**2 + (rot_coord_y)**2 + (rot_coord_z)**2)
        rad_grid_flat = rad_grid.flatten()

        '''
        To Do :
        Evaluate log P(r) for r at the nodes of the grid. Then interpolate at the center using 
        scipy.interpolate.interpn
        '''

        #----- Get the pressure profile in 1d
        rad = utils.sampling_array(self._Rmin, self._R_truncation, NptPd=100, unit=True)
        rad, p_rad = self.get_pressure_profile(rad)
        p_rmin = self.get_pressure_profile(self._Rmin)[1]
        
        #----- Interpolate the pressure to the flattened grid
        itpl = interp1d(rad.to_value('kpc'), p_rad.to_value('keV cm-3'),
                        kind='linear', fill_value='extrapolate')
        p_grid_flat = itpl(rad_grid_flat)

        #----- Truncate min and max radii
        p_grid_flat[rad_grid_flat <= self._Rmin.to_value('kpc')] = p_rmin.to_value('keV cm-3')
        p_grid_flat[rad_grid_flat > self._R_truncation.to_value('kpc')] = 0

        #----- Warning
        if self._silent is False:
            if self._R_truncation.to_value('kpc') < np.amax(rad_grid_flat):
                print('----- WARNING: the truncation radius is smaller than the box size.')
                print('               This induces a discountinuity wich may cause ringing.')
        
        #----- Reshape to the grid
        p_grid = p_grid_flat.reshape(Nz, Ny, Nx)
        
        return p_grid*u.keV*u.cm**-3


    #==================================================
    # Pressure fluctuation to grid
    #==================================================
    
    def get_pressure_cube_fluctuation(self,
                                      seed=None,
                                      kmin_input=None,
                                      kmax_input=None,
                                      force_isotropy=False,
                                      Npt=1000):
        """
        Fill the grid with pressure fluctuation, i.e. delta P / P(r)
        
        Parameters
        ----------
        - seed (bool): set to a number for reproducible fluctuations
        - kmin_input/kmax_input (flaot): the min and max k range, in kpc,
        used for sampling the power spectrum. If None, a default value based on 
        thre model is used
        - force_isotropy (bool): set to true to remove non isotropic k modes
        - Npt (int): the number of points used to sample the power spectrum

        Outputs
        ----------
        - fluctuation_cube (np.ndarray) : the output grid of delta P / P(r)
        
        """
        
        #----- Set a seed
        np.random.seed(seed)

        #----- Get the grid properties
        Nx, Ny, Nz, proj_reso, proj_reso, los_reso = self.get_3dgrid()

        #----- Bypass if no fluctuations
        if self._model_pressure_fluctuation['name'] == 'CutoffPowerLaw':
            if self._model_pressure_fluctuation['Norm'] == 0:
                fluctuation_cube = np.zeros((Nz, Ny, Nx))
                return fluctuation_cube

        #----- Get the k arrays along each axis
        k_x = np.fft.fftfreq(Nx, proj_reso) # 1/kpc
        k_y = np.fft.fftfreq(Ny, proj_reso)
        k_z = np.fft.fftfreq(Nz, los_reso)

        #----- Define the k grid and norm vector
        k3d_z, k3d_y, k3d_x = np.meshgrid(k_z, k_y, k_x, indexing='ij')
        k3d_norm = np.sqrt(k3d_x**2 + k3d_y**2 + k3d_z**2)
    
        #----- Get the Pk model on the grid
        k3d_norm_flat = k3d_norm.flatten()                      # Flatten the k3d array
        idx_sort = np.argsort(k3d_norm_flat)                    # Get the sorting index
        revidx = np.argsort(idx_sort)                           # Index to invert rearange by sorting
        k3d_norm_flat_sort = np.sort(k3d_norm_flat)             # Sort the k array
        _, P3d_flat_sort = self.get_pressure_fluctuation_spectrum(k3d_norm_flat_sort[1:]*u.kpc**-1)
        P3d_flat_sort = P3d_flat_sort.to_value('kpc3')          # Take the unit out
        P3d_flat_sort = np.append(np.array([0]), P3d_flat_sort) # Add P(k=0) = 0 back
        P3d_flat = P3d_flat_sort[revidx]                        # Unsort
        P3d_k_grid = P3d_flat.reshape(Nz,Ny,Nx)                 # Unflatten to k cube

        #----- Convert for lognormal
        if self._model_pressure_fluctuation['statistics'] == 'lognormal':
            P3d_k_grid = utils_pk.convert_pkln_to_pkgauss(P3d_k_grid, proj_reso, proj_reso, los_reso)
        
        #----- kill the unwanted mode: zero level and values beyond isotropic range if requested
        kmax_isosphere = self.get_kmax_isotropic().to_value('kpc-1')

        if force_isotropy:
            if not self.silent:
                print('            Non isotropic modes k>',kmax_isosphere,' kpc^-1 are set to zero.')
            P3d_k_grid[k3d_norm > kmax_isosphere] = 0

        #----- Compute the amplitude
        amplitude =  np.sqrt(P3d_k_grid / (proj_reso*proj_reso*los_reso))

        #----- Generate the 3D field
        field = np.random.normal(loc=0, scale=1, size=(Nz,Ny,Nx))
        fftfield = np.fft.fftn(field) * amplitude
        fluctuation_cube = np.real(np.fft.ifftn(fftfield))

        if self._model_pressure_fluctuation['statistics'] == 'lognormal':
            sig2 = np.mean(amplitude**2) # The mean of LN distrib is exp(mu+s^2/2) and sig^2 = np.mean(ampli^2)
            fluctuation_cube -= sig2/2   # So we want the mean to be -(mu + sig^2/2), with mu=0
            fluctuation_cube = np.exp(fluctuation_cube) # take exponential
            fluctuation_cube -= 1                       # Subtract 1 to have dP/P instead of dP/P+1 which is LN
        
        if not self.silent:
            print('----- INFO: fluctuation cube rms.')
            print('            Expected rms over the full k range:', self._model_pressure_fluctuation['Norm'])
            print('            Expected rms given the missing k range:', np.sqrt(np.mean(amplitude**2)))
            print('            Actual rms for this noise realization:', np.std(fluctuation_cube))

        return fluctuation_cube
        

    #==================================================
    # Compute SZ map
    #==================================================
    
    def get_sz_map(self,
                   seed=None,
                   no_fluctuations=False,
                   force_isotropy=False,
                   irfs_convolution_beam=None,
                   irfs_convolution_TF=None):
                   
        """
        Compute the SZ mock image.
        
        Parameters
        ----------
        - seed (bool): set to a number for reproducible fluctuations
        - no_fluctuations (bool): set to true when the pure spherical model is requested
        - force_isotropy (bool): set to true to remove non isotropic k modes
        - irfs_convolution_beam (quantity): PSF FWHM in unit homogeneous to arcsec. If given, 
        the model is convolved with the beam
        - irfs_convolution_TF (dictionary): if given, the model is convolved 
        with the transfer function       

        Outputs
        ----------
        - compton (np.ndarray) : the map in units of Compton parameter
        
        """

        #----- Get P(r) grid
        pressure_profile_cube = self.get_pressure_cube_profile()

        #----- Get delta P(x,y,z) grid
        if no_fluctuations:
            pressure_fluctuation_cube = pressure_profile_cube.value*0
        else:
            pressure_fluctuation_cube = self.get_pressure_cube_fluctuation(seed=seed, force_isotropy=force_isotropy)

        #----- Go to Compton
        intPdl = np.sum(pressure_profile_cube*(1 + pressure_fluctuation_cube), axis=0) * self._los_reso
        compton = (cst.sigma_T / (cst.m_e * cst.c**2) * intPdl).to_value('')
        
        #----- Convolution with instrument response
        if (irfs_convolution_beam is not None) or (irfs_convolution_TF is not None):
            if irfs_convolution_beam is not None:
                apps_beam = True
                psf_fwhm = irfs_convolution_beam.to_value('arcsec')
            else:
                apps_beam = False
                psf_fwhm = 0

            if irfs_convolution_TF is not None:
                apps_TF_LS = True
                TF = irfs_convolution_TF
            else:
                apps_TF_LS = False
                kref = np.linspace(0, 1, 1000)*u.arcsec**-1
                TF = {'k':kref, 'TF':kref.value*0+1}
                
            compton = utils_pk.apply_transfer_function(compton,
                                                       self.get_map_reso().to_value('arcsec'),
                                                       psf_fwhm, TF,
                                                       apps_TF_LS=apps_TF_LS, apps_beam=apps_beam)
            
        return compton


 
