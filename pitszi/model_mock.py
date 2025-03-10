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

from minot.ClusterTools import cluster_global

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
    - get_density_profile
    - get_temperature_profile
    - get_entropy_profile
    - get_Mhse_profile
    - get_overdensity_profile
    - get_Mgas_profile
    - get_fgas_profile
    - get_Ethermal_profile
    - get_pressure_cube_profile
    - get_density_cube_profile
    - get_temperature_cube_profile
    - get_entropy_cube_profile
    - get_pressure_fluctuation_spectrum
    - get_density_fluctuation_spectrum
    - get_temperature_fluctuation_spectrum
    - get_entropy_fluctuation_spectrum
    - get_pressure_cube_fluctuation
    - get_density_cube_fluctuation
    - get_temperature_cube_fluctuation
    - get_entropy_cube_fluctuation
    - get_sz_map
    - get_sx_map
    - get_Tx_map
    
    To Do
    ----------
    - Improve the cube central pixel definition
    - Include Sx normalization 
    - Define the spectra and fluctuation from the density instead of pressure to avoid issue with gamma=0

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
    # Get the electron density profile
    #==================================================

    def get_density_profile(self,
                            radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the thermal electron density profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - n_r (quantity): the electron density profile in unit of cm-3

        """
        
        # In case the input is not an array
        radius = utils.check_qarray(radius, unit='kpc')

        # get profile
        n_r = self._get_generic_profile(radius, self._model_density_profile)
        n_r[radius > self._R_truncation] *= 0

        return radius, n_r.to('cm-3')


    #==================================================
    # Get the electron temperature profile
    #==================================================
    
    def get_temperature_profile(self,
                                radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the thermal temperature profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - T_r (quantity): the temperature profile in unit of keV

        """
        
        # In case the input is not an array
        radius = utils.check_qarray(radius, unit='kpc')

        # Compute n and P
        radius, n_r = self.get_density_profile(radius=radius)
        radius, P_r = self.get_pressure_profile(radius=radius)

        # Get Temperature
        n_r[n_r <= 0] = np.nan
        T_r = P_r / n_r

        # Apply truncation
        T_r[radius > self._R_truncation] = np.nan
        
        return radius, T_r.to('keV')

    
    #==================================================
    # Get the entropy profile
    #==================================================
    
    def get_entropy_profile(self,
                            radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the entropy profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - K_r (quantity): the entropy profile in unit of keV cm2

        """

        # In case the input is not an array
        radius = utils.check_qarray(radius, unit='kpc')

        # Compute n and P
        radius, n_r = self.get_density_profile(radius=radius)
        radius, P_r = self.get_pressure_profile(radius=radius)

        # Get K
        n_r[n_r <= 0] = np.nan
        K_r = P_r / n_r**(5.0/3)

        # Apply truncation
        K_r[radius > self._R_truncation] = np.nan
        
        return radius, K_r.to('keV cm2')


    #==================================================
    # Get the hydrostatic mass profile
    #==================================================

    def get_Mhse_profile(self,
                         radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the hydrostatic mass profile using exact analytical expressions.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        
        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - Mhse_r (quantity): the hydrostatic mass profile in unit of Msun

        """

        # In case the input is not an array
        radius = utils.check_qarray(radius, unit='kpc')
        
        #---------- Mean molecular weights
        mu_gas,mu_e,mu_p,mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                         Z=self._metallicity_sol*self._abundance)

        #---------- Get the electron density profile
        radius, n_r = self.get_density_profile(radius=radius)

        #---------- Get dP/dr
        dpdr_r = self._get_generic_profile(radius, self._model_pressure_profile, derivative=True)
        dpdr_r[radius > self._R_truncation] *= 0

        #---------- Compute the mass
        n_r[n_r <= 0] = np.nan
        Mhse_r = -radius**2 / n_r * dpdr_r / (mu_gas*cst.m_p*cst.G)
        
        Mhse_r[radius > self._R_truncation] = np.nan
        
        return radius, Mhse_r.to('Msun')


    #==================================================
    # Get the HSE overdensity contrast profile
    #==================================================

    def get_overdensity_profile(self,
                                radius=np.logspace(0,4,100)*u.kpc,
                                bHSE=0.0):
        """
        Get the overdensity contrast profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - bHSE (flaat): hydrostatic mass bias to use (assumed constant)
        
        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - delta_r: the overdensity contrast profile

        """

        # In case the input is not an array
        radius = utils.check_qarray(radius, unit='kpc')

        # Compute delta from the mass profile
        r, mhse = self.get_Mhse_profile(radius)
        rho_c = self._cosmo.critical_density(self._redshift)
        delta_r = mhse/(1.0-bHSE) / (4.0/3.0*np.pi*radius**3 * rho_c)

        return radius, delta_r.to_value('')*u.adu
    
    
    #==================================================
    # Compute Mgas
    #==================================================
    
    def get_Mgas_profile(self,
                         radius=np.logspace(0,4,100)*u.kpc,
                         Npt_per_decade_integ=30):
        """
        Get the gas mass profile by integrating the density.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array
        - Npt_per_decade_integ (int): number of point per decade used for integration

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - Mgas_r (quantity): the gas mass profile 
        
        """

        # In case the input is not an array
        radius = utils.check_qarray(radius, unit='kpc')

        #---------- Mean molecular weights
        mu_gas,mu_e,mu_p,mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                         Z=self._metallicity_sol*self._abundance)
        
        #---------- Integrate the mass
        I_n_gas_r = np.zeros(len(radius))
        for i in range(len(radius)):
            # make sure we go well bellow rmax
            rmin = np.amin([self._Rmin.to_value('kpc'), radius.to_value('kpc')[i]/10.0])*u.kpc 
            rad = utils.sampling_array(rmin, radius[i], NptPd=Npt_per_decade_integ, unit=True)
            # To avoid ringing at Rtrunc, insert it if we are above
            if np.amax(rad) > self._R_truncation:
                rad = rad.insert(0, self._R_truncation)
                rad.sort()
            rad, n_r = self.get_density_profile(radius=rad)
            I_n_gas_r[i] = utils.trapz_loglog(4*np.pi*rad**2*n_r, rad)
        
        Mgas_r = mu_e*cst.m_p * I_n_gas_r

        return radius, Mgas_r.to('Msun')


    #==================================================
    # Compute fgas profile
    #==================================================

    def get_fgas_profile(self,
                         radius=np.logspace(0,4,100)*u.kpc,
                         bHSE=0.0,
                         Npt_per_decade_integ=30):
        """
        Get the gas fraction profile.
        
        Parameters
        ----------
        - radius : the physical 3d radius in units homogeneous to kpc, as a 1d array
        - bHSE (flaat): hydrostatic mass bias to use (assumed constant)
        - Npt_per_decade_integ (int): number of point per decade used for integration of Mgas

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - fgas_r (quantity): the gas mass profile 

        """

        # In case the input is not an array
        radius = utils.check_qarray(radius, unit='kpc')

        # Compute fgas from Mgas and Mhse
        r, mgas = self.get_Mgas_profile(radius, Npt_per_decade_integ=Npt_per_decade_integ)
        r, mhse = self.get_Mhse_profile(radius)       
        fgas_r = mgas.to_value('Msun') / mhse.to_value('Msun') * (1.0 - bHSE) 

        return radius, fgas_r*u.adu


    #==================================================
    # Compute thermal energy profile
    #==================================================

    def get_Ethermal_profile(self,
                             radius=np.logspace(0,4,100)*u.kpc,
                             Npt_per_decade_integ=30):
        """
        Compute the thermal energy profile
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array
        - Npt_per_decade_integ (int): number of point per decade used for integration

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - Uth (quantity) : the thermal energy, in GeV

        """
        
        # In case the input is not an array
        radius = utils.check_qarray(radius, unit='kpc')

        #---------- Mean molecular weights
        mu_gas,mu_e,mu_p,mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                         Z=self._metallicity_sol*self._abundance)

        #---------- Integrate the pressure in 3d
        Uth_r = np.zeros(len(radius))*u.erg
        for i in range(len(radius)):
            # make sure we go well bellow rmax
            rmin = np.amin([self._Rmin.to_value('kpc'), radius.to_value('kpc')[i]/10.0])*u.kpc 
            rad = utils.sampling_array(rmin, radius[i], NptPd=Npt_per_decade_integ, unit=True)
            # To avoid ringing at Rtrunc, insert it if we are above
            if np.amax(rad) > self._R_truncation:
                rad = rad.insert(0, self._R_truncation)
                rad.sort()
            rad, p_r = self.get_pressure_profile(radius=rad)
            Uth_r[i] = utils.trapz_loglog((3.0/2.0)*(mu_e/mu_gas) * 4*np.pi*rad**2 * p_r, rad)
                    
        return radius, Uth_r.to('erg')
    

    #==================================================
    # Generic ICM profile to grid
    #==================================================
    
    def _get_icm_cube_profile(self, thermo):
        """
        Fill the grid with the pressure profile.
        
        Parameters
        ----------
        - thermo (str): tell which profile use ('pressure', 'density', 'temperature', 'entropy')

        Outputs
        ----------
        p_grid (ndarray quantity) : the output profile grid with unit attached
        
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
        if thermo == 'pressure':
            rad, p_rad = self.get_pressure_profile(rad)
            p_rmin = self.get_pressure_profile(self._Rmin)[1]
        elif thermo == 'density':
            rad, p_rad = self.get_density_profile(rad)
            p_rmin = self.get_density_profile(self._Rmin)[1]
        elif thermo == 'temperature':
            rad, p_rad = self.get_temperature_profile(rad)
            p_rmin = self.get_density_profile(self._Rmin)[1]
        elif thermo == 'entropy':
            rad, p_rad = self.get_temperature_profile(rad)
            p_rmin = self.get_density_profile(self._Rmin)[1]
        else:
            raise ValueError("Only thermo='pressure', 'density', 'temperature', 'entropy' are implemented")
        unit = p_rad.unit
            
        #----- Interpolate the pressure to the flattened grid
        itpl = interp1d(rad.to_value('kpc'), p_rad.value,
                        kind='linear', fill_value='extrapolate')
        p_grid_flat = itpl(rad_grid_flat)

        #----- Truncate min and max radii
        p_grid_flat[rad_grid_flat <= self._Rmin.to_value('kpc')] = p_rmin.value
        p_grid_flat[rad_grid_flat > self._R_truncation.to_value('kpc')] = 0

        #----- Warning
        if self._silent is False:
            if self._R_truncation.to_value('kpc') < np.amax(rad_grid_flat):
                print('----- WARNING: the truncation radius is smaller than the box size.')
                print('               This induces a discountinuity wich may cause ringing.')
        
        #----- Reshape to the grid
        p_grid = p_grid_flat.reshape(Nz, Ny, Nx)
        
        return p_grid*unit
    
    
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

        return self._get_icm_cube_profile('pressure')

    
    #==================================================
    # Density profile to grid
    #==================================================
    
    def get_density_cube_profile(self):
        """
        Fill the grid with the density profile.
        
        Parameters
        ----------
        
        Outputs
        ----------
        p_grid (ndarray quantity) : the output pressure grid with cm-3 unit attached
        
        """

        return self._get_icm_cube_profile('density')


    #==================================================
    # Temperature profile to grid
    #==================================================
    
    def get_temperature_cube_profile(self):
        """
        Fill the grid with the temperature profile.
        
        Parameters
        ----------
        
        Outputs
        ----------
        p_grid (ndarray quantity) : the output temperature grid with keV unit attached
        
        """

        return self._get_icm_cube_profile('temperature')


    #==================================================
    # Pressure profile to grid
    #==================================================
    
    def get_entropy_cube_profile(self):
        """
        Fill the grid with the entropy profile.
        
        Parameters
        ----------
        
        Outputs
        ----------
        p_grid (ndarray quantity) : the output pressure grid with keV cm2 unit attached
        
        """

        return self._get_icm_cube_profile('entropy')


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
        - kmin/max_norm (quantity): the wavenumber range used for normalization
        - Npt_norm (int): number of point to generagte the spectrum used for normalization

        Outputs
        ----------
        - kvec (quantity): the physical wavenumber in units homogeneous to kpc-1, as a 1d array
        - P3d_k (quantity): the 3D power spectrum in unit homogeneous to kpc^3

        """

        # In case the input is not an array
        kvec = utils.check_qarray(kvec, unit='kpc-1')
        
        # compute the model
        P3d_k = self._get_generic_spectrum(kvec, self._model_pressure_fluctuation,
                                           kmin_norm=kmin_norm,
                                           kmax_norm=kmax_norm,
                                           Npt_norm=Npt_norm)

        return kvec, P3d_k.to('kpc3')
    
    
    #==================================================
    # Get the electron density fluctuation spectrum
    #==================================================

    def get_density_fluctuation_spectrum(self,
                                          kvec=np.logspace(-1,2,1000)*u.Mpc**-1,
                                          kmin_norm=None,
                                          kmax_norm=None,
                                          Npt_norm=10000):
        """
        Get the thermal electron density fluctuation spectrum (delta n / n).
        
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

        kvec, P3d_k = self.get_pressure_fluctuation_spectrum(kvec=kvec,
                                                             kmin_norm=kmin_norm,
                                                             kmax_norm=kmax_norm,
                                                             Npt_norm=Npt_norm)

        return kvec, P3d_k / self.model_gamma_fluctuation**2


    #==================================================
    # Get the temperature fluctuation spectrum
    #==================================================

    def get_temperature_fluctuation_spectrum(self,
                                             kvec=np.logspace(-1,2,1000)*u.Mpc**-1,
                                             kmin_norm=None,
                                             kmax_norm=None,
                                             Npt_norm=10000):
        """
        Get the temperature fluctuation spectrum (delta T / T).
        
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

        kvec, P3d_k = self.get_density_fluctuation_spectrum(kvec=kvec,
                                                            kmin_norm=kmin_norm,
                                                            kmax_norm=kmax_norm,
                                                            Npt_norm=Npt_norm)

        return kvec, P3d_k * (self.model_gamma_fluctuation - 1)**2


    #==================================================
    # Get the temperature fluctuation spectrum
    #==================================================

    def get_entropy_fluctuation_spectrum(self,
                                         kvec=np.logspace(-1,2,1000)*u.Mpc**-1,
                                         kmin_norm=None,
                                         kmax_norm=None,
                                         Npt_norm=10000):
        """
        Get the entropy fluctuation spectrum (delta K / K).
        
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

        kvec, P3d_k = self.get_density_fluctuation_spectrum(kvec=kvec,
                                                            kmin_norm=kmin_norm,
                                                            kmax_norm=kmax_norm,
                                                            Npt_norm=Npt_norm)

        return kvec, P3d_k * (self.model_gamma_fluctuation - 5.0/3)**2
    
    
    #==================================================
    # Pressure fluctuation to grid
    #==================================================
    
    def get_pressure_cube_fluctuation(self,
                                      kmin_input=None,
                                      kmax_input=None,
                                      force_isotropy=False,
                                      Npt=1000):
        """
        Fill the grid with pressure fluctuation, i.e. delta P / P(r)
        
        Parameters
        ----------
        - kmin_input/kmax_input (flaot): the min and max k range, in kpc,
        used for sampling the power spectrum. If None, a default value based on 
        thre model is used
        - force_isotropy (bool): set to true to remove non isotropic k modes
        - Npt (int): the number of points used to sample the power spectrum

        Outputs
        ----------
        - fluctuation_cube (np.ndarray) : the output grid of delta P / P(r)
        
        """

        #----- Model list for which norm is a parameter
        model_list = ['CutoffPowerLaw', 'ModifiedCutoffPowerLaw']
                
        #----- Set a seed
        np.random.seed(self._model_seed_fluctuation)

        #----- Get the grid properties
        Nx, Ny, Nz, proj_reso, proj_reso, los_reso = self.get_3dgrid()

        #----- Bypass if no fluctuations for some models
        if self._model_pressure_fluctuation['name'] in model_list:
            if self._model_pressure_fluctuation['Norm'] == 0:
                return np.zeros((Nz, Ny, Nx))

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
        elif self._model_pressure_fluctuation['statistics'] == 'gaussian':
            pass
        else:
            raise ValueError("Only gaussian or lognormal statistics implemented")
        
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
        elif self._model_pressure_fluctuation['statistics'] == 'gaussian':
            pass
        else:
            raise ValueError("Only gaussian or lognormal statistics implemented")
        
        if not self.silent:
            print('----- INFO: pressure fluctuation cube rms.')
            if self._model_pressure_fluctuation['name'] in model_list:
                print('            Expected rms over the full k range:', self._model_pressure_fluctuation['Norm'])
            print('            Expected rms given the missing k range:', np.sqrt(np.mean(amplitude**2)))
            print('            Actual rms for this noise realization:', np.std(fluctuation_cube))

        return fluctuation_cube
    
    
    #==================================================
    # Density fluctuation to grid
    #==================================================
    
    def get_density_cube_fluctuation(self,
                                     kmin_input=None,
                                     kmax_input=None,
                                     force_isotropy=False,
                                     Npt=1000):
        """
        Fill the grid with pressure fluctuation, i.e. delta P / P(r)
        
        Parameters
        ----------

        Outputs
        ----------
        - fluctuation_cube (np.ndarray): the output grid of delta n / n(r)
        
        """
        
        dP_P_cube = self.get_pressure_cube_fluctuation(kmin_input=kmin_input,
                                                       kmax_input=kmax_input,
                                                       force_isotropy=force_isotropy,
                                                       Npt=Npt)
        
        dn_n_cube = dP_P_cube / self.model_gamma_fluctuation
        
        return dn_n_cube

    
    #==================================================
    # Temperature fluctuation to grid
    #==================================================
    
    def get_temperature_cube_fluctuation(self,
                                         kmin_input=None,
                                         kmax_input=None,
                                         force_isotropy=False,
                                         Npt=1000):
        """
        Fill the grid with temperature fluctuation, i.e. delta T / T(r)
        
        Parameters
        ----------

        Outputs
        ----------
        - fluctuation_cube (np.ndarray): the output grid of delta T / T(r)
        
        """
        
        dn_n_cube = self.get_density_cube_fluctuation(kmin_input=kmin_input,
                                                      kmax_input=kmax_input,
                                                      force_isotropy=force_isotropy,
                                                      Npt=Npt)
        
        dT_T_cube = (self.model_gamma_fluctuation - 1) * dn_n_cube
        
        return dT_T_cube
    

    #==================================================
    # Entropy fluctuation to grid
    #==================================================
    
    def get_entropy_cube_fluctuation(self,
                                     kmin_input=None,
                                     kmax_input=None,
                                     force_isotropy=False,
                                     Npt=1000):
        """
        Fill the grid with entropy fluctuation, i.e. delta K / K(r)
        
        Parameters
        ----------

        Outputs
        ----------
        - fluctuation_cube (np.ndarray): the output grid of delta K / K(r)
        
        """
        
        dn_n_cube = self.get_density_cube_fluctuation(kmin_input=kmin_input,
                                                      kmax_input=kmax_input,
                                                      force_isotropy=force_isotropy,
                                                      Npt=Npt)
        
        dK_K_cube = (self.model_gamma_fluctuation - 5./3) * dn_n_cube
        
        return dK_K_cube
    

    #==================================================
    # Compute SZ map
    #==================================================
    
    def get_sz_map(self,
                   no_fluctuations=False,
                   new_seed=False,
                   force_isotropy=False,
                   irfs_convolution_beam=None,
                   irfs_convolution_TF=None):
                   
        """
        Compute the SZ mock image.
        
        Parameters
        ----------
        - no_fluctuations (bool): set to true when the pure spherical model is requested
        - new_seed (bool): regenerate the seed before computing the map
        - force_isotropy (bool): set to true to remove non isotropic k modes
        - irfs_convolution_beam (quantity): PSF FWHM in unit homogeneous to arcsec. If given, 
        the model is convolved with the beam
        - irfs_convolution_TF (dictionary): if given, the model is convolved 
        with the transfer function       
        
        Outputs
        ----------
        - compton (np.ndarray) : the map in units of Compton parameter
        
        """
        
        #----- Regenerate the seed if needed
        if new_seed: self.new_seed()
        
        #----- Get P(r) grid
        pressure_profile_cube = self.get_pressure_cube_profile()

        #----- Get delta P(x,y,z) grid
        if no_fluctuations:
            pressure_fluctuation_cube = pressure_profile_cube.value*0
        else:
            pressure_fluctuation_cube = self.get_pressure_cube_fluctuation(force_isotropy=force_isotropy)

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
    
    
    #==================================================
    # Compute Sx map
    #==================================================
    
    def get_sx_map(self,
                   no_fluctuations=False,
                   new_seed=False,
                   force_isotropy=False):
                   
        """
        Compute the Sx mock image.
        
        Parameters
        ----------
        - no_fluctuations (bool): set to true when the pure spherical model is requested
        - new_seed (bool): regenerate the seed before computing the map
        - force_isotropy (bool): set to true to remove non isotropic k modes
        
        Outputs
        ----------
        - sx (np.ndarray) : the map in units of TBD
        
        """
        
        #----- Regenerate the seed if needed
        if new_seed: self.new_seed()
        
        #----- Get n(r) and T(r) grid
        density_profile_cube = self.get_density_cube_profile()
        temperature_profile_cube = self.get_temperature_cube_profile()

        #----- Get delta n,T(x,y,z) grid
        if no_fluctuations:
            density_fluctuation_cube     = density_profile_cube.value*0
            temperature_fluctuation_cube = temperature_profile_cube.value*0
        else:
            density_fluctuation_cube     = self.get_density_cube_fluctuation(force_isotropy=force_isotropy)
            temperature_fluctuation_cube = self.get_temperature_cube_fluctuation(force_isotropy=force_isotropy)

        #----- Go to Compton
        n3d = density_profile_cube     * (1 + density_fluctuation_cube)
        T3d = temperature_profile_cube * (1 + temperature_fluctuation_cube)
        integrant = T3d**0.5 * n3d**2
        
        intdl = np.sum(integrant, axis=0) * self._los_reso
        Sx = intdl.value
        
        return Sx

    
    #==================================================
    # Compute Tx map
    #==================================================
    
    def get_Tx_map(self,
                   no_fluctuations=False,
                   new_seed=False,
                   force_isotropy=False):
                   
        """
        Compute the Tx mock image, as temperature weighted by n^2 T^1/2.
        
        Parameters
        ----------
        - no_fluctuations (bool): set to true when the pure spherical model is requested
        - new_seed (bool): regenerate the seed before computing the map
        - force_isotropy (bool): set to true to remove non isotropic k modes
        
        Outputs
        ----------
        - Tx (np.ndarray) : the map in units of keV
        
        """
        
        #----- Regenerate the seed if needed
        if new_seed: self.new_seed()
        
        #----- Get n(r) and T(r) grid
        density_profile_cube     = self.get_density_cube_profile()
        temperature_profile_cube = self.get_temperature_cube_profile()

        #----- Get delta n,T(x,y,z) grid
        if no_fluctuations:
            density_fluctuation_cube     = density_profile_cube.value*0
            temperature_fluctuation_cube = temperature_profile_cube.value*0
        else:
            density_fluctuation_cube     = self.get_density_cube_fluctuation(force_isotropy=force_isotropy)
            temperature_fluctuation_cube = self.get_temperature_cube_fluctuation(force_isotropy=force_isotropy)

        #----- Compute the mean temperature weighted by n^2
        n3d = density_profile_cube     * (1 + density_fluctuation_cube)
        T3d = temperature_profile_cube * (1 + temperature_fluctuation_cube)
        integrant1 = T3d * T3d**0.5*n3d**2
        integrant2 = T3d**0.5*n3d**2
        
        Tx = np.sum(integrant1, axis=0) / np.sum(integrant2, axis=0)
        
        return Tx.to('keV')

