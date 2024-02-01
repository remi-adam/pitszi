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
from scipy.spatial.transform import Rotation as R

from minot.ClusterTools import map_tools

from pitszi import utils


#==================================================
# Mock class
#==================================================

class ModelSampling(object):
    """ ModelSampling class
    This class serves as a parser to the main Model class, to
    include the subclass ModelSampling in this other file.

    Attributes
    ----------
    The attributes are the same as the Model class, see model_main.py

    Methods
    ----------
    - get_map_header
    - get_map_reso
    - get_map_fov
    - get_map_center
    - get_3dgrid
    
    """

    #==================================================
    # Extract the header
    #==================================================
    
    def get_map_header(self):
        """
        Extract the header of the map
        
        Parameters
        ----------

        Outputs
        ----------
        - header (astropy object): the header associated to the map

        """

        # Get the needed parameters in case of map header
        if self._map_header is not None:
            header = self._map_header
            
        # Get the needed parameters in case of set-by-hand map parameters
        elif (self._map_center is not None) and (self._map_reso is not None) and (self._map_fov is not None):
            header = map_tools.define_std_header(self._map_center.icrs.ra.to_value('deg'),
                                                 self._map_center.icrs.dec.to_value('deg'),
                                                 self._map_fov.to_value('deg')[0],
                                                 self._map_fov.to_value('deg')[1],
                                                 self._map_reso.to_value('deg'))
        
        # Otherwise there is a problem
        else:
            raise TypeError("A header, or the map_center & map_reso & map_fov should be defined.")

        return header        

    
    #==================================================
    # Extract the resolution
    #==================================================
    
    def get_map_reso(self, physical=False):
        """
        Extract the resolution of the map
        
        Parameters
        ----------
        - physical (bool): set to true to have the size in kpc

        Outputs
        ----------
        - reso (quantity): the resolution associated to the map

        """

        header = self.get_map_header()

        dx =  np.abs(header['CDELT1'])
        dy =  np.abs(header['CDELT2'])
        if dx != dy:
            raise ValueError('The map should have the same pixel resolution along x and y')

        if physical:
            map_reso = (dx*u.deg).to_value('radian')*self._D_ang.to('kpc')
        else:
            map_reso = dx*u.deg  
        
        return map_reso
        

    #==================================================
    # Extract the field of view
    #==================================================
    
    def get_map_fov(self, physical=False):
        """
        Extract the field of view of the map
        
        Parameters
        ----------
        - physical (bool): set to true to have the size in kpc

        Outputs
        ----------
        - fov (quantity): the field of view associated to the map

        """

        header = self.get_map_header()
        reso   = self.get_map_reso().to_value('deg')
        nx     = header['NAXIS1']
        ny     = header['NAXIS2']

        if physical:
            map_fov = [nx*reso*np.pi/180*self._D_ang.to_value('kpc'),
                       ny*reso*np.pi/180*self._D_ang.to_value('kpc')]*u.kpc
        else:
            map_fov = [nx*reso, ny*reso]*u.deg
        
        return map_fov

    
    #==================================================
    # Get the center of the map
    #==================================================
    
    def get_map_center(self):
        """
        Extract the coordinates of the center of the map
        
        Parameters
        ----------
        
        Outputs
        ----------
        - map_center (SkyCoord object): the center of the map
        
        """
        
        header = self.get_map_header()
        Nx = header['NAXIS1']
        Ny = header['NAXIS2']
        ramap, decmap = map_tools.get_radec_map(header) # indexing is xy in this function

        #----- Case of even even
        if (Nx/2.0 == int(Nx/2.0)) and (Ny/2.0 == int(Ny/2.0)):
            ramap_center  = np.mean(ramap[int(Ny/2.0)-1:int(Ny/2.0), int(Nx/2.0)-1:int(Nx/2.0)])
            decmap_center = np.mean(decmap[int(Ny/2.0)-1:int(Ny/2.0), int(Nx/2.0)-1:int(Nx/2.0)])
     
        #----- Case of odd odd
        if (Nx/2.0 != int(Nx/2.0)) and (Ny/2.0 != int(Ny/2.0)):
            ramap_center  = ramap[int(Ny/2.0), int(Nx/2.0)]
            decmap_center = decmap[int(Ny/2.0), int(Nx/2.0)]
     
        #----- Case of even odd
        if (Nx/2.0 == int(Nx/2.0)) and (Ny/2.0 != int(Ny/2.0)):
            ramap_center  = np.mean(ramap[int(Ny/2.0), int(Nx/2.0)-1:int(Nx/2.0)])
            decmap_center = np.mean(decmap[int(Ny/2.0), int(Nx/2.0)-1:int(Nx/2.0)])
     
        #----- Case of odd even
        if (Nx/2.0 != int(Nx/2.0)) and (Ny/2.0 == int(Ny/2.0)):
            ramap_center  = np.mean(ramap[int(Ny/2.0)-1:int(Ny/2.0), int(Nx/2.0)])
            decmap_center = np.mean(decmap[int(Ny/2.0)-1:int(Ny/2.0), int(Nx/2.0)])
                
        map_center = SkyCoord(ramap_center*u.deg, decmap_center*u.deg, frame='icrs')
     
        return map_center        

    
    #==================================================
    # Build the 3d grid
    #==================================================
    
    def get_3dgrid(self):
        """
        Build the physical 3D grid.
        
        Parameters
        ----------
        
        Outputs
        ----------
        - Nx,Ny,Nz (int): the grid number of pixel along each axis
        - proj_reso (float): the pixel size in the projected direction in kpc
        - los_reso (float): the pixel size alonog the line of sight in kpc
        
        """

        # Get the X,Y direction
        header = self.get_map_header()                       # header of the output map
        Nx = header['NAXIS1']
        Ny = header['NAXIS2']
        if np.abs(header['CDELT1']) != np.abs(header['CDELT2']):
            raise ValueError("Only maps with the same resolution towards x and y are allowed.")
        proj_reso_deg = np.abs(header['CDELT2'])
        proj_reso = self._D_ang.to_value('kpc')*proj_reso_deg*np.pi/180
        
        # Define the z axis
        Nz_min = (self._los_size / self._los_reso).to_value('')
        Nz = np.int(np.ceil(Nz_min))
        
        return Nx, Ny, Nz, proj_reso, proj_reso, self._los_reso.to_value('kpc')


    #==================================================
    # Give information relative to sampling
    #==================================================
    
    def give_sampling_information(self):
        """
        Output information relative to the sampling.
        
        Parameters
        ----------
        
        Outputs
        ----------
        Print information
        
        """

        Nx, Ny, Nz, proj_reso, proj_reso, los_reso = self.get_3dgrid()
        
        print('=====================================================')
        print('===== Information relative to the grid sampling =====')                
        print('=====================================================')
        print('   Grid size :', Nx, ',', Ny, ',', Nz)
        reso_kpc = self.get_map_reso(physical=True).to_value('kpc')
        reso_arcsec = self.get_map_reso(physical=False).to_value('arcsec')
        reso_kpc_s = "{:^10.1f}".format(reso_kpc)
        reso_arcsec_s = "{:^10.1f}".format(reso_arcsec)
        print('   Pixel size :  ', reso_kpc_s, ' kpc ; ', reso_arcsec_s, ' arcsec')

        fov_kpc = self.get_map_fov(physical=True).to_value('kpc')
        fov_arcsec = self.get_map_fov(physical=False).to_value('arcmin')
        print('   Fov size :  [', "{:^10.1f}".format(fov_kpc[0]),
              ',', "{:^10.1f}".format(fov_kpc[1]),
              '] kpc ; [',
              "{:^10.3f}".format(fov_arcsec[0]),',',
              "{:^10.3f}".format(fov_arcsec[1]),'] arcmin')
        print('   L.o.S. resolution :     ', "{:^10.1f}".format(los_reso),' kpc')
        print('   Map center :  ', self.get_map_center())
        k_proj = np.fft.fftfreq(Nx, reso_arcsec)
        print('   k min/max projected :     ',
              "{:^10.6f}".format(np.amin(k_proj[k_proj>0])),'/',
              "{:^10.6f}".format(np.amax(k_proj)),' 1/arcsec')
        k_proj = np.fft.fftfreq(Nx, proj_reso)
        print('   k min/max projected :     ',
              "{:^10.6f}".format(np.amin(k_proj[k_proj>0])),'/',
              "{:^10.6f}".format(np.amax(k_proj)), ' 1/kpc')
        k_z = np.fft.fftfreq(Nz, los_reso)
        conv = self.D_ang.to_value('kpc')/(1*u.rad).to_value('arcsec')
        print('   k min/max L.o.S. (eq.) :  ',
              "{:^10.6f}".format(conv*np.amin(k_z[k_z>0])),'/',
              "{:^10.6f}".format(conv*np.amax(k_z)), ' 1/arcsec')
        print('   k min/max L.o.S. :        ',
              "{:^10.6f}".format(np.amin(k_z[k_z>0])),'/',
              "{:^10.6f}".format(np.amax(k_z)), ' 1/kpc')
        print('=====================================================')


        
