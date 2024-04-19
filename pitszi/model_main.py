"""
This file contains the Model class. It is dedicated to the construction of a
Model object, definined by its physical properties and with associated methods
to compute derived properties or observables.

"""

#==================================================
# Requested imports
#==================================================

import os
import numpy as np
import astropy.units as u
from astropy.io import fits
import astropy.cosmology
from astropy.coordinates import SkyCoord
from astropy import constants as const
from astropy.wcs import WCS
from minot.ClusterTools import cluster_global
import pprint
import dill

from pitszi                  import title
from pitszi.model_library    import ModelLibrary
from pitszi.model_sampling   import ModelSampling
from pitszi.model_mock       import ModelMock


#==================================================
# Cluster Model class
#==================================================

class Model(ModelLibrary, ModelSampling, ModelMock):
    """ Model class.
    This class defines a model object. In addition to basic properties such as 
    mass and redshift, it includes the required physical properties for the modeling
    (pressure profile, pressure fluctuations).

    Attributes
    ----------  
    - silent (bool): print information if False, or not otherwise.
    - output_dir (str): directory where to output data files and plots.
    - cosmology (astropy.cosmology): background cosmological model. Can only be set
    when creating the Cluster object.
    - name (str): the name of the cluster
    - coord (SkyCoord object): the coordinate of the cluster.
    - redshift (float): redshift of the cluster center. Changing the redshift 
    on the fly propagate to cluster properties.
    - D_ang (quantity): can be access but not set directly. The redshift+cosmo
    defines this.
    - D_lum (quantity) : can be access but not set directly. The redshift+cosmo
    defines this.
    - M500 (quantity) : the mass enclosed within R500.
    - R500 (quantity) : the radius in which the density is 500 times the critical 
    density at the cluster redshift
    - theta500 (quantity): the angle corresponding to R500.
    - R_truncation (quantity): the radius at which the cluster stops (similar as virial radius)
    - theta_truncation (quantity): the angle corresponding to R_truncation.
    - Rmin (quantity): minimal scale considered numerically
    - helium_mass_fraction (float): the helium mass fraction of the gas (==Yp~0.25 in BBN)
    - metallicity_sol (float): the metallicity (default is Zprotosun == 0.0153)
    - abundance (float): the abundance (default is 0.3) in unit of Zprotosun
    - model_pressure_profile (dict): the model used for the thermal gas electron pressure 
    profile. It contains the name of the model and the associated model parameters. 
    - model_pressure_fluctuation (dict): the model used for the thermal gas electron pressure 
    fluctuations. It contains the name of the model and the associated model parameters. 
    - map_center (SkyCoord object): the map center coordinates.
    - map_reso (quantity): the map pixel size, homogeneous to degrees.
    - map_fov (list of quantity):  the map field of view as [FoV_x, FoV_y], homogeneous to deg.
    - map_header (standard header): this allows the user to provide a header directly.
    In this case, the map coordinates, field of view and resolution will be extracted 
    from the header and the projection can be arbitrary. If the header is not provided,
    then the projection will be standard RA-DEC tan projection.

    Methods
    ----------  
    Methods are split into the following files
    - title (title page)
    - model_library (model library related functions)
    - model_sampling (grid sampling related functions)
    - model_mock (mock generation related functions)

    ToDo
    ----------  
    - Improve the 3D pressure profile cube to avoid numercial issues near the center
    - deal with the fact that fluctuations can lead to negative pressure: add 'statistics' option

    """

    #==================================================
    # Initialize the cluster object
    #==================================================

    def __init__(self,
                 name='Cluster',
                 RA=0.0*u.deg, Dec=0.0*u.deg,
                 redshift=0.01,
                 M500=1e15*u.Unit('Msun'),
                 cosmology=astropy.cosmology.Planck15,
                 silent=False,
                 output_dir='./pitszi_output',
    ):
        """
        Initialize the cluster object. Several attributes of the class cannot 
        be defined externally because of intrications between parameters. For 
        instance, the cosmology cannot be changed on the fly because this would 
        mess up the internal consistency.
        
        Parameters
        ----------
        - name (str): cluster name 
        - RA, Dec (quantity): coordinates or the cluster in equatorial frame
        - redshift (float) : the cluster center cosmological redshift
        - M500 (quantity): the cluster mass 
        - cosmology (astropy.cosmology): the name of the cosmology to use.
        - silent (bool): set to true in order not to print informations when running 
        - output_dir (str): where to save outputs
        
        """
        
        #---------- Print the code header at launch
        if not silent:
            title.show_model()
            
        #---------- Admin
        self._silent     = silent
        self._output_dir = output_dir

        #---------- Check that the cosmology is indeed a cosmology object
        if hasattr(cosmology, 'h') and hasattr(cosmology, 'Om0'):
            self._cosmo = cosmology
        else:
            raise TypeError("Input cosmology must be an instance of astropy.cosmology")
        
        #---------- Global properties
        self._name     = name
        self._coord    = SkyCoord(RA, Dec, frame="icrs")
        self._redshift = redshift
        self._D_ang    = self._cosmo.angular_diameter_distance(self._redshift)
        self._D_lum    = self._cosmo.luminosity_distance(self._redshift)
        self._M500     = M500
        self._R500     = cluster_global.Mdelta_to_Rdelta(self._M500.to_value('Msun'),
                                                         self._redshift, delta=500,
                                                         cosmo=self._cosmo)*u.kpc
        self._theta500 = ((self._R500 / self._D_ang).to('') * u.rad).to('deg')

        # Initial x, y, z (l.o.s.) axes correspond to min, intermediate, maj axis
        # Euler angles 1, 2, 3 are about z axis, x axis, and z axis
        # see https://en.wikipedia.org/wiki/Euler_angles, https://arxiv.org/pdf/1702.00795.pdf
        self._triaxiality = {'min_to_maj_axis_ratio':1, 'int_to_maj_axis_ratio':1,
                             'euler_angle1':0*u.deg, 'euler_angle2':0*u.deg, 'euler_angle3':0*u.deg}
        
        #---------- Cluster boundary
        self._R_truncation     = 3*self._R500
        self._theta_truncation = ((self._R_truncation / self._D_ang).to('') * u.rad).to('deg')
        self._Rmin = 10.0*u.kpc

        #---------- ICM composition (default: protosolar from Lodders et al 2009: arxiv0901.1149)
        self._helium_mass_fraction = 0.2735
        self._metallicity_sol = 0.0153
        self._abundance = 0.3
        
        #---------- P3D physical properties
        self._model_pressure_profile = 1
        self.set_pressure_profile_universal_param(pressure_model='P13UPP')
        
        dPpar = self._validate_model_fluctuation_parameters({"name":'CutoffPowerLaw',
                                                             "Norm":0.25,
                                                             "slope":-11/3.0, 
                                                             "Linj":1*u.Mpc,
                                                             "Ldis":1*u.kpc})
        self._model_pressure_fluctuation = dPpar
        
        #---------- Sampling
        self._map_center = SkyCoord(RA, Dec, frame="icrs")
        self._map_reso   = 0.01*u.deg
        self._map_fov    = [5.0, 5.0]*u.deg
        self._map_header = None
        self._los_reso   = 10*u.kpc
        self._los_size   = 2 * 2*u.Mpc


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
        print('=============== Current Model() state ===============')
        print('=====================================================')
        
        pp = pprint.PrettyPrinter(indent=4)
        
        par = self.__dict__
        keys = list(par.keys())
        
        for k in range(len(keys)):
            print(('--- '+(keys[k])[1:]))
            print(('    '+str(par[keys[k]])))
            print(('    '+str(type(par[keys[k]]))+''))
        print('=====================================================')


    #==================================================
    # Save parameters
    #==================================================
    
    def save_model(self):
        """
        Save the current model object.
        
        Parameters
        ----------
            
        Outputs
        ----------
        The parameters are saved in the output directory

        """

        # Create the output directory if needed
        if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)

        # Save
        with open(self.output_dir+'/model_parameters.pkl', 'wb') as pfile:
            #pickle.dump(self.__dict__, pfile, pickle.HIGHEST_PROTOCOL)
            dill.dump(self.__dict__, pfile)

        # Text file for user
        par = self.__dict__
        keys = list(par.keys())
        with open(self.output_dir+'/model_parameters.txt', 'w') as txtfile:
            for k in range(len(keys)):
                txtfile.write('--- '+(keys[k])+'\n')
                txtfile.write('    '+str(par[keys[k]])+'\n')
                txtfile.write('    '+str(type(par[keys[k]]))+'\n')

                
    #==================================================
    # Load parameters
    #==================================================
    
    def load_model(self, param_file):
        """
        Read a given parameter file to re-initialize the model object.
        
        Parameters
        ----------
        param_file (str): the parameter file to be read
            
        Outputs
        ----------
            
        """

        with open(param_file, 'rb') as pfile:
            par = dill.load(pfile)
            
        self.__dict__ = par
        
        
    #==================================================
    # Get the hidden variable
    #==================================================
    
    #========== Admin
    @property
    def silent(self):
        return self._silent
    
    @property
    def output_dir(self):
        return self._output_dir
    
    #========== Cosmology
    @property
    def cosmo(self):
        return self._cosmo
    
    #========== Global properties
    @property
    def name(self):
        return self._name
    
    @property
    def coord(self):
        return self._coord
    
    @property
    def redshift(self):
        return self._redshift
    
    @property
    def D_ang(self):
        return self._D_ang
    
    @property
    def D_lum(self):
        return self._D_lum
    
    @property
    def M500(self):
        return self._M500
    
    @property
    def R500(self):
        return self._R500
    
    @property
    def theta500(self):
        return self._theta500

    @property
    def triaxiality(self):
        return self._triaxiality
 
    #========== Cluster boundary
    @property
    def R_truncation(self):
        return self._R_truncation
    
    @property
    def theta_truncation(self):
        return self._theta_truncation
    
    @property
    def Rmin(self):
        return self._Rmin

    #========== ICM composition
    @property
    def helium_mass_fraction(self):
        return self._helium_mass_fraction

    @property
    def abundance(self):
        return self._abundance

    @property
    def metallicity_sol(self):
        return self._metallicity_sol
    
    #========== ICM physics
    @property
    def model_pressure_profile(self):
        return self._model_pressure_profile
    
    @property
    def model_pressure_fluctuation(self):
        return self._model_pressure_fluctuation
    
    #========== Maps parameters
    @property
    def map_center(self):
        return self.get_map_center()
    
    @property
    def map_reso(self):
        return self.get_map_reso()
    
    @property
    def map_fov(self):
        return self.get_map_fov()
    
    @property
    def map_header(self):
        return self.get_map_header()

    @property
    def los_reso(self):
        return self._los_reso

    @property
    def los_size(self):
        return self._los_size

    
    #==================================================
    # Defines how the user can pass arguments and interconnections
    #==================================================
    
    #========== Admin
    @silent.setter
    def silent(self, value):
        # Check value and set
        if type(value) == bool:
            self._silent = value
        else:
            raise TypeError("The silent parameter should be a boolean.")

        # Information
        if not self._silent: print("Setting silent value")

    @output_dir.setter
    def output_dir(self, value):
        # Check value and set
        if type(value) == str:
            self._output_dir = value
        else:
            raise TypeError("The output_dir should be a string.")
        
        # Information
        if not self._silent: print("Setting output_dir value")

    #========== Cosmology
    @cosmo.setter
    def cosmo(self, value):
        message = ("The cosmology can only be set when defining the cluster object, "
                   "as clust = Cluster(cosmology=astropy.cosmology.YourCosmology).  "
                   "Doing nothing.                                                  ")
        if not self._silent: print(message)
    

    #========== Global properties
    @name.setter
    def name(self, value):
        # Check value and set
        if type(value) == str:
            self._name = value
        else:
            raise TypeError("The name should be a string.")

        # Information
        if not self._silent: print("Setting name value")

    @coord.setter
    def coord(self, value):
        # Case value is a SkyCoord object, nothing to be done
        if type(value) == astropy.coordinates.sky_coordinate.SkyCoord:
            self._coord = value

        # Case value is standard coordinates
        elif type(value) == dict:
            
            # It is not possible to have both RA-Dec and Glat-Glon, or just RA and not Dec, etc
            cond1 = 'RA'  in list(value.keys()) and 'Glat' in list(value.keys())
            cond2 = 'RA'  in list(value.keys()) and 'Glon' in list(value.keys())
            cond3 = 'Dec' in list(value.keys()) and 'Glat' in list(value.keys())
            cond4 = 'Dec' in list(value.keys()) and 'Glon' in list(value.keys())
            if cond1 or cond2 or cond3 or cond4:
                raise TypeError("The coordinates can be a coord object, or a {'RA','Dec'} or {'Glon', 'Glat'} dictionary.")
            
            # Case where RA-Dec is used
            if 'RA' in list(value.keys()) and 'Dec' in list(value.keys()):
                self._coord = SkyCoord(value['RA'], value['Dec'], frame="icrs")

            # Case where Glon-Glat is used
            elif 'Glon' in list(value.keys()) and 'Glat' in list(value.keys()):
                self._coord = SkyCoord(value['Glon'], value['Glat'], frame="galactic")

            # Otherwise, not appropriate value
            else:
                err_message = ("The coordinates can be a coord object, "
                               "a {'RA','Dec'} dictionary, or a {'Glon', 'Glat'} dictionary.")
                raise TypeError(err_message)

        # Case value is not accepted
        else:
            raise TypeError("The coordinates can be a coord object, a {'RA','Dec'} dictionary, or a {'Glon', 'Glat'} dictionary.")

        # Information
        if not self._silent: print("Setting coord value")

    @redshift.setter
    def redshift(self, value):
        # check type
        if type(value) != float and type(value) != int and type(value) != np.float64:
            raise TypeError("The redshift should be a int or a float.")
        
        # value check
        if value < 0 :
            raise ValueError("The redshift should be larger or equal to 0.")

        # Setting parameters
        self._redshift = value
        self._D_ang = self._cosmo.angular_diameter_distance(self._redshift)
        self._D_lum = self._cosmo.luminosity_distance(self._redshift)
        self._R500  = cluster_global.Mdelta_to_Rdelta(self._M500.to_value('Msun'),
                                                      self._redshift, delta=500, cosmo=self._cosmo)*u.kpc            
        self._theta500 = ((self._R500 / self._D_ang).to('') * u.rad).to('deg')
        self._theta_truncation = ((self._R_truncation / self._D_ang).to('') * u.rad).to('deg')
        
        # Information
        if not self._silent: print("Setting redshift value")
        if not self._silent: print("Setting: D_ang, D_lum, R500, theta500, theta_truncation ; Fixing: cosmo.")
        
    @D_ang.setter
    def D_ang(self, value):
        if not self._silent: print("The angular diameter distance cannot be set directly, the redshift has to be used instead.")
        if not self._silent: print("Doing nothing.                                                                            ")
        
    @D_lum.setter
    def D_lum(self, value):
        if not self._silent: print("The luminosity distance cannot be set directly, the redshift has to be used instead.")
        if not self._silent: print("Doing nothing.                                                                      ")
        
    @M500.setter
    def M500(self, value):
        # Type check
        try:
            test = value.to('Msun')
        except:
            raise TypeError("The mass M500 should be a quantity homogeneous to Msun.")

        # Value check
        if value <= 0 :
            raise ValueError("Mass M500 should be larger than 0")
        if value.to_value('Msun') < 1e10 :
            print("Warning! Your are setting the mass to a tiny value (i.e. not the cluster regime). This may lead to issues")
        
        # Setting parameters
        self._M500 = value
        self._R500 = cluster_global.Mdelta_to_Rdelta(self._M500.to_value('Msun'),
                                                     self._redshift, delta=500, cosmo=self._cosmo)*u.kpc
        self._theta500 = ((self._R500 / self._D_ang).to('') * u.rad).to('deg')
        
        # Information
        if not self._silent: print("Setting M500 value")
        if not self._silent: print("Setting: R500, theta500 ; Fixing: redshift, cosmo, D_ang")
        
    @R500.setter
    def R500(self, value):
        # Check type
        try:
            test = value.to('kpc')
        except:
            raise TypeError("The radius R500 should be a quantity homogeneous to kpc.")

        # check value
        if value < 0 :
            raise ValueError("Radius R500 should be larger than 0")

        # Setting parameter
        self._R500 = value
        self._theta500 = ((self._R500 / self._D_ang).to('') * u.rad).to('deg')
        self._M500 = cluster_global.Rdelta_to_Mdelta(self._R500.to_value('kpc'),
                                                     self._redshift, delta=500, cosmo=self._cosmo)*u.Msun
        
        # Information
        if not self._silent: print("Setting R500 value")
        if not self._silent: print("Setting: theta500, M500 ; Fixing: redshift, cosmo, D_ang")
        
    @theta500.setter
    def theta500(self, value):
        # Check type
        try:
            test = value.to('deg')
        except:
            raise TypeError("The angle theta500 should be a quantity homogeneous to deg.")
        
        # Check value
        if value <= 0 :
            raise ValueError("Angle theta500 should be larger than 0")

        # Setting parameters
        self._theta500 = value
        self._R500 = value.to_value('rad')*self._D_ang
        self._M500 = cluster_global.Rdelta_to_Mdelta(self._R500.to_value('kpc'),
                                                     self._redshift, delta=500, cosmo=self._cosmo)*u.Msun
        
        # Information
        if not self._silent: print("Setting theta500 value")
        if not self._silent: print("Setting: R500, M500 ; Fixing: redshift, cosmo, D_ang")

    @triaxiality.setter
    def triaxiality(self, value):
        # Check type
        if type(value) != dict :
            raise TypeError("The triaxiality should be a dictionary with relevant keys")

        cond1 = 'min_to_maj_axis_ratio' in list(value.keys()) and 'int_to_maj_axis_ratio' in list(value.keys())
        cond2 = 'euler_angle1' in list(value.keys()) and 'euler_angle2' in list(value.keys()) and 'euler_angle3' in list(value.keys())
        if not (cond1 and cond2):
            raise ValueError("The triaxiality should contain {min/int}_to_maj_axis_ratio and euler_angle{1,2,3} parameters")

        # Test units
        try:
            test = value['euler_angle1'].to('deg')
            test = value['euler_angle2'].to('deg')
            test = value['euler_angle3'].to('deg')
        except:
            raise ValueError("The euler_angle_{1,2,3} should have unit homogeneous to deg")

        # Test values
        cond1 = value['min_to_maj_axis_ratio'] <= 0 
        cond2 = value['min_to_maj_axis_ratio'] >  1
        cond3 = value['int_to_maj_axis_ratio'] <= 0 
        cond4 = value['int_to_maj_axis_ratio'] >  1
        cond5 = value['min_to_maj_axis_ratio'] > value['int_to_maj_axis_ratio']
        if cond1 or cond2 or cond3 or cond4 or cond5:
            raise ValueError("By definition, we should have 0 < min_to_maj_axis_ratio <= int_to_maj_axis_ratio <= 1")
        
        # Setting parameters
        self._triaxiality = {'min_to_maj_axis_ratio': value['min_to_maj_axis_ratio'],
                             'int_to_maj_axis_ratio': value['int_to_maj_axis_ratio'],
                             'euler_angle1': value['euler_angle1'].to('deg'),
                             'euler_angle2': value['euler_angle2'].to('deg'),
                             'euler_angle3': value['euler_angle3'].to('deg')}
        
        
    #========== Cluster boundary
    @R_truncation.setter
    def R_truncation(self, value):
        # Check type
        try:
            test = value.to('kpc')
        except:
            raise TypeError("The radius R_truncation should be a quantity homogeneous to kpc.")

        # check value
        #if value <= self._R500 :
        #    raise ValueError("Radius R_truncation should be larger than R500 for internal consistency.")

        # Set parameters
        self._R_truncation = value
        self._theta_truncation = ((self._R_truncation / self._D_ang).to('') * u.rad).to('deg')
        
        # Information
        if not self._silent: print("Setting R_truncation value")
        if not self._silent: print("Setting: theta_truncation ; Fixing: D_ang")
        
    @theta_truncation.setter
    def theta_truncation(self, value):
        # Check type
        try:
            test = value.to('deg')
        except:
            raise TypeError("The angle theta_truncation should be a quantity homogeneous to deg.")
        
        # check value
        if value <= self._theta500 :
            raise ValueError("Angle theta_truncation should be larger than theta500 for internal consistency.")

        # Set parameters
        self._theta_truncation = value
        self._R_truncation = value.to_value('rad') * self._D_ang
        
        # Information
        if not self._silent: print("Setting theta_truncation value")
        if not self._silent: print("Setting: R_truncation ; Fixing: D_ang")

    @Rmin.setter
    def Rmin(self, value):
        # Check type
        try:
            test = value.to('kpc')
        except:
            raise TypeError("The radius Rmin should be a quantity homogeneous to kpc.")

        # Check value
        if value.to_value('kpc') <= 0:
            raise TypeError("Rmin cannot be 0 (or less than 0) because integrations are in log space.")

        if value.to_value('kpc') < 1e-2:
            if not self._silent: 
                print("WARNING: the requested value of Rmin is very small. Rmin~kpc is expected")

        # Set parameters
        self._Rmin = value
        
        # Information
        if not self._silent: print("Setting Rmin value")

    #========== ICM composition
    @helium_mass_fraction.setter
    def helium_mass_fraction(self, value):
        # Check type
        if type(value) != float and type(value) != int and type(value) != np.float64:
            raise TypeError("The helium mass fraction should be a float or int")

        # Check value
        if value > 1.0 or value < 0.0:
            raise ValueError("The helium mass fraction should be between 0 and 1")

        # Set parameters
        self._helium_mass_fraction = value
        
        # Information
        if not self._silent: print("Setting helium mass fraction value")
        
    @metallicity_sol.setter
    def metallicity_sol(self, value):
        # Check type
        if type(value) != float and type(value) != int and type(value) != np.float64:
            raise TypeError("The metallicity should be a float")
        
        # Check value
        if value < 0.0:
            raise ValueError("The metallicity should be >= 0")

        # Set parameters
        self._metallicity_sol = value
        
        # Information
        if not self._silent: print("Setting metallicity value")

    @abundance.setter
    def abundance(self, value):
        # Check type
        if type(value) != float and type(value) != int and type(value) != np.float64:
            raise TypeError("The abundance should be a float")

        # Check value
        if value < 0.0:
            raise ValueError("The abundance should be >= 0")

        # Set parameters
        self._abundance = value
        
        # Information
        if not self._silent: print("Setting abundance value")
        
    #========== ICM physics
    @model_pressure_profile.setter
    def model_pressure_profile(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The model of pressure profile should be a dictionary with relevant keys")
        
        # Check the input parameters and use it
        Ppar = self._validate_model_profile_parameters(value, 'keV cm-3')
        self._model_pressure_profile = Ppar
        
        # Information
        if not self._silent: print("Setting model_pressure_profile value")
        if not self._silent: print("Fixing: R500 if involved")

    @model_pressure_fluctuation.setter
    def model_pressure_fluctuation(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The model of pressure fluctuation should be a dictionary with relevant keys")
        
        # Check the input parameters and use it
        Ppar = self._validate_model_fluctuation_parameters(value)
        self._model_pressure_fluctuation = Ppar
        
        # Information
        if not self._silent: print("Setting model_pressure_fluctuation value")
        if not self._silent: print("Fixing: R500 if involved")

        
    #========== Sampling
    @map_center.setter
    def map_center(self, value):
        err_msg = ("The coordinates can be a coord object, "
                   "or a {'RA','Dec'} or {'Glon', 'Glat'} dictionary.")
        
        # Case value is a SkyCoord object
        if type(value) == astropy.coordinates.sky_coordinate.SkyCoord:
            self._map_center = value
    
        # Case value is standard coordinates
        elif type(value) == dict:
            
            # It is not possible to have both RA-Dec and Glat-Glon, or just RA and not Dec, etc
            cond1 = 'RA'  in list(value.keys()) and 'Glat' in list(value.keys())
            cond2 = 'RA'  in list(value.keys()) and 'Glon' in list(value.keys())
            cond3 = 'Dec' in list(value.keys()) and 'Glat' in list(value.keys())
            cond4 = 'Dec' in list(value.keys()) and 'Glon' in list(value.keys())
            if cond1 or cond2 or cond3 or cond4:
                raise ValueError(err_msg)
            
            # Case where RA-Dec is used
            if 'RA' in list(value.keys()) and 'Dec' in list(value.keys()):
                self._map_center = SkyCoord(value['RA'], value['Dec'], frame="icrs")
    
            # Case where Glon-Glat is used
            elif 'Glon' in list(value.keys()) and 'Glat' in list(value.keys()):
                self._map_center = SkyCoord(value['Glon'], value['Glat'], frame="galactic")
    
            # Otherwise, not appropriate value
            else:
                raise TypeError(err_msg)
    
        # Case value is not accepted
        else:
            raise TypeError(err_msg)
    
        # Header to None
        self._map_header = None
    
        # Information
        if not self._silent: print("Setting the map coordinates")
        if not self._silent: print("Setting: map_header to None, as map properties are now set by hand")

    @map_reso.setter
    def map_reso(self, value):
        # check type
        try:
            test = value.to('deg')
        except:
            raise TypeError("The map resolution should be a quantity homogeneous to deg.")
    
        # check value
        if type(value.value) != float and type(value.value) != int and type(value.value) != np.float64:        
            raise TypeError("The map resolution should be a scalar, e.i. reso_x = reso_y.")
    
        # Set parameters
        self._map_reso = value
        self._map_header = None
        
        # Information
        if not self._silent: print("Setting the map resolution value")
        if not self._silent: print("Setting: map_header to None, as map properties are now set by hand")

    @map_fov.setter
    def map_fov(self, value):
        # check type
        try:
            test = value.to('deg')
        except:
            raise TypeError("The map field of view should be a quantity homogeneous to deg.")
    
        # Set parameters for single value application
        if type(value.value) == float or type(value.value) == int or type(value.value) == np.float64 :        
            self._map_fov = [value.to_value('deg'), value.to_value('deg')] * u.deg
    
        # Set parameters for single value application
        elif type(value.value) == np.ndarray:
            # check the dimension
            if len(value) == 2:
                self._map_fov = value
            else:
                raise TypeError("The map field of view is either a scalar, or a 2d list quantity.")
    
        # No other options
        else:
            raise TypeError("The map field of view is either a scalar, or a 2d list quantity.")
    
        # Set extra parameters
        self._map_header = None
    
        # Information
        if not self._silent: print("Setting the map field of view")
        if not self._silent: print("Setting: map_header to None, as map properties are now set by hand")

    @map_header.setter
    def map_header(self, value):
        # Check the header by reading it with WCS
        try:
            w = WCS(value)
            data_tpl = np.zeros((value['NAXIS2'], value['NAXIS1']))            
            header = w.to_header()
            hdu = fits.PrimaryHDU(header=header, data=data_tpl)
            header = hdu.header
        except:
            raise TypeError("Issue detected with the header, may not contain NAXIS1,2.")
        
        # set the value
        self._map_header = header
        self._map_center = None
        self._map_reso   = None
        self._map_fov    = None
    
        # Information
        if not self._silent: print("Setting the map header")
        if not self._silent: print("Setting: map_center, map_reso, map_fov to None, as the header will be used")

    @los_reso.setter
    def los_reso(self, value):
        # check type
        try:
            test = value.to('kpc')
        except:
            raise TypeError("The l.o.s. resolution should be a quantity homogeneous to kpc.")
    
        # check value
        if type(value.value) != float and type(value.value) != int and type(value.value) != np.float64:        
            raise TypeError("The l.o.s. resolution should be a scalar.")
    
        # Set parameters
        self._los_reso = value
        
        # Information
        if not self._silent: print("Setting the l.o.s. resolution value")

    @los_size.setter
    def los_size(self, value):
        # check type
        try:
            test = value.to('kpc')
        except:
            raise TypeError("The l.o.s. size should be a quantity homogeneous to kpc.")
    
        # check value
        if type(value.value) != float and type(value.value) != int and type(value.value) != np.float64:        
            raise TypeError("The l.o.s. size should be a scalar.")
    
        # Set parameters
        self._los_size = value
        
        # Information
        if not self._silent: print("Setting the l.o.s size value")

