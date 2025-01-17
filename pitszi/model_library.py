"""
This file deals with the model parametrization

"""

import astropy.units as u
import numpy as np
from scipy import interpolate
import warnings
from astropy import constants as const
from scipy.interpolate import interp1d

from minot.ClusterTools import cluster_global
from minot.ClusterTools import cluster_profile


#==================================================
# Library class
#==================================================

class ModelLibrary(object):
    """ ModelAdmin class
    This class searves as a parser to the main Model class, to
    include the subclass ModelAdmin in this other file. All the definitions of the
    model parameters should be here.

    Profile models are now:  ['GNFW', 'SVM', 'beta', 'doublebeta']

    Attributes
    ----------
    The attributes are the same as the Model class, see model_main.py

    Methods
    ----------
    - set_pressure_profile_universal_param
    - set_density_profile_universal_param
    - set_pressure_profile_polytropic_param
    - set_density_profile_polytropic_param
    - set_pressure_profile_isoT_param
    - set_density_profile_isoT_param
    - set_pressure_profile_from_temperature_model
    - set_density_profile_from_temperature_model
    - _validate_profile_model_parameters
    - _get_generic_profile

    To Do
    ----------

    """

    #==================================================
    # Set a given pressure UPP profile
    #==================================================

    def set_pressure_profile_universal_param(self, pressure_model='P13UPP'):
        """
        Set the parameters of the pressure profile:
        P0, c500 (and r_p given R500), gamma, alpha, beta
        
        Parameters
        ----------
        - pressure_model (str): available models are
            'A10UPP' (Universal Pressure Profile from Arnaud et al. 2010)
            'A10CC' (Cool-Core Profile from Arnaud et al. 2010)
            'A10MD' (Morphologically-Disturbed Profile from Arnaud et al. 2010)
            'P13UPP' (Planck Intermediate Paper V (2013) Universal Pressure Profile)
            'G19UPP' (Ghirardini et al 2019, all clusters)
            'G19CC' (Ghirardini et al 2019, cool-core clusters)
            'G19MD' (Ghirardini et al 2019, morphologically disturbed clusters)
        """

        # Arnaud et al. (2010) : Universal Pressure Profile parameters
        if pressure_model == 'A10UPP':
            if not self._silent: print('Setting gNFW Arnaud et al. (2010) UPP.')
            pppar = [8.403, 1.177, 0.3081, 1.0510, 5.4905]

        # Arnaud et al. (2010) : Cool-Core clusters parameters
        elif pressure_model == 'A10CC':
            if not self._silent: print('Setting gNFW Arnaud et al. (2010) cool-core.')
            pppar = [3.249, 1.128, 0.7736, 1.2223, 5.4905]

        # Arnaud et al. (2010) : Morphologically-Disturbed clusters parameters
        elif pressure_model == 'A10MD':
            if not self._silent: print('Setting gNFW Arnaud et al. (2010) morphologically disturbed.')
            pppar = [3.202, 1.083, 0.3798, 1.4063, 5.4905]

        # Planck Intermediate Paper V (2013) : Universal Pressure Profile parameters
        elif pressure_model == 'P13UPP':
            if not self._silent: print('Setting gNFW Planck coll. (2013) UPP.')
            pppar = [6.410, 1.810, 0.3100, 1.3300, 4.1300]

        # Ghirardini (2019) : Universal Pressure Profile parameters
        elif pressure_model == 'G19UPP':
            if not self._silent: print('Setting gNFW Ghirardini (2019) UPP.')
            pppar = [5.68, 1.49, 0.43, 1.33, 4.40]

        # Ghirardini (2019) : Universal Pressure Profile parameters
        elif pressure_model == 'G19CC':
            if not self._silent: print('Setting gNFW Ghirardini (2019) CC.')
            pppar = [6.03, 1.68, 0.51, 1.33, 4.37]

            # Ghirardini (2019) : Universal Pressure Profile parameters
        elif pressure_model == 'G19MD':
            if not self._silent: print('Setting gNFW Ghirardini (2019) MD.')
            pppar = [7.96, 1.79, 0.29, 1.33, 4.05]

        # No other profiles available
        else:
            raise ValueError('P(r) model not available. Use A10UPP,A10CC,A10MD,P13UPP,G19UPP,G19CC,G19MD.')

        # Compute the normalization
        if pressure_model in ['A10UPP', 'A10CC', 'A10MD', 'P13UPP']:
            mu_g,mu_e,mu_p,mu_a = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                       Z=self._metallicity_sol*self._abundance)
            try:
                fb = self._cosmo.Ob0/self._cosmo.Om0
            except:
                fb = 0.16
            Pnorm = cluster_global.gNFW_normalization(self._redshift,
                                                      self._M500.to_value('Msun'),
                                                      cosmo=self._cosmo,
                                                      mu=mu_g, mue=mu_e, fb=fb)
        if pressure_model in ['G19UPP', 'G19CC', 'G19MD']:
            E_z   = self._cosmo.efunc(self._redshift)
            h70   = self._cosmo.H0.value/70.0
            mu_g,mu_e,mu_p,mu_a = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                       Z=self._metallicity_sol*self._abundance)
            Pnorm = 3.426*1e-3 * (self._M500.to_value('Msun')*h70/1e15)**(2.0/3) * E_z**(8.0/3)
            try:
                fb = self._cosmo.Ob0/self._cosmo.Om0
            except:
                fb = 0.16
            Pnorm = Pnorm * (fb/0.16) * (mu_g/0.6) * (mu_e/1.14)
        
        # Set the parameters accordingly
        self._model_pressure_profile = {"name": 'GNFW',
                                        "P_0" : pppar[0]*Pnorm*u.Unit('keV cm-3'),
                                        "c500": pppar[1],
                                        "r_p" : self._R500/pppar[1],
                                        "a":pppar[3],
                                        "b":pppar[4],
                                        "c":pppar[2]}


    #==================================================
    # Set a given density UDP profile
    #==================================================
    
    def set_density_profile_universal_param(self, density_model='G19UDP'):
        """
        Set the parameters of the density profile:
        n0, rc, rs, alpha, beta, epsilon
        
        Parameters
        ----------
        - density_model (str): available models are 
            'G19UPP' (Ghirardini et al 2019, all clusters)
            'G19CC' (Ghirardini et al 2019, cool-core clusters)
            'G19MD' (Ghirardini et al 2019, morphologically disturbed clusters)
            'P22'   (Pratt et al. 2022), gNFW universal gas density profile
        """
        
        # Ghirardini (2019) : Universal density Profile parameters
        if density_model == 'G19UDP':
            if not self._silent: print('Setting SVM Ghirardini (2019) UPP.')
            dppar = [np.exp(-4.4), np.exp(-3.0), np.exp(-0.29), 0.89, 0.43, 2.86]
            gas_model = {'name' :    'SVM', 
                         'n_0' :     dppar[0]*u.cm**-3 * self._cosmo.efunc(self._redshift)**2,
                         'r_c' :     dppar[1]*self._R500, 
                         'r_s' :     dppar[2]*self._R500,
                         'alpha' :   dppar[3],
                         'beta' :    dppar[4] - dppar[3]/6.0, # Because not the same def of SVM
                         'epsilon' : dppar[5],
                         'gamma' :   3.0}
            
        # Ghirardini (2019) : Universal density Profile parameters
        elif density_model == 'G19CC':
            if not self._silent: print('Setting SVM Ghirardini (2019) CC.')
            dppar = [np.exp(-3.9), np.exp(-3.2) , np.exp(0.17), 0.80, 0.49, 4.67]
            gas_model = {'name' :    'SVM', 
                         'n_0' :     dppar[0]*u.cm**-3 * self._cosmo.efunc(self._redshift)**2,
                         'r_c' :     dppar[1]*self._R500, 
                         'r_s' :     dppar[2]*self._R500,
                         'alpha' :   dppar[3],
                         'beta' :    dppar[4] - dppar[3]/6.0, # Because not the same def of SVM
                         'epsilon' : dppar[5],
                         'gamma' :   3.0}
            
             # Ghirardini (2019) : Universal density Profile parameters
        elif density_model == 'G19MD':
            if not self._silent: print('Setting SVM Ghirardini (2019) MD.')
            dppar = [np.exp(-4.9), np.exp(-2.7), np.exp(-0.51), 0.70, 0.39, 2.6]
            gas_model = {'name' :    'SVM', 
                         'n_0' :     dppar[0]*u.cm**-3 * self._cosmo.efunc(self._redshift)**2,
                         'r_c' :     dppar[1]*self._R500, 
                         'r_s' :     dppar[2]*self._R500,
                         'alpha' :   dppar[3],
                         'beta' :    dppar[4] - dppar[3]/6.0, # Because not the same def of SVM
                         'epsilon' : dppar[5],
                         'gamma' :   3.0}

            # Pratt (2022) : Universal density profile parameters (and corrigendum)
        elif density_model == 'P22':
            dppar = [1.20, 1/0.28, 0.42, 1.52, 3*0.78]

            E_z = self._cosmo.efunc(self._redshift)
            h70 = self._cosmo.H0.value/70.0
            rho500 = 500*self._cosmo.critical_density(self._redshift)
            mu_g,mu_e,mu_p,mu_a = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                       Z=self._metallicity_sol*self._abundance)
            A_z_M = E_z**2.09 * (self._M500.to_value('Msun')/(5e14*h70**-1))**0.22
            Nnorm = A_z_M * dppar[0] * rho500/(mu_e*const.m_p)
            
            gas_model = {"name": 'GNFW',
                         "P_0" : Nnorm.to('cm-3'),
                         "c500": dppar[1],
                         "r_p" : self._R500/dppar[1],
                         "a":dppar[3],
                         "b":dppar[4],
                         "c":dppar[2]}
            
        # No other profiles available
        else:
            raise ValueError('Density profile requested model not available. Use G19UDP, G19CC, G19MD, or P22.')

        # Set the parameters accordingly
        self._model_density_profile = gas_model


    #==================================================
    # Set a given pressure polytropic profile
    #==================================================
    
    def set_pressure_profile_polytropic_param(self, polytropic_model='G19'):
        """
        Set the parameters of the pressure profile so that 
        the cluster follows the polytropc relation, using the 
        density as the input quantity. See
        Ghirardini et al. (2019) for details.
        
        Parameters
        ----------
        - polytropic_model (str): model to follow

        """

        #---------- Get the model parameters
        # Ghirardini et al. 2019 (see section 3 and tab 3)
        if polytropic_model == 'G19':

            E_z   = self._cosmo.efunc(self._redshift)
            h70   = self._cosmo.H0.value/70.0
            mu_g,mu_e,mu_p,mu_a = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                       Z=self._metallicity_sol*self._abundance)        
            try:
                fb = self._cosmo.Ob0/self._cosmo.Om0
            except:
                fb = 0.16
            
            P500  = 3.426*1e-3 * (self._M500.to_value('Msun')*h70/1e15)**(2.0/3) * E_z**(8.0/3)*(fb/0.16)*(mu_g/0.6)*(mu_e/1.14)
            P0    = np.exp(-2.94)
            n0    = np.exp(-10.3)
            Gamma = 1.19

            constant = P500*P0 * (n0*E_z**2)**(-Gamma)*u.keV*u.cm**-3

        # No other relation available
        else:
            raise ValueError('Only the Ghirardini et al. 2019 relation is available: polytropic_model="G19"')

        #---------- Set the model parameters
        # Get the density parameters
        Ppar = self._model_density_profile.copy()

        # Modify the parameters depending on the model
        if self._model_density_profile['name'] == 'GNFW':
            Ppar['P_0'] = constant * (Ppar['P_0'].to_value('cm**-3'))**Gamma
            Ppar['b']  *= Gamma
            Ppar['c']  *= Gamma
            
        elif self._model_density_profile['name'] == 'SVM':
            Ppar['n_0'] = constant * (Ppar['n_0'].to_value('cm**-3'))**Gamma
            Ppar['beta'] *= Gamma
            Ppar['alpha'] *= Gamma
            Ppar['epsilon'] *= Gamma
            
        elif self._model_density_profile['name'] == 'beta':
            Ppar['n_0'] = constant * (Ppar['n_0'].to_value('cm**-3'))**Gamma
            Ppar['beta'] *= Gamma
            
        elif self._model_density_profile['name'] == 'doublebeta':
            if self._silent is False:
                print('!!! Analytical polytropic transformation not available with doublebeta model. !!!')
                print('!!! Definition done via a User defined model in 0.1-10000 kpc!!!')
            rad, prof = self.get_density_profile(radius=np.logspace(np.log10(self._Rmin.to_value('kpc')/5),
                                                                    np.log10(self._R_truncation.to_value('kpc')*5),
                                                                    1000)*u.kpc)
            profile = constant * prof.to_value('cm-3')**Gamma
            Ppar = {'name':'User', 'radius':rad, 'profile':profile}

        elif self._model_density_profile['name'] == 'User':
             Ppar['profile'] = constant*(Ppar['profile'].to_value('cm-3'))**Gamma

        else:
            raise ValueError('Problem with pressure model list.')

        self._model_pressure_profile = Ppar
        

    #==================================================
    # Set a given density polytropic profile
    #==================================================
    
    def set_density_profile_polytropic_param(self, polytropic_model='G19'):
        """
        Set the parameters of the density profile so that 
        the cluster follows the polytropc relation using the 
        pressure as the input quantity. See
        Ghirardini et al. (2019) for details.
        
        Parameters
        ----------
        - polytropic_model (str): model to follow

        """

        #---------- Get the model parameters
        # Ghirardini et al. 2019 (see section 3 and tab 3)
        if polytropic_model == 'G19':

            E_z   = self._cosmo.efunc(self._redshift)
            h70   = self._cosmo.H0.value/70.0
            mu_g,mu_e,mu_p,mu_a = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                       Z=self._metallicity_sol*self._abundance)        
            try:
                fb = self._cosmo.Ob0/self._cosmo.Om0
            except:
                fb = 0.16
            
            P500  = 3.426*1e-3 * (self._M500.to_value('Msun')*h70/1e15)**(2.0/3) * E_z**(8.0/3)*(fb/0.16)*(mu_g/0.6)*(mu_e/1.14)
            P0    = np.exp(-2.94)
            n0    = np.exp(-10.3)
            Gamma = 1.19

            constant = n0*E_z**2*(P500*P0)**(-1/Gamma)*u.cm**-3

        # No other relation available
        else:
            raise ValueError('Only the Ghirardini et al. 2019 relation is available: polytropic_model="G19"')

        #---------- Set the model parameters
        # Get the pressure parameters
        Ppar = self._model_pressure_profile.copy()

        # Modify the parameters depending on the model
        if self._model_pressure_profile['name'] == 'GNFW':
            Ppar['P_0'] = constant * (Ppar['P_0'].to_value('keV cm**-3'))**(1/Gamma)
            Ppar['b']  *= 1/Gamma
            Ppar['c']  *= 1/Gamma
            
        elif self._model_pressure_profile['name'] == 'SVM':
            Ppar['n_0'] = constant * (Ppar['n_0'].to_value('keV cm**-3'))**(1/Gamma)
            Ppar['beta'] *= 1/Gamma
            Ppar['alpha'] *= 1/Gamma
            Ppar['epsilon'] *= 1/Gamma
            
        elif self._model_pressure_profile['name'] == 'beta':
            Ppar['n_0'] = constant * (Ppar['n_0'].to_value('keV cm**-3'))**(1/Gamma)
            Ppar['beta'] *= 1/Gamma
            
        elif self._model_pressure_profile['name'] == 'doublebeta':
            if self._silent is False:
                print('!!! Analytical polytropic transformation not available with doublebeta model. !!!')
                print('!!! Definition done via a User defined model in 0.1-10000 kpc!!!')
            rad, prof = self.get_pressure_profile(radius=np.logspace(np.log10(self._Rmin.to_value('kpc')/5),
                                                                     np.log10(self._R_truncation.to_value('kpc')*5),
                                                                     1000)*u.kpc)
            profile = constant * prof.to_value('keV cm-3')**(1/Gamma)
            Ppar = {'name':'User', 'radius':rad, 'profile':profile}

        elif self._model_pressure_profile['name'] == 'User':
             Ppar['profile'] = constant*(Ppar['profile'].to_value('keV cm-3'))**(1/Gamma)

        else:
            raise ValueError('Problem with pressure model list.')

        self._model_density_profile = Ppar


    #==================================================
    # Set a given pressure isothermal profile
    #==================================================
    
    def set_pressure_profile_isoT_param(self, kBT):
        """
        Set the parameters of the pressure profile so that 
        the cluster is iso thermal
        
        Parameters
        ----------
        - kBT (quantity): isothermal temperature

        """

        # check type of temperature
        try:
            test = kBT.to('keV')
        except:
            raise TypeError("The temperature should be a quantity homogeneous to keV.")

        # Get the density parameters
        Ppar = self._model_density_profile.copy()

        # Modify the parameters depending on the model
        if self._model_density_profile['name'] == 'GNFW':
            Ppar['P_0'] = (Ppar['P_0'] * kBT).to('keV cm-3')
            
        elif self._model_density_profile['name'] == 'SVM':
            Ppar['n_0'] = (Ppar['n_0'] * kBT).to('keV cm-3')

        elif self._model_density_profile['name'] == 'beta':
            Ppar['n_0'] = (Ppar['n_0'] * kBT).to('keV cm-3')

        elif self._model_density_profile['name'] == 'doublebeta':
            Ppar['n_01'] = (Ppar['n_01'] * kBT).to('keV cm-3')
            Ppar['n_02'] = (Ppar['n_02'] * kBT).to('keV cm-3')

        elif self._model_density_profile['name'] == 'User':
             Ppar['profile'] = (Ppar['profile'] * kBT).to('keV cm-3')
            
        else:
            raise ValueError('Problem with density model list.')

        self._model_pressure_profile = Ppar


    #==================================================
    # Set a given density isothermal profile
    #==================================================
    
    def set_density_profile_isoT_param(self, kBT):
        """
        Set the parameters of the density profile so that 
        the cluster is iso thermal
        
        Parameters
        ----------
        - kBT (quantity): isothermal temperature

        """

        # check type of temperature
        try:
            test = kBT.to('keV')
        except:
            raise TypeError("The temperature should be a quantity homogeneous to keV.")

        # Get the density parameters
        Ppar = self._model_pressure_profile.copy()

        # Modify the parameters depending on the model
        if self._model_pressure_profile['name'] == 'GNFW':
            Ppar['P_0'] = (Ppar['P_0'] / kBT).to('cm-3')
            
        elif self._model_pressure_profile['name'] == 'SVM':
            Ppar['n_0'] = (Ppar['n_0'] / kBT).to('cm-3')

        elif self._model_pressure_profile['name'] == 'beta':
            Ppar['n_0'] = (Ppar['n_0'] / kBT).to('cm-3')

        elif self._model_pressure_profile['name'] == 'doublebeta':
            Ppar['n_01'] = (Ppar['n_01'] / kBT).to('cm-3')
            Ppar['n_02'] = (Ppar['n_02'] / kBT).to('cm-3')

        elif self._model_pressure_profile['name'] == 'User':
             Ppar['profile'] = (Ppar['profile'] / kBT).to('cm-3')
            
        else:
            raise ValueError('Problem with density model list.')

        self._model_density_profile = Ppar
        

    #==================================================
    # Set a given pressure profile according to temperature
    #==================================================
    
    def set_pressure_profile_from_temperature_model(self, kBT_model):
        """
        Set the pressure profile so that 
        the cluster is defined via the density and the 
        temperature profiles.
        This function always uses a User defined model
        (i.e. implying interpolation) to define the pressure.
        
        Parameters
        ----------
        - kBT_model (dict): temperature profile

        """

        #---------- Inputs validation
        kBT_model = self._validate_model_profile_parameters(kBT_model, 'keV')

        #---------- Extract kBT_r and n_r
        radius = np.logspace(np.log10(self._Rmin.to_value('kpc')/5), np.log10(self._R_truncation.to_value('kpc')*5), 1000)*u.kpc
        kBT_r = self._get_generic_profile(radius, kBT_model, derivative=False)
        rad0, n_e_r = self.get_density_profile(radius)       
        
        #---------- Compute the pressure profile       
        profile = n_e_r * kBT_r
        Ppar = {'name':'User', 'radius':radius, 'profile':profile.to('keV cm-3')}
        
        #---------- Set the density model
        self._model_pressure_profile = Ppar


    #==================================================
    # Set a given density profile according to temperature
    #==================================================
    
    def set_density_profile_from_temperature_model(self, kBT_model):
        """
        Set the density profile so that 
        the cluster is defined via the pressure and the 
        temperature profiles.
        This function always uses a User defined model
        (i.e. implying interpolation) to define the density.
        
        Parameters
        ----------
        - kBT_model (dict): temperature profile

        """

        #---------- Inputs
        kBT_model = self._validate_model_profile_parameters(kBT_model, 'keV')
        
        #---------- Extract kBT_r and P_r
        radius = np.logspace(np.log10(self._Rmin.to_value('kpc')/5), np.log10(self._R_truncation.to_value('kpc')*5), 1000)*u.kpc
        kBT_r = self._get_generic_profile(radius, kBT_model, derivative=False)
        rad0, p_e_r = self.get_pressure_profile(radius)       
        
        #---------- Compute the pressure profile       
        profile = p_e_r / kBT_r
        Dpar = {'name':'User', 'radius':radius, 'profile':profile.to('cm-3')}        

        #---------- Set the density model
        self._model_density_profile = Dpar

        
    #==================================================
    # Validate profile model parameters
    #==================================================
    
    def _validate_model_profile_parameters(self, inpar, unit):
        """
        Check the profile parameters.
        
        Parameters
        ----------
        - inpar (dict): a dictionary containing the input parameters
        - unit (str): contain the unit of the profile, e.g. keV/cm-3 for pressure
        
        Outputs
        ----------
        - outpar (dict): a dictionary with output parameters

        """

        # List of available authorized models
        model_list = ['NFW', 'GNFW', 'SVM', 'beta', 'doublebeta', 'User']
        
        # Deal with unit
        if unit == '' or unit == None:
            hasunit = False
        else:
            hasunit = True

        # Check that the input is a dictionary
        if type(inpar) != dict :
            raise TypeError("The model should be a dictionary containing the name key and relevant parameters")
        
        # Check that input contains a name
        if 'name' not in list(inpar.keys()) :
            raise ValueError("The model dictionary should contain a 'name' field")
            
        # Check that the name is in the acceptable name list
        if not inpar['name'] in model_list:
            print('The profile model can be:')
            print(model_list)
            raise ValueError("The requested model is not available")

        #---------- Deal with the case of NFW
        if inpar['name'] == 'NFW':
            # Check the content of the dictionary
            cond1a = 'r_s' in list(inpar.keys()) and 'rho_0' in list(inpar.keys())
            cond1b = not('M' in list(inpar.keys()) or 'c' in list(inpar.keys()) or 'delta' in list(inpar.keys()))
            cond2a = 'M' in list(inpar.keys()) and 'c' in list(inpar.keys()) and 'delta' in list(inpar.keys())
            cond2b = not('rho_0' in list(inpar.keys()) or 'r_s' in list(inpar.keys()))

            if (cond1a and cond1b):
                case1 = True
            else:
                case1 = False
            if (cond2a and cond2b):
                case2 = True
            else:
                case2 = False
            
            if not ((cond1a and cond1b) or (cond2a and cond2b)):
                raise ValueError("The NFW model should contain: {'rho_0', 'r_s'} or {'M', 'c', 'delta'} .")

            # Case of generic profile with rho0 and rs
            if case1:
                # Check units and values
                if hasunit:
                    try:
                        test = inpar['rho_0'].to(unit)
                    except:
                        raise TypeError("rho_0 should be homogeneous to "+unit)
                
                if inpar['rho_0'] < 0:
                    raise ValueError("rho_0 should be >=0")
                
                try:
                    test = inpar['r_s'].to('kpc')
                except:
                    raise TypeError("r_s should be homogeneous to kpc")
                    
                if inpar['r_s'] <= 0:
                    raise ValueError("r_s should be >0")
                
                # All good at this stage, setting parameters
                if hasunit:
                    rho0 = inpar['rho_0'].to(unit)
                else:
                    rho0 = inpar['rho_0']*u.adu

                r_s = inpar['r_s'].to('kpc')

            # Case of physical profile with mass and concentration
            if case2:
                # Check units and values
                if hasunit:
                    try:
                        test = (u.Unit(unit)).to('Msun kpc-3')
                    except:
                        raise TypeError("The unit should be left empty or homogeneous to Msun/kpc^3 for a physically defined NFW")

                try:
                    test = inpar['M'].to('Msun')
                except:
                    raise TypeError("M should be homogeneous to Msun")
                    
                if inpar['M'] <= 0:
                    raise ValueError("M should be >0")
                
                if inpar['c'] <= 0:
                    raise ValueError("c should be >0")

                if inpar['delta'] < 100 or inpar['delta'] > 5000:
                    raise ValueError("delta is restricted to 100 < delta < 5000 to avoid numerical faillures")

                # Compute r_s and rho_0
                R_delta = cluster_global.Mdelta_to_Rdelta(inpar['M'].to_value('Msun'), self._redshift,
                                                          delta=inpar['delta'], cosmo=self._cosmo)
                r_s = (R_delta / inpar['c']) * u.kpc
                overdens = cluster_global.concentration_to_deltac_NFW(inpar['c'], Delta=inpar['delta'])
                rho0 = overdens * self._cosmo.critical_density(self._redshift).to('Msun kpc-3')

            # Set the parameters
            outpar = {"name": 'NFW',
                      "rho_0" : rho0,
                      "r_s"   : r_s}
            
        #---------- Deal with the case of GNFW
        elif inpar['name'] == 'GNFW':
            # Check the content of the dictionary
            cond1 = 'P_0' in list(inpar.keys()) and 'a' in list(inpar.keys()) and 'b' in list(inpar.keys()) and 'c' in list(inpar.keys())
            cond2 = 'c500' in list(inpar.keys()) or 'r_p' in list(inpar.keys())
            cond3 = not('c500' in list(inpar.keys()) and 'r_p' in list(inpar.keys()))

            if not (cond1 and cond2 and cond3):
                raise ValueError("The GNFW model should contain: {'P_0','c500' or 'r_p' (not both),'a','b','c'}.")
            
            # Check units and values
            if hasunit:
                try:
                    test = inpar['P_0'].to(unit)
                except:
                    raise TypeError("P_0 should be homogeneous to "+unit)

            if inpar['P_0'] < 0:
                raise ValueError("P_0 should be >=0")

            if 'r_p' in list(inpar.keys()):
                try:
                    test = inpar['r_p'].to('kpc')
                except:
                    raise TypeError("r_p should be homogeneous to kpc")
                
                if inpar['r_p'] <= 0:
                    raise ValueError("r_p should be >0")

            if 'c500' in list(inpar.keys()):
                if inpar['c500'] <=0:
                    raise ValueError("c500 should be > 0")

            # All good at this stage, setting parameters
            if 'c500' in list(inpar.keys()):
                c500 = inpar['c500']
                r_p = self._R500/inpar['c500']
            if 'r_p' in list(inpar.keys()):
                c500 = (self._R500/inpar['r_p']).to_value('')
                r_p = inpar['r_p']
            if hasunit:
                P0 = inpar['P_0'].to(unit)
            else:
                P0 = inpar['P_0']*u.adu
                
            outpar = {"name": 'GNFW',
                      "P_0" : P0,
                      "c500": c500,
                      "r_p" : r_p.to('kpc'),
                      "a"   : inpar['a'],
                      "b"   : inpar['b'],
                      "c"   : inpar['c']}
            
        #---------- Deal with the case of SVM
        if inpar['name'] == 'SVM':
            # Check the content of the dictionary
            cond1 = 'n_0' in list(inpar.keys()) and 'r_c' in list(inpar.keys()) and 'beta' in list(inpar.keys())
            cond2 = 'r_s' in list(inpar.keys()) and 'alpha' in list(inpar.keys()) and 'epsilon' in list(inpar.keys()) and 'gamma' in list(inpar.keys())
            if not (cond1 and cond2):
                raise ValueError("The SVM model should contain: {'n_0','beta','r_c','r_s', 'alpha', 'gamma', 'epsilon'}.")
 
            # Check units
            if hasunit:
                try:
                    test = inpar['n_0'].to(unit)
                except:
                    raise TypeError("n_0 should be homogeneous to "+unit)
            try:
                test = inpar['r_c'].to('kpc')
            except:
                raise TypeError("r_c should be homogeneous to kpc")
            try:
                test = inpar['r_s'].to('kpc')
            except:
                raise TypeError("r_s should be homogeneous to kpc")
            
            # Check values
            if inpar['n_0'] < 0:
                raise ValueError("n_0 should be >= 0")
            if inpar['r_c'] <= 0:
                raise ValueError("r_c should be larger than 0")
            if inpar['r_s'] <= 0:
                raise ValueError("r_s should be larger than 0")

            if hasunit:
                n0 = inpar['n_0'].to(unit)
            else:
                n0 = inpar['n_0']*u.adu
            
            # All good at this stage, setting parameters
            outpar = {"name"   : 'SVM',
                      "n_0"    : n0,
                      "r_c"    : inpar['r_c'].to('kpc'),
                      "r_s"    : inpar['r_s'].to('kpc'),
                      "alpha"  : inpar['alpha'],
                      "beta"   : inpar['beta'],
                      "gamma"  : inpar['gamma'],
                      "epsilon": inpar['epsilon']}
            
        #---------- Deal with the case of beta
        if inpar['name'] == 'beta':
            # Check the content of the dictionary
            cond1 = 'n_0' in list(inpar.keys()) and 'r_c' in list(inpar.keys()) and 'beta' in list(inpar.keys())
            if not cond1:
                raise ValueError("The beta model should contain: {'n_0','beta','r_c'}.")

            # Check units
            if hasunit:
                try:
                    test = inpar['n_0'].to(unit)
                except:
                    raise TypeError("n_0 should be homogeneous to "+unit)
            try:
                test = inpar['r_c'].to('kpc')
            except:
                raise TypeError("r_c should be homogeneous to kpc")

            # Check values
            if inpar['n_0'] < 0:
                raise ValueError("n_0 should be >= 0")
            if inpar['r_c'] <= 0:
                raise ValueError("r_c should be larger than 0")

            if hasunit:
                n0 = inpar['n_0'].to(unit)
            else:
                n0 = inpar['n_0']*u.adu
                
            # All good at this stage, setting parameters
            outpar = {"name"   : 'beta',
                      "n_0"    : n0,
                      "r_c"    : inpar['r_c'].to('kpc'),
                      "beta"   : inpar['beta']}
            
        #---------- Deal with the case of doublebeta
        if inpar['name'] == 'doublebeta':
            # Check the content of the dictionary
            cond1 = 'n_01' in list(inpar.keys()) and 'r_c1' in list(inpar.keys()) and 'beta1' in list(inpar.keys())
            cond2 = 'n_02' in list(inpar.keys()) and 'r_c2' in list(inpar.keys()) and 'beta2' in list(inpar.keys())
            if not (cond1 and cond2):
                raise ValueError("The double beta model should contain: {'n_01','beta1','r_c1','n_02','beta2','r_c2'}.")

            # Check units
            if hasunit:
                try:
                    test = inpar['n_01'].to(unit)
                except:
                    raise TypeError("n_01 should be homogeneous to "+unit)
                try:
                    test = inpar['n_02'].to(unit)
                except:
                    raise TypeError("n_02 should be homogeneous to "+unit)
            try:
                test = inpar['r_c1'].to('kpc')
            except:
                raise TypeError("r_c1 should be homogeneous to kpc")
            try:
                test = inpar['r_c2'].to('kpc')
            except:
                raise TypeError("r_c2 should be homogeneous to kpc")

            # Check values
            if inpar['n_01'] < 0:
                raise ValueError("n_01 should be >= 0")
            if inpar['r_c1'] <= 0:
                raise ValueError("r_c1 should be larger than 0")
            if inpar['n_02'] < 0:
                raise ValueError("n_02 should be >= 0")
            if inpar['r_c2'] <= 0:
                raise ValueError("r_c2 should be larger than 0")

            if hasunit:
                n01 = inpar['n_01'].to(unit)
                n02 = inpar['n_02'].to(unit)
            else:
                n01 = inpar['n_01']*u.adu
                n02 = inpar['n_02']*u.adu
            
            # All good at this stage, setting parameters
            outpar = {"name"  : 'doublebeta',
                      "n_01"  : n01,
                      "r_c1"  : inpar['r_c1'].to('kpc'),
                      "beta1" : inpar['beta1'],
                      "n_02"  : n02,
                      "r_c2"  : inpar['r_c2'].to('kpc'),
                      "beta2" : inpar['beta2']}

        #---------- Deal with the case of User
        if inpar['name'] == 'User':
            # Check the content of the dictionary
            cond1 = 'radius' in list(inpar.keys()) and 'profile' in list(inpar.keys())
            if not cond1:
                raise ValueError("The User model should contain: {'radius','profile'}.")

            # Check units
            if hasunit:
                try:
                    test = inpar['profile'].to(unit)
                except:
                    raise TypeError("profile should be homogeneous to "+unit)
            try:
                test = inpar['radius'].to('kpc')
            except:
                raise TypeError("radius should be homogeneous to kpc")

            # Check values
            if np.amin(inpar['radius']) < 0:
                raise ValueError("radius should be >= 0")
            if np.amin(inpar['profile']) < 0:
                raise ValueError("profile should be larger >= 0")

            if hasunit:
                prof = inpar['profile'].to(unit)
            else:
                prof = inpar['profile']*u.adu

            # All good at this stage, setting parameters
            outpar = {"name"    : 'User',
                      "radius"  : inpar['radius'].to('kpc'),
                      "profile" : prof}
            
        return outpar


    #==================================================
    # Validate fluctuation model parameters
    #==================================================
    
    def _validate_model_fluctuation_parameters(self, inpar):
        """
        Check the pressure fluctuation model parameters.
        
        Parameters
        ----------
        - inpar (dict): a dictionary containing the input parameters
        - unit (str): contain the unit of the fluctuation, if any
        
        Outputs
        ----------
        - outpar (dict): a dictionary with output parameters

        """

        # List of available authorized models
        model_list = ['CutoffPowerLaw', 'ModifiedCutoffPowerLaw', 'User']
        stats_list = ['gaussian', 'lognormal']
        
        # Check that the input is a dictionary
        if type(inpar) != dict :
            raise TypeError("The model should be a dictionary containing the name key and relevant parameters")
        
        # Check that input contains a name
        if 'name' not in list(inpar.keys()) :
            raise ValueError("The model dictionary should contain a 'name' field")
            
        # Check that the name is in the acceptable name list
        if not inpar['name'] in model_list:
            print('The fluctuation model can be:')
            print(model_list)
            raise ValueError("The requested model is not available")

        # Check that input contains a statistics
        if 'statistics' not in list(inpar.keys()) :
            raise ValueError("The model dictionary should contain a 'statistics' field")
            
        # Check that the statistics is acceptable
        if not inpar['statistics'] in stats_list:
            print('The fluctuation "statistics" can be:')
            print(stats_list)
            raise ValueError("The requested statistics are not available")

        #---------- Deal with the case of CutoffPowerLaw
        if inpar['name'] == 'CutoffPowerLaw':
            # Check the content of the dictionary
            cond1 = 'Norm' in list(inpar.keys()) and 'slope' in list(inpar.keys())
            cond2 = 'Linj' in list(inpar.keys()) and 'Ldis' in list(inpar.keys())
            if not (cond1 and cond2):
                raise ValueError("The CutoffPowerLaw model should contain: {'Norm','slope','Linj','Ldis'}.")

            # Check units
            try:
                test = inpar['Linj'].to('kpc')
            except:
                raise TypeError("Linj should be homogeneous to kpc")
            try:
                test = inpar['Ldis'].to('kpc')
            except:
                raise TypeError("Ldis should be homogeneous to kpc")
            if type(inpar['Norm']) == u.quantity.Quantity:
                raise TypeError("Norm is the rms of the relative fluctuation, should be dimenssionless")
            if type(inpar['slope']) == u.quantity.Quantity:
                raise TypeError("slope should be dimenssionless")

            # Check values
            if inpar['Norm'] < 0:
                raise ValueError("Norm should be >= 0")
            if inpar['Linj'] < 0:
                raise ValueError("Linj should be >= 0")
            if inpar['Ldis'] < 0:
                raise ValueError("Ldis should be >= 0")

            # All good at this stage, setting parameters
            outpar = {"name" : 'CutoffPowerLaw',
                      "statistics": inpar['statistics'],
                      "Norm" : inpar['Norm'],
                      "slope": inpar['slope'],
                      "Linj" : inpar['Linj'].to('kpc'),
                      "Ldis" : inpar['Ldis'].to('kpc')}

        #---------- Deal with the case of ModifiedCutoffPowerLaw (Eq 4 in C. Romero 2024)
        if inpar['name'] == 'ModifiedCutoffPowerLaw':
            # Check the content of the dictionary
            cond1 = 'Norm' in list(inpar.keys()) and 'slope' in list(inpar.keys())
            cond2 = 'Linj' in list(inpar.keys()) and 'Ldis' in list(inpar.keys())
            cond3 = 'Ninj' in list(inpar.keys()) and 'Ndis' in list(inpar.keys())
            if not (cond1 and cond2 and cond3):
                raise ValueError("The ModifiedCutoffPowerLaw model should contain: {'Norm','slope','Linj','Ldis','Ninj','Ndis'}.")

            # Check units
            try:
                test = inpar['Linj'].to('kpc')
            except:
                raise TypeError("Linj should be homogeneous to kpc")
            try:
                test = inpar['Ldis'].to('kpc')
            except:
                raise TypeError("Ldis should be homogeneous to kpc")
            if type(inpar['Norm']) == u.quantity.Quantity:
                raise TypeError("Norm is the rms of the relative fluctuation, should be dimenssionless")
            if type(inpar['slope']) == u.quantity.Quantity:
                raise TypeError("slope should be dimenssionless")
            if type(inpar['Ndis']) == u.quantity.Quantity:
                raise TypeError("Ndis should be dimenssionless")
            if type(inpar['Ninj']) == u.quantity.Quantity:
                raise TypeError("Ndis should be dimenssionless")
            
            # Check values
            if inpar['Norm'] < 0:
                raise ValueError("Norm should be >= 0")
            if inpar['Linj'] < 0:
                raise ValueError("Linj should be >= 0")
            if inpar['Ldis'] < 0:
                raise ValueError("Ldis should be >= 0")

            # All good at this stage, setting parameters
            outpar = {"name" : 'ModifiedCutoffPowerLaw',
                      "statistics": inpar['statistics'],
                      "Norm" : inpar['Norm'],
                      "slope": inpar['slope'],
                      "Linj" : inpar['Linj'].to('kpc'),
                      "Ldis" : inpar['Ldis'].to('kpc'),
                      "Ninj": inpar['Ninj'],
                      "Ndis": inpar['Ndis']}

        #---------- Deal with the case of User
        if inpar['name'] == 'User':
            # Check the content of the dictionary
            cond1 = 'k' in list(inpar.keys()) and 'pk' in list(inpar.keys())
            if not cond1:
                raise ValueError("The User model should contain: {'k','pk'}.")

            # Check units
            try:
                test = inpar['k'].to('kpc-1')
            except:
                raise TypeError("k should be homogeneous to 1/kpc")
            try:
                test = inpar['pk'].to('kpc3')
            except:
                raise TypeError("pk should be homogeneous to kpc^3")

            # Check values
            if np.amin(inpar['k']) < 0:
                raise ValueError("k should be >= 0")
            if np.amin(inpar['pk']) < 0:
                raise ValueError("pk should be larger >= 0")

            # All good at this stage, setting parameters
            outpar = {"name"   : 'User',
                      "radius" : inpar['k'].to('kpc-1'),
                      "pk"     : inpar['pk'].to('kpc3')}

        return outpar


    #==================================================
    # Get the generic model profile
    #==================================================

    def _get_generic_profile(self, radius, model, derivative=False):
        """
        Get the generic profile profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array
        - model (dict): dictionary containing the model parameters
        - derivative (bool): to get the derivative of the profile
        
        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - p_r (quantity): the profile

        """

        model_list = ['NFW', 'GNFW', 'SVM', 'beta', 'doublebeta', 'User']

        if not model['name'] in model_list:
            print('The profile model can :')
            print(model_list)
            raise ValueError("The requested model has not been implemented")

        r3d_kpc = radius.to_value('kpc')

        #---------- Case of GNFW profile
        if model['name'] == 'NFW':
            unit = model["rho_0"].unit
            
            rho0 = model["rho_0"].to_value(unit)
            rs = model["r_s"].to_value('kpc')

            if derivative:
                prof_r = cluster_profile.NFW_model_derivative(r3d_kpc, rho0, rs) * unit*u.Unit('kpc-1')
            else:
                prof_r = cluster_profile.NFW_model(r3d_kpc, rho0, rs)*unit

        #---------- Case of GNFW profile
        elif model['name'] == 'GNFW':
            unit = model["P_0"].unit
            
            P0 = model["P_0"].to_value(unit)
            rp = model["r_p"].to_value('kpc')
            a  = model["a"]
            b  = model["b"]
            c  = model["c"]

            if derivative:
                prof_r = cluster_profile.gNFW_model_derivative(r3d_kpc, P0, rp,
                                                               slope_a=a,
                                                               slope_b=b,
                                                               slope_c=c) * unit*u.Unit('kpc-1')
            else:
                prof_r = cluster_profile.gNFW_model(r3d_kpc, P0, rp, slope_a=a, slope_b=b, slope_c=c)*unit

        #---------- Case of SVM model
        elif model['name'] == 'SVM':
            unit = model["n_0"].unit
            
            n0      = model["n_0"].to_value(unit)
            rc      = model["r_c"].to_value('kpc')
            rs      = model["r_s"].to_value('kpc')
            alpha   = model["alpha"]
            beta    = model["beta"]
            gamma   = model["gamma"]
            epsilon = model["epsilon"]

            if derivative:
                prof_r = cluster_profile.svm_model_derivative(r3d_kpc, n0, rc, beta,
                                                              rs, gamma, epsilon, alpha) * unit*u.Unit('kpc-1')
            else:
                prof_r = cluster_profile.svm_model(r3d_kpc, n0, rc, beta, rs, gamma, epsilon, alpha)*unit

        #---------- beta model
        elif model['name'] == 'beta':
            unit = model["n_0"].unit

            n0      = model["n_0"].to_value(unit)
            rc      = model["r_c"].to_value('kpc')
            beta    = model["beta"]

            if derivative:
                prof_r = cluster_profile.beta_model_derivative(r3d_kpc, n0, rc, beta) * unit*u.Unit('kpc-1')
            else:
                prof_r = cluster_profile.beta_model(r3d_kpc, n0, rc, beta)*unit
            
        #---------- double beta model
        elif model['name'] == 'doublebeta':
            unit1 = model["n_01"].unit
            unit2 = model["n_02"].unit

            n01      = model["n_01"].to_value(unit1)
            rc1      = model["r_c1"].to_value('kpc')
            beta1    = model["beta1"]
            n02      = model["n_02"].to_value(unit2)
            rc2      = model["r_c2"].to_value('kpc')
            beta2    = model["beta2"]

            if derivative:
                prof_r1 = cluster_profile.beta_model_derivative(r3d_kpc, n01, rc1, beta1) * unit1*u.Unit('kpc-1')
                prof_r2 = cluster_profile.beta_model_derivative(r3d_kpc, n02, rc2, beta2) * unit2*u.Unit('kpc-1')
                prof_r = prof_r1 + prof_r2
            else:
                prof_r1 = cluster_profile.beta_model(r3d_kpc, n01, rc1, beta1)*unit1
                prof_r2 = cluster_profile.beta_model(r3d_kpc, n02, rc2, beta2)*unit2
                prof_r = prof_r1 + prof_r2

        #---------- User model
        elif model['name'] == 'User':
            user_r = model["radius"].to_value('kpc')
            user_p = model["profile"].value

            # Warning
            if np.amin(user_r) > np.amin(r3d_kpc) or np.amax(user_r) < np.amax(r3d_kpc):
                if self._silent == False:
                    print('WARNING: User model interpolated beyond the provided range!')
            
            # Case of derivative needed
            if derivative:
                # Compute the derivative
                user_der = np.gradient(user_p, user_r)
                f = interpolate.interp1d(user_r, user_der, kind='linear', fill_value='extrapolate')
                prof_r = f(r3d_kpc)
                
                # Add the unit
                prof_r = prof_r*u.Unit('kpc-1')*model["profile"].unit

            # Standard case
            else:
                # log-log interpolation of the user profile
                with warnings.catch_warnings(): # Warning raised in the case of log(0)
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    f = interpolate.interp1d(np.log10(user_r),
                                             np.log10(user_p), kind='linear', fill_value='extrapolate')
                    pitpl_r = 10**f(np.log10(r3d_kpc))

                # Correct for nan (correspond to user_p == 0 in log)
                pitpl_r[np.isnan(pitpl_r)] = 0
                
                # Correct for negative value
                pitpl_r[pitpl_r<0] = 0

                # Add the unit
                prof_r = pitpl_r*model["profile"].unit

        #---------- Otherwise nothing is done
        else :
            if not self._silent: print('The requested model has not been implemented.')

        return prof_r

