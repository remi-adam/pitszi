__version__ = '0.3.0'

try:
    from .model_main import Model
    from .data_main import Data
    from .inference_radial_main import InferenceRadial
    from .inference_fluctuation_main import InferenceFluctuation

except ImportError:
    print('WARNING: Could not import submodules                             ')
    print('     from .model_main import Model                               ')
    print('     from .data_main import Data                                 ')
    print('     from .inference_radial_main import InferenceRadial          ')
    print('     from .inference_fluctuation_main import InferenceFluctuation')
    print('                                                                 ')
    print('         You may try (re)installing dependencies                 ')
    print('         by hand. For example running:                           ')
    print('             $ conda install numpy                               ')
    print('             $ conda install astropy                             ')
    print('             $ conda install scipy                               ')
    print('             $ conda install matplotlib                          ')
    print('             $ conda install emcee                               ')
    print('             $ conda install corner                              ')
    print('             $ conda install pandas                              ')
    print('             $ conda install seaborn                             ')
    print('             $ conda install dill                                ')
    print('             $ conda install pathos                              ')
    print('             $ pip install minot                                 ')
