__version__ = '0.1.0'

try:
    from .model_main import Model
    #from .main_fitter import ClusterFitter
except ImportError:
    print('Install error to be solved...')
