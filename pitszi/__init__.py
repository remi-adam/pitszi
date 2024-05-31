__version__ = '0.1.0'

try:
    from .model_main                 import Model
    from .data_main                  import Data
    from .inference_radial_main      import InferenceRadial
    from .inference_fluctuation_main import InferenceFluctuation
except ImportError:
    print('Install error to be solved...')
