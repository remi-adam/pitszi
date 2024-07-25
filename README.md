#  PITSZI: Probing ICM Turbulence from Sunyaev-Zel'dovich Imaging 
Software dedicated to model intracluster medium pressure fluctuations, generate Monte Carlo Sunyaev-Zel'dovich data, and fit the model to input data.

                                                            
## Overview of the physical processes and structure of the code
<figure>
	<img src="/overview.png" width="600" />
	<figcaption> Figure 1. Overview of the code structure.</figcaption>
</figure>

<p style="margin-bottom:3cm;"> </p>


## Content
The pitszi directory contains the main code, including:

- model_main.py : 
	main code entry to use the class Model
    
- model_library.py : 
        subclass that defines model libraries and tools

- model_sampling.py : 
        subclass that deals with the sampling of the model
   
- model_mock.py : 
        subclass used to generate mock images

- data_main.py : 
	class Data used to define input data and usefull associated functions

- inference_radial_main.py : 
  	class InferenceRadial used to constrain the pressure radial model (from Model class) given input data (from Data class)

- inference_radial_fitting.py :
	subclass of inference_radial_main, used for fitting

- inference_fluctuation_main.py : 
  	class InferenceFluctuation used to constrain the pressure fluctuation model (from Model class) given input data (from Data class)

- inference_fluctuation_fitting.py :
	subclass of inference_fluctuation_main, used for fitting

- physics_main.py : 
  	libraries to be used for infering nonthermal ICM information from pressure fluctuations
  
- utils.py, utils_pk.py, utils_fitting.py, utils_plot.py : 
	library of useful functions

- title.py : 
	title for the package

- notebook :
	Repository where to find Jupyter notebook used for validation/example/developments. 


## Installation
You can use pip to install the package:

```
pip install pitszi
```

#### Reference
PITSZI: Probing ICM Turbulence from Sunyaev-Zel'dovich Imaging -- Application to the triple merging cluster MACS J0717.5+3745, 
Adam et al. (in prep)

#### History
- Version 0.1.0 --> Initial upload

