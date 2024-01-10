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
	main code entry defining the class Model
    
- model_library.py : 
        subclass that defines model libraries and tools

- model_sampling.py : 
        subclass that deals with the sampling of the model
   
- model_mock.py : 
        subclass used to generate mock images

- data_main.py : 
	class used to define input data and usefull associated functions

- inference_main.py : 
  	class used to constrain the model (from Model class) given input data (from Data class)
  
- utils.py : 
	library of useful functions

- title.py : 
	title for the package

- notebook :
	Repository where to find Jupyter notebook used for validation/example/developments. 


## Installation
TBD

#### Reference
TBD

#### History
- Version 0.1.0 --> Initial upload

