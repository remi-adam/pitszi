#  PITSZI: Probing ICM Turbulence from Sunyaev-Zel'dovich Imaging 
Software dedicated to model intracluster medium pressure fluctuations, generate Monte Carlo Sunyaev-Zel'dovich data, and fit the model to input data.

                                                            
## Overview of the physical processes and structure of the code
<figure>
	<img src="/overview1.png" width="600" />
	<figcaption> Figure 1. Overview of the parametrization, physical processes, and observables dependencies.</figcaption>
</figure>

<p style="margin-bottom:3cm;"> </p>


## Content
The minot directory contains the main code, including:

- main.py : 
	main code that defines the class Cluster
    
- admin.py : 
        subclass that defines administrative tools
   
- modpar.py : 
        subclass that handles model parameters functions 
        
- imgsim.py : 
    subclass that handles the observational properties of the cluster
    
- plots.py : 
        plotting tools for automatic outputs

- title.py : 
	title for the package

- notebook :
	Repository where to find Jupyter notebook used for validation/example. 


## Installation
You can use

#### Reference
In case you use the software.

#### History
- Version 0.1.0 --> Initial release

