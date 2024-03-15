"""
This file contains generic utilities for various making calculations used in pitszi

"""

import numpy as np


#==================================================
# Check array
#==================================================

def check_qarray(qarr, unit=None):
    """
    Make sure quantity array are arrays

    Parameters
    ----------
    - qarr (quantity): array or float, homogeneous to some unit

    Outputs
    ----------
    - qarr (quantity): quantity array

    """

    if unit is None:
        if type(qarr) == float or type(qarr) == np.float64:
            qarr = np.array([qarr])
            
    else:
        try:
            test = qarr.to(unit)
        except:
            raise TypeError("Unvalid unit for qarr")

        if type(qarr.to_value()) == float or type(qarr.to_value()) == np.float64:
            qarr = np.array([qarr.to_value()]) * qarr.unit

    return qarr


#==================================================
# Def array based on point per decade, min and max
#==================================================

def sampling_array(xmin, xmax, NptPd=10, unit=False):
    """
    Make an array with a given number of point per decade
    from xmin to xmax

    Parameters
    ----------
    - xmin (quantity): min value of array
    - xmax (quantity): max value of array
    - NptPd (int): the number of point per decade

    Outputs
    ----------
    - array (quantity): the array

    """

    if unit:
        my_unit = xmin.unit
        array = np.logspace(np.log10(xmin.to_value(my_unit)),
                            np.log10(xmax.to_value(my_unit)),
                            int(NptPd*(np.log10(xmax.to_value(my_unit)/xmin.to_value(my_unit)))))*my_unit
    else:
        array = np.logspace(np.log10(xmin), np.log10(xmax), int(NptPd*(np.log10(xmax/xmin))))

    return array


#==================================================
# Integration loglog space with trapezoidale rule
#==================================================

def trapz_loglog(y, x, axis=-1, intervals=False):
    """
    Integrate along the given axis using the composite trapezoidal rule in
    loglog space. Integrate y(x) along given axis in loglog space. y can be a function
    with multiple dimension. This follows the script in the Naima package.
    
    Parameters
    ----------
    - y (array_like): Input array to integrate.
    - x (array_like):  optional. Independent variable to integrate over.
    - axis (int): Specify the axis.
    - intervals (bool): Return array of shape x not the total integral, default: False
    
    Returns
    -------
    - trapz (float): Definite integral as approximated by trapezoidal rule in loglog space.
    """
    
    log10 = np.log10
    
    #----- Check for units
    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.0
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.0

    y = np.asanyarray(y)
    x = np.asanyarray(x)

    #----- Define the slices
    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)
    slice1, slice2 = tuple(slice1), tuple(slice2)

    #----- arrays with uncertainties contain objects, remove tiny elements
    if y.dtype == "O":
        from uncertainties.unumpy import log10
        # uncertainties.unumpy.log10 can't deal with tiny values see
        # https://github.com/gammapy/gammapy/issues/687, so we filter out the values
        # here. As the values are so small it doesn't affect the final result.
        # the sqrt is taken to create a margin, because of the later division
        # y[slice2] / y[slice1]
        valid = y > np.sqrt(np.finfo(float).tiny)
        x, y = x[valid], y[valid]

    #----- reshaping x
    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)
        
    #-----
    with np.errstate(invalid="ignore", divide="ignore"):
        # Compute the power law indices in each integration bin
        b = log10(y[slice2] / y[slice1]) / log10(x[slice2] / x[slice1])
        
        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use normal
        # powerlaw integration
        trapzs = np.where(np.abs(b + 1.0) > 1e-10,
                          (y[slice1] * (x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1]))
                          / (b + 1),
                          x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]))
        
    tozero = (y[slice1] == 0.0) + (y[slice2] == 0.0) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.0
    
    if intervals:
        return trapzs * x_unit * y_unit
    
    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit
    
    return ret


#==================================================
# Extract correlation matrix from covarariance
#==================================================

def correlation_from_covariance(covariance):
    """
    Compute the correlation matrix from the covariance
    
    Parameters
    ----------
    - covariance (2d array): the covariance matrix

    Output
    ------
    - correlation (2d array): the correlation matrix

    """
    
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    
    return correlation


#==================================================
# Extract covariance matrix from correlation + std
#==================================================

def covariance_from_correlation(correlation_matrix, std):
    """
    Convert correlation matrix to covariance matrix.
    
    Parameters
    ----------
    - correlation_matrix (2d array): numpy array, the correlation matrix
    - std (1d array): numpy array, standard dev along the diagonal of the covariance matrix
    
    Returns:
    - cov_matrix (2d array): numpy array, the covariance matrix
    """

    cov_matrix = correlation_matrix * np.outer(std, std)
    
    return cov_matrix

