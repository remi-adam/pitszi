"""
This file contains utilities related to plots used in pitszi

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from astropy.wcs import WCS
import astropy.units as u
import matplotlib.pyplot as plt
import corner
import seaborn as sns
import pandas as pd
import warnings

from pitszi import utils


#==================================================
# Plot MCMC chains
#==================================================

def chains_1Dplots(param_chains,
                   parname,
                   output_file):
    """
    Plot related to MCMC chains
        
    Parameters
    ----------
    - param_chains (np array): parameters as Nchain x Npar x Nsample
    - parname (list): list of parameter names
    - output_file (str): output file name

    Output
    ------
    Plots are saved in the output directory

    """

    Npar = len(param_chains[0,0,:])
    Nchain = len(param_chains[:,0,0])

    # Chains
    fig, axes = plt.subplots(Npar, figsize=(8, 2*Npar), sharex=True)
    for i in range(Npar):
        if Npar>1:
            ax = axes[i]
        else:
            ax = axes
            
        for j in range(Nchain):
            ax.plot(param_chains[j, :, i], alpha=0.5)
        ax.set_xlim(0, len(param_chains[0,:,0]))
        ax.set_ylabel(parname[i])
        ax.grid()
    if Npar>1:
        axes[-1].set_xlabel("step number")
    else:
        axes.set_xlabel("step number")
        
    fig.savefig(output_file)
    plt.close()


#==================================================
# Plots 1D chain histograms
#==================================================

def chains_1Dhist(param_chains,
                  parname,
                  out_files,
                  conf=68.0,
                  par_best=None,
                  truth=None):
    """
    Plot related to MCMC chains
        
    Parameters
    ----------
    - param_chains (np array): parameters as Nchain x Npar x Nsample
    - parname (list): list of parameter names
    - out_files (str): file name list for each param
    - par_best (float): best-fit parameter
    - truth (float): the true known value from input simulation

    Output
    ------
    Plots are saved in the output file

    """
    
    Nbin_hist = 40
    Npar = len(param_chains[0,0,:])
    for ipar in range(Npar):
        if par_best is not None:
            par_best_i = par_best[ipar]
        else:
            par_best_i = None
        if truth is not None:
            truth_i = truth[ipar]
        else:
            truth_i = None
    
        flatchain = param_chains[:,:,ipar].flatten()
        
        seaborn_1d(flatchain.astype("float64"),
                   output_fig=out_files[ipar],
                   ci=conf/100, truth=truth_i, best=par_best_i,
                   label=parname[ipar],
                   gridsize=100, alpha=(0.2, 0.4), 
                   figsize=(10,10), fontsize=12,
                   cols=[('blue','grey', 'orange')])
        plt.close("all")


#==================================================
# Plot corner plots
#==================================================

def chains_2Dplots_corner(param_chains,
                          parname,
                          out_file,
                          smooth=1,
                          Nbin_hist=40,
                          truth=None):
    """
    Plot related to MCMC chains
        
    Parameters
    ----------
    - param_chains (np array): parameters as Nchain x Npar x Nsample
    - parname (list): list of parameter names
    - out_file (str): file where to save plot
    - smooth (float): smoothing lenght
    - Nbin_hist (int): number of bins in histogram
    - truth (list): the true values of the recovered parameters

    Output
    ------
    Plots are saved in the output file

    """

    Npar = len(param_chains[0,0,:])
    Nchain = len(param_chains[:,0,0])
    par_flat = param_chains.reshape(param_chains.shape[0]*param_chains.shape[1], param_chains.shape[2])

    # Corner plot using corner
    figure = corner.corner(par_flat,
                           bins=Nbin_hist,
                           color='k',
                           smooth=smooth,
                           labels=parname,
                           quantiles=(0.16, 0.84),
                           levels=[0.68, 0.95])
    if truth is not None:
        axes = np.array(figure.axes).reshape((Npar, Npar))
        for yi in range(Npar):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(truth[xi], color="r")
                ax.axhline(truth[yi], color="r")
                ax.plot(truth[xi], truth[yi], "sr")
    figure.savefig(out_file)
    plt.close("all")


#==================================================
# Plot corner plots with seaborn
#==================================================

def chains_2Dplots_sns(param_chains,
                       parname,
                       out_file,
                       smooth=1,
                       gridsize=100,
                       truth=None):
    """
    Plot related to MCMC chains
        
    Parameters
    ----------
    - param_chains (np array): parameters as Nchain x Npar x Nsample
    - parname (list): list of parameter names
    - out_file (str): file where to save plot
    - smooth (float): smoothing lenght
    - gridsize (int): number of bins in the grid
    - truth (list): the true values of the recovered parameters

    Output
    ------
    Plots are saved in the output file

    """

    Npar = len(param_chains[0,0,:])
    Nchain = len(param_chains[:,0,0])
    par_flat = param_chains.reshape(param_chains.shape[0]*param_chains.shape[1], param_chains.shape[2])

    # Correct scale to have numbers ~1
    power = np.round(np.log10(np.abs(np.mean(par_flat,axis=0))))
    par_flat_scale = par_flat * 10**-power[np.newaxis,:]
    parname_scale = []
    for i in range(len(parname)): parname_scale.append('$10^{'+str(int(-power[i]))+'} '+parname[i]+'$')
    
    # Corner plot using seaborn
    df = pd.DataFrame(par_flat_scale, columns=parname_scale)
    seaborn_corner(df, output_fig=out_file,
                   n_levels=15, cols=[('royalblue', 'k', 'grey', 'Blues')], 
                   ci2d=[0.68, 0.95],
                   smoothing1d=smooth, smoothing2d=smooth, gridsize=gridsize,
                   truth=truth, truth_style='star',
                   limits=None,
                   linewidth=2.0, alpha=(0.1, 0.3, 1.0), figsize=((Npar+1)*3,(Npar+1)*3))
    plt.close("all")
    

#==================================================
# 1D distribution plot function using seaborn
#==================================================

def seaborn_1d(chains, output_fig=None, ci=0.68, truth=None,
               best=None, label=None,
               gridsize=100, alpha=(0.2, 0.4), 
               figsize=(10,10), fontsize=12,
               cols=[('blue','grey', 'orange')]):
    '''
    This function plots 1D distributions of MC chains
    
    Parameters:
    -----------
    - chain (array): the chains sampling the considered parameter
    - output_fig (str): full path to output plot
    - ci (float): confidence interval considered
    - truth (float): the expected truth for overplot
    - best (float list): list of float that contain best fit models to overplot
    - label (str): the label of the parameter
    - gridsize (int): the size of the kde grid
    - alpha (tupple): alpha values for the histogram and the 
    overplotted confidence interval
    - figsize (tupple): the size of the figure
    - fontsize (int): the font size
    - cols (tupple): the colors of the histogram, confidence interval 
    values, and confidence interval filled region

    Output:
    -------
    - Plots
    '''

    # Check type
    if type(chains) is not list:
        chains = [chains]

    # Plot dat
    Ndat = len(chains)            # Number of datasets

    # Make sure there are enough colors
    icol = 0
    while len(cols) < Ndat:
        cols.append(cols[icol])
        icol = icol+1

    fig = plt.figure(0, figsize=(8, 6))
    #----- initial plots of histograms + kde
    for idx, ch in enumerate(chains, start=0):
        sns.histplot(ch, kde=True, kde_kws={'cut':3}, color=cols[idx][0], edgecolor=cols[idx][1], 
                     alpha=alpha[0], stat='density')
    ax = plt.gca()
    ymax = ax.get_ylim()[1]
    
    #----- show limits
    for idx, ch in enumerate(chains, start=0):
        if ci is not None:
            perc = np.percentile(ch, [100 - (100-ci*100)/2.0, 50.0, (100-ci*100)/2.0])
            # Get the KDE line for filling below
            if len(ax.lines) > 0:
                try:
                    xkde = ax.lines[idx].get_xdata()
                    ykde = ax.lines[idx].get_ydata()
                    wkeep = (xkde < perc[0]) * (xkde > perc[2])
                    xkde_itpl = np.append(np.append(perc[2], xkde[wkeep]), perc[0])
                    itpl = interp1d(xkde, ykde)
                    ykde_itpl = itpl(xkde_itpl)
                    perc_max = itpl(perc)
                except:
                    perc_max = perc*0+ymax
                    xkde_itpl = perc*1+0
                    ykde_itpl = ymax*1+0
            else:
                perc_max = perc*0+ymax
                xkde_itpl = perc*1+0
                ykde_itpl = ymax*1+0
            
            ax.vlines(perc[0], 0.0, perc_max[0], linestyle='--', color=cols[idx][1])
            ax.vlines(perc[2], 0.0, perc_max[2], linestyle='--', color=cols[idx][1])
            
            if idx == 0:
                ax.vlines(perc[1], 0.0, perc_max[1], linestyle='-.', label='Median', color=cols[idx][1])
                ax.fill_between(xkde_itpl, 0*ykde_itpl, y2=ykde_itpl, alpha=alpha[1], 
                                color=cols[idx][2], label=str(ci*100)+'% CL')
            else:
                ax.vlines(perc[1], 0.0, perc_max[1], linestyle='-.', color=cols[idx][1])
                ax.fill_between(xkde_itpl, 0*ykde_itpl, y2=ykde_itpl, alpha=alpha[1], color=cols[idx][2])
    
        # Show best fit value                   
        if best is not None:
            if type(best) is not list:
                best = [best]
            ax.vlines(best[idx], 0, ymax, linestyle='-', label='Best-fit', linewidth=2, color=cols[idx][1])
        
    # Show expected value                        
    if truth is not None:
        ax.vlines(truth, 0, ymax, linestyle=':', label='Truth', color='k')
    
    # label and ticks
    if label is not None:
        ax.set_xlabel(label)
        ax.set_ylabel('Probability density')
    ax.set_yticks([])
    ax.set_ylim(0, ymax)
               
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
               
    ax.legend(fontsize=fontsize)
    
    if output_fig is not None:
        plt.savefig(output_fig)


#==================================================
# Corner plot function using seaborn
#==================================================

def seaborn_corner(dfs, output_fig=None, ci2d=[0.95, 0.68], ci1d=0.68,
                   truth=None, truth_style='star', labels=None,
                   smoothing1d=1,smoothing2d=1,
                   gridsize=100, linewidth=0.75, alpha=(0.3, 0.3, 1.0), n_levels=None,
                   zoom=1.0/10, add_grid=True,
                   limits=None,
                   figsize=(10,10), fontsize=12,
                   cols = [('orange',None,'orange','Oranges'),
                           ('green',None,'green','Greens'), 
                           ('magenta',None,'magenta','RdPu'),
                           ('purple',None,'purple','Purples'), 
                           ('blue',None,'blue','Blues'),
                           ('k',None,'k','Greys')]):
    '''
    This function plots corner plot of MC chains
    
    Parameters:
    -----------
    - dfs (list): list of pandas dataframe used for the plot
    - output_fig (str): full path to the figure to save
    - ci2d (list): confidence intervals to be considered in 2d
    - ci1d (list): confidence interval to be considered for the histogram
    - truth (list): list of expected value for the parameters
    - truth_style (str): either 'line' or 'star'
    - labels (list): list of label for the datasets
    - gridsize (int): the number of cells in the grid (higher=nicer=slower)
    - smoothing1d (float): the width of the smoothing kernel for 1d distribution
    - smoothing2d (float): the width of the smoothing kernel for 2d distribution
    - linewidth (float): linewidth of the contours
    - alpha (tuple): alpha parameters for the histogram, histogram CI, and contours plot
    - n_levels (int): if set, will draw a 'diffuse' filled contour plot with n_levels
    - zoom (float): controle the axis limits wrt the plotted distribution.
    The give nnumber corresponds to the fractional size of the 2D distribution 
    to add on each side. If negative, will zoom in the plot.
    - add_grid (bool): add the grid in the plot
    - limits (list of tupple): the limit for each parameters
    - figsize (tuple): the size of the figure
    - fontsize (int): the font size
    - cols (list of 4-tupples): deal with the colors for the dataframes. Each tupple
    is for histogram, histogram edges, filled contours, contour edges
    
    Output:
    -------
    - Plots
    '''
    
    # Check type
    if type(dfs) is not list:
        dfs = [dfs]
    
    # Plot length
    Npar = len(dfs[0].columns) # Number of parameters
    Ndat = len(dfs)            # Number of datasets

    # Percentiles
    levels = 1.0-np.array(ci2d)
    levels = np.append(levels, 1.0)
    levels.sort()

    if n_levels is None:
        n_levels = copy.copy(levels)
    
    # Make sure there are enough colors
    icol = 0
    while len(cols) < Ndat:
        cols.append(cols[icol])
        icol = icol+1
    
    # Figure
    plt.figure(figsize=figsize)
    for ip in range(Npar):
        for jp in range(Npar):
            #----- Diagonal histogram
            if ip == jp:
                plt.subplot(Npar,Npar,ip*Npar+jp+1)
                xlims1, xlims2 = [], []
                ylims = []
                # Get the range
                for idx, df in enumerate(dfs, start=0):
                    xlims1.append(np.nanmin(df[df.columns[ip]]))
                    xlims2.append(np.nanmax(df[df.columns[ip]]))
                xmin = np.nanmin(np.array(xlims1))
                xmax = np.nanmax(np.array(xlims2))
                Dx = (xmax - xmin)*zoom
                for idx, df in enumerate(dfs, start=0):
                    if labels is not None:
                        sns.histplot(x=df.columns[ip], data=df, kde=True,
                                     kde_kws={'cut':3, 'bw_adjust':smoothing1d},
                                     color=cols[idx][0], binrange=[xmin-Dx,xmax+Dx],
                                     alpha=alpha[0], edgecolor=cols[idx][1], stat='density', label=labels[idx])
                    else:
                        sns.histplot(x=df.columns[ip], data=df, kde=True,
                                     kde_kws={'cut':3, 'bw_adjust':smoothing1d},
                                     color=cols[idx][0], binrange=[xmin-Dx,xmax+Dx],
                                     alpha=alpha[0], edgecolor=cols[idx][1], stat='density')
                ax = plt.gca()
                ylims.append(ax.get_ylim()[1])
                ax.set_xlim(xmin-Dx, xmax+Dx)
                if limits is not None:
                    ax.set_xlim(limits[ip][0], limits[ip][1])
                ax.set_ylim(0, np.nanmax(np.array(ylims)))

                
                if ci1d is not None:
                    for idx, df in enumerate(dfs, start=0):
                        try:
                            perc = np.percentile(df[df.columns[ip]], [100-(100-ci1d*100)/2.0, (100-ci1d*100)/2.0])
                            # Get the KDE line for filling below
                            xkde = ax.lines[idx].get_xdata()
                            ykde = ax.lines[idx].get_ydata()
                            wkeep = (xkde < perc[0]) * (xkde > perc[1])
                            xkde_itpl = np.append(np.append(perc[1], xkde[wkeep]), perc[0])
                            itpl = interp1d(xkde, ykde)
                            ykde_itpl = itpl(xkde_itpl)
                            perc_max = itpl(perc)
                            
                            ax.vlines(perc[0], 0.0, perc_max[0], linestyle='--', color=cols[idx][0])
                            ax.vlines(perc[1], 0.0, perc_max[1], linestyle='--', color=cols[idx][0])
                            ax.fill_between(xkde_itpl, 0*ykde_itpl, y2=ykde_itpl,
                                            alpha=alpha[1], color=cols[idx][0])
                        except:
                            warnings.warn('could not get and fill bellow KDE line')
                        
                if add_grid:
                    #ax.xaxis.set_major_locator(MultipleLocator((xmax+Dx-(xmin-Dx))/5.0))
                    ax.grid(True, axis='x', linestyle='--')
                else:
                    ax.grid(False)
                        
                if truth is not None:
                    ax.vlines(truth[ip], ax.get_ylim()[0], ax.get_ylim()[1], linestyle=':', color='k')
                    
                plt.yticks([])
                plt.ylabel(None)
                if jp<Npar-1:
                    #plt.xticks([])
                    ax.set_xticklabels([])
                    plt.xlabel(None)
                if ip == 0 and labels is not None:
                    plt.legend(loc='upper left')
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(fontsize)
                    
            #----- Off diagonal 2d plots
            if ip>jp:
                plt.subplot(Npar,Npar,ip*Npar+jp+1)
                xlims1, xlims2 = [], []
                ylims1, ylims2 = [], []
                for idx, df in enumerate(dfs, start=0):
                    xlims1.append(np.nanmin(df[df.columns[jp]]))
                    xlims2.append(np.nanmax(df[df.columns[jp]]))
                    ylims1.append(np.nanmin(df[df.columns[ip]]))
                    ylims2.append(np.nanmax(df[df.columns[ip]]))
                    sns.kdeplot(x=df.columns[jp], y=df.columns[ip], data=df, gridsize=gridsize, 
                                n_levels=n_levels, levels=levels, thresh=levels[0], fill=True, 
                                cmap=cols[idx][3], alpha=alpha[2],
                                bw_adjust=smoothing2d)
                    sns.kdeplot(x=df.columns[jp], y=df.columns[ip], data=df, gridsize=gridsize, 
                                levels=levels[0:-1], color=cols[idx][2], linewidths=linewidth,
                                bw_adjust=smoothing2d)
                ax = plt.gca()
                xmin = np.nanmin(np.array(xlims1))
                xmax = np.nanmax(np.array(xlims2))
                Dx = (xmax - xmin)*zoom
                ymin = np.nanmin(np.array(ylims1))
                ymax = np.nanmax(np.array(ylims2))
                Dy = (ymax - ymin)*zoom

                ax.set_xlim(xmin-Dx, xmax+Dx)
                ax.set_ylim(ymin-Dy, ymax+Dy)
                if limits is not None:
                    ax.set_xlim(limits[jp][0], limits[jp][1])
                    ax.set_ylim(limits[ip][0], limits[ip][1])
                
                if add_grid:
                    #ax.xaxis.set_major_locator(MultipleLocator((xmax+Dx-(xmin-Dx))/5.0))
                    #ax.yaxis.set_major_locator(MultipleLocator((ymax+Dy-(ymin-Dy))/5.0))
                    ax.grid(True, linestyle='--')
                else:
                    ax.grid(False)

                if truth is not None:
                    if truth_style is 'line':
                        ax.vlines(truth[jp], ax.get_ylim()[0], ax.get_ylim()[1], linestyle=':', color='k')
                        ax.hlines(truth[ip], ax.get_xlim()[0], ax.get_xlim()[1], linestyle=':', color='k')
                    if truth_style is 'star':
                        ax.plot(truth[jp], truth[ip], linestyle='', marker="*", color='k', markersize=10)
                    
                if jp > 0:
                    #plt.yticks([])
                    ax.set_yticklabels([])
                    plt.ylabel(None)
                if ip<Npar-1:
                    #ax.set_xticks([])
                    ax.set_xticklabels([])
                    ax.set_xlabel(None)
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(fontsize)
                    
    plt.tight_layout(h_pad=0.0,w_pad=0.0)
    if output_fig is not None:
        plt.savefig(output_fig)


#==================================================
# Plot of the best-fit map
#==================================================

def show_fit_result_ymap(figfile,
                         image, header, noise_mc,
                         model_ymap_sph,
                         mask=None,
                         visu_smooth=10,
                         cmap='Spectral_r',
                         contcol='k'):
    '''
    This function plots the best fit map model
    
    Parameters:
    ----------
    - figfile (str): name of the file to produce
    - image (2d array): the data image
    - header (str): the header
    - noise_mc (3d array): the noise MC realization
    - model_ymap_sph (2d image): the best fit model
    - mask (2d array): the mask if any
    - visu_smooth (float): the smoothing FWHM in arcsec
    - cmap (str): colormap
    - contcol (str): the contour color
    
    Output:
    -------
    - Plots produced
    '''

    #----- Get extra data info
    if mask is None:
        mask = image*0+1

    #----- Compute the rms at given scale
    rms = np.std(gaussian_filter(noise_mc,
                                 sigma=np.array([0,1,1])*visu_smooth/2.35/header['CDELT2']/3600),
                 axis=0)
    levels = [-30,-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,
              2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

    #----- Compute plot range
    rng_map = [np.amin(mask*gaussian_filter(image,
                                            sigma=visu_smooth/2.35/header['CDELT2']/3600)*1e5),
               np.amax(mask*gaussian_filter(image,
                                            sigma=visu_smooth/2.35/header['CDELT2']/3600)*1e5)]
    
    stdres = np.std((gaussian_filter(image-model_ymap_sph,
                                     sigma=visu_smooth/2.35/header['CDELT2']/3600)*1e5)[mask > 0])
    rng_res = np.array([-1,1]) * stdres * 3
    
    #----- Show the maps
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(0, figsize=(17, 5))

    # Input data
    ax = plt.subplot(1, 3, 1, projection=WCS(header))
    plt.imshow(mask * gaussian_filter(image,
                                      sigma=visu_smooth/2.35/header['CDELT2']/3600)*1e5,
               cmap=cmap, vmin=rng_map[0], vmax=rng_map[1])
    cb = plt.colorbar()
    plt.contour(mask*gaussian_filter(image,
                                     sigma=visu_smooth/2.35/header['CDELT2']/3600)/rms,
                levels=levels, colors=contcol)
    plt.title('y-Compton data')
    plt.xlabel('R.A. (deg)')
    plt.ylabel('Dec. (deg)')
    
    # Best model
    ax = plt.subplot(1, 3, 2, projection=WCS(header))
    plt.imshow(gaussian_filter(model_ymap_sph,
                               sigma=visu_smooth/2.35/header['CDELT2']/3600)*1e5,
               cmap=cmap, vmin=rng_map[0], vmax=rng_map[1])
    cb = plt.colorbar()
    plt.title('y-Compton best-fit')
    plt.xlabel('R.A. (deg)')
    plt.ylabel(' ')

    # Best model residual
    ax = plt.subplot(1, 3, 3, projection=WCS(header))
    plt.imshow(gaussian_filter(image - model_ymap_sph,
                               sigma=visu_smooth/2.35/header['CDELT2']/3600)*1e5*mask,
               cmap=cmap,
               vmin=rng_res[0], vmax=rng_res[1])
    cb = plt.colorbar()
    plt.contour(mask*gaussian_filter(image-model_ymap_sph,
                                     sigma=visu_smooth/2.35/header['CDELT2']/3600)/rms,
                levels=levels, colors=contcol)
    plt.title('y-Compton residual')
    plt.xlabel('R.A. (deg)')
    plt.ylabel(' ')
    plt.savefig(figfile)
    plt.close()
    

#==================================================
# Plot of the best-fit profile
#==================================================

def show_fit_ycompton_profile(figfile,
                              r2d,
                              data_yprof, data_yprof_err,
                              prof_best,
                              prof_mc,
                              true_compton_profile=None):
    '''
    This function plots the pressure profile results
    
    Parameters:
    ----------
    - figfile (str): name of the file to produce
    - r2d (1d array): the radius in arcmin
    - data_yprof (1d array): the data points
    - data_yprof_err (1d array): the data error
    - prof_best (1d array): the best-fit y compton
    - prof_mc (2d array): the MC resampling of the profile (Nsample x len(r2d))
    - true_y_profile (dict): pass a dictionary containing the profile to compare with
    in the form {'r':array in arcmin, 'p':array in [y]}

    Output:
    -------
    - Plots produced
    '''
    
    #----- Compute model uncertainties
    ci1 = 68.0
    prof_perc1 = np.percentile(prof_mc,
                              (100-(100-ci1)/2.0, 50, (100-ci1)/2.0), axis=0)
    ci2 = 95.0
    prof_perc2 = np.percentile(prof_mc,
                              (100-(100-ci2)/2.0, 50, (100-ci2)/2.0), axis=0)
    
    #----- Plot the result
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(0, figsize=(7, 5))
    plt.errorbar(r2d, 1e5*data_yprof, 1e5*data_yprof_err, marker='o', ls='', color='k', label='Data')
    plt.plot(r2d, 1e5*prof_best, color='r', ls='-', label='Best-fit model')
    plt.plot(r2d, 1e5*prof_perc1[1,:], color='b', ls='--', label='Median, 68% and 95% C.I.')
    plt.fill_between(r2d, 1e5*prof_perc1[0,:], y2=1e5*prof_perc1[2,:], alpha=0.3, color='blue')
    plt.fill_between(r2d, 1e5*prof_perc2[0,:], y2=1e5*prof_perc2[2,:], alpha=0.15, color='blue')

    if true_compton_profile is not None:
        plt.plot(true_compton_profile['r'], true_compton_profile['y']*1e5, label='True profile', color='orange')
    plt.xlabel('Radius (arcmin)')
    plt.ylabel(r'$10^5 \times y$-Compton')
    plt.xlim(0, np.amax(r2d[np.isfinite(data_yprof)])*1.2)
    plt.legend(fontsize=12)
    plt.savefig(figfile)
    plt.close()
        

#==================================================
# Plot of the best-fit profile
#==================================================

def show_fit_result_pressure_profile(figfile,
                                     r3d,
                                     prof_best,
                                     prof_mc,
                                     true_pressure_profile=None):
    '''
    This function plots the pressure profile results
    
    Parameters:
    ----------
    - figfile (str): name of the file to produce
    - r3d (1d array): the radius in kpc
    - prof_best (1d array): the best-fit pressure in keV/cm^3
    - prof_mc (2d array): the MC resampling of the profile (Nsample x len(r3d))
    - true_pressure_profile (dict): pass a dictionary containing the profile to compare with
    in the form {'r':array in kpc, 'p':array in keV cm-3}
    
    Output:
    -------
    - Plots produced
    '''

    #----- Compute model uncertainties
    ci1 = 68.0
    prof_perc1 = np.percentile(prof_mc,
                              (100-(100-ci1)/2.0, 50, (100-ci1)/2.0), axis=0)
    ci2 = 95.0
    prof_perc2 = np.percentile(prof_mc,
                               (100-(100-ci2)/2.0, 50, (100-ci2)/2.0), axis=0)
    
    #----- Plot the result
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(0, figsize=(7, 5))
    plt.plot(r3d, prof_best, color='r', ls='-', label='Best-fit model')
    plt.plot(r3d, prof_perc1[1,:], color='b', ls='--', label='Median, 68% and 95% C.I.')
    plt.fill_between(r3d, prof_perc2[0,:], y2=prof_perc2[2,:], alpha=0.15, color='blue')
    plt.fill_between(r3d, prof_perc1[0,:], y2=prof_perc1[2,:], alpha=0.3, color='blue')
    if true_pressure_profile is not None:
        plt.plot(true_pressure_profile['r'], true_pressure_profile['p'], label='True profile', color='orange')
    plt.xlabel('Radius (kpc)')
    plt.ylabel(r'$P_e(r)$ (keV/cm$^3$)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(np.amin(r3d), np.amax(r3d))
    plt.legend(fontsize=12)
    plt.savefig(figfile)
    plt.close()

    
#==================================================
# Plot of contraint on Pk3d
#==================================================

def show_fit_result_pk3d(figfile,
                         k3d,
                         pk3d_best,
                         pk3d_mc,
                         true_pk3d=None):
    '''
    This function plots the deprojected constraint on Pk3d
    given the 2d fit.
    
    Parameters:
    ----------
    - figfile (str): name of the file to produce
    - k3d (1d array): k vector in physical scale
    - pk3d_best (1d array): best fit model for Pk3d
    - pk3d_mc (2d array): Pk3d for parameter sampling
    - true_pk3d (dict): pass a dictionary containing the Pk3d to compare with
    in the form {'k':array in kpc-1, 'pk':array in kpc3}
    
    Output:
    -------
    - Plots produced

    '''
    
    #----- Compute model uncertainties
    ci1 = 68.0
    pk3d_perc1 = np.percentile(pk3d_mc,
                               (100-(100-ci1)/2.0, 50, (100-ci1)/2.0), axis=0)
    ci2 = 95.0
    pk3d_perc2 = np.percentile(pk3d_mc,
                               (100-(100-ci2)/2.0, 50, (100-ci2)/2.0), axis=0)

    #----- Plot the result
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(0, figsize=(7, 5))
    plt.loglog(k3d, np.sqrt(4*np.pi*k3d**3*pk3d_best), color='r', ls='-', label='Best-fit')
    plt.loglog(k3d, np.sqrt(4*np.pi*k3d**3*pk3d_perc1[1,:]), color='b', ls='--', label='Median, 68% and 95% C.I.')

    plt.fill_between(k3d, np.sqrt(4*np.pi*k3d**3*pk3d_perc1[0,:]), np.sqrt(4*np.pi*k3d**3*pk3d_perc1[2,:]),
                     color='blue', alpha=0.3)
    plt.fill_between(k3d, np.sqrt(4*np.pi*k3d**3*pk3d_perc2[0,:]), np.sqrt(4*np.pi*k3d**3*pk3d_perc2[2,:]),
                     color='blue', alpha=0.15)
    
    if true_pk3d is not None:
        plt.plot(true_pk3d['k'], np.sqrt(4*np.pi*true_pk3d['k']**3*true_pk3d['pk']),
                 label='True $P_k$', color='orange')
    plt.xlabel(r'$k$ (kpc$^{-1}$)')
    plt.ylabel(r'$\sqrt{4 \pi k^3 P(k)}$')
    plt.ylim(np.amax(np.sqrt(4*np.pi*k3d**3*pk3d_perc1[1,:]))*1e-3,
             np.amax(np.sqrt(4*np.pi*k3d**3*pk3d_perc1[1,:]))*5)
    plt.xlim(np.amin(k3d), np.amax(k3d))
    plt.legend(fontsize=12)
    plt.savefig(figfile)
    plt.close()


#==================================================
# Plot of contraint on Pk2d
#==================================================

def show_fit_result_pk2d(figfile,
                         k2d,
                         pk2d_data,
                         pk2d_modref,
                         pk2d_data_err_model,
                         pk2d_data_err_noise,
                         pk2d_data_err_bkg,
                         pk2d_mc,
                         pk2d_noise_best, pk2d_noise_mc,                         
                         pk2d_bkg_best, pk2d_bkg_mc,                         
                         true_pk2d=None):
    '''
    This function plots the deprojected constraint on Pk3d
    given the 2d fit.
    
    Parameters:
    ----------
    - figfile (str): name of the file to produce
    - k2d (1d array): k vector in physical scale
    - pk2d_data (1d array): data points
    - pk2d_modref (1d array): reference model used for uncertainty computation
    - pk2d_data_err_model (1d array): uncertainty acssociated with the model
    - pk2d_data_err_noise (1d array): uncertainty associated with the noise
    - pk2d_data_err_bkg (1d array): uncertainty associated with the bkg
    - pk2d_mc (2d array): MC ressampling for the cluster model contribution
    - pk2d_noise_best (1d array): bestfit noise contribution
    - pk2d_noise_mc (2d array): MC ressampling for the noise contribution
    - pk2d_bkg_best (1d array): bestfit bkg contribution
    - pk2d_bkg_mc (2d array): MC ressampling for the bkg contribution
    - true_pk2d (dict): pass a dictionary containing the Pk3d to compare with
    in the form {'k':array in arcsec-1, 'pk':array in arcsec2}
    
    Output:
    -------
    - Plots produced

    '''
    
    #----- Compute model uncertainties
    ci1 = 68.0
    pk2d_perc1 = np.percentile(pk2d_mc,
                              (100-(100-ci1)/2.0, 50, (100-ci1)/2.0), axis=0)
    pk2d_noise_perc1 = np.percentile(pk2d_noise_mc,
                                    (100-(100-ci1)/2.0, 50, (100-ci1)/2.0), axis=0)
    pk2d_bkg_perc1 = np.percentile(pk2d_bkg_mc,
                                   (100-(100-ci1)/2.0, 50, (100-ci1)/2.0), axis=0)

    ci2 = 95.0
    pk2d_perc2 = np.percentile(pk2d_mc,
                              (100-(100-ci2)/2.0, 50, (100-ci2)/2.0), axis=0)
    pk2d_noise_perc2 = np.percentile(pk2d_noise_mc,
                                    (100-(100-ci2)/2.0, 50, (100-ci2)/2.0), axis=0)
    pk2d_bkg_perc2 = np.percentile(pk2d_bkg_mc,
                                    (100-(100-ci2)/2.0, 50, (100-ci2)/2.0), axis=0)

    #----- Compute uncertainty
    err_noise = np.sqrt(2*np.pi*k2d**2)/(2*np.sqrt(pk2d_data))*pk2d_data_err_noise
    err_nbkg  = np.sqrt(2*np.pi*k2d**2)/(2*np.sqrt(pk2d_data))*(pk2d_data_err_noise**2
                                                                + pk2d_data_err_bkg**2)**0.5
    err_tot   = np.sqrt(2*np.pi*k2d**2)/(2*np.sqrt(pk2d_data))*(pk2d_data_err_model**2
                                                                + pk2d_data_err_noise**2
                                                                + pk2d_data_err_bkg**2)**0.5
    
    #----- Plot the result
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(0, figsize=(7, 5))
    frame1 = fig.add_axes((.15,.3,.8,.6))

    # Data
    plt.errorbar(k2d, np.sqrt(2*np.pi*k2d**2*pk2d_data), err_tot, marker='.', ls='', color='lightgrey',
                 label='Data (noise+bkg+model uncertainties)')
    plt.errorbar(k2d, np.sqrt(2*np.pi*k2d**2*pk2d_data), err_nbkg, marker='.', ls='', color='grey',
                 label='Data (noise+bkg uncertainty)')
    plt.errorbar(k2d, np.sqrt(2*np.pi*k2d**2*pk2d_data), err_noise, marker='.', ls='', color='k',
                 label='Data (noise uncertainty)')

    # Model reference
    plt.plot(k2d, np.sqrt(2*np.pi*k2d**2*(pk2d_modref)), color='magenta', label='Best-fit model and intrinsic scatter')
    up, low = pk2d_modref+pk2d_data_err_model, pk2d_modref-pk2d_data_err_model
    low[low<=0] = 0
    plt.fill_between(k2d, np.sqrt(2*np.pi*k2d**2*(up)), y2=np.sqrt(2*np.pi*k2d**2*(low)),
                     color='magenta', alpha=0.3)
    
    # Model fit
    plt.loglog(k2d, np.sqrt(2*np.pi*k2d**2*pk2d_perc1[1,:]), color='b', ls='--', label='Model median, 68% and 95% C.I.')
    plt.fill_between(k2d, np.sqrt(2*np.pi*k2d**2*pk2d_perc1[0,:]), np.sqrt(2*np.pi*k2d**2*pk2d_perc1[2,:]),
                     color='blue', alpha=0.3)
    plt.fill_between(k2d, np.sqrt(2*np.pi*k2d**2*pk2d_perc2[0,:]), np.sqrt(2*np.pi*k2d**2*pk2d_perc2[2,:]),
                     color='blue', alpha=0.15)
    
    # Noise fit
    plt.loglog(k2d, np.sqrt(2*np.pi*k2d**2*pk2d_noise_perc1[1,:]), color='darkcyan', ls='--',
               label='Noise median, 68% and 95% C.I.')
    plt.fill_between(k2d, np.sqrt(2*np.pi*k2d**2*pk2d_noise_perc1[0,:]),
                     np.sqrt(2*np.pi*k2d**2*pk2d_noise_perc1[2,:]),
                     color='darkcyan', alpha=0.3)
    plt.fill_between(k2d, np.sqrt(2*np.pi*k2d**2*pk2d_noise_perc2[0,:]),
                     np.sqrt(2*np.pi*k2d**2*pk2d_noise_perc2[2,:]),
                     color='darkcyan', alpha=0.15)

    # Bkg fit
    plt.loglog(k2d, np.sqrt(2*np.pi*k2d**2*pk2d_bkg_perc1[1,:]), color='darkorange', ls='--',
               label='Bkg median, 68% and 95% C.I.')
    plt.fill_between(k2d, np.sqrt(2*np.pi*k2d**2*pk2d_bkg_perc1[0,:]),
                     np.sqrt(2*np.pi*k2d**2*pk2d_bkg_perc1[2,:]),
                     color='darkorange', alpha=0.3)
    plt.fill_between(k2d, np.sqrt(2*np.pi*k2d**2*pk2d_bkg_perc2[0,:]),
                     np.sqrt(2*np.pi*k2d**2*pk2d_bkg_perc2[2,:]),
                     color='darkorange', alpha=0.15)

    # Total
    plt.loglog(k2d, np.sqrt(2*np.pi*k2d**2*(pk2d_noise_best + pk2d_bkg_best + pk2d_modref)),
               color='red', ls='-', label='Total best-fit')
        
    if true_pk2d is not None:
        plt.plot(true_pk2d['k'], np.sqrt(2*np.pi*true_pk2d['k']**2*true_pk2d['pk']),
                 label='True $P_k$', color='orange')
    plt.ylabel(r'$\sqrt{2 \pi k^2 P(k)}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(np.amin(k2d)*0.9, np.amax(k2d)*1.1)
    
    plt.ylim(np.nanmax(np.sqrt(2*np.pi*k2d**2*pk2d_data)+err_tot)*1e-3,
             np.nanmax(np.sqrt(2*np.pi*k2d**2*pk2d_data)+err_tot)*2)
    
    plt.legend(fontsize=10)

    frame2 = fig.add_axes((.15,.1,.8,.2))
    dat = np.sqrt(2*np.pi*k2d**2*pk2d_data)
    mod = np.sqrt(2*np.pi*k2d**2*(pk2d_noise_best + pk2d_bkg_best + pk2d_modref))
    plt.plot(k2d, (dat-mod)/err_tot, marker='o', color='lightgrey')
    plt.plot(k2d, (dat-mod)/err_nbkg, marker='o', color='grey')
    plt.plot(k2d, (dat-mod)/err_noise, marker='o', color='k')
    plt.axhline(0, color='k', linestyle='-')
    plt.axhline(+3, color='k', linestyle='--')
    plt.axhline(-3, color='k', linestyle='--')
    plt.axhline(+5, color='k', linestyle='--')
    plt.axhline(-5, color='k', linestyle='--')
    plt.xscale('log')
    plt.xlabel(r'$k$ (kpc$^{-1}$)')
    plt.ylabel('Residual ($\sigma$)')
    
    plt.savefig(figfile)
    plt.close()


#==================================================
# Plot of delta_y / y fitted
#==================================================

def show_input_delta_ymap(figfile,
                          image,
                          data_image,
                          model_ymap_sph1,
                          model_ymap_sph2,
                          w8,
                          header,
                          noise_mc,
                          mask=None,
                          visu_smooth=10,
                          cmap='Spectral_r',
                          contcol='k'):
    '''
    This function plots the best fit map model
    
    Parameters:
    ----------
    - figfile (str): name of the file to produce
    - image (2d array): the data image
    - header (str): the header
    - model_ymap_sph1 (2d image): the best fit model
    - model_ymap_sph2 (2d image): the best fit model deconvolved
    - w8 (2d array): the weight map
    - mask (2d array): the mask if any
    - visu_smooth (float): the smoothing FWHM in arcsec
    - cmap (str): colormap
    - contcol (str): contour color
    
    Output:
    -------
    - Plots produced
    '''

    #----- Get extra data info
    if mask is None:
        mask = image*0+1

    #----- Compute the rms at given scale
    rms = np.std(gaussian_filter(noise_mc,
                                 sigma=np.array([0,1,1])*visu_smooth/2.35/header['CDELT2']/3600),
                 axis=0)
    rms_img = np.std(gaussian_filter(noise_mc*w8[np.newaxis,:,:]/model_ymap_sph2,
                                     sigma=np.array([0,1,1])*visu_smooth/2.35/header['CDELT2']/3600),
                 axis=0)
    rms_img[rms_img <=0] = np.nan
    levels = [-30,-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,
              2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

    #---------- Plot
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(0, figsize=(23, 5))

    # Input data
    ax = plt.subplot(1, 4, 1, projection=WCS(header))
    plt.imshow(mask * gaussian_filter(image,
                                      sigma=visu_smooth/2.35/header['CDELT2']/3600)*1e5,cmap=cmap)
    cb = plt.colorbar()
    plt.contour(mask*gaussian_filter(image,
                                     sigma=visu_smooth/2.35/header['CDELT2']/3600)/rms,
                levels=levels, colors=contcol)
    plt.title(r'y-Compton data ($\times 10^5$)')
    plt.xlabel('R.A. (deg)')
    plt.ylabel('Dec. (deg)')
    
    # Best model
    ax = plt.subplot(1, 4, 2, projection=WCS(header))
    plt.imshow(model_ymap_sph1*1e5,cmap=cmap)
    cb = plt.colorbar()
    plt.title(r'y-Compton best-fit ($\times 10^5$)')
    plt.xlabel('R.A. (deg)')
    plt.ylabel(' ')

    # w8
    ax = plt.subplot(1, 4, 3, projection=WCS(header))
    plt.imshow(w8,cmap=cmap)
    cb = plt.colorbar()
    plt.title('Weight')
    plt.xlabel('R.A. (deg)')
    plt.ylabel(' ')
    
    # dy/y w8
    ax = plt.subplot(1, 4, 4, projection=WCS(header))
    plt.imshow(gaussian_filter(data_image,
                               sigma=visu_smooth/2.35/header['CDELT2']/3600),cmap=cmap)
    cb = plt.colorbar()
    plt.contour(gaussian_filter(data_image,
                                sigma=visu_smooth/2.35/header['CDELT2']/3600)/rms_img,
                levels=levels, colors=contcol)
    
    plt.title(r'$\Delta y / y \times W$')
    plt.xlabel('R.A. (deg)')
    plt.ylabel(' ')
    
    plt.savefig(figfile)
    plt.close()


#==================================================
# Plot of covariance
#==================================================

def show_fit_result_covariance(figfile,
                               covmat_data,
                               covmat_bfmodel,
                               covmat_model=None,
                               covmat_bkg=None):
    '''
    This function plots the best fit map model
    
    Parameters:
    ----------
    - figfile (str): name of the file to produce
    - covmat_data (2d array): bin to bin noise covariance matrix
    - covmat_bfmodel (2d array): bin to bin best fit model covariance matrix
    - covmat_model (2d array): bin to bin reference model covariance matrix
    - covmat_nkg (2d array): bin to bin background covariance matrix

    Output:
    -------
    - Plots produced
    '''
    
    Nrow = 2
    if covmat_model is not None: Nrow += 1
    if covmat_bkg   is not None: Nrow += 1
    
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(0, figsize=(12, 5*Nrow))

    # Covariance data
    ax = plt.subplot(Nrow, 2, 1)
    plt.imshow(covmat_data)
    cb = plt.colorbar()
    plt.title(r'Noise covariance matrix')
    plt.xlabel('k bin')
    plt.ylabel('k bin')
    
    # Correlation data
    ax = plt.subplot(Nrow, 2, 2)
    plt.imshow(utils.correlation_from_covariance(covmat_data))
    cb = plt.colorbar()
    plt.title(r'Noise correlation matrix')
    plt.xlabel('k bin')
    plt.ylabel('k bin')

    # Covariance best fit model
    ax = plt.subplot(Nrow, 2, 3)
    plt.imshow(covmat_bfmodel)
    cb = plt.colorbar()
    plt.title(r'Best fit model covariance matrix')
    plt.xlabel('k bin')
    plt.ylabel('k bin')
    
    # Correlation best fit model
    ax = plt.subplot(Nrow, 2, 4)
    plt.imshow(utils.correlation_from_covariance(covmat_bfmodel))
    cb = plt.colorbar()
    plt.title(r'Best fit model correlation matrix')
    plt.xlabel('k bin')
    plt.ylabel('k bin')

    idx = 0
    if covmat_model is not None:
        idx += 2
        # Covariance model
        ax = plt.subplot(Nrow, 2, 5)
        plt.imshow(covmat_model)
        cb = plt.colorbar()
        plt.title(r'Ref model covariance matrix')
        plt.xlabel('k bin')
        plt.ylabel('k bin')
        
        # Correlation model
        ax = plt.subplot(Nrow, 2, 6)
        plt.imshow(utils.correlation_from_covariance(covmat_model))
        cb = plt.colorbar()
        plt.title(r'Ref model correlation matrix')
        plt.xlabel('k bin')
        plt.ylabel('k bin')

    if covmat_bkg is not None:
    
        # Covariance model
        ax = plt.subplot(Nrow, 2, 5+idx)
        plt.imshow(covmat_bkg)
        cb = plt.colorbar()
        plt.title(r'Background covariance matrix')
        plt.xlabel('k bin')
        plt.ylabel('k bin')
        
        # Correlation model
        ax = plt.subplot(Nrow, 2, 6+idx)
        plt.imshow(utils.correlation_from_covariance(covmat_bkg))
        cb = plt.colorbar()
        plt.title(r'Background correlation matrix')
        plt.xlabel('k bin')
        plt.ylabel('k bin')
    
    plt.savefig(figfile)
    plt.close()
