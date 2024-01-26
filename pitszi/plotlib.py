"""
This file contains utilities related to plots used in pitszi

"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import corner
import seaborn as sns
import pandas as pd
import warnings

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
        ax = axes[i]
        for j in range(Nchain):
            ax.plot(param_chains[j, :, i], alpha=0.5)
        ax.set_xlim(0, len(param_chains[0,:,0]))
        ax.set_ylabel(parname[i])
        ax.grid()
    axes[-1].set_xlabel("step number")
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
            par_besti = par_best[ipar]
        else:
            par_besti = None
        seaborn_1d(param_chains[:,:,ipar].flatten(),
                   output_fig=out_files[ipar],
                   ci=conf/100, truth=truth[ipar], best=par_besti,
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

    # Corner plot using seaborn
    df = pd.DataFrame(par_flat, columns=parname)
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


