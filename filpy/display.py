import numpy as np
import matplotlib.pyplot as plt
from .typing import * 
from .typing import Axes, Figure, AxesArray

__all__ = [
            'myplot',
            'quickplot',
            'v_lines',
            'h_lines',
            'plot_image',
            'show_image'
          ]

def v_lines(ax: Axes, lines: Union[float, list[float]], colors: Union[Optional[str], list[Optional[str]]] = None, linestyles: Union[Optional[str], list[Optional[str]]] = None, labels: Union[Optional[str],list[Optional[str]]] = None,**pltkwargs) -> None:
    """Plot vertical license

    Parameters
    ----------
    ax : Axes
        subplot
    lines : Union[float, list[float]]
        line value or a list of lines
    colors : Union[Optional[str], list[Optional[str]]], optional
        color/s of the value/s, by default None
    linestyles : Union[Optional[str], list[Optional[str]]], optional
        line style/s of the value/s, by default None
    labels : Union[Optional[str],list[Optional[str]]], optional
        label/s of the value/s, by default None
    """
    if not isinstance(lines,list):
        lines = [lines]
    if not isinstance(colors,list):
        colors = [colors]*len(lines)
    if not isinstance(linestyles,list):
        linestyles = [linestyles]*len(lines)
    if not isinstance(labels,list):
        labels = [labels]*len(lines)

    if len(colors) < len(lines):
        colors += [colors[-1]]*(len(lines)-len(colors))
    if len(linestyles) < len(lines):
        linestyles += [linestyles[-1]]*(len(lines)-len(linestyles))
    if len(labels) < len(lines):
        labels += [labels[-1]]*(len(lines)-len(labels))

    for line, color, linestyle, label in zip(lines,colors,linestyles,labels):
        ax.axvline(line,color=color,linestyle=linestyle,label=label,**pltkwargs)

def h_lines(ax: Axes, lines: Union[float, list[float]], colors: Union[Optional[str], list[Optional[str]]] = None, linestyles: Union[Optional[str], list[Optional[str]]] = None, labels: Union[Optional[str],list[Optional[str]]] = None,**pltkwargs) -> None:
    """Plot horizontal license

    Parameters
    ----------
    ax : Axes
        subplot
    lines : Union[float, list[float]]
        line value or a list of lines
    colors : Union[Optional[str], list[Optional[str]]], optional
        color/s of the value/s, by default None
    linestyles : Union[Optional[str], list[Optional[str]]], optional
        line style/s of the value/s, by default None
    labels : Union[Optional[str],list[Optional[str]]], optional
        label/s of the value/s, by default None
    """
    if not isinstance(lines,list):
        lines = [lines]
    if not isinstance(colors,list):
        colors = [colors]*len(lines)
    if not isinstance(linestyles,list):
        linestyles = [linestyles]*len(lines)
    if not isinstance(labels,list):
        labels = [labels]*len(lines)

    if len(colors) < len(lines):
        colors += [colors[-1]]*(len(lines)-len(colors))
    if len(linestyles) < len(lines):
        linestyles += [linestyles[-1]]*(len(lines)-len(linestyles))
    if len(labels) < len(lines):
        labels += [labels[-1]]*(len(lines)-len(labels))

    for line, color, linestyle, label in zip(lines,colors,linestyles,labels):
        ax.axhline(line,color=color,linestyle=linestyle,label=label,**pltkwargs)

def myplot(fig: Figure, ax: Axes, data: Union[Sequence[NDArray], NDArray], title: str = '', suptitle: str = '',xlabel: str = '', ylabel: str = '',grid: bool = False, fontsize: int = 18,**pltkwargs) -> None:
    """Plot data

    Parameters
    ----------
    fig : Figure
        figure
    ax : Axes
        subplot
    data : Union[Sequence[NDArray], NDArray]
        data to plot
    title : str, optional
        the title, by default ''
    suptitle : str, optional
        the suptitle, by default ''
    xlabel : str, optional
        xlabel, by default ''
    ylabel : str, optional
        ylabel, by default ''
    grid : bool, optional
        _description_, by default False
    fontsize : int, optional
        _description_, by default 18
    """
    title_font = fontsize + 2 
    fig.suptitle(suptitle,fontsize=title_font)
    ax.set_title(title,fontsize=title_font)
    if not isinstance(data,Sequence):
        data = [np.arange(len(data)),data]
    ax.errorbar(*data,**pltkwargs)
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    if grid: 
        ax.grid(which='both',linestyle='--',alpha=0.2,color='grey')



def quickplot(data: Union[Sequence[NDArray], NDArray], numfig: Optional[int] = None, figsize: tuple[int,int] = (10,7), output: bool = False, title: str = '', suptitle: str = '',xlabel: str = '', ylabel: str = '',grid: bool = True, fontsize: int = 18,**pltkwargs) -> Optional[tuple[Figure, Axes]]:
    """Display a plot quickly.

    Parameters
    ----------
    data : Sequence[NDArray] | NDArray
        one can pass a single array of data (on y axis) or 
        x data and y data or in addition the corresponding uncertainty/ies
    numfig : int, optional
        figure number, by default `None`
    fmt : str, optional
        makers format, by default `'-'`
    title : str, optional
        title of the figure, by default `''`
    labels : list[str], optional
        axes label [x,y] format, by default `['','']`
    dim : list[int], optional
        figure size, by default `[10,7]`
    grid : bool, optional
        to plot the grid, by default `False`
    **pltargs
        parameters of `matplotlib.pyplot.errorbar()`
    
    Notes
    -----
    You can choose to make a simple plot or adding some stuff.

    (I wrote it only because of my laziness in writing code).

    """
    fig = plt.figure(num=numfig,figsize=figsize)
    ax = fig.add_subplot()
    myplot(fig=fig,ax=ax,
           data=data,
           title=title,
           suptitle=suptitle,
           xlabel=xlabel,
           ylabel=ylabel,
           grid=grid,
           fontsize=fontsize,
           **pltkwargs)
    if output:
        return fig, ax

def plot_image(fig: Figure, ax: Axes, data: NDArray, subtitle: str = '',colorbar: bool = True, **figkwargs) -> None:
    """To plot fits images

    Parameters
    ----------
    fig : Figure
        figure variable
    ax : Axes
        axes variable
    data : Spectrum
        target
    subtitle : str, optional 
        subtitle, by default `''`
    colobar : bool, optional
        if `True` the colorbar is displayed too
    **figkwargs
        parameters of `matplotlib.pyplot.imshow()`
    """
    if 'cmap' not in figkwargs.keys():
        figkwargs['cmap'] = 'gray'
    if 'origin' not in figkwargs.keys():
        figkwargs['origin'] = 'lower'
    if 'fontsize' not in figkwargs.keys():
        figkwargs['fontsize'] = 18
    font_size = figkwargs['fontsize']
    figkwargs.pop('fontsize')
    if 'colorbar_pos' not in figkwargs.keys():
        figkwargs['colorbar_pos'] = 'bottom'
    colorbar_pos = figkwargs['colorbar_pos']
    figkwargs.pop('colorbar_pos')

    # if 'xlabel' not in figkwargs.keys():
    #     figkwargs['xlabel'] = True
    # if 'ylabel' not in figkwargs.keys():
    #     figkwargs['ylabel'] = True
    # if figkwargs['xlabel']: ax.set_xlabel('x [px]', fontsize=font_size)
    # if figkwargs['ylabel']: ax.set_ylabel('y [px]', fontsize=font_size)
    # figkwargs.pop('xlabel')
    # figkwargs.pop('ylabel')
    ax.set_title(subtitle, fontsize=font_size)
    if 'aspect' not in figkwargs.keys():
        figkwargs['aspect'] = 'equal'
    if 'barlabel' not in figkwargs.keys():
        figkwargs['barlabel'] = 'intensity [a.u.]'
    bar_label = figkwargs['barlabel']
    figkwargs.pop('barlabel')
    image = ax.imshow(data,**figkwargs)
    # cbar = fig.colorbar(image, ax=ax, cmap=color, orientation=orientation)
    # cbar.set_label(bar_label,fontsize=font_size)

    if colorbar:
        if colorbar_pos == 'right':
            # adjusting the position and the size of colorbar
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)  
            # generating the colorbar
            cbar = fig.colorbar(image, ax=ax, cmap=figkwargs['cmap'], cax=colorbar_axes)
        elif colorbar_pos == 'bottom':
            fig.subplots_adjust(bottom=0.2)
            cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.05])
            cbar = fig.colorbar(image,cax=cbar_ax, cmap=figkwargs['cmap'], orientation='horizontal')
        cbar.set_label(bar_label,fontsize=font_size)


##*
def show_image(data: Union[NDArray, list[Optional[NDArray]]], num_plots: tuple[int,int] = (1,1), dim: tuple[float,float] = (10,10), title: str = '',subtitles: Union[list[str],str] = '', show: bool = False, fignum: Optional[int] = None, projection: Union[Optional[str],list[Optional[str]]] = None, sharex: Union[list[int],int] = -1, sharey: Union[list[int],int] = -1, **figkwargs) ->  Optional[tuple[Figure, Union[Axes, AxesArray]]]:
    """To plot quickly one or a set of 2D pictures
    
    Parameters
    ----------
    show : bool, optional
        if `True` it displays the figure and the function returns `None`, by default `False`
    projection : str | None, optional
    **figkwargs
        parameters of `plot_image()` and `matplotlib.pyplot.imshow()`
    
    Returns
    -------


    Parameters
    ----------
    data : NDArray | list[NDArray | None]
        target(s). If an element is `None` then that space remains empty
    num_plots : tuple[int,int], optional
        shape of grid of plots, by default `(1,1)`
    dim : tuple[int,int], optional
        figure size, by default `(10,7)`
    title : str, optional
        the title of the image, by default `''`
    subtitles : list[str] | str, optional
        the title of each subplot, by default `''`
    show : bool, optional
        if `True` it displays the figure and the function returns `None`, by default `False`
    projection : str | list[str | None] | None, optional
        the projection of each subplot, 
    **figkwargs
        parameters of `plot_image()` and `matplotlib.pyplot.imshow()`

    Returns
    -------
    fig : Figure
        figure if `show` is `False`
    axs : Axes | AxesArray
        axes if `show` is `False`
    """
    fig = plt.figure(num=fignum,figsize=dim)
    if num_plots != (1,1): 
        len_plots = sum(num_plots)
        if len(data) != len_plots:
            data += [None]*(len_plots-len(data))
        if isinstance(projection,str) or (projection is None):
            projection = [projection]*len_plots
        elif len_plots > len(projection):
            projection += [None]*(len_plots - len(projection)) 
        for key, val in figkwargs.items():
            if not isinstance(val,list):
                figkwargs[key] = [val]*len_plots
            elif len_plots > len(val):
                figkwargs[key] += [None]*(len_plots - len(val))
        if isinstance(subtitles,str):
            subtitles = [subtitles]*len_plots
        elif len_plots > len(subtitles):
            subtitles += ['']*(len_plots - len(subtitles))

        if isinstance(sharex,int): sharex = [sharex]*len_plots
        if isinstance(sharey,int): sharey = [sharey]*len_plots

        fig.suptitle(title, fontsize=20)
        axs = [None]*len_plots
        for i, elem in enumerate(data):
            if elem is not None:
                shx = axs[sharex[i]] 
                shy = axs[sharey[i]]                     
                ax = fig.add_subplot(*num_plots,i+1,projection=projection[i],sharex=shx,sharey=shy)
                axs[i] = ax
                sub = subtitles[i]
                elem_kwargs = { key: val[i] for key, val in figkwargs.items()}
                print(elem_kwargs)
                plot_image(fig,ax,elem,subtitle=sub,**elem_kwargs)
        axs = np.asarray(axs).reshape(*num_plots)
    else: 
        if subtitles == '' and title != '':
            subtitles = title
        else:
            fig.suptitle(title, fontsize=20)
        axs = fig.add_subplot(1,1,1,projection=projection) 
        plot_image(fig,axs,data,subtitle=subtitles,**figkwargs)
    if show: plt.show()
    else: return fig, axs
