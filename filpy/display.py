import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Sequence, Any



def quickplot(data: Sequence[ndarray] | ndarray, numfig: int = None, fmt: str = '-', title: str = '', labels: Sequence[str] = ('',''), dim: Sequence[int] = [10,7], grid: bool = False,**pltargs) -> None:
    """Function to display a plot quickly.

    Parameters
    ----------
    data : Sequence[ndarray] | ndarray
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
    if 'fontsize' not in pltargs.keys():
        pltargs['fontsize'] = 18
    font_size = pltargs['fontsize']
    pltargs.pop('fontsize')
    xl,yl = labels
    plt.figure(numfig,figsize=dim)
    plt.title(title,fontsize=font_size+2)
    if isinstance(data,ndarray): data = [np.arange(len(data)),data]
    plt.errorbar(*data,fmt=fmt,**pltargs)
    plt.xlabel(xl,fontsize=font_size)
    plt.ylabel(yl,fontsize=font_size)
    if grid: plt.grid(which='both',linestyle='--',alpha=0.2,color='grey')


def plot_image(fig: Figure, ax: Axes, data: ndarray, v: int = 0, subtitle: str = '',colorbar: bool = True, **figargs) -> None:
    """To plot fits images

    Parameters
    ----------
    fig : Figure
        figure variable
    ax : Axes
        axes variable
    data : Spectrum
        target
    v : int, optional
        color-code, by default `-1`
    subtitle : str 
        subtitle, by default `''`
    **figargs
        parameters of `matplotlib.pyplot.imshow()`
    """
    if 'cmap' not in figargs.keys():
        figargs['cmap'] = 'gray'
    if 'origin' not in figargs.keys():
        figargs['origin'] = 'lower'
    if 'fontsize' not in figargs.keys():
        figargs['fontsize'] = 18
    font_size = figargs['fontsize']
    figargs.pop('fontsize')
    if 'colorbar_pos' not in figargs.keys():
        figargs['colorbar_pos'] = 'bottom'
    colorbar_pos = figargs['colorbar_pos']
    figargs.pop('colorbar_pos')

    # if 'xlabel' not in figargs.keys():
    #     figargs['xlabel'] = True
    # if 'ylabel' not in figargs.keys():
    #     figargs['ylabel'] = True
    # if figargs['xlabel']: ax.set_xlabel('x [px]', fontsize=font_size)
    # if figargs['ylabel']: ax.set_ylabel('y [px]', fontsize=font_size)
    # figargs.pop('xlabel')
    # figargs.pop('ylabel')
    ax.set_title(subtitle, fontsize=font_size)
    if 'aspect' not in figargs.keys():
        figargs['aspect'] = 'equal'
    if 'barlabel' not in figargs.keys():
        figargs['barlabel'] = 'intensity [a.u.]'
    bar_label = figargs['barlabel']
    figargs.pop('barlabel')
    image = ax.imshow(data,**figargs)
    # cbar = fig.colorbar(image, ax=ax, cmap=color, orientation=orientation)
    # cbar.set_label(bar_label,fontsize=font_size)

    if colorbar:
        if colorbar_pos == 'right':
            # adjusting the position and the size of colorbar
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)  
            # generating the colorbar
            cbar = fig.colorbar(image, ax=ax, cmap=figargs['cmap'], cax=colorbar_axes)
        elif colorbar_pos == 'bottom':
            fig.subplots_adjust(bottom=0.2)
            cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.05])
            cbar = fig.colorbar(image,cax=cbar_ax, cmap=figargs['cmap'], orientation='horizontal')
        cbar.set_label(bar_label,fontsize=font_size)


##*
def show_image(data: ndarray | list[ndarray], num_plots: Sequence[int] = (1,1), dim: Sequence[int] = (10,10), title: str = '',subtitles: list[str] | None = None, show: bool = False, projection: str | None = None, **figargs) -> None | tuple[Figure, Axes | ndarray]:
    """To plot quickly one or a set of fits pictures
    
    Parameters
    ----------
    data : Spectrum
        target
    num_plots : Sequence[int], optional
        shape of grid of plots, by default `(1,1)`
    dim : Sequence[int], optional
        figure size, by default `(10,7)`
    title : str, optional
        title of the image, by default `''`
    show : bool, optional
        if `True` it displays the figure, by default `False`
    **figargs
        parameters of `plot_image()` and `matplotlib.pyplot.imshow()`
    
    Returns
    -------
    fig : Figure
        figure
    axs : Axes | ndarray
        axes
    """
    fig, axs = plt.subplots(*num_plots, figsize=dim, subplot_kw={'projection': projection}) 
    if num_plots != (1,1): 
        if subtitles is None:
            subtitles = ['']*len(data)
        fig.suptitle(title, fontsize=20)
        for (ax,elem,sub) in zip(axs,data,subtitles):
            plot_image(fig,ax,elem,subtitle=sub,**figargs)
    else: 
        if subtitles is None:
            subtitles = title
        else:
            fig.suptitle(title, fontsize=20)
            subtitles = subtitles[0]
        plot_image(fig,axs,data,subtitle=subtitles,**figargs)
    if show: plt.show()
    else: return fig, axs
