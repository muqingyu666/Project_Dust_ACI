#                  ___====-_  _-====___
#            _--^^^#####//      \\#####^^^--_
#         _-^##########// (    ) \\##########^-_
#        -############//  |\^^/|  \\############-
#      _/############//   (@::@)   \\############\_
#     /#############((     \\//     ))#############\
#    -###############\\    (oo)    //###############-
#   -#################\\  / VV \  //#################-
#  -###################\\/      \//###################-
# _#/|##########/\######(   /\   )######/\##########|\#_
# |/ |#/\#/\#/\/  \#/\##\  |  |  /##/\#/  \/\#/\#/\#| \|
# `  |/  V  V  `   V  \#\| |  | |/#/  V   '  V  V  \|  '
#    `   `  `      `   / | |  | | \   '      '  '   '
#                     (  | |  | |  )
#                    __\ | |  | | /__
#                   (vvv(VVV)(VVV)vvv)
#                       神兽保佑
#                      代码无BUG!

"""

    Code to test if filter data for 10% lowermost and 90% highermost
    can reveal the anormal cirrus signal in 2020.
        
    Owner: Mu Qingyu
    version 1.0
        version 1.1: 2022-11-28
        - we only analyze the filtered data for 10% lowermost and 90% highermost
        - In order to extract maximum signal of contrail, only april to july 
          cld and pc1 data are used
          
    Created: 2022-10-27
    
    Including the following parts:

        1) Read in basic PCA & Cirrus data (include cirrus morphology and microphysics)
                
        2) Filter anormal hcf data within lowermost 10% or highermost 90%
        
        3) Plot the filtered data to verify the anormal cirrus signal
        
        4) Calculate the mean and std of filtered data, cv of filtered data
        
"""

# import modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from muqy_20221026_func_filter_hcf_anormal_data import (
    filter_data_PC1_gap_lowermost_highermost_error as filter_data_PC1_gap_lowermost_highermost_error,
)

# --------- import done ------------
# --------- Plot style -------------
# Set parameter to avoid warning
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.style.use("seaborn-v0_8-ticks")
mpl.rc("font", family="Times New Roman")


# ----------------------------------
# Read in filtered data
def read_filtered_data_out(
    file_name: str = "Cld_match_PC_gap_IWP_AOD_constrain_mean_2010_2020.nc",
):
    Cld_match_PC_gap_IWP_AOD_constrain_mean = xr.open_dataarray(
        "/RAID01/data/Filtered_data/" + file_name
    )

    return Cld_match_PC_gap_IWP_AOD_constrain_mean


# Read the filtered data
# Read the Dust constrain data
# ----------------------------------
# Filtered data with 1% extreme value of IWP and AOD masked
# Read the Dust constrain HCF data
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_HCF = read_filtered_data_out(
    file_name="CERES_SSF_HCF_match_PC_gap_IWP_AOD_constrain_mean_2005_2020_Dust_AOD_mask_1_IWP_AOD_no_antarc_new_PC_IWP_gaps_4_AOD_gaps.nc"
)
# Read the Dust constrain cld icerad data
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR = read_filtered_data_out(
    file_name="CERES_SSF_IPR_match_PC_gap_IWP_AOD_constrain_mean_2005_2020_Dust_AOD_mask_1_IWP_AOD_no_antarc_new_PC_IWP_gaps_4_AOD_gaps.nc"
)
# Read the Dust constrain cld top pressure data
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_CTP = read_filtered_data_out(
    file_name="CERES_SSF_CTP_match_PC_gap_IWP_AOD_constrain_mean_2005_2020_Dust_AOD_mask_1_IWP_AOD_no_antarc_new_PC_IWP_gaps_4_AOD_gaps.nc"
)
# Read the Dust constrain cld optical depth data
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_COD = read_filtered_data_out(
    file_name="CERES_SSF_COT_match_PC_gap_IWP_AOD_constrain_mean_2005_2020_Dust_AOD_mask_1_IWP_AOD_no_antarc_new_PC_IWP_gaps_4_AOD_gaps.nc"
)

# Extract AOD bin, PC1 bin, IWP bin values
AOD_gaps_values = (
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_HCF.AOD_bin.values
)
PC1_gaps_values = (
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_HCF.PC_bin.values
)
IWP_gaps_values = (
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_HCF.IWP_bin.values
)

######################################################################################
###### Plot 3d filled figure to represent the cloud amount in each PC1 interval ######
######################################################################################

# color nan values with self-defined color
def create_colormap_with_nan(
    cmap_name: str, nan_color: str = "silver"
) -> ListedColormap:
    """
    Create a colormap with NaN values colored with a self-defined color.

    Args:
        cmap_name (str): The name of the colormap to use.
        nan_color (str, optional): The color to use for NaN values. Defaults to "silver".

    Returns:
        ListedColormap: A colormap with NaN values colored with the specified color.

    Notes:
        The function creates a colormap with NaN values colored with a self-defined color. It uses the specified colormap name to create a colormap, and then sets the color for NaN values to the specified color. The function returns the resulting colormap.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> cmap = create_colormap_with_nan("viridis", "gray")
        >>> plt.imshow([[1, 2], [3, np.nan]], cmap=cmap)
    """
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.arange(cmap.N))
    cmap_with_nan = ListedColormap(colors)
    cmap_with_nan.set_bad(nan_color)
    return cmap_with_nan


def plot_3d_colored_IWP_PC1_AOD_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    high_cloud_amount_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
    savefig_str,
    aod_range,
    vmin,
    vmax,
    AOD_bin_values,
    PC1_bin_values,
    IWP_bin_values,
    cmap="Spectral_r",
) -> None:
    """
    Create a 3D plot with 2D pcolormesh color fill maps representing high cloud amount for each AOD interval.

    Args:
        Cld_match_PC_gap_IWP_AOD_constrain_mean (numpy.array): Cloud data under each restriction, shape (AOD_bin, IWP_bin, PC_bin, lat, lon)
        high_cloud_amount_mean (numpy.array): Mean high cloud amount for each AOD interval, shape (AOD_bin,)
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        zlabel (str): Label for the z-axis
        colobar_label (str): Label for the color bar
        savefig_str (str): String for saving the figure
        aod_range (tuple): Tuple defining the start and end indices of the AOD cases to plot
        vmin (float): Minimum value for the color scale
        vmax (float): Maximum value for the color scale
        cmap (str, optional): The name of the colormap to use. Defaults to "Spectral_r".

    Returns:
        None: The function displays the plot using matplotlib.pyplot.show().

    Raises:
        ValueError: If the input arrays are not of the expected shape.

    Notes:
        The function creates a 3D plot with 2D pcolormesh color fill maps representing high cloud amount for each AOD interval. It takes as input the cloud data under each restriction, the mean high cloud amount for each AOD interval, the labels for the x, y, and z axes, the label for the color bar, the string for saving the figure, the range of AOD cases to plot, the minimum and maximum values for the color scale, and the name of the colormap to use. The function displays the plot using matplotlib.pyplot.show().

    Examples:
        >>> import numpy as np
        >>> Cld_match_PC_gap_IWP_AOD_constrain_mean = np.random.rand(3, 4, 5, 6, 7)
        >>> high_cloud_amount_mean = np.random.rand(3)
        >>> xlabel = "PC1"
        >>> ylabel = "IWP"
        >>> zlabel = "AOD"
        >>> colobar_label = "High cloud amount"
        >>> savefig_str = "3d_colored_IWP_PC1_AOD_min_max_version.png"
        >>> aod_range = (1, 3)
        >>> vmin = 0.0
        >>> vmax = 1.0
        >>> cmap = "Spectral_r"
        >>> plot_3d_colored_IWP_PC1_AOD_min_max_version(
        ...     Cld_match_PC_gap_IWP_AOD_constrain_mean,
        ...     high_cloud_amount_mean,
        ...     xlabel,
        ...     ylabel,
        ...     zlabel,
        ...     colobar_label,
        ...     savefig_str,
        ...     aod_range,
        ...     vmin,
        ...     vmax,
        ...     cmap
        ... )
    """
    fig = plt.figure(figsize=(16, 14), dpi=400)
    ax = fig.add_subplot(111, projection="3d")

    AOD_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[0]
    IWP_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[1]
    PC_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[2]

    X, Y = np.meshgrid(range(PC_bin), range(IWP_bin))

    cmap_with_nan = create_colormap_with_nan(cmap)

    for aod_num in range(aod_range[0], aod_range[1]):
        Z = aod_num * np.ones_like(X)
        ax.plot_surface(
            Z,
            X,
            Y,
            rstride=1,
            cstride=1,
            facecolors=cmap_with_nan(
                (high_cloud_amount_mean[aod_num] - vmin)
                / (vmax - vmin)
            ),
            shade=False,
            edgecolor="none",
            alpha=0.95,
            antialiased=False,
            linewidth=0,
        )

    # Define the number of ticks you want for y and z axes (for example 5)
    n_ticks_y = 5
    n_ticks_z = 7

    # Define tick positions for each axis

    xticks = np.arange(
        aod_range[0], aod_range[1]
    )  # Only three AOD bins per 3D image
    yticks = np.linspace(0, PC_bin - 1, n_ticks_y, dtype=int)
    zticks = np.linspace(0, IWP_bin - 1, n_ticks_z, dtype=int)

    # Set tick positions
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)

    # Set tick labels using the provided bin values and format them with three decimal places
    ax.set_xticklabels(
        ["{:.3f}".format(val) for val in AOD_bin_values[xticks]]
    )
    ax.set_yticklabels(
        ["{:.3f}".format(val) for val in PC1_bin_values[yticks]]
    )
    ax.set_zticklabels(
        ["{:.1f}".format(val) for val in IWP_bin_values[zticks]]
    )

    # set labels and distance from axis
    ax.set_xlabel(xlabel, labelpad=27, fontsize=16.5)
    ax.set_ylabel(ylabel, labelpad=27, fontsize=16.5)
    ax.set_zlabel(zlabel, labelpad=27, fontsize=16.5)

    # set tick label distance from axis
    ax.tick_params(axis="x", which="major", pad=13, labelsize=14)
    ax.tick_params(axis="y", which="major", pad=13, labelsize=14)
    ax.tick_params(axis="z", which="major", pad=13, labelsize=14)

    ax.grid(False)

    m = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r)
    m.set_cmap(cmap_with_nan)
    m.set_array(high_cloud_amount_mean)
    m.set_clim(vmin, vmax)
    cb = fig.colorbar(
        m, shrink=0.3, aspect=9, pad=0.01, label=colobar_label
    )

    # set colorbar tick label size
    cb.set_label(colobar_label, fontsize=16.5)
    cb.ax.tick_params(labelsize=14)

    ax.view_init(elev=13, azim=-48)
    ax.dist = 12

    plt.savefig(savefig_str)
    plt.show()


def plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
    vmin,
    vmax,
    cmap="Spectral_r",
) -> None:
    """
    Plot two 3D fill plots with custom colormaps and color scales.

    Args:
        Cld_match_PC_gap_IWP_AOD_constrain_mean (numpy.array): Cloud data under each restriction, shape (AOD_bin, IWP_bin, PC_bin, lat, lon)
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        zlabel (str): Label for the z-axis.
        colobar_label (str): Label for the color bar.
        vmin (float): Minimum value for the color scale.
        vmax (float): Maximum value for the color scale.
        cmap (str, optional): The name of the colormap to use. Defaults to "Spectral_r".

    Returns:
        None: The function displays the plot using matplotlib.pyplot.show().

    Raises:
        ValueError: If the input arrays are not of the expected shape.

    Notes:
        The function plots two 3D fill plots with custom colormaps and color scales. It takes as input the cloud data under each restriction, the labels for the x, y, and z axes, the label for the color bar, the minimum and maximum values for the color scale, and the name of the colormap to use. The function displays the plot using matplotlib.pyplot.show().

    Examples:
        >>> import numpy as np
        >>> Cld_match_PC_gap_IWP_AOD_constrain_mean = np.random.rand(3, 4, 5, 6, 7)
        >>> xlabel = "PC1"
        >>> ylabel = "IWP"
        >>> zlabel = "AOD"
        >>> colobar_label = "High cloud amount"
        >>> vmin = 0.0
        >>> vmax = 1.0
        >>> cmap = "Spectral_r"
        >>> plot_both_3d_fill_plot_min_max_version(
        ...     Cld_match_PC_gap_IWP_AOD_constrain_mean,
        ...     xlabel,
        ...     ylabel,
        ...     zlabel,
        ...     colobar_label,
        ...     vmin,
        ...     vmax,
        ...     cmap
        ... )
    """  # Calculate the mean high cloud amount for each AOD interval, IWP, and PC1 bin
    high_cloud_amount_mean = np.nanmean(
        Cld_match_PC_gap_IWP_AOD_constrain_mean, axis=(3, 4)
    )

    plot_3d_colored_IWP_PC1_AOD_min_max_version(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        high_cloud_amount_mean,
        xlabel,
        ylabel,
        zlabel,
        colobar_label,
        "subplot_1.png",
        (0, 2),
        vmin,  # Add this parameter to define the minimum value for the color scale
        vmax,  # Add this parameter to define the maximum value for the color scale
        AOD_bin_values=Cld_match_PC_gap_IWP_AOD_constrain_mean.AOD_bin.values,
        PC1_bin_values=Cld_match_PC_gap_IWP_AOD_constrain_mean.PC_bin.values,
        IWP_bin_values=Cld_match_PC_gap_IWP_AOD_constrain_mean.IWP_bin.values,
        cmap=cmap,  # Add this parameter to define the custom colormap
    )
    plot_3d_colored_IWP_PC1_AOD_min_max_version(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        high_cloud_amount_mean,
        xlabel,
        ylabel,
        zlabel,
        colobar_label,
        "subplot_2.png",
        (2, 4),
        vmin,  # Add this parameter to define the minimum value for the color scale
        vmax,  # Add this parameter to define the maximum value for the color scale
        AOD_bin_values=Cld_match_PC_gap_IWP_AOD_constrain_mean.AOD_bin.values,
        PC1_bin_values=Cld_match_PC_gap_IWP_AOD_constrain_mean.PC_bin.values,
        IWP_bin_values=Cld_match_PC_gap_IWP_AOD_constrain_mean.IWP_bin.values,
        cmap=cmap,  # Add this parameter to define the custom colormap
    )


# Call the function with different AOD ranges and save each figure separately
# -----------------------------------------------
# Plot the dust-AOD constrained data
# high cloud area
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_HCF,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CERES_SSF Cloud Area Fraction",
    vmin=0,
    vmax=59,
)

# ice effective radius
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CERES_SSF Ice Particle Radius (microns)",
    vmin=13,
    vmax=38,
)

# cloud top pressure
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_CTP,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CERES_SSF Cloud Top Pressure (hPa)",
    vmin=170,
    vmax=296,
)

# cloud optical depth
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_COD,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CERES_SSF Cloud Optical Depth",
    vmin=0,
    vmax=15,
)

# -----------------------------------------------------------------------
# Plot diverse IWP regions, mask out the IWP regions we dont need
# -----------------------------------------------------------------------

# Low IWP region
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_low_IWP = CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR.where(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR["IWP_bin"]
    <= 1,
    drop=True,
)

# Mid IWP region
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_mid_IWP = CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR.where(
    (
        CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR[
            "IWP_bin"
        ]
        > 1
    )
    & (
        CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR[
            "IWP_bin"
        ]
        < 80
    ),
    drop=True,
)

# High IWP region
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_high_IWP = CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR.where(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR["IWP_bin"]
    >= 80,
    drop=True,
)

# Plot low IWP region 3D constrain layout
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_low_IWP,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CERES_SSF Cloud Effective Radius (microns)",
    vmin=13,
    vmax=38,
)

# Plot mid IWP region 3D constrain layout
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_mid_IWP,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CERES_SSF Cloud Effective Radius (microns)",
    vmin=13,
    vmax=38,
)

# Plot high IWP region 3D constrain layout
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_high_IWP,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CERES_SSF Cloud Effective Radius (microns)",
    vmin=13,
    vmax=38,
)


# -----------------------------------------------------------------------
# Plot spatial distribution of different AOD gap values
# -----------------------------------------------------------------------
# Divide 3, Dust AOD data
# Divide AOD data as well


def plot_spatial_distribution(
    data, var_name, title, AOD_intervals, cmap="jet"
):
    """
    Plots a spatial distribution of a given variable using a custom colormap.

    Args:
        data (numpy.ndarray): The data to plot.
        var_name (str): The name of the variable being plotted.
        title (str): The title of the plot.
        AOD_intervals (list): A list of intervals to use for the colormap.
        cmap (str, optional): The name of the colormap to use. Defaults to "jet".

    Returns:
        None
    """
    from matplotlib.colors import BoundaryNorm

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    from matplotlib.colors import LinearSegmentedColormap

    # Define colors for each interval (use as many colors as the number of intervals)
    colors = ["green", "blue", "yellow", "orange", "red", "purple"]

    # Create custom colormap
    cmap = plt.cm.get_cmap(cmap, len(AOD_intervals) - 1)

    # Set up the norm and boundaries for the colormap
    norm = BoundaryNorm(AOD_intervals, cmap.N)

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12, 8),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax1.set_facecolor("silver")
    b = ax1.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
    )
    ax1.coastlines(resolution="50m", lw=0.9)
    ax1.set_title(title, fontsize=24)

    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.top_labels = False
    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.65,
        extend="both",
    )
    cb2.set_label(label=var_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


# Read the data
(
    # pc
    PC_all,
    PC_years,
    # cld
    Cld_all,
    Cld_years,
    # iwp
    IWP_data,
    IWP_years,
) = read_PC1_CERES_from_netcdf(CERES_Cld_dataset_name="Cldicerad")

# CERES_Cld_dataset = [
#     "Cldarea",
#     "Cldicerad",
#     "Cldeff_hgth",
#     "Cldpress_base",
#     "Cldhgth_top",
#     "Cldtau",
#     "Cldtau_lin",
#     "IWP",
#     "Cldemissir",
# ]

# AOD file
data_merra2_2010_2020_new_lon = xr.open_dataset(
    "/RAID01/data/merra2/merra_2_daily_2010_2020_new_lon.nc"
)

# Plot the distribution gap for Dust AOD
AOD_temp = data_merra2_2010_2020_new_lon["DUEXTTAU"].values

divide_AOD = DividePCByDataVolume(
    dataarray_main=AOD_temp,
    n=6,
)
AOD_gap = divide_AOD.main_gap()

Dust_2010_2020_mean = np.nanmean(AOD_temp, axis=0)

plot_spatial_distribution(
    data=Dust_2010_2020_mean,
    var_name="Dust AOD",
    title="Spatial Distribution of Dust AOD Section",
    AOD_intervals=AOD_gap,
    cmap="RdYlBu_r",
)

# Plot the distribution gap for IWP
PC_temp = PC_all

divide_PC = DividePCByDataVolume(
    dataarray_main=PC_temp,
    n=30,
)
PC_gap = divide_PC.main_gap()

PC_mean = np.nanmean(PC_temp, axis=0)

plot_spatial_distribution(
    data=PC_mean,
    var_name="PC1",
    title="Spatial Distribution of PC1 Section",
    AOD_intervals=PC_gap,
    cmap="RdYlBu_r",
)

# Plot the distribution gap for PC1
IWP_temp = IWP_data

divide_IWP = DividePCByDataVolume(
    dataarray_main=IWP_temp,
    n=30,
)
IWP_gap = divide_IWP.main_gap()

IWP_mean = np.nanmean(IWP_temp, axis=0)

plot_spatial_distribution(
    data=IWP_mean,
    var_name="IWP",
    title="Spatial Distribution of IWP Section",
    AOD_intervals=IWP_gap,
    cmap="RdYlBu_r",
)


# -----------------------------------------------------------------------
#  Plot spatial distribution of global vars distribution
# -----------------------------------------------------------------------


def plot_global_spatial_distribution(
    data,
    var_name,
    title,
):
    """
    Plots the spatial distribution of a given variable on a global map.

    Args:
    - data (numpy.ndarray): The data to be plotted.
    - var_name (str): The name of the variable being plotted.
    - title (str): The title of the plot.

    Returns:
    - None
    """
    from matplotlib.colors import BoundaryNorm

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    from matplotlib.colors import LinearSegmentedColormap

    # Create custom colormap
    cmap = plt.cm.get_cmap("RdBu_r")

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(12, 8),
        constrained_layout=True,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    ax1 = plt.subplot(
        111,
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax1.set_facecolor("silver")
    b = ax1.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
    )
    ax1.coastlines(resolution="50m", lw=0.9)
    ax1.set_title(title, fontsize=24)

    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.top_labels = False
    cb2 = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.65,
        extend="both",
    )
    cb2.set_label(label=var_name, size=24)
    cb2.ax.tick_params(labelsize=24)

    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}


# verify the input data
plot_global_spatial_distribution(
    IWP_mean,
    "IWP",
    "IWP Spatial Distribution",
)

