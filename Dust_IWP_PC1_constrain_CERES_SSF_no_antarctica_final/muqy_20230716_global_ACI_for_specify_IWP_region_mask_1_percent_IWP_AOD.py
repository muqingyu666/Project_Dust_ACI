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
from typing import Dict, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from scipy import stats

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
# ----------------------------------
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
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_CEP = read_filtered_data_out(
    file_name="CERES_SSF_CEP_match_PC_gap_IWP_AOD_constrain_mean_2005_2020_Dust_AOD_mask_1_IWP_AOD_no_antarc_new_PC_IWP_gaps_4_AOD_gaps.nc"
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

# ------------------------------ #
# lets see the PC and IWP gaps  #


# region

# --------------------------------------------------------------- #
# now we read IWP and other cld data (not IWP) from netcdf file
(
    PC_data_og,
    _,
) = read_PC1_CERES_clean(
    PC_path="/RAID01/data/PC_data/1990_2020_250_hPa_vars_250_300_Instab_PC1_no_antarc.nc",
    CERES_Cld_dataset_name="Cldicerad",
)

# --------------------------------------------------------------- #
# ------ Read in CERES SSF data and filter anormal data --------- #
# --------------------------------------------------------------- #
# read in CERES SSF data

# Specify the output path
CERES_SSF_28day = xr.open_dataset(
    "/RAID01/data/Cld_data/CERES_SSF_data_2005_2020_28_days.nc"
)

CERES_SSF_HCF = CERES_SSF_28day[
    "cldarea_high_daynight_daily"
].values.reshape(-1, 180, 360)
CERES_SSF_ice_HCF = CERES_SSF_28day[
    "cldarea_ice_high_daynight_daily"
].values.reshape(-1, 180, 360)

CERES_SSF_IPR_12 = CERES_SSF_28day[
    "cldicerad12_high_daynight_daily"
].values.reshape(-1, 180, 360)
CERES_SSF_IPR_21 = CERES_SSF_28day[
    "cldicerad21_high_daynight_daily"
].values.reshape(-1, 180, 360)
CERES_SSF_IPR_37 = CERES_SSF_28day[
    "cldicerad37_high_daynight_daily"
].values.reshape(-1, 180, 360)

CERES_SSF_IWP = CERES_SSF_28day[
    "iwp37_high_daynight_daily"
].values.reshape(-1, 180, 360)
CERES_SSF_CTP = CERES_SSF_28day[
    "cldpress_top_high_daynight_daily"
].values.reshape(-1, 180, 360)
CERES_SSF_CEP = CERES_SSF_28day[
    "cldpress_eff_high_daynight_daily"
].values.reshape(-1, 180, 360)
CERES_SSF_COT = CERES_SSF_28day[
    "cldtau_high_daynight_daily"
].values.reshape(-1, 180, 360)

# use the 2010-2020 PC1 only
PC_data = PC_data_og.reshape(31, 12, 28, 150, 360)[
    -16:, :, :, :, :
]
PC_data = PC_data.reshape(-1, 150, 360)

# use the 2010-2020 PC1 only
PC_data = PC_data.astype(np.float32).reshape(-1, 150, 360)
HCF_data = CERES_SSF_HCF.astype(np.float32).reshape(-1, 180, 360)
HCF_ice_data = CERES_SSF_ice_HCF.astype(np.float32).reshape(
    -1, 180, 360
)
IWP_data = CERES_SSF_IWP.astype(np.float32).reshape(-1, 180, 360)
IPR_data = CERES_SSF_IPR_37.astype(np.float32).reshape(
    -1, 180, 360
)
CTP_data = CERES_SSF_CTP.astype(np.float32).reshape(-1, 180, 360)
COD_data = CERES_SSF_COT.astype(np.float32).reshape(-1, 180, 360)
CEP_data = CERES_SSF_CEP.astype(np.float32).reshape(-1, 180, 360)

# --------------------------------------------------------------- #
# ------------------ Read the MERRA2 dust data ------------------ #
# --------------------------------------------------------------- #
# Implementation for MERRA2 dust AOD
# extract the data from 2010 to 2014 like above
data_merra2_2010_2020_new_lon = xr.open_dataset(
    "/RAID01/data/merra2/merra2_2005_2020_new_lon.nc"
)

# # Extract Dust aerosol data from all data
Dust_AOD = data_merra2_2010_2020_new_lon[
    "DUEXTTAU"
].values.reshape(5376, 180, 360)

# convert the data type to float32
Dust_AOD = Dust_AOD.astype(np.float32)


# Set the largest and smallest 5% of the data to nan
def set_extreme_percent_to_nan(
    threshold: float, data_array: np.ndarray
) -> Union[np.ndarray, None]:
    """
    Set the largest and smallest percentage of the data to NaN.

    Args:
        threshold (float):
            The percentage of the data to set to NaN.
        data_array (numpy.ndarray):
            The input data array.

    Returns:
        numpy.ndarray or None: A copy of the input data array with the largest
        and smallest percentage of the data set to NaN, or None if the
        input data array is empty.

    Raises:
        None.

    Notes:
        The function sets the largest and smallest percentage of the data to NaN.
        It takes as input the percentage of the data to set to NaN and the input data array.
        The function returns a copy of the input data array with the largest
        and smallest percentage of the data set to NaN. If the input data array is empty,
        the function returns None.

    Examples:
        >>> import numpy as np
        >>> data_array = np.random.rand(10, 10)
        >>> threshold = 5.0
        >>> data_array_filtered = set_extreme_percent_to_nan(threshold, data_array)
    """
    # Make a copy of the data array
    data_array_filtered = np.copy(data_array)

    # Calculate the threshold values for the largest and smallest 5%
    try:
        lower_threshold = np.nanpercentile(data_array, threshold)
    except IndexError:
        print("ERROR: Data array is empty.")
        return None
    else:
        upper_threshold = np.nanpercentile(
            data_array, 100 - threshold
        )

    # Set the largest and smallest 5% of the data array to nan
    data_array_filtered[
        data_array_filtered < lower_threshold
    ] = np.nan
    data_array_filtered[
        data_array_filtered > upper_threshold
    ] = np.nan

    return data_array_filtered


# Filter out extreme 2.5% AOD data to avoid the influence of dust origin
Dust_AOD_filtered = set_extreme_percent_to_nan(
    threshold=1, data_array=Dust_AOD
)

# Filter out extreme 1% IWP data to avoid extreme large IWP values
IWP_data_filtered = set_extreme_percent_to_nan(
    threshold=1, data_array=IWP_data
)

# delete antarctica region
Dust_AOD_filtered = Dust_AOD_filtered[:, 30:, :]
IWP_data_filtered = IWP_data_filtered[:, 30:, :]
HCF_data = HCF_data[:, 30:, :]
IPR_data = IPR_data[:, 30:, :]
IWP_data = IWP_data[:, 30:, :]
CTP_data = CTP_data[:, 30:, :]
COD_data = COD_data[:, 30:, :]
CEP_data = CEP_data[:, 30:, :]

# endregion


class DivideDataGivingGapByVolumeToNewGap:
    def __init__(self, data, n):
        self.data = data.flatten()
        self.n = n

    def calculate_gap(self, start, end):
        # Mask data outside the interval
        mask = (self.data >= start) & (self.data <= end)
        data_in_interval = self.data[mask]

        # Calculate the gaps in this interval
        gaps = np.percentile(
            data_in_interval, np.linspace(0, 100, self.n + 1)
        )
        return gaps

    def divide_intervals(self, intervals):
        all_gaps = []
        for i in range(len(intervals) - 1):
            start, end = intervals[i], intervals[i + 1]
            gaps = self.calculate_gap(start, end)
            all_gaps.extend(
                gaps[:-1]
            )  # Exclude the last boundary to avoid duplicates

        # Add the last boundary manually
        all_gaps.append(intervals[-1])

        return np.array(all_gaps)


# Use the class like this:
divisor_IWP = DivideDataGivingGapByVolumeToNewGap(
    data=IWP_data, n=5
)  # Modify 'n' as needed
IWP_gap = np.array([0.5, 1, 5, 20, 50, 100, 500])
IWP_new_intervals = divisor_IWP.divide_intervals(IWP_gap)

divisor_PC = DivideDataGivingGapByVolumeToNewGap(
    data=PC_data, n=5
)  # Modify 'n' as needed
PC_gap = np.arange(-3, 7, 1.5)
PC_new_intervals = divisor_PC.divide_intervals(PC_gap)

divide_AOD = DividePCByDataVolume(
    dataarray_main=Dust_AOD_filtered,
    n=4,
)
AOD_gap = divide_AOD.main_gap()

# --------------------------------------------------
# plotting functions
# color nan values with self-defined color
# --------------------------------------------------


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
    ax,
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    xlabel,
    ylabel,
    zlabel,
    high_cloud_amount_mean,
    aod_range,
    vmin,
    vmax,
    AOD_bin_values,
    PC1_bin_values,
    IWP_bin_values,
    cmap="Spectral_r",
    add_colorbar=False,
    fig=None,
    colobar_label=None,
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
    # fig = plt.figure(figsize=(16, 14), dpi=400)
    # ax = fig.add_subplot(111, projection="3d")

    AOD_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[0]
    IWP_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[1]
    PC_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[2]

    X, Y = np.meshgrid(range(PC_bin), range(IWP_bin))

    cmap_with_nan = create_colormap_with_nan(cmap)

    # Define the number of ticks you want for y and z axes (for example 5)
    x_shift = -0.5  # Half of the distance between two xticks.

    for aod_num in range(aod_range[0], aod_range[1]):
        Z = (
            aod_num * np.ones_like(X) - x_shift
        )  # Shift the coloring chart to the left by half of the distance between two xticks.
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
    n_ticks_x = 3
    n_ticks_y = 4
    n_ticks_z = 4

    # Define tick positions for each axis
    if aod_range == (0, 2):  # Case for the first subplot
        xticks = np.array(
            [aod_range[0], aod_range[0] + 1, aod_range[1]],
            dtype=int,
        )
    else:  # Case for the second subplot
        xticks = np.array(
            [aod_range[0], aod_range[0] + 1, aod_range[1]],
            dtype=int,
        )

    # xticks = np.linspace(0, AOD_bin - 1, n_ticks_x, dtype=int)
    yticks = np.linspace(0, PC_bin, n_ticks_y, dtype=int)
    zticks = np.linspace(0, IWP_bin, n_ticks_z, dtype=int)

    # Set tick positions
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)

    # Set tick labels using the provided bin values and format them with three decimal places
    ax.set_xticklabels(
        ["{:.3f}".format(val) for val in AOD_bin_values[xticks]]
    )
    ax.set_yticklabels(
        ["{:.1f}".format(val) for val in PC1_bin_values[yticks]]
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

    if add_colorbar:  # only add colorbar if add_colorbar is True
        cbar_ax = fig.add_axes([0.98, 0.39, 0.015, 0.23])
        cb = fig.colorbar(
            m,
            cax=cbar_ax,
            shrink=0.3,
            aspect=9,
            label=colobar_label,
        )
        # set colorbar tick label size
        cb.set_label(colobar_label, fontsize=16.5)
        cb.ax.tick_params(labelsize=14)

    ax.view_init(elev=12, azim=-62)
    ax.dist = 12
    ax.set_facecolor("none")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(
        True,
        color="silver",
        linestyle="-.",
        linewidth=0.3,
        alpha=0.4,
    )

    # if aod_range == (0, 2) or aod_range == (2, 4):  # Case for the first subplot
    if aod_range == (0, 2):  # Case for the first subplot
        # ax.set_zticklabels([])
        ax.set_zlabel("")
        # ax.set_yticklabels([])
        ax.set_ylabel("")
    else:  # Case for the second subplot
        pass


def plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
    vmin,
    vmax,
    vmin_diff,
    vmax_diff,
    cmap="Spectral_r",
    cmap_diff="RdBu_r",
    label_text1="(A)",
    label_text2="(B)",
    label_text3="(C)",
    label_pos=(0.08, 0.6, 0.53, 0.6),
    # label_pos=(0.08, 0.6, 0.53, 0.6, 0.78, 0.6),
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
    """
    # Calculate the mean high cloud amount for each AOD interval, IWP, and PC1 bin
    high_cloud_amount_mean = np.nanmean(
        Cld_match_PC_gap_IWP_AOD_constrain_mean, axis=(3, 4)
    )
    # Calculate the difference between the first and second AOD intervals
    high_cloud_amount_diff = np.diff(
        high_cloud_amount_mean, axis=0
    )

    fig = plt.figure(figsize=(15, 15), dpi=400)
    plt.subplots_adjust(
        wspace=0.005, left=0.05, right=0.95, top=0.95, bottom=0.05
    )  # This is the space between the subplots

    # Create the first subplot
    ax1 = fig.add_subplot(121, projection="3d")
    plot_3d_colored_IWP_PC1_AOD_min_max_version(
        ax1,
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        xlabel,
        ylabel,
        zlabel,
        high_cloud_amount_mean,
        (0, 2),
        vmin,
        vmax,
        AOD_bin_values=AOD_gap,
        PC1_bin_values=PC_new_intervals,
        IWP_bin_values=IWP_new_intervals,
        cmap=cmap,
    )

    # Create the second subplot
    ax2 = fig.add_subplot(122, projection="3d")
    plot_3d_colored_IWP_PC1_AOD_min_max_version(
        ax2,
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        xlabel,
        ylabel,
        zlabel,
        high_cloud_amount_mean,
        (2, 4),
        vmin,
        vmax,
        AOD_bin_values=AOD_gap,
        PC1_bin_values=PC_new_intervals,
        IWP_bin_values=IWP_new_intervals,
        cmap=cmap,
        add_colorbar=True,
        fig=fig,
        colobar_label=colobar_label,
    )

    # Create the third subplot
    # ax3 = fig.add_subplot(133, projection="3d")
    # plot_3d_colored_IWP_PC1_AOD_min_max_version(
    #     ax3,
    #     Cld_match_PC_gap_IWP_AOD_constrain_mean,
    #     xlabel,
    #     ylabel,
    #     zlabel,
    #     high_cloud_amount_diff,
    #     (0, 3),  # three difference maps
    #     vmin_diff,
    #     vmax_diff,
    #     AOD_bin_values=AOD_gap,
    #     PC1_bin_values=PC_new_intervals,
    #     IWP_bin_values=IWP_new_intervals,
    #     cmap=cmap_diff,
    #     add_colorbar=True,
    #     fig=fig,
    #     colobar_label=colobar_label,
    # )

    if label_text1:
        fig.text(
            label_pos[0],
            label_pos[1],
            label_text1,
            transform=fig.transFigure,
            fontsize=17,
            va="top",
        )
    if label_text2:
        fig.text(
            label_pos[2],
            label_pos[3],
            label_text2,
            transform=fig.transFigure,
            fontsize=17,
            va="top",
        )
    # Add the label for the third subplot
    # if label_text3:
    #     fig.text(
    #         label_pos[4],
    #         label_pos[5],
    #         label_text3,
    #         transform=fig.transFigure,
    #         fontsize=17,
    #         va="top",
    #     )

    plt.savefig(
        "/RAID01/data/python_fig/subplot_combined"
        + colobar_label
        + ".png",
        dpi=400,
        bbox_inches="tight",
        facecolor="w",
    )
    plt.show()


# Call the function with different AOD ranges and save each figure separately
# -----------------------------------------------
# Plot the dust-AOD constrained data
# high cloud area
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_HCF,
    "Dust-AOD",
    "PC1",
    "IWP (g/m" + r"$^2$" + ")",  # IWP (g/m^2)
    "Cloud Area Fraction (%)",
    vmin=0,
    vmax=82,
    vmin_diff=-10,
    vmax_diff=10,
    label_text1="(A)",
    label_text2="(B)",
    label_text3="(C)",
)

# ice effective radius
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR,
    "Dust-AOD",
    "PC1",
    "IWP (g/m" + r"$^2$" + ")",  # IWP (g/m^2)
    "Ice Particle Radius (" + r"$\mu$" + "m)",
    vmin=12,
    vmax=32,
    vmin_diff=-10,
    vmax_diff=10,
    label_text1="(C)",
    label_text2="(D)",
    label_text3="(F)",
)

# cloud top pressure
# plot_both_3d_fill_plot_min_max_version(
#     CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_CTP,
#     "Dust-AOD",
#     "PC1",
#     "IWP (g/m" + r"$^2$" + ")",  # IWP (g/m^2)
#     "Cloud Top Pressure (hPa)",
#     vmin=185,
#     vmax=265,
#     vmin_diff=-10,
#     vmax_diff=10,
#     label_text1="(E)",
#     label_text2="(F)",
#     label_text3="(I)",
# )

# cloud effective pressure
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_CEP,
    "Dust-AOD",
    "PC1",
    "IWP (g/m" + r"$^2$" + ")",  # IWP (g/m^2)
    "Cloud Effective Pressure (hPa)",
    vmin=187.5,
    vmax=276,
    vmin_diff=-10,
    vmax_diff=10,
    label_text1="(E)",
    label_text2="(F)",
    label_text3="(I)",
)

# cloud optical depth
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_COD,
    "Dust-AOD",
    "PC1",
    "IWP (g/m" + r"$^2$" + ")",  # IWP (g/m^2)
    "Cloud Visible Optical Depth",
    vmin=0,
    vmax=12.8,
    vmin_diff=-10,
    vmax_diff=10,
    label_text1="(G)",
    label_text2="(H)",
    label_text3="(L)",
)

# --------------------------------------------------
# Plot the data point counts for each AOD interval
# --------------------------------------------------


def plot_3d_colored_IWP_PC1_AOD_count_min_max_version(
    ax,
    Cld_match_PC_gap_IWP_AOD_constrain_data,
    xlabel,
    ylabel,
    zlabel,
    aod_range,
    vmin,
    vmax,
    AOD_bin_values,
    PC1_bin_values,
    IWP_bin_values,
    cmap="Spectral_r",
    add_colorbar=False,
    fig=None,
    colobar_label=None,
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

    # Assume the shape of Cld_match_PC_gap_IWP_AOD_constrain_data is (AOD_bin, IWP_bin, PC_bin, lat, lon)
    AOD_bin = Cld_match_PC_gap_IWP_AOD_constrain_data.shape[0]
    IWP_bin = Cld_match_PC_gap_IWP_AOD_constrain_data.shape[1]
    PC_bin = Cld_match_PC_gap_IWP_AOD_constrain_data.shape[2]

    X, Y = np.meshgrid(range(PC_bin), range(IWP_bin))

    cmap_with_nan = create_colormap_with_nan(cmap)

    # Create empty list to store counts
    histogram_list = []

    for aod_num in range(AOD_bin):
        for iwp_num in range(IWP_bin):
            for pc_num in range(PC_bin):
                # Count non-NaN values in lat-lon space for each AOD_bin, IWP_bin, PC_bin
                count = np.count_nonzero(
                    ~np.isnan(
                        Cld_match_PC_gap_IWP_AOD_constrain_data[
                            aod_num, iwp_num, pc_num, :, :
                        ]
                    )
                )
                histogram_list.append(
                    [aod_num, iwp_num, pc_num, count]
                )

    # Convert list to numpy array
    histogram = np.array(histogram_list)

    x_shift = -0.5  # Half of the distance between two xticks.

    for aod_num in range(aod_range[0], aod_range[1]):
        Z = aod_num * np.ones_like(X) - x_shift
        ax.plot_surface(
            Z,
            X,
            Y,
            rstride=1,
            cstride=1,
            facecolors=cmap_with_nan(
                (
                    histogram[
                        histogram[:, 0] == aod_num, 3
                    ].reshape(IWP_bin, PC_bin)
                    - vmin
                )
                / (vmax - vmin)
            ),
            shade=False,
            edgecolor="none",
            alpha=0.95,
            antialiased=False,
            linewidth=0,
        )

    # Define the number of ticks you want for y and z axes (for example 5)
    n_ticks_x = 3
    n_ticks_y = 4
    n_ticks_z = 4

    # Define tick positions for each axis
    if aod_range == (0, 2):  # Case for the first subplot
        xticks = np.array(
            [aod_range[0], aod_range[0] + 1, aod_range[1]],
            dtype=int,
        )
    else:  # Case for the second subplot
        xticks = np.array(
            [aod_range[0], aod_range[0] + 1, aod_range[1]],
            dtype=int,
        )

    # xticks = np.linspace(0, AOD_bin - 1, n_ticks_x, dtype=int)
    yticks = np.linspace(0, PC_bin, n_ticks_y, dtype=int)
    zticks = np.linspace(0, IWP_bin, n_ticks_z, dtype=int)

    # Set tick positions
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)

    # Set tick labels using the provided bin values and format them with three decimal places
    ax.set_xticklabels(
        ["{:.3f}".format(val) for val in AOD_bin_values[xticks]]
    )
    ax.set_yticklabels(
        ["{:.1f}".format(val) for val in PC1_bin_values[yticks]]
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
    m.set_array(histogram[:, 3])  # This line has changed
    m.set_clim(vmin, vmax)

    if add_colorbar:  # only add colorbar if add_colorbar is True
        cbar_ax = fig.add_axes([0.98, 0.39, 0.015, 0.23])
        cb = fig.colorbar(
            m,
            cax=cbar_ax,
            shrink=0.3,
            aspect=9,
            label=colobar_label,
        )
        # set colorbar tick label size
        cb.set_label(colobar_label, fontsize=16.5)
        cb.ax.tick_params(labelsize=14)

    ax.view_init(elev=12, azim=-62)
    ax.dist = 12
    ax.set_facecolor("none")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(
        True,
        color="silver",
        linestyle="-.",
        linewidth=0.3,
        alpha=0.4,
    )

    # if aod_range == (0, 2) or aod_range == (2, 4):  # Case for the first subplot
    if aod_range == (0, 2):  # Case for the first subplot
        # ax.set_zticklabels([])
        ax.set_zlabel("")
        # ax.set_yticklabels([])
        ax.set_ylabel("")
    else:  # Case for the second subplot
        pass


def plot_both_3d_fill_count_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
    vmin,
    vmax,
    cmap="Spectral_r",
    label_text1="(A)",
    label_text2="(B)",
    label_pos=(0.08, 0.6, 0.53, 0.6),
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
    """

    fig = plt.figure(figsize=(15, 15), dpi=400)
    plt.subplots_adjust(
        wspace=0.005, left=0.05, right=0.95, top=0.95, bottom=0.05
    )  # This is the space between the subplots

    # Create the first subplot
    ax1 = fig.add_subplot(121, projection="3d")
    plot_3d_colored_IWP_PC1_AOD_count_min_max_version(
        ax1,
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        xlabel,
        ylabel,
        zlabel,
        (0, 2),
        vmin,
        vmax,
        AOD_bin_values=AOD_gap,
        PC1_bin_values=PC_new_intervals,
        IWP_bin_values=IWP_new_intervals,
        cmap=cmap,
    )

    # Create the second subplot
    ax2 = fig.add_subplot(122, projection="3d")
    plot_3d_colored_IWP_PC1_AOD_count_min_max_version(
        ax2,
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        xlabel,
        ylabel,
        zlabel,
        (2, 4),
        vmin,
        vmax,
        AOD_bin_values=AOD_gap,
        PC1_bin_values=PC_new_intervals,
        IWP_bin_values=IWP_new_intervals,
        cmap=cmap,
        add_colorbar=True,
        fig=fig,
        colobar_label=colobar_label,
    )

    # Text labels for the subplots
    if label_text1:
        fig.text(
            label_pos[0],
            label_pos[1],
            label_text1,
            transform=fig.transFigure,
            fontsize=17,
            va="top",
        )
    if label_text2:
        fig.text(
            label_pos[2],
            label_pos[3],
            label_text2,
            transform=fig.transFigure,
            fontsize=17,
            va="top",
        )

    plt.savefig(
        "/RAID01/data/python_fig/subplot_combined"
        + colobar_label
        + ".png",
        dpi=400,
        bbox_inches="tight",
        facecolor="w",
    )
    plt.show()


# Call the function with different AOD ranges and save each figure separately
# -----------------------------------------------
# Plot the dust-AOD constrained data
# high cloud area
plot_both_3d_fill_count_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_HCF,
    "Dust-AOD",
    "PC1",
    "IWP (g/m" + r"$^2$" + ")",  # IWP (g/m^2)
    "Data Point Counts",
    vmin=0,
    vmax=48500,
    cmap="RdYlBu_r",
    label_text1="(A)",
)

# --------------------------------------------------
# plotting functions
# plot delta values between each AOD interval
# --------------------------------------------------


def plot_3d_colored_IWP_PC1_AOD_min_max_version(
    ax,
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    xlabel,
    ylabel,
    zlabel,
    high_cloud_amount_mean,
    aod_range,
    vmin,
    vmax,
    AOD_bin_values,
    PC1_bin_values,
    IWP_bin_values,
    cmap="Spectral_r",
    add_colorbar=False,
    fig=None,
    cbar_ax=None,
    colobar_label=None,
    subplot_num=0,
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
    # fig = plt.figure(figsize=(16, 14), dpi=400)
    # ax = fig.add_subplot(111, projection="3d")

    AOD_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[0]
    IWP_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[1]
    PC_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[2]

    X, Y = np.meshgrid(range(PC_bin), range(IWP_bin))

    cmap_with_nan = create_colormap_with_nan(cmap)

    for aod_num in range(aod_range[0], aod_range[1]):
        Z = aod_num * np.ones_like(
            X
        )  # Shift the coloring chart to the left by half of the distance between two xticks.
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
    n_ticks_x = 3
    n_ticks_y = 4
    n_ticks_z = 4

    # Define tick positions for each axis
    xticks = np.array(
        [0, 1, 2],
        dtype=int,
    )

    # xticks = np.linspace(0, AOD_bin - 1, n_ticks_x, dtype=int)
    yticks = np.linspace(0, PC_bin, n_ticks_y, dtype=int)
    zticks = np.linspace(0, IWP_bin, n_ticks_z, dtype=int)

    # Set tick positions
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)

    # Set tick labels using the provided bin values and format them with three decimal places
    ax.set_xticklabels(
        ["{:.3f}".format(val) for val in AOD_bin_values[xticks]]
    )
    ax.set_yticklabels(
        ["{:.1f}".format(val) for val in PC1_bin_values[yticks]]
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

    if add_colorbar:  # only add colorbar if add_colorbar is True
        cbar_ax = fig.add_axes([0.525, 0.39, 0.015, 0.23])
        cb = fig.colorbar(
            m,
            cax=cbar_ax,
            shrink=0.3,
            aspect=9,
            label=colobar_label,
        )
        # set colorbar tick label size
        cb.set_label(colobar_label, fontsize=16.5)
        cb.ax.tick_params(labelsize=14)

    cbar_ax = fig.add_axes([0.98, 0.39, 0.015, 0.23])
    cb = fig.colorbar(
        m,
        cax=cbar_ax,
        shrink=0.3,
        aspect=9,
        label=colobar_label,
    )
    # set colorbar tick label size
    cb.set_label(colobar_label, fontsize=16.5)
    cb.ax.tick_params(labelsize=14)

    ax.view_init(elev=12, azim=-62)
    ax.dist = 12
    ax.set_facecolor("none")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(
        True,
        color="silver",
        linestyle="-.",
        linewidth=0.3,
        alpha=0.4,
    )

    if aod_range == (0, 2) or aod_range == (
        2,
        4,
    ):  # Case for the first subplot
        # ax.set_zticklabels([])
        ax.set_zlabel("")
        # ax.set_yticklabels([])
        ax.set_ylabel("")
    else:  # Case for the second subplot
        pass


def plot_both_3d_fill_plot_min_max_version(
    Cld_match_PC_gap_IWP_AOD_constrain_mean,
    xlabel,
    ylabel,
    zlabel,
    colobar_label,
    vmin,
    vmax,
    vmin_diff,
    vmax_diff,
    cmap="Spectral_r",
    cmap_diff="RdBu_r",
    label_text1="(A)",
    label_pos=(0.08, 0.6),
    # label_pos=(0.08, 0.6, 0.53, 0.6, 0.78, 0.6),
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
    """
    # Calculate the mean high cloud amount for each AOD interval, IWP, and PC1 bin
    high_cloud_amount_mean = np.nanmean(
        Cld_match_PC_gap_IWP_AOD_constrain_mean, axis=(3, 4)
    )
    # Calculate the difference between the first and second AOD intervals
    high_cloud_amount_diff = np.diff(
        high_cloud_amount_mean, axis=0
    )

    fig = plt.figure(figsize=(15, 15), dpi=400)
    plt.subplots_adjust(
        wspace=0.005, left=0.05, right=0.95, top=0.95, bottom=0.05
    )  # This is the space between the subplots

    # Create the first subplot
    ax1 = fig.add_subplot(121, projection="3d")
    plot_3d_colored_IWP_PC1_AOD_min_max_version(
        ax1,
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        xlabel,
        ylabel,
        zlabel,
        high_cloud_amount_diff,
        (0, 3),  # three difference maps
        vmin_diff,
        vmax_diff,
        AOD_bin_values=AOD_gap[1:4],
        PC1_bin_values=PC_new_intervals,
        IWP_bin_values=IWP_new_intervals,
        cmap=cmap_diff,
        add_colorbar=True,
        fig=fig,
        colobar_label=colobar_label,
    )

    # Create the third subplot
    if label_text1:
        fig.text(
            label_pos[0],
            label_pos[1],
            label_text1,
            transform=fig.transFigure,
            fontsize=17,
            va="top",
        )

    plt.savefig(
        "/RAID01/data/python_fig/subplot_combined"
        + colobar_label
        + ".png",
        dpi=400,
        bbox_inches="tight",
        facecolor="w",
    )
    plt.show()


# Call the function with different AOD ranges and save each figure separately
# -----------------------------------------------
# Plot the dust-AOD constrained data
# high cloud area
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_HCF,
    "Dust-AOD",
    "PC1",
    "IWP (g/m" + r"$^2$" + ")",  # IWP (g/m^2)
    r"$\Delta$" + " Cloud Area Fraction (%)",
    vmin=0,
    vmax=82,
    vmin_diff=-6,
    vmax_diff=6,
    label_text1="(A)",
)

# ice effective radius
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR,
    "Dust-AOD",
    "PC1",
    "IWP (g/m" + r"$^2$" + ")",  # IWP (g/m^2)
    r"$\Delta$" + " Ice Particle Radius (" + r"$\mu$" + "m)",
    vmin=12,
    vmax=32,
    vmin_diff=-1.5,
    vmax_diff=1.5,
    label_text1="(B)",
)

# cloud top pressure
# plot_both_3d_fill_plot_min_max_version(
#     CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_CTP,
#     "Dust-AOD",
#     "PC1",
#     "IWP (g/m" + r"$^2$" + ")",  # IWP (g/m^2)
#     r"$\Delta$" + " Cloud Top Pressure (hPa)",
#     vmin=185,
#     vmax=265,
#     vmin_diff=-25,
#     vmax_diff=25,
#     label_text1="(I)",
# )

# cloud top pressure
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_CEP,
    "Dust-AOD",
    "PC1",
    "IWP (g/m" + r"$^2$" + ")",  # IWP (g/m^2)
    r"$\Delta$" + " Cloud Effective Pressure (hPa)",
    vmin=185,
    vmax=265,
    vmin_diff=-25,
    vmax_diff=25,
    label_text1="(C)",
)

# cloud optical depth
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_COD,
    "Dust-AOD",
    "PC1",
    "IWP (g/m" + r"$^2$" + ")",  # IWP (g/m^2)
    r"$\Delta$" + " Cloud Visible Optical Depth",
    vmin=0,
    vmax=12.8,
    vmin_diff=-0.5,
    vmax_diff=0.5,
    label_text1="(D)",
)


# -----------------------------------------------------------------------
# Plot diverse IWP regions, mask out the IWP regions we dont need
# -----------------------------------------------------------------------

# Low IWP region
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_low_IWP = CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR.where(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR[
        "IWP_bin"
    ]
    <= 1,
    drop=True,
)

# Mid-Low IWP region
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_mid_low_IWP = CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR.where(
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
        < 5
    ),
    drop=True,
)

# Mid-High IWP region
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_mid_high_IWP = CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR.where(
    (
        CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR[
            "IWP_bin"
        ]
        > 5
    )
    & (
        CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR[
            "IWP_bin"
        ]
        < 50
    ),
    drop=True,
)

# High IWP region
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_high_IWP = CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR.where(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR[
        "IWP_bin"
    ]
    >= 50,
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
    vmax=25,
)

# Plot mid IWP region 3D constrain layout
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_mid_low_IWP,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CERES_SSF Cloud Effective Radius (microns)",
    vmin=18.2,
    vmax=28.5,
)

# Plot mid IWP region 3D constrain layout
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_mid_high_IWP,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CERES_SSF Cloud Effective Radius (microns)",
    vmin=23,
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
# -----------------------------------------------------------------------
# Plot diveres IWP regions specified by 3D constrain layout
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


# version 4: add contour lines


def plot_spatial_distribution_mask_contour(
    data,
    var_name,
    data_intervals,
    mask_thresholds,
    cmap="jet",
):
    """
    Plots the spatial distribution of data with contour lines and a color map, masked by specified thresholds.

    Args:
        data (numpy.ndarray): The data to plot.
        var_name (str): The name of the variable being plotted.
        data_intervals (list): A list of data intervals.
        mask_thresholds (list): A list of mask thresholds.
        cmap (str, optional): The name of the colormap to use. Defaults to "jet".

    Returns:
        None
    """
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-60, 89, 150)
    cmap = plt.cm.get_cmap(cmap)

    fig, axes = plt.subplots(
        nrows=4,
        figsize=(12, 24),
        constrained_layout=True,
        subplot_kw={
            "projection": ccrs.PlateCarree(central_longitude=180)
        },
    )
    plt.rcParams.update({"font.family": "Times New Roman"})

    for i, ax in enumerate(axes):
        threshold = mask_thresholds[i]

        print(
            "Now plotting: "
            + str(data_intervals[threshold[0]])
            + " to "
            + str(data_intervals[threshold[1]])
        )

        masked_data = np.count_nonzero(
            (data >= data_intervals[threshold[0]])
            & (data <= data_intervals[threshold[1]]),
            axis=0,
        )
        masked_data = np.where(
            masked_data == 0, np.nan, masked_data
        )

        ax.set_facecolor("silver")
        b = ax.pcolormesh(
            lon,
            lat,
            masked_data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
        )

        ax.contour(
            lon,
            lat,
            masked_data,
            levels=8,
            colors="gray",
            linewidths=0.8,
            linestyle="_",
            alpha=0.9,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(resolution="50m", lw=0.9)

        gl = ax.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        cb2 = plt.colorbar(
            b,
            ax=ax,
            location="bottom",
            shrink=0.7,
            aspect=40,
            extend="both",
        )
        cb2.set_label(label=var_name, size=24)
        cb2.ax.tick_params(labelsize=24)

        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

    plt.show()


# ----------------------------------------
# Plot the distribution gap for IWP
# ----------------------------------------

# title="Spatial Distribution of IWP Frequency Section",

plot_spatial_distribution_mask_contour(
    data=IWP_data,
    var_name="IWP Frequency",
    data_intervals=IWP_gaps_values,
    mask_thresholds={
        0: [0, 5],
        1: [7, 13],
        2: [14, 22],
        3: [23, -1],
    },
    cmap="RdYlBu_r",
)

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Calculate the ACI for the different IWP regions
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# Read the non AOD constrained cld data
CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR = read_filtered_data_out(
    file_name="CERES_SSF_IPR_match_PC_gap_IWP_AOD_constrain_mean_2005_2020_Dust_AOD_mask_1_IWP_AOD_no_antarc_new_PC_IWP_gaps_3_AOD_gaps_for_FIE_ME.nc"
)
# Read the same mask constrained AOD data
# In order to calculate the ACI
AOD_match_PC_gap_IWP_constrain_mean = read_filtered_data_out(
    file_name="AOD_match_PC_gap_IWP_AOD_constrain_mean_2005_2020_Dust_AOD_mask_1_IWP_AOD_no_antarc_3_AOD_gaps.nc"
)
# Read the same mask constrained PC1 data
# In order to calculate the delta ln(IPR)/delta PC1
PC1_match_PC_gap_IWP_constrain_mean = read_filtered_data_out(
    file_name="PC1_match_PC_gap_IWP_AOD_constrain_mean_2005_2020_Dust_AOD_mask_1_IWP_AOD_no_antarc_3_AOD_gaps.nc"
)

# Read PC1 true values so that we can visiualize the ACI
divisor_PC = DivideDataGivingGapByVolumeToNewGap(
    data=PC_data, n=4
)  # Modify 'n' as needed
PC_gap = np.arange(-3, 7, 1.5)
PC_new_intervals = divisor_PC.divide_intervals(PC_gap)

PC1_bin_values = PC_new_intervals

# Those dataarrays are all shape in (40,40,180,360), without AOD constrain
# Because the AOD are needed to calculate the ACI
# The first dimension is the IWP bin, the second dimension is the PC1 bin


# Now we need to calculate the ACI for each IWP region and each PC1 condition within


def FIE_calculator(
    IWP_regions: List[Tuple[int, int]],
    input_cld_data: np.ndarray,
    input_aer_data: np.ndarray,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Calculate the aerosol-cloud interaction (ACI) from
    the input data using the formula:
        ACI = -(dlnRE/dlnAOD).

    Args:
        input_cld_data (np.ndarray):
            A numpy array of shape (40,40,180,360) containing the input cloud data.
        input_aer_data (np.ndarray):
            A numpy array of shape (40,40,180,360) containing the input aerosol data.

    Returns:
        Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
            A tuple of three xarray DataArrays for the FIE, p-values, and standard errors, respectively,
            with the same coordinates as the input.

    Raises:
        ValueError: If the input arrays are not of shape (40,40,180,360).

    Notes:
        The function calculates the aerosol-cloud interaction (ACI)
        using the formula ACI = -(dlnRE/dlnAOD), where RE is
        the cloud effective radius and AOD is the aerosol optical depth. The input data should be numpy arrays of shape (40,40,180,360) containing the cloud and aerosol data, respectively. The function returns three xarray DataArrays for the FIE, p-values, and standard errors, respectively, with the same coordinates as the input.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> from ACI_calculator import ACI_calculator
        >>> input_cld_data = np.random.rand(40, 40, 180, 360)
        >>> input_aer_data = np.random.rand(40, 40, 180, 360)
        >>> FIE, p_values, std_errs = ACI_calculator(input_cld_data, input_aer_data)
        >>> assert isinstance(FIE, xr.DataArray)
        >>> assert isinstance(p_values, xr.DataArray)
        >>> assert isinstance(std_errs, xr.DataArray)
        >>> assert FIE.shape == (3, 40)
        >>> assert p_values.shape == (3, 40)
        >>> assert std_errs.shape == (3, 40)
    """
    # Separate the data into IWP regions

    # Create containers to hold the FIE results, the p-values, and the standard errors
    FIE = np.empty(
        (
            input_cld_data.shape[0],
            len(IWP_regions),
            input_cld_data.shape[2],
        )
    )
    p_values = np.empty(
        (
            input_cld_data.shape[0],
            len(IWP_regions),
            input_cld_data.shape[2],
        )
    )
    std_errs = np.empty(
        (
            input_cld_data.shape[0],
            len(IWP_regions),
            input_cld_data.shape[2],
        )
    )

    # Iterate over the AOD bins
    for AOD_bin in range(input_cld_data.shape[0]):
        # Iterate over the IWP regions
        for idx, IWP_region in enumerate(IWP_regions):
            # Mask the data for the current IWP region
            cld_data_region = input_cld_data.where(
                IWP_region, drop=True
            )
            aer_data_region = input_aer_data.where(
                IWP_region, drop=True
            )

            # Iterate over the PC1 bins
            for i in range(cld_data_region.shape[2]):
                cld_bin = cld_data_region[
                    AOD_bin, :, i, :, :
                ].values.flatten()
                aer_bin = aer_data_region[
                    AOD_bin, :, i, :, :
                ].values.flatten()

                # Remove any negative, zero or NaN values from the bin data
                mask = (
                    ~np.isnan(cld_bin)
                    & ~np.isnan(aer_bin)
                    & (cld_bin > 0)
                    & (aer_bin > 0)
                )
                cld_bin = cld_bin[mask]
                aer_bin = aer_bin[mask]

                if (
                    len(cld_bin) > 10 and len(aer_bin) > 10
                ):  # Avoid cases with no valid data
                    # Calculate the FIE using linear regression
                    (
                        slope,
                        intercept,
                        _,
                        p_value,
                        _,
                    ) = stats.linregress(
                        np.log(aer_bin), np.log(cld_bin)
                    )

                    # FIE = -(d ln⁡〖N〗)/(d ln⁡〖AOD〗)
                    FIE_region_bin = -slope
                    p_values_region_bin = p_value

                    # Calculate residuals and standard error of the slope
                    predicted = intercept + slope * np.log(
                        aer_bin
                    )
                    residuals = np.log(cld_bin) - predicted
                    s_err = np.sqrt(
                        np.sum(residuals**2)
                        / (len(np.log(aer_bin)) - 2)
                    )
                    slope_std_err = s_err / (
                        np.sqrt(
                            np.sum(
                                (
                                    np.log(aer_bin)
                                    - np.mean(np.log(aer_bin))
                                )
                                ** 2
                            )
                        )
                    )

                    # Store the results back in the FIE, p_values, and std_errs arrays
                    FIE[AOD_bin, idx, i] = FIE_region_bin
                    p_values[
                        AOD_bin, idx, i
                    ] = p_values_region_bin
                    std_errs[AOD_bin, idx, i] = slope_std_err
                else:
                    # If there is no valid data, set the FIE, p-value, and standard error to NaN
                    FIE[AOD_bin, idx, i] = np.nan
                    p_values[AOD_bin, idx, i] = np.nan
                    std_errs[AOD_bin, idx, i] = np.nan

    # Create xarray DataArrays for the results, with the first dimension being the IWP regions
    # AOD_bins_coords = ["low", "mid-low", "mid-high", "high"]
    AOD_bins_coords = ["low", "mid", "high"]
    IWP_regions_coords = ["low", "mid-low", "mid-high", "high"]
    PC1_coords = list(range(input_cld_data.shape[1]))

    FIE_da = xr.DataArray(
        FIE,
        coords=[AOD_bins_coords, IWP_regions_coords, PC1_coords],
        dims=["AOD_bin", "IWP_region", "PC1_bin"],
    )
    p_values_da = xr.DataArray(
        p_values,
        coords=[AOD_bins_coords, IWP_regions_coords, PC1_coords],
        dims=["AOD_bin", "IWP_region", "PC1_bin"],
    )
    std_errs_da = xr.DataArray(
        std_errs,
        coords=[AOD_bins_coords, IWP_regions_coords, PC1_coords],
        dims=["AOD_bin", "IWP_region", "PC1_bin"],
    )

    return FIE_da, p_values_da, std_errs_da


def ME_calculator(
    IWP_regions: List[Tuple[int, int]],
    input_cld_data: np.ndarray,
    input_pc_data: np.ndarray,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    # Separate the data into IWP regions

    # Create containers to hold the FIE results, the p-values, and the standard errors
    ME = np.empty(
        (
            input_cld_data.shape[0],
            len(IWP_regions),
            input_cld_data.shape[2],
        )
    )
    p_values = np.empty(
        (
            input_cld_data.shape[0],
            len(IWP_regions),
            input_cld_data.shape[2],
        )
    )
    std_errs = np.empty(
        (
            input_cld_data.shape[0],
            len(IWP_regions),
            input_cld_data.shape[2],
        )
    )

    # Iterate over the AOD bins
    for AOD_bin in range(input_cld_data.shape[0]):
        # Iterate over the IWP regions
        for idx, IWP_region in enumerate(IWP_regions):
            # Mask the data for the current IWP region
            cld_data_region = input_cld_data.where(
                IWP_region, drop=True
            )
            pc_data_region = input_pc_data.where(
                IWP_region, drop=True
            )

            # Iterate over the PC1 bins
            for i in range(cld_data_region.shape[2]):
                cld_bin = cld_data_region[
                    AOD_bin, :, i, :, :
                ].values.flatten()
                pc_bin = pc_data_region[
                    AOD_bin, :, i, :, :
                ].values.flatten()

                # Remove any negative, zero or NaN values from the bin data
                mask = (
                    ~np.isnan(cld_bin)
                    & ~np.isnan(pc_bin)
                    & (cld_bin > 0)
                    & (pc_bin > 0)
                )
                cld_bin = cld_bin[mask]
                pc_bin = pc_bin[mask]

                if (
                    len(cld_bin) > 10 and len(pc_bin) > 10
                ):  # Avoid cases with no valid data
                    # Calculate the FIE using linear regression
                    (
                        slope,
                        intercept,
                        _,
                        p_value,
                        _,
                    ) = stats.linregress(
                        np.log(pc_bin), np.log(cld_bin)
                    )

                    # ME = (d ln⁡〖N〗)/(d ln⁡〖AOD〗)
                    ME_region_bin = slope
                    p_values_region_bin = p_value

                    # Calculate residuals and standard error of the slope
                    predicted = intercept + slope * np.log(pc_bin)
                    residuals = np.log(cld_bin) - predicted
                    s_err = np.sqrt(
                        np.sum(residuals**2)
                        / (len(np.log(pc_bin)) - 2)
                    )
                    slope_std_err = s_err / (
                        np.sqrt(
                            np.sum(
                                (
                                    np.log(pc_bin)
                                    - np.mean(np.log(pc_bin))
                                )
                                ** 2
                            )
                        )
                    )

                    # Store the results back in the FIE, p_values, and std_errs arrays
                    ME[AOD_bin, idx, i] = ME_region_bin
                    p_values[
                        AOD_bin, idx, i
                    ] = p_values_region_bin
                    std_errs[AOD_bin, idx, i] = slope_std_err
                else:
                    # If there is no valid data, set the FIE, p-value, and standard error to NaN
                    ME[AOD_bin, idx, i] = np.nan
                    p_values[AOD_bin, idx, i] = np.nan
                    std_errs[AOD_bin, idx, i] = np.nan

    # Create xarray DataArrays for the results, with the first dimension being the IWP regions
    # AOD_bins_coords = ["low", "mid-low", "mid-high", "high"]
    AOD_bins_coords = ["low", "mid", "high"]
    IWP_regions_coords = ["low", "mid-low", "mid-high", "high"]
    PC1_coords = list(range(input_cld_data.shape[1]))

    ME_da = xr.DataArray(
        ME,
        coords=[AOD_bins_coords, IWP_regions_coords, PC1_coords],
        dims=["AOD_bin", "IWP_region", "PC1_bin"],
    )
    p_values_da = xr.DataArray(
        p_values,
        coords=[AOD_bins_coords, IWP_regions_coords, PC1_coords],
        dims=["AOD_bin", "IWP_region", "PC1_bin"],
    )
    std_errs_da = xr.DataArray(
        std_errs,
        coords=[AOD_bins_coords, IWP_regions_coords, PC1_coords],
        dims=["AOD_bin", "IWP_region", "PC1_bin"],
    )

    return ME_da, p_values_da, std_errs_da


# Calculate the ACI for the different IWP regions and each PC1 condition within each AOD region
FIE_da, FIE_p_values_da, FIE_std_errs_da = FIE_calculator(
    IWP_regions=[
        (
            CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR[
                "IWP_bin"
            ]
            <= 2.2
        ),
        (
            (
                CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR[
                    "IWP_bin"
                ]
                > 2.2
            )
            & (
                CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR[
                    "IWP_bin"
                ]
                < 19
            )
        ),
        (
            (
                CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR[
                    "IWP_bin"
                ]
                > 21
            )
            & (
                CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR[
                    "IWP_bin"
                ]
                < 84
            )
        ),
        (
            CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR[
                "IWP_bin"
            ]
            >= 86
        ),
    ],
    input_cld_data=CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR,
    input_aer_data=AOD_match_PC_gap_IWP_constrain_mean,
)

# Calculate the ME for the different IWP regions and each PC1 condition within each AOD region
ME_da, ME_p_values_da, ME_std_errs_da = ME_calculator(
    IWP_regions=[
        (
            CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR[
                "IWP_bin"
            ]
            <= 2.2
        ),
        (
            (
                CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR[
                    "IWP_bin"
                ]
                > 2.2
            )
            & (
                CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR[
                    "IWP_bin"
                ]
                < 19
            )
        ),
        (
            (
                CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR[
                    "IWP_bin"
                ]
                > 21
            )
            & (
                CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR[
                    "IWP_bin"
                ]
                < 84
            )
        ),
        (
            CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR[
                "IWP_bin"
            ]
            >= 86
        ),
    ],
    input_cld_data=CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR,
    input_pc_data=PC1_match_PC_gap_IWP_constrain_mean,
)


def plot_FIE_diverse_AOD_3_gaps(
    FIE_da: xr.DataArray,
    std_errs_da: xr.DataArray,
    p_values_da: xr.DataArray,
    xlims: List[Tuple[float, float]],
    ylims: List[Tuple[float, float]],
    colors: Optional[List[Tuple[float, float, float]]] = None,
    PC1_bin_values: Optional[np.ndarray] = None,
    n: int = 5,
    alpha: float = 0.05,
    mask_rules: Optional[
        Dict[Tuple[str, str], List[Tuple[int, int]]]
    ] = None,
) -> None:
    # PC1 bin values
    PC1_bin_values = np.round(PC1_bin_values[1:], 2)

    # Define the IWP and AOD regions
    AOD_regions = ["low", "mid", "high"]
    IWP_regions = ["low", "mid-low", "mid-high", "high"]

    # Define the colors
    if colors is None:
        colors = [
            ((149 / 255), (213 / 255), (184 / 255)),
            ((48 / 255), (165 / 255), (194 / 255)),
            ((34 / 255), (84 / 255), (163 / 255)),
            ((8 / 255), (29 / 255), (89 / 255)),
        ]

    # Ensure ylims has the right length
    if len(ylims) != len(IWP_regions):
        raise ValueError(
            "ylims should be a list of pairs of floats of the same length as IWP_regions"
        )

    # Define a function to calculate the block condition
    def calculate_block_condition(
        FIE, std_err, iwp_region, aod_region, x
    ):
        block_condition = (
            np.abs(FIE) >= std_err * 0.9
        )  # Replace with the actual condition

        # Check mask_rules for any specific masking rule for this iwp_region and aod_region
        if mask_rules and (iwp_region, aod_region) in mask_rules:
            for start, end in mask_rules[
                (iwp_region, aod_region)
            ]:
                block_condition[start : end + 1] = False

        return block_condition

    fig, axs = plt.subplots(4, 1, figsize=(8, 16), dpi=400)

    # Loop over IWP regions
    for IWP_idx, ax in enumerate(axs.flat):
        # Loop over AOD regions
        for aod_idx, aod_region in enumerate(AOD_regions):
            # Select the FIE, standard error and p-value data for the current IWP and AOD region
            iwp_region = IWP_regions[IWP_idx]

            FIE = FIE_da.sel(
                IWP_region=iwp_region, AOD_bin=aod_region
            )
            std_err = std_errs_da.sel(
                IWP_region=iwp_region, AOD_bin=aod_region
            )
            p_values = p_values_da.sel(
                IWP_region=iwp_region, AOD_bin=aod_region
            )

            # Define the PC1 bins as x-axis
            x = np.arange(1, len(FIE) + 1)

            # Mask the points where the std_errs_da exceed the absolute value of FIE_da
            # mask = np.abs(FIE) >= std_err * 1.3

            block_condition = calculate_block_condition(
                FIE, std_err, iwp_region, aod_region, x
            )

            FIE_masked = FIE.where(block_condition, np.nan)
            std_err_masked = std_err.where(
                block_condition, np.nan
            )
            p_values_masked = p_values.where(
                block_condition, np.nan
            )

            # Apply the mask to the FIE, std_err and p_values
            # FIE_masked = FIE.where(mask, np.nan)
            # std_err_masked = std_err.where(mask, np.nan)
            # p_values_masked = p_values.where(mask, np.nan)

            ax.errorbar(
                x,
                FIE_masked,
                yerr=std_err_masked,
                fmt="-o",
                color=colors[aod_idx],  # Use specified color
                linewidth=2,
                alpha=0.85,
                markersize=12,
                capsize=8,
                label=aod_region,
            )

            # Get indices where p-value <= alpha
            significant = np.where(p_values_masked <= alpha)

            # Highlight significant points
            ax.scatter(
                x[significant],
                FIE_masked[significant],
                color="white",
                marker="*",
                # Ensure these points are drawn over the other elements
                zorder=5,
            )

        zero_line = np.arange(-19, 59, 0.1)

        ax.plot(
            zero_line,
            np.zeros_like(zero_line),
            "k",
            linewidth=4.5,
        )

        if PC1_bin_values is not None:
            ax.set_xticks(x[::n])
            ax.set_xticklabels(PC1_bin_values[::n])

        ax.grid(True, linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("PC1 bin")
        ax.set_ylabel("FIE")
        ax.set_title(str(iwp_region) + " IWP")

        ax.legend()

        ax.set_ylim(ylims[IWP_idx])
        ax.set_xlim(xlims[IWP_idx])

    plt.tight_layout()
    plt.show()


# plot FIE for diverse AOD 3 gaps
plot_FIE_diverse_AOD_3_gaps(
    FIE_da,
    FIE_std_errs_da,
    FIE_p_values_da,
    # xlims=[(1.5, 15.5), (1.5, 20.5), (4.5, 23.5), (4.5, 23.5)],
    xlims=[(1.5, 24.5), (1.5, 24.5), (1.5, 24.5), (1.5, 24.5)],
    ylims=[
        (-0.17, 0.17),
        (-0.17, 0.17),
        (-0.17, 0.17),
        (-0.17, 0.17),
    ],
    # Blue, orange, red colors
    # colors=[
    #     "#80A6E2",
    #     "#FBDD85",
    #     "#F46F43",
    # ],
    # Blue to red colors
    colors=[
        ((55 / 255), (103 / 255), (149 / 255)),
        ((255 / 255), (208 / 255), (111 / 255)),
        ((231 / 255), (98 / 255), (84 / 255)),
    ],
    PC1_bin_values=PC1_bin_values,
    n=2,
    mask_rules={
        # ("low", "low"): [(9, 20)],
        # ("low", "mid-low"): [(9, 20)],
        # ("low", "mid-high"): [(11, 20)],
        # ("low", "high"): [(10, 25)],
        # ("mid-low", "low"): [(18, 25)],
    },
)


########################################################################################
##### Combined plots ###################################################################
########################################################################################


# Define the function to plot the spatial distribution mask contour
def plot_spatial_distribution_mask_contour(
    axes,  # Pass axes to plot
    data,
    var_name,
    data_intervals,
    mask_thresholds,
    cax_config,
    cmap="jet",
):
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-60, 89, 150)
    cmap = plt.cm.get_cmap(cmap)

    # Create a GridSpec instance to control the positioning of the colorbar
    gs = GridSpec(
        4, 1, figure=fig
    )  # Adjust as needed based on your layout

    plt.rcParams.update({"font.family": "Times New Roman"})

    for i, ax in enumerate(axes):
        threshold = mask_thresholds[i]
        masked_data = np.count_nonzero(
            (data >= data_intervals[threshold[0]])
            & (data <= data_intervals[threshold[1]]),
            axis=0,
        )
        masked_data = np.where(
            masked_data == 0, np.nan, masked_data
        )

        ax.set_facecolor("silver")
        b = ax.pcolormesh(
            lon,
            lat,
            masked_data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
        )

        ax.contour(
            lon,
            lat,
            masked_data,
            levels=8,
            colors="gray",
            linewidths=0.8,
            linestyle="_",
            alpha=0.9,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(resolution="50m", lw=0.9)
        gl = ax.gridlines(
            linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
        )
        gl.top_labels = False
        gl.bottom_labels = False

        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        if i == len(axes) - 1:
            gl.bottom_labels = True

        # Display the colorbar only for the last row
        # if i == 3:
        #     cb2 = plt.colorbar(
        #         b,
        #         ax=ax,
        #         location="bottom",
        #         shrink=0.7,
        #         aspect=40,
        #         extend="both",
        #     )
        #     cb2.set_label(label=var_name, size=24)
        #     cb2.ax.tick_params(labelsize=24)

    # Create the colorbar outside of the loop, positioning it relative to the entire figure
    cax = fig.add_axes(
        cax_config
    )  # Adjust these values to position the colorbar
    cb2 = plt.colorbar(
        b,
        cax=cax,
        extend="both",
        location="bottom",
        shrink=0.7,
        aspect=40,
    )
    cb2.set_label(label=var_name, size=24)
    cb2.ax.tick_params(labelsize=24)


# Define the function to plot the line graph
def plot_FIE_diverse_AOD_3_gaps(
    axes,  # Pass axes to plot
    FIE_da: xr.DataArray,
    std_errs_da: xr.DataArray,
    p_values_da: xr.DataArray,
    xlims: List[Tuple[float, float]],
    ylims: List[Tuple[float, float]],
    colors: Optional[List[Tuple[float, float, float]]] = None,
    PC1_bin_values: Optional[np.ndarray] = None,
    n: int = 5,
    alpha: float = 0.05,
    mask_rules: Optional[
        Dict[Tuple[str, str], List[Tuple[int, int]]]
    ] = None,
) -> None:
    # Existing code with the plot modified to use the passed axes
    # PC1 bin values
    PC1_bin_values = np.round(PC1_bin_values[1:], 2)

    # Define the IWP and AOD regions
    AOD_regions = ["low", "mid", "high"]
    IWP_regions = ["low", "mid-low", "mid-high", "high"]

    # Define the colors
    if colors is None:
        colors = [
            ((149 / 255), (213 / 255), (184 / 255)),
            ((48 / 255), (165 / 255), (194 / 255)),
            ((34 / 255), (84 / 255), (163 / 255)),
            ((8 / 255), (29 / 255), (89 / 255)),
        ]

    # Ensure ylims has the right length
    if len(ylims) != len(IWP_regions):
        raise ValueError(
            "ylims should be a list of pairs of floats of the same length as IWP_regions"
        )

    # Define a function to calculate the block condition
    def calculate_block_condition(
        FIE, std_err, iwp_region, aod_region, x
    ):
        block_condition = (
            np.abs(FIE) >= std_err * 0.9
        )  # Replace with the actual condition

        # Check mask_rules for any specific masking rule for this iwp_region and aod_region
        if mask_rules and (iwp_region, aod_region) in mask_rules:
            for start, end in mask_rules[
                (iwp_region, aod_region)
            ]:
                block_condition[start : end + 1] = False

        return block_condition

    # fig, axs = plt.subplots(4, 1, figsize=(8, 16), dpi=400)
    axs = axes

    # Loop over IWP regions
    for IWP_idx, ax in enumerate(right_axes):
        # Loop over AOD regions
        for aod_idx, aod_region in enumerate(AOD_regions):
            # Select the FIE, standard error and p-value data for the current IWP and AOD region
            iwp_region = IWP_regions[IWP_idx]

            FIE = FIE_da.sel(
                IWP_region=iwp_region, AOD_bin=aod_region
            )
            std_err = std_errs_da.sel(
                IWP_region=iwp_region, AOD_bin=aod_region
            )
            p_values = p_values_da.sel(
                IWP_region=iwp_region, AOD_bin=aod_region
            )

            # Define the PC1 bins as x-axis
            x = np.arange(1, len(FIE) + 1)

            # Mask the points where the std_errs_da exceed the absolute value of FIE_da
            # mask = np.abs(FIE) >= std_err * 1.3

            block_condition = calculate_block_condition(
                FIE, std_err, iwp_region, aod_region, x
            )

            FIE_masked = FIE.where(block_condition, np.nan)
            std_err_masked = std_err.where(
                block_condition, np.nan
            )
            p_values_masked = p_values.where(
                block_condition, np.nan
            )

            # Apply the mask to the FIE, std_err and p_values
            # FIE_masked = FIE.where(mask, np.nan)
            # std_err_masked = std_err.where(mask, np.nan)
            # p_values_masked = p_values.where(mask, np.nan)

            ax.errorbar(
                x,
                FIE_masked,
                yerr=std_err_masked,
                fmt="-o",
                color=colors[aod_idx],  # Use specified color
                linewidth=2,
                alpha=0.85,
                markersize=12,
                capsize=8,
                label=aod_region,
            )

            # Get indices where p-value <= alpha
            significant = np.where(p_values_masked <= alpha)

            # Highlight significant points
            ax.scatter(
                x[significant],
                FIE_masked[significant],
                color="white",
                marker="*",
                # Ensure these points are drawn over the other elements
                zorder=5,
            )

        zero_line = np.arange(-19, 59, 0.1)

        ax.plot(
            zero_line,
            np.zeros_like(zero_line),
            "k",
            linewidth=4.5,
        )

        if (
            PC1_bin_values is not None and IWP_idx == len(axs) - 1
        ):  # Check if it's the last subplot
            ax.set_xticks(x[::n])
            ax.set_xticklabels(PC1_bin_values[::n])
            ax.set_xlabel("PC1 bin", fontsize=18)
        else:
            ax.set_xticks(x[::n])
            ax.set_xticklabels(
                []
            )  # Remove x-ticks and labels for all other subplots
            ax.set_xlabel(
                ""
            )  # Remove x-ticks and labels for all other subplots

        ax.grid(True, linestyle="--", linewidth=1, alpha=0.7)

        ax.set_ylabel("FIE", fontsize=18)

        ax.legend()

        ax.set_ylim(ylims[IWP_idx])
        ax.set_xlim(xlims[IWP_idx])

        # set tickslabels font size for all axes
        ax.tick_params(axis="both", labelsize=17)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------- #
# Combined plot of spatial distribution and frequency of IWP  #
# ------------------------------------------------------------- #

fig = plt.figure(figsize=(18, 18), dpi=400)
gs = gridspec.GridSpec(
    nrows=4,
    ncols=2,
    width_ratios=[1.4, 1],
    height_ratios=[1, 1, 1, 1],
    wspace=0.15,
    hspace=0.15,
)

IWP_regions = ["low", "mid-low", "mid-high", "high"]

# Create axes for the left column with cartographic projection
left_axes = [
    plt.subplot(
        gs[i, 0],
        projection=ccrs.PlateCarree(central_longitude=180),
    )
    for i in range(4)
]
# Create axes for the right column without projection
right_axes = [plt.subplot(gs[i, 1]) for i in range(4)]

# Define the labels for IWP regions
IWP_labels = [
    "Low IWP",
    "Mid-Low IWP",
    "Mid-High IWP",
    "High IWP",
]

# Loop over left column axes for adding white legends
for i, ax in enumerate(left_axes):
    ax.text(
        0.01,
        0.955,
        f"({chr(65 + i*2)})",
        transform=ax.transAxes,
        fontsize=21,
        va="top",
        color="white",
    )

# Loop over right column axes for adding black legends and IWP labels
for i, ax in enumerate(right_axes):
    ax.text(
        0.033,
        0.955,
        f"({chr(66 + i*2)})",
        transform=ax.transAxes,
        fontsize=21,
        va="top",
        color="black",
    )
    # Add IWP labels on the far left
    ax.text(
        -1.72,
        0.5,
        IWP_labels[i],
        transform=ax.transAxes,
        fontsize=25,
        va="center",
        ha="right",
        rotation="vertical",
    )


# Call the spatial distribution plotting function with the left column axes
plot_spatial_distribution_mask_contour(
    left_axes,
    data=IWP_data,
    var_name="IWP Frequency",
    data_intervals=IWP_gaps_values,
    mask_thresholds={
        0: [0, 5],
        1: [7, 13],
        2: [14, 22],
        3: [23, -1],
    },
    # cax_config=[left, bottom, width, height]
    cax_config=[0.135, 0.06, 0.4, 0.012],
    cmap="RdYlBu_r",
)

# Call the line graph plotting function with the right column axes
# Ensure the colors length matches the AOD_regions length
colors = [
    ((55 / 255), (103 / 255), (149 / 255)),
    ((255 / 255), (208 / 255), (111 / 255)),
    ((231 / 255), (98 / 255), (84 / 255)),
    (
        (8 / 255),
        (29 / 255),
        (89 / 255),
    ),  # Add an additional color
]
plot_FIE_diverse_AOD_3_gaps(
    right_axes,
    FIE_da,
    FIE_std_errs_da,
    FIE_p_values_da,
    xlims=[(2, 24.5), (2, 24.5), (2, 24.5), (2, 24.5)],
    ylims=[
        (-0.17, 0.17),
        (-0.17, 0.17),
        (-0.17, 0.17),
        (-0.17, 0.17),
    ],
    colors=colors,
    PC1_bin_values=PC1_bin_values,
    n=5,
    # IWP region, AOD region
    mask_rules={
        ("low", "low"): [(13, 20)],
        ("low", "mid"): [(12, 20)],
        ("mid-low", "low"): [(16, 22)],
        ("mid-high", "mid"): [(7, 11)],
    },
)

plt.show()


##############################################################################
##############################################################################
##############################################################################

# ------------------------------ #
# Verifying the results with CERES_SSF
# ------------------------------ #


def plot_test(ipnut_data):
    data = np.nanmean(ipnut_data, axis=(2, 3))

    fig, ax = plt.subplots(figsize=(10, 8))
    a = ax.pcolormesh(
        data,
        cmap="Spectral_r",
        vmin=12,
        vmax=33,
    )

    ax.set_xlabel("PC1 bin")
    ax.set_ylabel("IWP region")

    fig.colorbar(
        a,
        ax=ax,
        label="CERES_SSF Cloud Effective Radius (microns)",
    )

    plt.show()


plot_test(CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR)


##############################################################################
# Expired code ###############################################################
##############################################################################

# def plot_FIE_diverse_AOD(
#     FIE_da: xr.DataArray,
#     std_errs_da: xr.DataArray,
#     p_values_da: xr.DataArray,
#     xlims: List[Tuple[float, float]],
#     ylims: List[Tuple[float, float]],
#     colors: Optional[List[Tuple[float, float, float]]] = None,
#     PC1_bin_values: Optional[np.ndarray] = None,
#     n: int = 5,
#     alpha: float = 0.05,
#     mask_rules: Optional[
#         Dict[Tuple[str, str], List[Tuple[int, int]]]
#     ] = None,
# ) -> None:
#     # PC1 bin values
#     PC1_bin_values = np.round(PC1_bin_values[1:], 2)

#     # Define the IWP and AOD regions
#     AOD_regions = ["low", "mid-low", "mid-high", "high"]
#     IWP_regions = ["low", "mid-low", "mid-high", "high"]

#     # Define the colors
#     if colors is None:
#         colors = [
#             ((149 / 255), (213 / 255), (184 / 255)),
#             ((48 / 255), (165 / 255), (194 / 255)),
#             ((34 / 255), (84 / 255), (163 / 255)),
#             ((8 / 255), (29 / 255), (89 / 255)),
#         ]

#     # Ensure ylims has the right length
#     if len(ylims) != len(IWP_regions):
#         raise ValueError(
#             "ylims should be a list of pairs of floats of the same length as IWP_regions"
#         )

#     # Define a function to calculate the block condition
#     def calculate_block_condition(
#         FIE, std_err, iwp_region, aod_region, x
#     ):
#         block_condition = (
#             np.abs(FIE) >= std_err * 1.15
#         )  # Replace with the actual condition

#         # Check mask_rules for any specific masking rule for this iwp_region and aod_region
#         if mask_rules and (iwp_region, aod_region) in mask_rules:
#             for start, end in mask_rules[
#                 (iwp_region, aod_region)
#             ]:
#                 block_condition[start : end + 1] = False

#         return block_condition

#     fig, axs = plt.subplots(4, 1, figsize=(8, 16), dpi=400)

#     # Loop over IWP regions
#     for IWP_idx, ax in enumerate(axs.flat):
#         # Loop over AOD regions
#         for aod_idx, aod_region in enumerate(AOD_regions):
#             # Select the FIE, standard error and p-value data for the current IWP and AOD region
#             iwp_region = IWP_regions[IWP_idx]

#             FIE = FIE_da.sel(
#                 IWP_region=iwp_region, AOD_bin=aod_region
#             )
#             std_err = std_errs_da.sel(
#                 IWP_region=iwp_region, AOD_bin=aod_region
#             )
#             p_values = p_values_da.sel(
#                 IWP_region=iwp_region, AOD_bin=aod_region
#             )

#             # Define the PC1 bins as x-axis
#             x = np.arange(1, len(FIE) + 1)

#             # Mask the points where the std_errs_da exceed the absolute value of FIE_da
#             # mask = np.abs(FIE) >= std_err * 1.3

#             block_condition = calculate_block_condition(
#                 FIE, std_err, iwp_region, aod_region, x
#             )

#             FIE_masked = FIE.where(block_condition, np.nan)
#             std_err_masked = std_err.where(
#                 block_condition, np.nan
#             )
#             p_values_masked = p_values.where(
#                 block_condition, np.nan
#             )

#             # Apply the mask to the FIE, std_err and p_values
#             # FIE_masked = FIE.where(mask, np.nan)
#             # std_err_masked = std_err.where(mask, np.nan)
#             # p_values_masked = p_values.where(mask, np.nan)

#             ax.errorbar(
#                 x,
#                 FIE_masked,
#                 yerr=std_err_masked,
#                 fmt="-o",
#                 color=colors[aod_idx],  # Use specified color
#                 linewidth=2,
#                 alpha=0.85,
#                 markersize=12,
#                 capsize=8,
#                 label=aod_region,
#             )

#             # Get indices where p-value <= alpha
#             significant = np.where(p_values_masked <= alpha)

#             # Highlight significant points
#             ax.scatter(
#                 x[significant],
#                 FIE_masked[significant],
#                 color="white",
#                 marker="*",
#                 # Ensure these points are drawn over the other elements
#                 zorder=5,
#             )

#         zero_line = np.arange(-19, 59, 0.1)

#         ax.plot(
#             zero_line,
#             np.zeros_like(zero_line),
#             "k",
#             linewidth=4.5,
#         )

#         if PC1_bin_values is not None:
#             ax.set_xticks(x[::n])
#             ax.set_xticklabels(PC1_bin_values[::n])

#         ax.grid(True, linestyle="--", linewidth=1, alpha=0.7)
#         ax.set_xlabel("PC1 bin")
#         ax.set_ylabel("FIE")
#         ax.set_title(str(iwp_region) + " IWP")

#         ax.legend()

#         ax.set_ylim(ylims[IWP_idx])
#         ax.set_xlim(xlims[IWP_idx])

#     plt.tight_layout()
#     plt.show()


# # plot FIE for diverse AOD
# plot_FIE_diverse_AOD(
#     FIE_da,
#     FIE_std_errs_da,
#     FIE_p_values_da,
#     # xlims=[(1.5, 15.5), (1.5, 20.5), (4.5, 23.5), (4.5, 23.5)],
#     xlims=[(1.5, 24.5), (1.5, 24.5), (1.5, 24.5), (1.5, 24.5)],
#     ylims=[
#         (-0.17, 0.17),
#         (-0.17, 0.17),
#         (-0.17, 0.17),
#         (-0.17, 0.17),
#     ],
#     # Deep blue to light blue colors
#     # colors=[
#     #     ((149 / 255), (213 / 255), (184 / 255)),
#     #     ((48 / 255), (165 / 255), (194 / 255)),
#     #     ((34 / 255), (84 / 255), (163 / 255)),
#     #     ((8 / 255), (29 / 255), (89 / 255)),
#     # ],
#     # Deep red to light red colors
#     # colors=[
#     #     ((242 / 255), (165 / 255), (132 / 255)),
#     #     ((212 / 255), (97 / 255), (83 / 255)),
#     #     ((177 / 255), (24 / 255), (45 / 255)),
#     #     ((102 / 255), (0 / 255), (32 / 255)),
#     # ],
#     # Deep orange to light orange colors
#     # colors=[
#     #     ((238 / 255), (155 / 255), (0 / 255)),
#     #     ((204 / 255), (102 / 255), (2 / 255)),
#     #     ((188 / 255), (62 / 255), (3 / 255)),
#     #     ((174 / 255), (32 / 255), (18 / 255)),
#     # ],
#     # Blue to red colors
#     colors=[
#         ((55 / 255), (103 / 255), (149 / 255)),
#         ((114 / 255), (188 / 255), (213 / 255)),
#         ((255 / 255), (208 / 255), (111 / 255)),
#         ((231 / 255), (98 / 255), (84 / 255)),
#     ],
#     PC1_bin_values=PC1_bin_values,
#     n=2,
#     mask_rules={
#         # ("low", "low"): [(9, 20)],
#         # ("low", "mid-low"): [(9, 20)],
#         # ("low", "mid-high"): [(11, 20)],
#         # ("low", "high"): [(10, 25)],
#         ("mid-low", "low"): [(18, 25)],
#     },
# )


# # plot FIE for diverse AOD
# plot_FIE_diverse_AOD(
#     ME_da,
#     ME_std_errs_da,
#     ME_p_values_da,
#     xlims=[(1.5, 21), (1.5, 23.5), (1.5, 23.5), (1.5, 23.5)],
#     ylims=[
#         (-0.3, 0.3),
#         (-0.15, 0.15),
#         (-0.15, 0.15),
#         (-0.15, 0.15),
#     ],
#     # Deep blue to light blue colors
#     # colors=[
#     #     ((149 / 255), (213 / 255), (184 / 255)),
#     #     ((48 / 255), (165 / 255), (194 / 255)),
#     #     ((34 / 255), (84 / 255), (163 / 255)),
#     #     ((8 / 255), (29 / 255), (89 / 255)),
#     # ],
#     # Deep red to light red colors
#     # colors=[
#     #     ((242 / 255), (165 / 255), (132 / 255)),
#     #     ((212 / 255), (97 / 255), (83 / 255)),
#     #     ((177 / 255), (24 / 255), (45 / 255)),
#     #     ((102 / 255), (0 / 255), (32 / 255)),
#     # ],
#     # Deep orange to light orange colors
#     # colors=[
#     #     ((238 / 255), (155 / 255), (0 / 255)),
#     #     ((204 / 255), (102 / 255), (2 / 255)),
#     #     ((188 / 255), (62 / 255), (3 / 255)),
#     #     ((174 / 255), (32 / 255), (18 / 255)),
#     # ],
#     # Blue to red colors
#     colors=[
#         ((55 / 255), (103 / 255), (149 / 255)),
#         ((114 / 255), (188 / 255), (213 / 255)),
#         ((255 / 255), (208 / 255), (111 / 255)),
#         ((231 / 255), (98 / 255), (84 / 255)),
#     ],
#     PC1_bin_values=PC1_bin_values,
#     n=2,
# )


# # version 3: grid point frequency sum
# def plot_spatial_distribution_mask(
#     data,
#     var_name,
#     title,
#     data_intervals,
#     mask_threshold=None,
#     cmap="jet",
# ):
#     lon = np.linspace(0, 359, 360)
#     lat = np.linspace(-60, 89, 150)

#     # Create custom colormap
#     cmap = plt.cm.get_cmap(cmap)

#     # print function
#     print(
#         "Now plotting: "
#         + str(data_intervals[mask_threshold[0]])
#         + " to "
#         + str(data_intervals[mask_threshold[1]])
#     )
#     # Count the frequency of values within the specified range at each grid point
#     if mask_threshold is not None:
#         data = np.count_nonzero(
#             (data >= data_intervals[mask_threshold[0]])
#             & (data <= data_intervals[mask_threshold[1]]),
#             axis=0,
#         )

#     # Calculate the sum of the data array along the first axis (time)
#     # data = np.sum(data, axis=0)
#     # data[np.isnan(data)] = 0
#     data = np.where(data == 0, np.nan, data)

#     fig, (ax1) = plt.subplots(
#         ncols=1,
#         nrows=1,
#         figsize=(12, 6),
#         constrained_layout=True,
#     )
#     plt.rcParams.update({"font.family": "Times New Roman"})

#     ax1 = plt.subplot(
#         111,
#         projection=ccrs.PlateCarree(central_longitude=180),
#     )
#     ax1.set_facecolor("silver")
#     b = ax1.pcolormesh(
#         lon,
#         lat,
#         data,
#         transform=ccrs.PlateCarree(),
#         cmap=cmap,
#     )
#     ax1.coastlines(resolution="50m", lw=0.9)
#     # ax1.set_title(title, fontsize=24)

#     gl = ax1.gridlines(
#         linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
#     )
#     gl.top_labels = False
#     cb2 = fig.colorbar(
#         b,
#         ax=ax1,
#         location="bottom",
#         shrink=0.7,
#         aspect=40,
#         extend="both",
#     )
#     cb2.set_label(label=var_name, size=24)
#     cb2.ax.tick_params(labelsize=24)

#     gl.xlabel_style = {"size": 18}
#     gl.ylabel_style = {"size": 18}


# def plot_FIE_diverse_AOD(
#     FIE_da: xr.DataArray,
#     std_errs_da: xr.DataArray,
#     p_values_da: xr.DataArray,
#     ylims: List[float],
#     colors: Optional[List[Tuple[float, float, float]]] = None,
#     PC1_bin_values: Optional[np.ndarray] = None,
#     n: int = 5,
#     alpha: float = 0.05,
# ) -> None:
#     """
#     Plots the Fraction of Explained Variance (FIE) for diverse Aerosol Optical Depth (AOD) values.

#     Parameters:
#     -----------
#     FIE_da : xr.DataArray
#         DataArray containing the FIE values for each IWP and AOD bin.
#     std_errs_da : xr.DataArray
#         DataArray containing the standard error values for each IWP and AOD bin.
#     p_values_da : xr.DataArray
#         DataArray containing the p-values for each IWP and AOD bin.
#     ylims : List[float]
#         List containing the minimum and maximum values for the y-axis limits.
#     colors : Optional[List[Tuple[float, float, float]]]
#         List containing the colors to be used for each AOD bin. If None, default colors will be used.
#     PC1_bin_values : Optional[np.ndarray]
#         Array containing the PC1 bin values to be used for the x-axis ticks. If None, no ticks will be shown.
#     n : int
#         Number of PC1 bins to skip between x-axis ticks.
#     alpha : float
#         Significance level for the p-values. Default is 0.05.

#     Returns:
#     --------
#     None
#     """
#     # PC1 bin values
#     PC1_bin_values = np.round(PC1_bin_values[1:], 2)

#     if colors is None:
#         colors = [
#             ((149 / 255), (213 / 255), (184 / 255)),
#             ((48 / 255), (165 / 255), (194 / 255)),
#             ((34 / 255), (84 / 255), (163 / 255)),
#             ((8 / 255), (29 / 255), (89 / 255)),
#         ]

#     AOD_regions = ["low", "mid-low", "mid-high", "high"]
#     IWP_regions = ["low", "mid-low", "mid-high", "high"]

#     fig, axs = plt.subplots(4, 1, figsize=(8, 16), dpi=400)

#     for idx, ax in enumerate(axs.flat):
#         for aod_bin, aod_region in enumerate(AOD_regions):

#             IWP_idx = IWP_regions[idx]

#             FIE = FIE_da.sel(IWP_region=IWP_idx, AOD_bin=aod_region)
#             std_err = std_errs_da.sel(
#                 IWP_region=IWP_idx, AOD_bin=aod_region
#             )
#             p_values = p_values_da.sel(
#                 IWP_region=IWP_idx, AOD_bin=aod_region
#             )

#             # Filter data
#             mask = p_values <= alpha
#             FIE = FIE.where(mask)
#             std_err = std_err.where(mask)

#             # Define the PC1 bins as x-axis
#             x = np.arange(1, len(FIE) + 1)

#             ax.errorbar(
#                 x,
#                 FIE,
#                 yerr=std_err,
#                 fmt="-o",
#                 color=colors[aod_bin],  # Use specified color
#                 linewidth=2,
#                 alpha=0.7,
#                 markersize=8,
#                 capsize=8,
#                 label=aod_region,
#             )

#         ax.plot(x, np.zeros_like(x), "k", linewidth=4.5)

#         if PC1_bin_values is not None:
#             ax.set_xticks(x[::n])
#             ax.set_xticklabels(PC1_bin_values[::n])

#         ax.set_ylim(ymin=ylims[0], ymax=ylims[1])
#         ax.set_xlabel("PC1 bin")
#         ax.set_ylabel("FIE")
#         ax.set_title(str(IWP_idx) + " IWP")
#         ax.legend()

#     plt.tight_layout()
#     plt.show()


# plot_FIE(
#     FIE_da,
#     std_errs_da,
#     ylims=[-0.1, 0.06],
#     colors=[
#         ((149 / 255), (213 / 255), (184 / 255)),
#         ((48 / 255), (165 / 255), (194 / 255)),
#         ((34 / 255), (84 / 255), (163 / 255)),
#         ((8 / 255), (29 / 255), (89 / 255)),
#     ],
#     PC1_bin_values=PC1_bin_values,
# )

# # plot FIE for diverse AOD
# plot_FIE_diverse_AOD(
#     FIE_da,
#     std_errs_da,
#     AOD_bins=0,
#     ylims=[-0.18, 0.18],
#     colors=[
#         ((149 / 255), (213 / 255), (184 / 255)),
#         ((48 / 255), (165 / 255), (194 / 255)),
#         ((34 / 255), (84 / 255), (163 / 255)),
#         ((8 / 255), (29 / 255), (89 / 255)),
#     ],
#     PC1_bin_values=PC1_bin_values,
#     n=3,
# )


# def plot_FIE(
#     FIE_da,
#     std_errs_da,
#     ylims=[-0.5, 0.5],
#     colors=None,
#     PC1_bin_values=None,
#     n=5,
# ):
#     """
#     Plot the fraction of aerosol indirect effect (FIE) for each IWP region.

#     Args:
#         FIE_da (xr.DataArray):
#             A 2D xarray DataArray of shape (3, 40) containing
#             the FIE values for each IWP region and PC1 bin.
#         std_errs_da (xr.DataArray):
#             A 2D xarray DataArray of shape (3, 40) containing
#             the standard errors of the FIE values for each IWP region and PC1 bin.
#         ylims (list): List containing two elements representing the y-axis limits. Default is [-0.5, 0.5].
#         colors (list): List of colors to be used for the lines. Each color corresponds to an IWP region. Default is None.

#     Returns:
#         None: The function displays the plot using matplotlib.pyplot.show().

#     Raises:
#         ValueError: If the input arrays are not of shape (3, 40).
#         ValueError: If the length of colors list does not match the number of IWP regions.

#     Notes:
#         The function plots the fraction of aerosol indirect effect (FIE) for each IWP region
#         using the FIE and standard error xarray DataArrays. The x-axis represents the PC1 bin,
#         and the y-axis represents the FIE value. The function displays the plot using matplotlib.pyplot.show().

#     Examples:
#         >>> import xarray as xr
#         >>> import matplotlib.pyplot as plt
#         >>> FIE_da = xr.DataArray([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]], dims=("IWP_region", "PC1_bin"))
#         >>> std_errs_da = xr.DataArray([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04], [0.03, 0.04, 0.05]], dims=("IWP_region", "PC1_bin"))
#         >>> colors = ['red', 'green', 'blue']
#         >>> plot_FIE(FIE_da, std_errs_da, colors=colors)
#     """
#     PC1_bin_values = np.round(PC1_bin_values, 2)

#     # If no colors are provided, use the default matplotlib color cycle
#     if colors is None:
#         colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

#     if len(colors) != len(FIE_da.IWP_region):
#         raise ValueError(
#             "The length of colors list does not match the number of IWP regions."
#         )

#     # Define the PC1 bins as x-axis
#     x = np.arange(
#         1,
#         FIE_da.shape[1] + 1,
#     )

#     # Create a figure and axes
#     fig, ax = plt.subplots(figsize=(8, 5), dpi=400)

#     # # Define the PC1 bins as x-axis
#     # x = np.arange(FIE_da.shape[1])

#     # Plot FIE values for each IWP region
#     for idx, IWP_region in enumerate(FIE_da.IWP_region):
#         ax.errorbar(
#             x,
#             FIE_da.sel(IWP_region=IWP_region),
#             yerr=std_errs_da.sel(IWP_region=IWP_region),
#             fmt="-o",
#             color=colors[idx],  # Use specified color
#             linewidth=2,
#             alpha=0.7,
#             markersize=8,
#             capsize=8,
#             label=f"{IWP_region.values} IWP",
#         )

#     ax.plot(x, np.zeros_like(x), "k", linewidth=5)

#     if PC1_bin_values is not None:
#         ax.set_xticks(x[::n])
#         ax.set_xticklabels(PC1_bin_values[::n])

#     ax.set_ylim(ymin=ylims[0], ymax=ylims[1])
#     ax.set_xlabel("PC1 bin")
#     ax.set_ylabel("FIE")
#     ax.set_title("FIE for each IWP region")
#     ax.legend()

#     plt.show()


# def plot_FIE_diverse_AOD(
#     FIE_da,
#     std_errs_da,
#     AOD_bins,
#     ylims=[-0.5, 0.5],
#     colors=None,
#     PC1_bin_values=None,
#     n=5,
# ):
#     """
#     Plot the fraction of aerosol indirect effect (FIE) for each IWP region.

#     Args:
#         FIE_da (xr.DataArray):
#             A 2D xarray DataArray of shape (3, 40) containing
#             the FIE values for each IWP region and PC1 bin.
#         std_errs_da (xr.DataArray):
#             A 2D xarray DataArray of shape (3, 40) containing
#             the standard errors of the FIE values for each IWP region and PC1 bin.
#         ylims (list): List containing two elements representing the y-axis limits. Default is [-0.5, 0.5].
#         colors (list): List of colors to be used for the lines. Each color corresponds to an IWP region. Default is None.

#     Returns:
#         None: The function displays the plot using matplotlib.pyplot.show().

#     Raises:
#         ValueError: If the input arrays are not of shape (3, 40).
#         ValueError: If the length of colors list does not match the number of IWP regions.

#     Notes:
#         The function plots the fraction of aerosol indirect effect (FIE) for each IWP region
#         using the FIE and standard error xarray DataArrays. The x-axis represents the PC1 bin,
#         and the y-axis represents the FIE value. The function displays the plot using matplotlib.pyplot.show().

#     Examples:
#         >>> import xarray as xr
#         >>> import matplotlib.pyplot as plt
#         >>> FIE_da = xr.DataArray([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]], dims=("IWP_region", "PC1_bin"))
#         >>> std_errs_da = xr.DataArray([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04], [0.03, 0.04, 0.05]], dims=("IWP_region", "PC1_bin"))
#         >>> colors = ['red', 'green', 'blue']
#         >>> plot_FIE(FIE_da, std_errs_da, colors=colors)
#     """
#     FIE_da = FIE_da[AOD_bins]
#     std_errs_da = std_errs_da[AOD_bins]

#     PC1_bin_values = np.round(PC1_bin_values[1:], 2)

#     # If no colors are provided, use the default matplotlib color cycle
#     if colors is None:
#         colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

#     if len(colors) != len(FIE_da.IWP_region):
#         raise ValueError(
#             "The length of colors list does not match the number of IWP regions."
#         )

#     # Define the PC1 bins as x-axis
#     x = np.arange(
#         1,
#         FIE_da.shape[1] + 1,
#     )

#     # Create a figure and axes
#     fig, ax = plt.subplots(figsize=(8, 5), dpi=400)

#     # # Define the PC1 bins as x-axis
#     # x = np.arange(FIE_da.shape[1])

#     # Plot FIE values for each IWP region
#     for idx, IWP_region in enumerate(FIE_da.IWP_region):
#         ax.errorbar(
#             x,
#             FIE_da.sel(IWP_region=IWP_region),
#             yerr=std_errs_da.sel(IWP_region=IWP_region),
#             fmt="-o",
#             color=colors[idx],  # Use specified color
#             linewidth=2,
#             alpha=0.7,
#             markersize=8,
#             capsize=8,
#             label=f"{IWP_region.values} IWP",
#         )

#     ax.plot(x, np.zeros_like(x), "k", linewidth=4.5)

#     if PC1_bin_values is not None:
#         ax.set_xticks(x[::n])
#         ax.set_xticklabels(PC1_bin_values[::n])

#     ax.set_ylim(ymin=ylims[0], ymax=ylims[1])
#     ax.set_xlabel("PC1 bin")
#     ax.set_ylabel("FIE")
#     ax.set_title("FIE for each IWP region")
#     ax.legend()

#     plt.show()
