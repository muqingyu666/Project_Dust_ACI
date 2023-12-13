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
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from muqy_20221026_func_filter_hcf_anormal_data import (
    filter_data_PC1_gap_lowermost_highermost_error as filter_data_PC1_gap_lowermost_highermost_error,
)
from scipy import stats

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
# Pristine filtered data
# Read the Dust constrain HCF data
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_HCF = read_filtered_data_out(
    file_name="CERES_SSF_HCF_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_Dust_AOD_pristine.nc"
)
# Read the Dust constrain cld icerad data
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR = read_filtered_data_out(
    file_name="CERES_SSF_IPR_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_Dust_AOD_pristine.nc"
)
# Read the Dust constrain cld top pressure data
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_CTP = read_filtered_data_out(
    file_name="CERES_SSF_CTP_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_Dust_AOD_pristine.nc"
)
# Read the Dust constrain cld optical depth data
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_COD = read_filtered_data_out(
    file_name="CERES_SSF_COT_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_Dust_AOD_pristine.nc"
)

# Read AOD data
# -------------------------------------
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
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    AOD_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[0]
    IWP_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[1]
    PC_bin = Cld_match_PC_gap_IWP_AOD_constrain_mean.shape[2]

    X, Y = np.meshgrid(range(PC_bin), range(IWP_bin))

    # Add this line after the function's docstring to create a colormap that handles NaN values
    cmap_with_nan = create_colormap_with_nan(cmap)

    # Iterate over the specified AOD range
    for aod_num in range(aod_range[0], aod_range[1]):
        Z = aod_num * np.ones_like(X)

        # Plot the 2D pcolormesh color fill map for the current AOD interval
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

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # Turn off the grid lines
    ax.grid(False)

    # Add color bar
    m = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r)
    m.set_cmap(cmap_with_nan)
    m.set_array(high_cloud_amount_mean)
    m.set_clim(vmin, vmax)
    fig.colorbar(
        m, shrink=0.3, aspect=9, pad=0.01, label=colobar_label
    )

    ax.view_init(elev=10, azim=-65)
    ax.dist = 12

    # Save the figure
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
        (0, 3),
        vmin,  # Add this parameter to define the minimum value for the color scale
        vmax,  # Add this parameter to define the maximum value for the color scale
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
        (3, 6),
        vmin,  # Add this parameter to define the minimum value for the color scale
        vmax,  # Add this parameter to define the maximum value for the color scale
        cmap=cmap,  # Add this parameter to define the custom colormap
    )


# Call the function with different AOD ranges and save each figure separately
# -----------------------------------------------
# Plot the dust-AOD constrained data
# high cloud area
# plot_both_3d_fill_plot_min_max_version(
#     CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_HCF,
#     "Dust-AOD",
#     "PC1",
#     "IWP",
#     "CERES_SSF Cloud Mask",
#     vmin=0,
#     vmax=0.93,
# )

# # ice effective radius
# plot_both_3d_fill_plot_min_max_version(
#     CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR,
#     "Dust-AOD",
#     "PC1",
#     "IWP",
#     "CERES_SSF Cloud Effective Radius (microns)",
#     vmin=12,
#     vmax=33,
# )

# # cloud top pressure
# plot_both_3d_fill_plot_min_max_version(
#     CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_CTP,
#     "Dust-AOD",
#     "PC1",
#     "IWP",
#     "CERES_SSF Cloud Top Pressure (hPa)",
#     vmin=330,
#     vmax=830,
# )

# # cloud optical depth
# plot_both_3d_fill_plot_min_max_version(
#     CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_COD,
#     "Dust-AOD",
#     "PC1",
#     "IWP",
#     "CERES_SSF Cloud Optical Depth",
#     vmin=2,
#     vmax=55,
# )

# -----------------------------------------------------------------------
# Plot diverse IWP regions, mask out the IWP regions we dont need
# -----------------------------------------------------------------------

# Low IWP region
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_low_IWP = (
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR.where(
        CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR["IWP_bin"]
        <= 15,
        drop=True,
    )
)

# Mid IWP region
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_mid_IWP = (
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR.where(
        (
            CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR[
                "IWP_bin"
            ]
            > 15
        )
        & (
            CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR[
                "IWP_bin"
            ]
            < 200
        ),
        drop=True,
    )
)

# High IWP region
CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_high_IWP = (
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR.where(
        CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR["IWP_bin"]
        >= 200,
        drop=True,
    )
)

# Plot low IWP region 3D constrain layout
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_low_IWP,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CERES_SSF Cloud Effective Radius (microns)",
    vmin=12,
    vmax=33,
)

# Plot mid IWP region 3D constrain layout
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_mid_IWP,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CERES_SSF Cloud Effective Radius (microns)",
    vmin=12,
    vmax=33,
)

# Plot high IWP region 3D constrain layout
plot_both_3d_fill_plot_min_max_version(
    CERES_SSF_match_PC_gap_IWP_AOD_constrain_mean_Dust_IPR_masked_high_IWP,
    "Dust-AOD",
    "PC1",
    "IWP",
    "CERES_SSF Cloud Effective Radius (microns)",
    vmin=12,
    vmax=33,
)

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Calculate the ACI for the different IWP regions
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# Read the non AOD constrained cld data
CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR = read_filtered_data_out(
    file_name="CERES_SSF_IPR_match_PC_gap_IWP_constrain_mean_2010_2020_Dust_AOD_pristine.nc"
)
# Read the same mask constrained AOD data
# In order to calculate the ACI
AOD_match_PC_gap_IWP_constrain_mean = read_filtered_data_out(
    file_name="AOD_match_PC_gap_IWP_constrain_mean_2010_2020_Dust_AOD_pristine.nc"
)
# Those dataarrays are all shape in (40,40,180,360), without AOD constrain
# Because the AOD are needed to calculate the ACI
# The first dimension is the IWP bin, the second dimension is the PC1 bin


# Now we need to calculate the ACI for each IWP region and each PC1 condition within


def ACI_calculator(
    input_cld_data: np.ndarray, input_aer_data: np.ndarray
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
    # Create containers to hold the FIE results, the p-values, and the standard errors
    FIE = np.zeros((3, input_cld_data.shape[1]))
    p_values = np.zeros((3, input_cld_data.shape[1]))
    std_errs = np.zeros((3, input_cld_data.shape[1]))

    # Separate the data into IWP regions
    IWP_regions = [
        (input_cld_data["IWP_bin"] <= 15),
        (
            (input_cld_data["IWP_bin"] > 15)
            & (input_cld_data["IWP_bin"] < 200)
        ),
        (input_cld_data["IWP_bin"] >= 200),
    ]

    for idx, IWP_region in enumerate(IWP_regions):
        # Mask the data for the current IWP region
        cld_data_region = input_cld_data.where(IWP_region, drop=True)
        aer_data_region = input_aer_data.where(IWP_region, drop=True)

        # Iterate over the PC1 bins
        for i in range(cld_data_region.shape[1]):
            cld_bin = cld_data_region[:, i, :, :].values.flatten()
            aer_bin = aer_data_region[:, i, :, :].values.flatten()

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
                len(cld_bin) > 0 and len(aer_bin) > 0
            ):  # Avoid cases with no valid data
                # Calculate the FIE using linear regression
                slope, intercept, _, p_value, _ = stats.linregress(
                    np.log(aer_bin), np.log(cld_bin)
                )

                FIE_region_bin = (
                    -slope
                )  # FIE = -(d ln⁡〖N〗)/(d ln⁡〖AOD〗)
                p_values_region_bin = p_value

                # Calculate residuals and standard error of the slope
                predicted = intercept + slope * np.log(aer_bin)
                residuals = np.log(cld_bin) - predicted
                s_err = np.sqrt(
                    np.sum(residuals**2) / (len(np.log(aer_bin)) - 2)
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
                FIE[idx, i] = FIE_region_bin
                p_values[idx, i] = p_values_region_bin
                std_errs[idx, i] = slope_std_err
            else:
                continue  # Skip this bin if there's no data after masking

    # Create xarray DataArrays for the results, with the first dimension being the IWP regions
    IWP_regions_coords = ["low", "mid", "high"]
    PC1_coords = list(range(input_cld_data.shape[1]))

    FIE_da = xr.DataArray(
        FIE,
        coords=[IWP_regions_coords, PC1_coords],
        dims=["IWP_region", "PC1_bin"],
    )
    p_values_da = xr.DataArray(
        p_values,
        coords=[IWP_regions_coords, PC1_coords],
        dims=["IWP_region", "PC1_bin"],
    )
    std_errs_da = xr.DataArray(
        std_errs,
        coords=[IWP_regions_coords, PC1_coords],
        dims=["IWP_region", "PC1_bin"],
    )

    return FIE_da, p_values_da, std_errs_da


def plot_FIE(FIE_da, std_errs_da):
    """
    Plot the fraction of aerosol indirect effect (FIE) for each IWP region.

    Args:
        FIE_da (xr.DataArray):
            A 2D xarray DataArray of shape (3, 40) containing
            the FIE values for each IWP region and PC1 bin.
        std_errs_da (xr.DataArray):
            A 2D xarray DataArray of shape (3, 40) containing
            the standard errors of the FIE values for each IWP region and PC1 bin.

    Returns:
        None: The function displays the plot using matplotlib.pyplot.show().

    Raises:
        ValueError: If the input arrays are not of shape (3, 40).

    Notes:
        The function plots the fraction of aerosol indirect effect (FIE) for each IWP region
        using the FIE and standard error xarray DataArrays. The x-axis represents the PC1 bin,
        and the y-axis represents the FIE value. The function displays the plot using matplotlib.pyplot.show().

    Examples:
        >>> import xarray as xr
        >>> import matplotlib.pyplot as plt
        >>> FIE_da = xr.DataArray([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]], dims=("IWP_region", "PC1_bin"))
        >>> std_errs_da = xr.DataArray([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04], [0.03, 0.04, 0.05]], dims=("IWP_region", "PC1_bin"))
        >>> plot_FIE(FIE_da, std_errs_da)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define the PC1 bins as x-axis
    x = np.arange(FIE_da.shape[1])

    # Plot FIE values for each IWP region
    for idx, IWP_region in enumerate(FIE_da.IWP_region):
        ax.errorbar(
            x,
            FIE_da.sel(IWP_region=IWP_region),
            yerr=std_errs_da.sel(IWP_region=IWP_region),
            fmt="-o",
            linewidth=2,
            alpha=0.7,
            markersize=8,
            capsize=8,
            label=f"{IWP_region.values} IWP",
        )

    ax.plot(x, np.zeros_like(x), "k", linewidth=5)

    ax.set_ylim(-0.081, 0.081)
    ax.set_xlabel("PC1 bin")
    ax.set_ylabel("FIE")
    ax.set_title("FIE for each IWP region")
    ax.legend()

    plt.show()


# Calculate the ACI for the different IWP regions and each PC1 condition within
FIE_da, p_values_da, std_errs_da = ACI_calculator(
    input_cld_data=CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR,
    input_aer_data=AOD_match_PC_gap_IWP_constrain_mean,
)


plot_FIE(FIE_da, std_errs_da)

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
        a, ax=ax, label="CERES_SSF Cloud Effective Radius (microns)"
    )

    plt.show()


plot_test(CERES_SSF_match_PC_gap_IWP_constrain_mean_IPR)
