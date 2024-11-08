# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2023-10-24 22:16:04
# @Last Modified by:   Muqy
# @Last Modified time: 2024-09-29 10:45

# --------------------------------------------------------------- #
# import modules

from typing import Union

import matplotlib as mpl
from matplotlib.colors import ListedColormap
import numpy as np
import xarray as xr

# ----------  importing dcmap from my util ----------#
from muqy_20220413_util_useful_functions import dcmap as dcmap
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

# ---------- Read PCA&CLD data from netcdf file --------

# 1990_2020_250_hPa_vars_250_300_Instab_PC1_no_antarc
PC_path="/RAID01/data/PC_data/1990_2020_250_hPa_vars_250_300_Instab_PC1_no_antarc.nc"

# Data shape is (372, 28, 180, 360), which means month, day, lat, lon
# Lets reshape it to (31, 12, 28, 180, 360), which means year, month, day, lat, lon
# Aka 1990-2020, 31 years, 12 months, 28 days, 180 lat, 360 lon
PC_data = xr.open_dataset(PC_path)["PC1"].values.reshape(31, 12, 28, 150, 360)

# Only use the data from 2005 to 2020
PC_data = PC_data[-16:, :, :, :, :].reshape(-1, 150, 360)


# --------------------------------------------------------------- #
# ------ Read in CERES SSF data and filter anormal data --------- #
# --------------------------------------------------------------- #
# read in CERES SSF data

# Specify the output path
CERES_SSF_28day = xr.open_dataset(
    "/RAID01/data/Cld_data/CERES_SSF_Aqua_data_2003_2020_28_days.nc"
)

CERES_SSF_HCF = CERES_SSF_28day[
    "cldarea_high_daynight_daily"
].values.reshape(18, 12, 28, 180, 360)[-16:].reshape(-1, 180, 360)
CERES_SSF_CEP = CERES_SSF_28day[
    "cldpress_eff_high_daynight_daily"
].values.reshape(18,12,28, 180, 360)[-16:].reshape(-1, 180, 360)

CERES_SSF_IPR_37 = CERES_SSF_28day[
    "cldicerad37_high_daynight_daily"
].values.reshape(18,12,28, 180, 360)[-16:].reshape(-1, 180, 360)
CERES_SSF_COD = CERES_SSF_28day[
    "cldtau_high_daynight_daily"
].values.reshape(18,12,28, 180, 360)[-16:].reshape(-1, 180, 360)

CERES_SSF_IWP = CERES_SSF_28day[
    "iwp37_high_daynight_daily"
].values.reshape(18, 12, 28, 180, 360)[-16:].reshape(-1, 180, 360)

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
    threshold=1, data_array=CERES_SSF_IWP
)

# delete antarctica region
Dust_AOD_filtered = Dust_AOD_filtered[:, 30:, :]
IWP_data_filtered = IWP_data_filtered[:, 30:, :]
CERES_SSF_HCF = CERES_SSF_HCF[:, 30:, :]
CERES_SSF_IPR_37 = CERES_SSF_IPR_37[:, 30:, :]
CERES_SSF_IWP = CERES_SSF_IWP[:, 30:, :]
CERES_SSF_CEP = CERES_SSF_CEP[:, 30:, :]
CERES_SSF_COD = CERES_SSF_COD[:, 30:, :]

#########################################
##### start seperate time test ##########
#########################################

# ------ Segmentation of cloud data within each PC interval ---------------------------------
### triout for IWP constrain the same time with PC1 gap constrain ####
# first we need to divide IWP data into n intervals
# what i think is 6 parts of IWP, 0-1, 1-2, 2-3, 3-4, 4-5, 5-6
# Filtered version of IWP data and AOD data
# ------------------ Divide IWP data into 6 intervals ------------------
# divideIWP = DividePCByDataVolume(
#     dataarray_main=IWP_data_filtered,
#     n=10,
# )
# IWP_gap = divideIWP.main_gap()

IWP_gap = np.array([0.5, 5, 50, 100])

# Filter the data by AOD gap
divideAOD = DividePCByDataVolume(
    dataarray_main=Dust_AOD_filtered,
    n=3,
)
AOD_gap = divideAOD.main_gap()

PC1_gap = np.array([-3, 0, 3, 6])

filter_data_fit_PC1_gap_IWP_constrain = (
    Filter_data_fit_PC1_gap_plot_IWP_AOD_constrain(
        start=-3,
        end=6,
        gap=0.06,
        lat=np.arange(150),
        lon=np.arange(360),
    )
)

# empty array for storing the data
HCF_match_PC_gap_IWP_constrain_mean = np.empty((4, 150, 150, 360))
HCF_ice_match_PC_gap_IWP_constrain_mean = np.empty(
    (4, 150, 150, 360)
)
IPR_match_PC_gap_IWP_constrain_mean = np.empty((4, 150, 150, 360))
CTP_match_PC_gap_IWP_constrain_mean = np.empty((4, 150, 150, 360))
COD_match_PC_gap_IWP_constrain_mean = np.empty((4, 150, 150, 360))
CEP_match_PC_gap_IWP_constrain_mean = np.empty((4, 150, 150, 360))

HCF_match_PC_gap_AOD_constrain_mean = np.empty((4, 150, 150, 360))
HCF_ice_match_PC_gap_AOD_constrain_mean = np.empty(
    (4, 150, 150, 360)
)
IPR_match_PC_gap_AOD_constrain_mean = np.empty((4, 150, 150, 360))
CTP_match_PC_gap_AOD_constrain_mean = np.empty((4, 150, 150, 360))
COD_match_PC_gap_AOD_constrain_mean = np.empty((4, 150, 150, 360))
CEP_match_PC_gap_AOD_constrain_mean = np.empty((4, 150, 150, 360))

# Main loop for the data
for i in range(0, len(IWP_gap) - 1):
    # HCF data consttrained by IWP and AOD gap
    print("IWP gap: ", IWP_gap[i], " to ", IWP_gap[i + 1])
    HCF_match_PC_gap_IWP_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_IWP(
        Cld_data=CERES_SSF_HCF.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        IWP_data=IWP_data_filtered.reshape(-1, 150, 360),
        IWP_min=IWP_gap[i],
        IWP_max=IWP_gap[i + 1],
    )

    print("AOD gap: ", AOD_gap[i], " to ", AOD_gap[i + 1])
    HCF_match_PC_gap_AOD_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_AOD(
        Cld_data=CERES_SSF_HCF.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        AOD_data=Dust_AOD_filtered.reshape(-1, 150, 360),
        AOD_min=AOD_gap[i],
        AOD_max=AOD_gap[i + 1],
    )

    # IPR data consttrained by IWP and AOD gap
    print("IWP gap: ", IWP_gap[i], " to ", IWP_gap[i + 1])
    IPR_match_PC_gap_IWP_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_IWP(
        Cld_data=CERES_SSF_IPR_37.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        IWP_data=IWP_data_filtered.reshape(-1, 150, 360),
        IWP_min=IWP_gap[i],
        IWP_max=IWP_gap[i + 1],
    )

    print("AOD gap: ", AOD_gap[i], " to ", AOD_gap[i + 1])
    IPR_match_PC_gap_AOD_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_AOD(
        Cld_data=CERES_SSF_IPR_37.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        AOD_data=Dust_AOD_filtered.reshape(-1, 150, 360),
        AOD_min=AOD_gap[i],
        AOD_max=AOD_gap[i + 1],
    )

    # CEP data consttrained by IWP and AOD gap
    print("IWP gap: ", IWP_gap[i], " to ", IWP_gap[i + 1])
    CEP_match_PC_gap_IWP_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_IWP(
        Cld_data=CERES_SSF_CEP.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        IWP_data=IWP_data_filtered.reshape(-1, 150, 360),
        IWP_min=IWP_gap[i],
        IWP_max=IWP_gap[i + 1],
    )

    print("AOD gap: ", AOD_gap[i], " to ", AOD_gap[i + 1])
    CEP_match_PC_gap_AOD_constrain_mean[
        i
    ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_AOD(
        Cld_data=CERES_SSF_CEP.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        AOD_data=Dust_AOD_filtered.reshape(-1, 150, 360),
        AOD_min=AOD_gap[i],
        AOD_max=AOD_gap[i + 1],
    )

    # # COD data consttrained by IWP and AOD gap
    # print("IWP gap: ", IWP_gap[i], " to ", IWP_gap[i + 1])
    # COD_match_PC_gap_IWP_constrain_mean[
    #     i
    # ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_IWP(
    #     Cld_data=CERES_SSF_COD.reshape(-1, 150, 360),
    #     PC_data=PC_all.reshape(-1, 150, 360),
    #     IWP_data=IWP_data_filtered.reshape(-1, 150, 360),
    #     IWP_min=IWP_gap[i],
    #     IWP_max=IWP_gap[i + 1],
    # )

    # print("AOD gap: ", AOD_gap[i], " to ", AOD_gap[i + 1])
    # COD_match_PC_gap_AOD_constrain_mean[
    #     i
    # ] = filter_data_fit_PC1_gap_IWP_constrain.Filter_data_fit_PC1_gap_AOD(
    #     Cld_data=CERES_SSF_COD.reshape(-1, 150, 360),
    #     PC_data=PC_all.reshape(-1, 150, 360),
    #     AOD_data=Dust_AOD_filtered.reshape(-1, 150, 360),
    #     AOD_min=AOD_gap[i],
    #     AOD_max=AOD_gap[i + 1],
    # )


# save all the data to netcdf4 format

# Create a Dataset with DataArrays
ds = xr.Dataset(
    {
        "HCF_IWP_bin": (
            ("IWP_bin", "PC1_bin", "latitude", "longitude"),
            HCF_match_PC_gap_IWP_constrain_mean,
        ),
        "HCF_ice_IWP_bin": (
            ("IWP_bin", "PC1_bin", "latitude", "longitude"),
            HCF_ice_match_PC_gap_IWP_constrain_mean,
        ),
        "IPR_IWP_bin": (
            ("IWP_bin", "PC1_bin", "latitude", "longitude"),
            IPR_match_PC_gap_IWP_constrain_mean,
        ),
        "CTP_IWP_bin": (
            ("IWP_bin", "PC1_bin", "latitude", "longitude"),
            CTP_match_PC_gap_IWP_constrain_mean,
        ),
        "HCF_AOD_bin": (
            ("AOD_bin", "PC1_bin", "latitude", "longitude"),
            HCF_match_PC_gap_AOD_constrain_mean,
        ),
        "HCF_ice_AOD_bin": (
            ("AOD_bin", "PC1_bin", "latitude", "longitude"),
            HCF_ice_match_PC_gap_AOD_constrain_mean,
        ),
        "IPR_AOD_bin": (
            ("AOD_bin", "PC1_bin", "latitude", "longitude"),
            IPR_match_PC_gap_AOD_constrain_mean,
        ),
        "CTP_AOD_bin": (
            ("AOD_bin", "PC1_bin", "latitude", "longitude"),
            CTP_match_PC_gap_AOD_constrain_mean,
        ),
        "CEP_IWP_bin": (
            ("IWP_bin", "PC1_bin", "latitude", "longitude"),
            COD_match_PC_gap_IWP_constrain_mean,
        ),
        "CEP_AOD_bin": (
            ("AOD_bin", "PC1_bin", "latitude", "longitude"),
            CEP_match_PC_gap_AOD_constrain_mean,
        ),
        "COD_IWP_bin": (
            ("IWP_bin", "PC1_bin", "latitude", "longitude"),
            COD_match_PC_gap_IWP_constrain_mean,
        ),
        "COD_AOD_bin": (
            ("AOD_bin", "PC1_bin", "latitude", "longitude"),
            COD_match_PC_gap_AOD_constrain_mean,
        ),
    },
)

# Save to a netCDF file
ds.to_netcdf(
    "/RAID01/data/Filtered_data/CERES_SSF_under_diverse_IWP_AOD_conditions_filtered_1_percent_no_antarctica_250hPa_PC1_4_gaps.nc"
    # "/RAID01/data/Filtered_data/CERES_SSF_under_diverse_IWP_AOD_conditions_filtered_1_percent_no_antarctica_best_corr_spatial.nc"
)


CERES_SSF_cld_under_diverser_IWP_AOD_conditions = xr.open_dataset(
    "/RAID01/data/Filtered_data/CERES_SSF_under_diverse_IWP_AOD_conditions_filtered_1_percent_no_antarctica_250hPa_PC1_4_gaps.nc"
)

# Extract the data
HCF_match_PC_gap_IWP_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "HCF_IWP_bin"
    ].values
)
IPR_match_PC_gap_IWP_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "IPR_IWP_bin"
    ].values
)
CEP_match_PC_gap_IWP_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "CEP_IWP_bin"
    ].values
)
HCF_match_PC_gap_AOD_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "HCF_AOD_bin"
    ].values
)
IPR_match_PC_gap_AOD_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "IPR_AOD_bin"
    ].values
)
CEP_match_PC_gap_AOD_constrain_mean = (
    CERES_SSF_cld_under_diverser_IWP_AOD_conditions[
        "CEP_AOD_bin"
    ].values
)

# ---------------------------------------------------------------
# Data shape will be (4, 150) means 4 IWP/DAOD gaps and 150 PC1 bins
HCF_match_PC_gap_IWP_constrain_mean_curvs = np.nanmean(HCF_match_PC_gap_IWP_constrain_mean, axis = (2,3))
IPR_match_PC_gap_IWP_constrain_mean_curvs = np.nanmean(IPR_match_PC_gap_IWP_constrain_mean, axis = (2,3))
CEP_match_PC_gap_IWP_constrain_mean_curvs = np.nanmean(CEP_match_PC_gap_IWP_constrain_mean, axis = (2,3))
HCF_match_PC_gap_AOD_constrain_mean_curvs = np.nanmean(HCF_match_PC_gap_AOD_constrain_mean, axis = (2,3))
IPR_match_PC_gap_AOD_constrain_mean_curvs = np.nanmean(IPR_match_PC_gap_AOD_constrain_mean, axis = (2,3))
CEP_match_PC_gap_AOD_constrain_mean_curvs = np.nanmean(CEP_match_PC_gap_AOD_constrain_mean, axis = (2,3))

# Calculate the standard deviation, std array shape will be (4, 150), same as the mean array
HCF_match_PC_gap_IWP_constrain_mean_std = np.nanstd(HCF_match_PC_gap_IWP_constrain_mean, axis=(2, 3))
IPR_match_PC_gap_IWP_constrain_mean_std = np.nanstd(IPR_match_PC_gap_IWP_constrain_mean, axis=(2, 3))
CEP_match_PC_gap_IWP_constrain_mean_std = np.nanstd(CEP_match_PC_gap_IWP_constrain_mean, axis=(2, 3))
HCF_match_PC_gap_AOD_constrain_mean_std = np.nanstd(HCF_match_PC_gap_AOD_constrain_mean, axis=(2, 3))
IPR_match_PC_gap_AOD_constrain_mean_std = np.nanstd(IPR_match_PC_gap_AOD_constrain_mean, axis=(2, 3))
CEP_match_PC_gap_AOD_constrain_mean_std = np.nanstd(CEP_match_PC_gap_AOD_constrain_mean, axis=(2, 3))

#######################################################################
## Use box plot to quantify the cld distribution within each PC interval
## Under the IWP constrain ####
#######################################################################

# ------------------------------------------------------------
# Plot 3D error fill plot for different IWP conditions
# ------------------------------------------------------------


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


XTICK_LABELS = {-4: -3, -1: 0, 2: 3, 5: 6}


COLORS = [
    (90 / 255, 199 / 255, 183 / 255),
    (0 / 255, 147 / 255, 196 / 255),
    (0 / 255, 105 / 255, 151 / 255),
    (0 / 255, 41 / 255, 84 / 255),
]


def error_fill_3d_plot_no_legend_1f_4_gaps(
    data_curvs,
    data_std,
    xlabel,
    ylabel,
    zlabel,
    IWP_or_AOD_bin_values,
    elev_angle,
    azim_angle,
    savefig_str,
    savefig_path,
    annotations: dict,
    zmin: float = None,
    zmax: float = None,
):
    fig = plt.figure(figsize=(12, 10), dpi=400)
    ax = fig.add_subplot(111, projection="3d")

    for bin in range(data_curvs.shape[0]):
        data_y = np.round(data_curvs[bin], 4)
        data_x = np.round(np.arange(-3, 6, 0.06), 3)
        data_up = data_y + data_std[bin]
        data_down = data_y - data_std[bin]
        iwp_condition = np.ones_like(data_x) * bin

        ax.plot(
            data_x,
            iwp_condition,
            data_y,
            linewidth=2,
            color=COLORS[bin % len(COLORS)],
        )
        ax.add_collection3d(
            plt.fill_between(
                data_x,
                data_down,
                data_up,
                facecolor=COLORS[bin % len(COLORS)],
                alpha=0.38,
            ),
            zs=bin,
            zdir="y",
        )

    ax.grid(
        True,
        color="silver",
        linestyle="-.",
        linewidth=0.3,
        alpha=0.4,
    )
    yticks = np.linspace(
        0, len(IWP_or_AOD_bin_values) - 1, 7, dtype=int
    )
    xticks = list(XTICK_LABELS.keys())

    ax.set(
        yticks=yticks,
        xticks=xticks,
        zlim=(zmin, zmax),
    )
    ax.set_yticklabels(
        [
            "{:.1f}".format(val)
            for val in IWP_or_AOD_bin_values[yticks]
        ]
    )
    ax.set_xticklabels([XTICK_LABELS[val] for val in xticks])
    ax.set_xlim(-4.3, 5.3)

    ax.view_init(elev=elev_angle, azim=azim_angle)
    ax.set_facecolor("none")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # set font size for tick labels
    ax.tick_params(axis="x", which="major", pad=8, labelsize=17)
    ax.tick_params(axis="y", which="major", pad=8, labelsize=17)
    ax.tick_params(axis="z", which="major", pad=8, labelsize=17)

    ax.set_xlabel(xlabel, fontsize=20, labelpad=22)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=22)
    ax.set_zlabel(zlabel, fontsize=20, labelpad=22)

    ax.set_proj_type("persp")
    ax.set_box_aspect((1, 1, 0.8))

    # Annotate figure
    for key, value in annotations.items():
        fig.text(*key, value, va="top", fontsize=27)

    plt.savefig(
        os.path.join(savefig_path, savefig_str),
        dpi=400,
        facecolor="w",
        edgecolor="w",
        # bbox_inches="tight",
    )

    plt.show()


COLORS = [
    (255 / 255, 157 / 255, 0 / 255),
    (255 / 255, 76 / 255, 0 / 255),
    (181 / 255, 0 / 255, 0 / 255),
    (95 / 255, 0 / 255, 0 / 255),
]


def error_fill_3d_plot_no_legend_3f_4_gaps(
    data_list,
    xlabel,
    ylabel,
    zlabel,
    IWP_or_AOD_bin_values,
    elev_angle,
    azim_angle,
    savefig_str,
    savefig_path,
    annotations: dict,
    zmin: float = None,
    zmax: float = None,
):
    fig = plt.figure(figsize=(12, 10), dpi=400)
    ax = fig.add_subplot(111, projection="3d")

    for i, data in enumerate(data_list):
        data = data.reshape(data.shape[0], -1)
        data_y = np.round(np.nanmean(data, axis=1), 3)
        data_x = np.round(np.arange(-3, 6, 0.06), 3)
        data_std = np.nanstd(data, axis=1)
        data_up = data_y + data_std
        data_down = data_y - data_std
        iwp_condition = np.ones_like(data_x) * i

        ax.plot(
            data_x,
            iwp_condition,
            data_y,
            linewidth=2,
            color=COLORS[i % len(COLORS)],
        )
        ax.add_collection3d(
            plt.fill_between(
                data_x,
                data_down,
                data_up,
                facecolor=COLORS[i % len(COLORS)],
                alpha=0.38,
            ),
            zs=i,
            zdir="y",
        )

    ax.grid(
        True,
        color="silver",
        linestyle="-.",
        linewidth=0.3,
        alpha=0.4,
    )
    yticks = np.linspace(
        0, len(IWP_or_AOD_bin_values) - 1, 7, dtype=int
    )
    xticks = list(XTICK_LABELS.keys())

    ax.set(
        yticks=yticks,
        xticks=xticks,
        zlim=(zmin, zmax),
    )
    ax.set_yticklabels(
        [
            "{:.3f}".format(val)
            for val in IWP_or_AOD_bin_values[yticks]
        ]
    )
    ax.set_xticklabels([XTICK_LABELS[val] for val in xticks])
    ax.set_xlim(-4.3, 5.3)

    ax.view_init(elev=elev_angle, azim=azim_angle)
    ax.set_facecolor("none")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # set font size for tick labels
    ax.tick_params(axis="x", which="major", pad=8, labelsize=17)
    ax.tick_params(axis="y", which="major", pad=8, labelsize=17)
    ax.tick_params(axis="z", which="major", pad=8, labelsize=17)

    ax.set_xlabel(xlabel, fontsize=20, labelpad=22)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=22)
    ax.set_zlabel(zlabel, fontsize=20, labelpad=22)

    ax.set_proj_type("persp")
    ax.set_box_aspect((1, 1, 0.8))

    # Annotate figure
    for key, value in annotations.items():
        fig.text(*key, value, va="top", fontsize=27)

    plt.savefig(
        os.path.join(savefig_path, savefig_str),
        dpi=400,
        facecolor="w",
        edgecolor="w",
        # bbox_inches="tight",
    )

    plt.show()


# -------------------------------------------------------------
# ------------------- Plotting --------------------------------
# Call the error_fill_3d_plot function with your data

# set font for mathrm
plt.rcParams["mathtext.fontset"] = "cm"

annotations = {
    (0.15, 0.75): "(A)",
}

savefig_path = (
    "/RAID01/data/python_fig/fill_between_plot_cld_var/"
)
if not os.path.exists(savefig_path):
    os.makedirs(savefig_path)

error_fill_3d_plot_no_legend_1f_4_gaps(
    data_curvs=HCF_match_PC_gap_IWP_constrain_mean_curvs,
    data_std=HCF_match_PC_gap_IWP_constrain_mean_std,
    xlabel="PC1",
    ylabel="IWP (g/m" + r"$^2$)",
    zlabel="Cld Area (%)",
    IWP_or_AOD_bin_values=IWP_gap,
    elev_angle=13,
    azim_angle=22,
    savefig_str="HCF_IWP_constrain_3D",
    savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
    annotations=annotations,
    zmin=-8,
    zmax=85,
)

annotations = {
    (0.15, 0.75): "(B)",
}

# plot the constrained ice particle radiusd
# IWP constrained
error_fill_3d_plot_no_legend_1f_4_gaps(
    data_curvs=IPR_match_PC_gap_IWP_constrain_mean_curvs,
    data_std=IPR_match_PC_gap_IWP_constrain_mean_std,
    xlabel="PC1",
    ylabel="IWP (g/m" + r"$^2$)",
    zlabel="Ice Particle Radius " + r"$(\mu m)$",
    IWP_or_AOD_bin_values=IWP_gap,
    elev_angle=13,
    azim_angle=22,
    savefig_str="HCF_IPR_constrain_3D",
    savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
    annotations=annotations,
    zmin=10,
    zmax=36,
)

annotations = {
    (0.15, 0.75): "(C)",
}

# Plot the constrained cloud effect height
# IWP constrained
error_fill_3d_plot_no_legend_1f_4_gaps(
    data_curvs=CEP_match_PC_gap_IWP_constrain_mean_curvs,
    data_std=CEP_match_PC_gap_IWP_constrain_mean_std,
    xlabel="PC1",
    ylabel="IWP (g/m" + r"$^2$)",
    zlabel="Cloud Effective Pressure (hPa)",
    IWP_or_AOD_bin_values=IWP_gap,
    elev_angle=13,
    azim_angle=22,
    savefig_str="HCF_CEP_constrain_3D",
    savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
    annotations=annotations,
    zmin=175,
    zmax=300,
)

# -------------------------------------------------------------
# ------------------- Plotting --------------------------------
# Call the error_fill_3d_plot function with your data
annotations = {
    (0.15, 0.75): "(D)",
}

error_fill_3d_plot_no_legend_3f_4_gaps(
    data_list=HCF_list_AOD,
    xlabel="PC1",
    ylabel="AOD",
    zlabel="Cld Area (%)",
    IWP_or_AOD_bin_values=AOD_gap,
    elev_angle=13,
    azim_angle=22,
    savefig_str="HCF_AOD_constrain_3D",
    savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
    annotations=annotations,
    zmin=-10,
    zmax=95,
)

# plot the constrained ice particle radiusd
# AOD constrained
annotations = {
    (0.15, 0.75): "(E)",
}

error_fill_3d_plot_no_legend_3f_4_gaps(
    data_list=IPR_list_AOD,
    xlabel="PC1",
    ylabel="AOD",
    zlabel="Ice Particle Radius " + r"$(\mu m)$",
    IWP_or_AOD_bin_values=AOD_gap,
    elev_angle=13,
    azim_angle=22,
    savefig_str="IPR_AOD_constrain_3D",
    savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
    annotations=annotations,
    zmin=14,
    zmax=33,
)

# Plot the constrained cloud effect height
# AOD constrained
annotations = {
    (0.15, 0.75): "(F)",
}

error_fill_3d_plot_no_legend_3f_4_gaps(
    data_list=CEP_list_AOD,
    xlabel="PC1",
    ylabel="AOD",
    zlabel="Cloud Effective Pressure (hPa)",
    IWP_or_AOD_bin_values=AOD_gap,
    elev_angle=13,
    azim_angle=22,
    savefig_str="CEP_AOD_constrain_3D",
    savefig_path="/RAID01/data/python_fig/fill_between_plot_cld_var/",
    annotations=annotations,
    zmin=175,
    zmax=290,
)
