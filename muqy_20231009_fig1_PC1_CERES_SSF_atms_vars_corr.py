# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2023-10-16 13:35:59
# @Last Modified by:   Muqy
# @Last Modified time: 2024-02-16 09:40:51

"""

    Code to check the correlation between PC1 and atmospheric variables
    
    Owner: Mu Qingyu
    version 1.0
    Created: 2022-12-08
    
    Including the following parts:
        
        1) Read the PC1 and atmospheric variables data
        
        2) Plot the correlation between PC1 and atmospheric variables
        
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from muqy_20220413_util_useful_functions import dcmap as dcmap
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from scipy import stats

# --------- import done ------------
# --------- Plot style -------------
# Set parameter to avoid warning
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.style.use("seaborn-v0_8-ticks")
mpl.rc("font", family="Times New Roman")


# 1) Read the PC1 and atmospheric variables data


def open_and_reshape_netcdf(
    path: str, variables: list, shape: tuple
) -> list:
    """
    Opens a netCDF file at the given path and reshapes the specified variables to the given shape.

    Args:
        path (str): The path to the netCDF file.
        variables (list): A list of variable names to reshape.
        shape (tuple): The desired shape of the variables.

    Returns:
        list: A list of the reshaped variables.
    """
    data = xr.open_dataset(path)
    reshaped_data = [
        data[var].values.reshape(shape) for var in variables
    ]
    return reshaped_data


def flatten_data(data):
    """
    Reshapes the input data array to a 3D array with shape (-1, 180, 360).
    Only the last 16 elements of the input array are used.
    Which means the last 16 years, 2005-2020.

    Args:
    data: numpy array of shape (n,) where n >= 16

    Returns:
    numpy array of shape (-1, 180, 360)
    """
    data = data[-16:]
    return data.reshape(-1, 180, 360)


def get_correlation(PC_data, Cld_data):
    """
    Calculates the Pearson correlation coefficient and p-value between two arrays of data.

    Parameters:
    PC_data (array-like): The first array of data.
    Cld_data (array-like): The second array of data.

    Returns:
    correlation (float): The Pearson correlation coefficient between the two arrays.
    p_value (float): The two-sided p-value for a hypothesis test whose null hypothesis is that the correlation coefficient is zero.
    """
    valid_indices = np.isfinite(PC_data) & np.isfinite(Cld_data)
    valid_PC = PC_data[valid_indices]
    valid_Cld = Cld_data[valid_indices]
    correlation, p_value = stats.pearsonr(valid_PC, valid_Cld)
    print("Correlation: ", correlation)
    print("P_value: ", p_value)
    return correlation, p_value


def common_correlation_calculation(
    valid_indices, PC_data, Cld_data, i, j
):
    """
    Calculates the Pearson correlation coefficient between the PC data and cloud data
    for the given valid indices, i, and j.

    Parameters:
    valid_indices (array-like): Indices of valid data points
    PC_data (array-like): Principal component data
    Cld_data (array-like): Cloud data
    i (int): Index of the first dimension of the data arrays
    j (int): Index of the second dimension of the data arrays

    Returns:
    tuple: Pearson correlation coefficient and p-value
    """
    return stats.pearsonr(
        PC_data[valid_indices, i, j],
        Cld_data[valid_indices, i, j],
    )


def get_correlation_spatial(PC_data, Cld_data):
    """
    Calculates the spatial correlation and p-value between the PC data and cloud data.

    Parameters:
    PC_data (numpy.ndarray): A 3D numpy array of PC data.
    Cld_data (numpy.ndarray): A 3D numpy array of cloud data.

    Returns:
    Correlation (numpy.ndarray): A 2D numpy array of the spatial correlation between the PC data and cloud data.
    P_value (numpy.ndarray): A 2D numpy array of the p-value of the correlation between the PC data and cloud data.
    """
    shape_1, shape_2 = Cld_data.shape[1], Cld_data.shape[2]
    Correlation = np.zeros((shape_1, shape_2))
    P_value = np.zeros((shape_1, shape_2))

    for i in range(shape_1):
        for j in range(shape_2):
            valid_indices = np.isfinite(
                PC_data[:, i, j]
            ) & np.isfinite(Cld_data[:, i, j])
            if np.count_nonzero(valid_indices) == 0:
                Correlation[i, j] = np.nan
                P_value[i, j] = np.nan
                continue

            (
                Correlation[i, j],
                P_value[i, j],
            ) = common_correlation_calculation(
                valid_indices, PC_data, Cld_data, i, j
            )

    return Correlation, P_value


def get_correlation_spatial_mean(PC_data, Cld_data):
    """
    Calculates the spatial mean of the correlation and p-value between two datasets.

    Args:
    - PC_data (numpy.ndarray): The first dataset.
    - Cld_data (numpy.ndarray): The second dataset.

    Returns:
    - Correlation (float): The spatial mean of the correlation between the two datasets.
    - P_value (float): The spatial mean of the p-value between the two datasets.
    """
    Correlation, P_value = get_correlation_spatial(PC_data, Cld_data)

    print("Correlation spatially mean: ", np.nanmean(Correlation))
    print("P_value: ", np.nanmean(P_value))

    return np.nanmean(Correlation), np.nanmean(P_value)


pressure_levels = [
    "150",
    "175",
    "200",
    "225",
    "250",
    "300",
    "350",
    "400",
]

base_path = "../Data_python/ERA5_var_data/"

# get the xarray dataset
atms_4vars = xr.open_dataset(
    base_path + "1990_2020_250_hPa_vars_250_300_Instab.nc"
)

# extract each vars
T = atms_4vars["T"].values[-16:].reshape(-1, 180, 360)
RH = atms_4vars["RH"].values[-16:].reshape(-1, 180, 360)
W = atms_4vars["W"].values[-16:].reshape(-1, 180, 360)
Instab = atms_4vars["Instab"].values[-16:].reshape(-1, 180, 360)


##########################################################################################
# --------------------------------------------------------------- #
# ------ Read in CERES SSF data and filter anormal data --------- #
# --------------------------------------------------------------- #
# read in CERES SSF data

# region
# Specify the output path
CERES_SSF_28day = xr.open_dataset(
    "../Data_python/Cld_data/CERES_SSF_data_2005_2020_28_days.nc"
)

CERES_SSF_HCF = CERES_SSF_28day[
    "cldarea_high_daynight_daily"
].values.reshape(-1, 180, 360)

# read in PC1 data
PC_best_corr_spatial = read_PC1_clean(
    PC_path="../Data_python/PC_data/1990_2020_300hPa_250_300_instab_PC1.nc"
)

# extract the last 11 years of the data
PC_best_corr_spatial = PC_best_corr_spatial.reshape(
    31, 12, 28, 180, 360
)[-16:].reshape(-1, 180, 360)

# endregion

##############################################################################
######## Var - PC1 correlation test ########################################
##############################################################################


def calculate_correlations(HCF_data, variables):
    """
    Calculates the standard correlation, spatial mean correlation, and spatial correlation
    between the given HCF data and the specified variables.

    Args:
        HCF_data (pandas.DataFrame): The HCF data to calculate correlations with.
        variables (dict): A dictionary of variable names and their corresponding data.

    Returns:
        dict: A dictionary containing the calculated correlations for each variable.
    """
    results = {}
    for var_name, var_data in variables.items():
        # Calculate standard correlation
        results[f"corr_{var_name}_HCF"], _ = get_correlation(
            HCF_data, var_data
        )

        # Calculate spatial mean correlation
        (
            results[f"corr_{var_name}_HCF_spatial_mean"],
            _,
        ) = get_correlation_spatial_mean(HCF_data, var_data)

        # Calculate spatial correlation
        (
            results[f"corr_{var_name}_HCF_spatial"],
            _,
        ) = get_correlation_spatial(HCF_data, var_data)
    return results


variables = {
    "PC_best_corr_spatial": PC_best_corr_spatial,
    # "T": T,
    # "RH": RH,
    # "W": W,
    # "Instab": Instab,
}

results = calculate_correlations(CERES_SSF_HCF, variables)

##############################################################################
# Convert the dictionary to a DataFrame. The keys in the inner dictionaries will be used as column names,
# and the keys in the outer dictionary will be used as row labels


def plot_full_hemisphere_self_cmap(
    data,
    min,
    max,
    cb_label,
    cmap_file="../Color_python/test.txt",
):
    """
    Plot the data on the full hemisphere

    Parameters
    ----------
    data : numpy.ndarray
        The data to be plotted
    min : float
        The minimum value of the data
    max : float
        The maximum value of the data
    cb_label : str
        The label of the colorbar
    cmap_file : str, optional
        The path of the color map file, by default "../Color_python/test.txt"
    """
    # set the font
    plt.rcParams.update({"font.family": "Times New Roman"})

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    cmap.set_under((27 / 255, 44 / 255, 98 / 255))

    fig = plt.figure(figsize=(12, 7), dpi=400)

    ax1 = plt.subplot(
        111,
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    b = ax1.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=min,
        vmax=max,
    )
    ax1.coastlines(resolution="50m", lw=0.9)

    gl = ax1.gridlines(
        linestyle="-.", lw=0.2, alpha=0.5, draw_labels=True
    )
    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    cb = fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.65,
        extend="both",
    )
    cb.set_label(label=cb_label, size=24)
    cb.ax.tick_params(labelsize=24)

    cb.ax.tick_params(labelsize=24)


# ---------------------------------------------------------------------------- #
# ------------------- Plot the correlation ------------------- #
# ---------------------------------------------------------------------------- #

plot_full_hemisphere_self_cmap(
    data=results["corr_PC_best_corr_spatial_HCF_spatial"],
    min=-1,
    max=1,
    cb_label="Correlation Coef",
)

plot_full_hemisphere_self_cmap(
    data=results["corr_T_HCF_spatial"],
    min=-1,
    max=1,
    cb_label="Correlation Coef",
)

plot_full_hemisphere_self_cmap(
    data=results["corr_RH_HCF_spatial"],
    min=-1,
    max=1,
    cb_label="Correlation Coef",
)

plot_full_hemisphere_self_cmap(
    data=results["corr_W_HCF_spatial"],
    min=-1,
    max=1,
    cb_label="Correlation Coef",
)

plot_full_hemisphere_self_cmap(
    data=results["corr_Instab_HCF_spatial"],
    min=-1,
    max=1,
    cb_label="Correlation Coef",
)
