# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2023-07-16 13:27:33
# @Last Modified by:   Muqy
# @Last Modified time: 2024-11-08 10:33

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

# read data
ds_PC = xr.open_dataset(
    "/RAID01/data/PC_data/1990_2020_250_hPa_vars_250_300_Instab_PC1_no_antarc.nc"
)

ds_vars = xr.open_dataset(
    "/RAID01/data/ERA5_Var_data/1990_2020_250_hPa_vars_250_300_Instab.nc"
)
ds_vars_0 = xr.open_dataset(
    "/RAID01/data/ERA5_Var_data/1990_2020_300_hPa_vars_250_300_Instab.nc"
)

# read PC1
PC1_data = ds_PC["PC1"]

# read all atms variables
Temperature_250 = ds_vars["T"]
Temperature_300 = ds_vars_0["T"]
RelativeHumidity_250 = ds_vars["RH"]
VerticalVelocity_250 = ds_vars["W"]
Instability_250_300 = ds_vars["Instab"]

# exclude polar region from all atms variables
Temperature_250 = Temperature_250.sel(lat=slice(-60, 90))
Temperature_300 = Temperature_300.sel(lat=slice(-60, 90))
RelativeHumidity_250 = RelativeHumidity_250.sel(lat=slice(-60, 90))
VerticalVelocity_250 = VerticalVelocity_250.sel(lat=slice(-60, 90))
Instability_250_300 = Instability_250_300.sel(lat=slice(-60, 90))

# reshape all data to (time, lat, lon)
PC1_data = PC1_data.values.reshape(-1, 150, 360)
Temperature_250 = Temperature_250.values.reshape(-1, 150, 360)
Temperature_300 = Temperature_300.values.reshape(-1, 150, 360)
RelativeHumidity_250 = RelativeHumidity_250.values.reshape(-1, 150, 360)
VerticalVelocity_250 = VerticalVelocity_250.values.reshape(-1, 150, 360)
Instability_250_300 = Instability_250_300.values.reshape(-1, 150, 360)

##############################################################################
#### Plot atms variables vary with PC1 ######################################
##############################################################################

# Set the start, end, and gap values
start = -2.5
end = 5.5
gap = 0.05

# Initialize the FilterAtmosDataFitPCgap class with lat and lon
filter_atmos_fit_PC1 = FilterAtmosDataFitPCgap(
    start, end, gap, lat=np.arange(150), lon=np.arange(360)
)

# Apply the Filter_data_fit_PC1_gap_new method to your data
RelativeH_filtered = filter_atmos_fit_PC1.Filter_data_fit_PC1_gap_new(
    Atms_data=RelativeHumidity_250,
    PC_data=PC1_data,
)
Temperature_filtered = filter_atmos_fit_PC1.Filter_data_fit_PC1_gap_new(
    Atms_data=Temperature_300,
    PC_data=PC1_data,
)
Wvelocity_filtered = filter_atmos_fit_PC1.Filter_data_fit_PC1_gap_new(
    Atms_data=VerticalVelocity_250,
    PC_data=PC1_data,
)
Stability_filtered = filter_atmos_fit_PC1.Filter_data_fit_PC1_gap_new(
    Atms_data=Instability_250_300,
    PC_data=PC1_data,
)

# --------------------------------------------------------------------------- #
# ------------------- Plot each var within each PC1 bin ------------------- #
# --------------------------------------------------------------------------- #


def plot_error_fill_between(datasets, var_names, PC1_bin):
    """
    Plot an error fill_between plot.

    Parameters
    ----------
    datasets : list of numpy.array
        List of filtered atmospheric datasets
    var_names : list of str
        List of variable names
    PC1_bin : numpy.array
        The PC1 bin values for x-axis
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 7), sharex=True
    )  # Set up a 2x2 grid of subplots
    axes = axes.ravel()  # Flatten axes for easy iteration
    labels = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
    ]  # Labels for each subplot
    colors = [
        "#0071C2",
        "#D75615",
        "#EDB11A",
        "#7E318A",
    ]  # Define colors for each line

    handles = []  # Place holder for legend handles

    for i, (ax, data, var_name, label, color) in enumerate(
        zip(axes, datasets, var_names, labels, colors)
    ):
        # Calculate mean and standard error
        mean = np.mean(data, axis=(1, 2))
        std_err = np.std(data, axis=(1, 2)) / np.sqrt(
            np.prod(data.shape[1:])
        )

        # Create plot
        (line,) = ax.plot(PC1_bin, mean, label=var_name, color=color)
        ax.fill_between(
            PC1_bin,
            mean - std_err,
            mean + std_err,
            color=color,
            alpha=0.2,
        )
        handles.append(line)  # Append line to legend handles
        if i >= 2:  # Only set x-label for bottom two subplots
            ax.set_xlabel("PC1 bin")
        ax.set_ylabel(var_name)
        ax.text(
            -0.08,
            0.99,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
        )  # Add label in the upper left corner

    fig.legend(
        handles, var_names, loc="lower center", ncol=4
    )  # Add figure legend
    plt.tight_layout()
    plt.subplots_adjust(
        bottom=0.12
    )  # Adjust subplot to make space for legend
    plt.show()


# Define PC1 bin and variable names
PC1_bin = np.linspace(-2.5, 5.5, 160)
var_names = [
    "Relative Humidity (%)",
    "Temperature (K)",
    "Vertical Velocity (m/s)",
    "Instability",
]

# Call the plotting function
plot_error_fill_between(
    [
        RelativeH_filtered,
        Temperature_filtered,
        Wvelocity_filtered,
        Stability_filtered,
    ],
    var_names,
    PC1_bin,
)


# --------------------------------------------------------------------------- #
# ------------------- Plot each var within each PC1 bin ------------------- #
# --------------------------------------------------------------------------- #


def plot_error_fill_between(datasets, var_names, PC1_bin):
    """
    Plot an error fill_between plot.

    Parameters
    ----------
    datasets : list of numpy.array
        List of filtered atmospheric datasets
    var_names : list of str
        List of variable names
    PC1_bin : numpy.array
        The PC1 bin values for x-axis
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 7), sharex=True
    )  # Set up a 2x2 grid of subplots
    axes = axes.ravel()  # Flatten axes for easy iteration
    labels = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
    ]  # Labels for each subplot
    colors = [
        "#0071C2",
        "#D75615",
        "#EDB11A",
        "#7E318A",
    ]  # Define colors for each line

    handles = []  # Place holder for legend handles

    for i, (ax, data, var_name, label, color) in enumerate(
        zip(axes, datasets, var_names, labels, colors)
    ):
        # Calculate mean and standard error
        mean = np.nanmean(data, axis=(1, 2))
        # std_err = np.nanstd(data, axis=(1, 2)) / np.sqrt(
        #     np.nanprod(data.shape[1:])
        # )
        std_err = np.nanstd(data, axis=(1, 2))

        # Create plot
        (line,) = ax.plot(PC1_bin, mean, label=var_name, color=color)
        ax.fill_between(
            PC1_bin,
            mean - std_err,
            mean + std_err,
            color=color,
            alpha=0.2,
        )
        handles.append(line)  # Append line to legend handles
        if i >= 2:  # Only set x-label for bottom two subplots
            ax.set_xlabel("PC1 bin")
        ax.set_ylabel(var_name)
        ax.text(
            -0.08,
            0.99,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
        )  # Add label in the upper left corner

    fig.legend(
        handles, var_names, loc="lower center", ncol=4
    )  # Add figure legend
    plt.tight_layout()
    plt.subplots_adjust(
        bottom=0.12
    )  # Adjust subplot to make space for legend
    plt.show()


# Define PC1 bin and variable names
PC1_bin = np.linspace(-2.5, 5.5, 160)
var_names = [
    "Relative Humidity (%)",
    "Temperature (K)",
    "Vertical Velocity (m/s)",
    "Instability",
]

# Call the plotting function
plot_error_fill_between(
    [
        RelativeH_filtered,
        Temperature_filtered,
        Wvelocity_filtered,
        Stability_filtered,
    ],
    var_names,
    PC1_bin,
)
