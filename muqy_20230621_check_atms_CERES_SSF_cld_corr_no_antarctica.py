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


def open_and_reshape_netcdf(path, variables, shape):
    data = xr.open_dataset(path)
    reshaped_data = [
        data[var].values.reshape(shape) for var in variables
    ]
    return reshaped_data


def flatten_data(data):
    data = data[-11:]
    return data.reshape(-1, 180, 360)


def get_correlation(PC_data, Cld_data):
    valid_indices = np.isfinite(
        PC_data.flatten()[:]
    ) & np.isfinite(Cld_data.flatten()[:])
    correlation, p_value = stats.pearsonr(
        PC_data.flatten()[valid_indices],
        Cld_data.flatten()[valid_indices],
    )
    print("Correlation: ", correlation)
    print("P_value: ", p_value)
    return correlation, p_value


def get_correlation_spatial(PC_data, Cld_data):
    """
    Calc the correlation between PC1 and Cld data

    Parameters
    ----------
    PC_data : np.array
        PC1 data
    Cld_data : np.array
        Cld data

    Returns
    -------
    Correlation : np.array
        Correlation between PC1 and Cld data
    P_value : np.array
        P_value between PC1 and Cld data
    """

    Correlation = np.empty((Cld_data.shape[1], Cld_data.shape[2]))
    P_value = np.empty((Cld_data.shape[1], Cld_data.shape[2]))

    for i in range(Cld_data.shape[1]):
        for j in range(Cld_data.shape[2]):
            # Only include indices where both PC_data and Cld_data are finite
            valid_indices = np.isfinite(
                PC_data[:, i, j]
            ) & np.isfinite(Cld_data[:, i, j])

            # if a grid point is all NaNs, then skip it and leave it as NaN
            if np.count_nonzero(valid_indices) == 0:
                Correlation[i, j] = np.nan
                P_value[i, j] = np.nan
                continue

            Correlation[i, j], P_value[i, j] = stats.pearsonr(
                PC_data[valid_indices, i, j],
                Cld_data[valid_indices, i, j],
            )

    return Correlation, P_value


def get_correlation_spatial_mean(PC_data, Cld_data):
    """
    Calc the correlation between PC1 and Cld data

    Parameters
    ----------
    PC_data : np.array
        PC1 data
    Cld_data : np.array
        Cld data

    Returns
    -------
    Correlation : np.array
        Correlation between PC1 and Cld data
    P_value : np.array
        P_value between PC1 and Cld data
    """

    Correlation = np.empty((Cld_data.shape[1], Cld_data.shape[2]))
    P_value = np.empty((Cld_data.shape[1], Cld_data.shape[2]))

    for i in range(Cld_data.shape[1]):
        for j in range(Cld_data.shape[2]):
            # Only include indices where both PC_data and Cld_data are finite
            valid_indices = np.isfinite(
                PC_data[:, i, j]
            ) & np.isfinite(Cld_data[:, i, j])

            # if a grid point is all NaNs, then skip it and leave it as NaN
            if np.count_nonzero(valid_indices) == 0:
                Correlation[i, j] = np.nan
                P_value[i, j] = np.nan
                continue

            Correlation[i, j], P_value[i, j] = stats.pearsonr(
                PC_data[valid_indices, i, j],
                Cld_data[valid_indices, i, j],
            )

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
base_path = "/RAID01/data/ERA5_Var_data/2010_2020_"
variables = ["RelativeH", "Temperature", "Wvelocity"]
shape = (31, 12, 28, 180, 360)

reshaped_data = {
    level: [
        flatten_data(data)
        for data in open_and_reshape_netcdf(
            f"{base_path}{level}hPa_3_paras.nc", variables, shape
        )
    ]
    for level in pressure_levels
}

# copy the reshaped_data to reshaped_data_no_antarctica
reshaped_data_no_antarctica = reshaped_data.copy()

# For each pressure level and variable...
for level in reshaped_data:
    for i in range(len(variables)):
        # Get the 3D array
        var_3D = reshaped_data[level][i]

        # Determine the index corresponding to 60S
        ny = var_3D.shape[1]  # number of latitude points
        sixty_s_index = int(
            ny * (30 / 180)
        )  # index for 60S assuming latitude ranges from -90 to 90

        # Slice array to exclude values south of 60S
        var_3D = var_3D[:, sixty_s_index:, :]

        # Update the data in the reshaped_data dictionary
        reshaped_data_no_antarctica[level][i] = var_3D


##########################################################################################
# --------------------------------------------------------------- #
# ------ Read in CERES SSF data and filter anormal data --------- #
# --------------------------------------------------------------- #
# read in CERES SSF data

# region
# Specify the output path
CERES_SSF_28day = xr.open_dataset(
    "/RAID01/data/Cld_data/CERES_SSF_data_2010_2020_28_days.nc"
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

# read in PC1 data
(
    PC_best_corr_spatial,
    Cld_all,
) = read_PC1_CERES_clean(
    PC_path="/RAID01/data/PC_data/1990_2020_best_spatialcorr_var_PC1_no_antarctica_250_300Instab.nc",
    CERES_Cld_dataset_name="Cldicerad",
)

(
    PC_best_corr,
    _,
) = read_PC1_CERES_clean(
    PC_path="/RAID01/data/PC_data/1990_2020_best_corr_var_PC1_no_antarctica_250_300Instab.nc",
    CERES_Cld_dataset_name="Cldicerad",
)

(
    PC_og,
    _,
) = read_PC1_CERES_clean(
    PC_path="/RAID01/data/PC_data/1990_2020_250_hPa_vars_250_300_Instab_PC1_no_antarc.nc",
    CERES_Cld_dataset_name="Cldicerad",
)

# extract the last 11 years of the data
PC_best_corr_spatial = PC_best_corr_spatial.reshape(
    31, 12, 28, 150, 360
)[-11:, :, :, :, :]
PC_best_corr = PC_best_corr.reshape(31, 12, 28, 150, 360)[
    -11:, :, :, :, :
]
PC_og = PC_og.reshape(31, 12, 28, 150, 360)[-11:, :, :, :, :]


# Reshape the PC data into -1, 180, 360
PC_best_corr_spatial = PC_best_corr_spatial.reshape(-1, 150, 360)
PC_best_corr = PC_best_corr.reshape(-1, 150, 360)
PC_og = PC_og.reshape(-1, 150, 360)

# flatten all cld data to (-1,180,360)
HCF_data = CERES_SSF_HCF.reshape(-1, 180, 360)[:, 30:, :]

# endregion

##############################################################################
######## Var - PC1 correlation test ########################################
##############################################################################

# Create an empty dictionary to hold correlation data
corr_data = {}

for level in pressure_levels:
    RH_data, T_data, W_data = reshaped_data_no_antarctica[level]
    print(f"\nCorrelations for {level}hPa data:")

    print("Correlation between RH and Cld:")
    corr_RH_Cld, _ = get_correlation(HCF_data, RH_data)

    print("Correlation between T and Cld:")
    corr_T_Cld, _ = get_correlation(HCF_data, T_data)

    print("Correlation between W and Cld:")
    corr_W_Cld, _ = get_correlation(HCF_data, W_data)

    # Store the correlation data in the dictionary
    corr_data[level] = {
        "Corr_RH_HCF": corr_RH_Cld,
        "Corr_T_HCF": corr_T_Cld,
        "Corr_W_HCF": corr_W_Cld,
    }

# Convert the correlation data to a pandas DataFrame
corr_df_1D_best = pd.DataFrame(corr_data)

# --------------------------------------------------------------- #
# ----- Calculate the correlation between PC1 and Cld ------------ #
# ------------ Spatially mean over the globe --------------------- #
# --------------------------------------------------------------- #

# Create an empty dictionary to hold correlation data
corr_data_spatial = {}

for level in pressure_levels:
    RH_data, T_data, W_data = reshaped_data_no_antarctica[level]
    print(f"\nCorrelations for {level}hPa data:")

    print("Correlation between RH and Cld:")
    corr_RH_Cld, _ = get_correlation_spatial(HCF_data, RH_data)

    print("Correlation between T and Cld:")
    corr_T_Cld, _ = get_correlation_spatial(HCF_data, T_data)

    print("Correlation between W and Cld:")
    corr_W_Cld, _ = get_correlation_spatial(HCF_data, W_data)

    # Store the correlation data in the dictionary
    corr_data_spatial[level] = {
        "Corr_RH_HCF": corr_RH_Cld,
        "Corr_T_HCF": corr_T_Cld,
        "Corr_W_HCF": corr_W_Cld,
    }


# --------------------------------------------------------------- #
# ----- Calculate the correlation between PC1 and Cld ------------ #
# ------------ Spatially over the globe --------------------- #
# --------------------------------------------------------------- #

# Create an empty dictionary to hold correlation data
corr_data = {}

for level in pressure_levels:
    RH_data, T_data, W_data = reshaped_data_no_antarctica[level]
    print(f"\nCorrelations for {level}hPa data:")

    print("Correlation between RH and Cld:")
    corr_RH_Cld, _ = get_correlation_spatial_mean(
        HCF_data, RH_data
    )

    print("Correlation between T and Cld:")
    corr_T_Cld, _ = get_correlation_spatial_mean(HCF_data, T_data)

    print("Correlation between W and Cld:")
    corr_W_Cld, _ = get_correlation_spatial_mean(HCF_data, W_data)

    # Store the correlation data in the dictionary
    corr_data[level] = {
        "Corr_RH_HCF": corr_RH_Cld,
        "Corr_T_HCF": corr_T_Cld,
        "Corr_W_HCF": corr_W_Cld,
    }

# Convert the correlation data to a pandas DataFrame
corr_df_spatial_best = pd.DataFrame(corr_data)

# --------------------------------------------------------------- #
# ----- Calculate the correlation between PC1 and Cld ------------ #
# ------------ 1 Dimensional corr over the globe ----------------- #
# ------------ Spatially mean over the globe --------------------- #
# --------------------------------------------------------------- #

# Calc PC and HCF correlation
corr_PC_HCF_best_corr, _ = get_correlation_spatial(
    HCF_data, PC_best_corr
)
corr_PC_HCF_best_corr_spatial, _ = get_correlation_spatial(
    HCF_data, PC_best_corr_spatial
)
corr_PC_HCF_og, _ = get_correlation_spatial(HCF_data, PC_og)

# Calc PC and HCF correlation
corr_PC_HCF, _ = get_correlation_spatial_mean(
    HCF_data, PC_best_corr
)
corr_PC_HCF_spatial, _ = get_correlation_spatial_mean(
    HCF_data, PC_best_corr_spatial
)
corr_PC_HCF_og, _ = get_correlation_spatial_mean(HCF_data, PC_og)

# Calc PC and HCF correlation OG
corr_PC_HCF_og, _ = get_correlation_spatial(
    HCF_data, PC_og[:, 30:, :]
)


##############################################################################
# Convert the dictionary to a DataFrame. The keys in the inner dictionaries will be used as column names,
# and the keys in the outer dictionary will be used as row labels
corr_df = pd.DataFrame(corr_data)


# 2) Calc and Plot the correlation between PC1 and atmospheric variables
def calc_correlation_pvalue_PC1_Cld(PC_data, Cld_data):
    """
    Calc the correlation between PC1 and Cld data

    Parameters
    ----------
    PC_data : np.array
        PC1 data
    Cld_data : np.array
        Cld data

    Returns
    -------
    Correlation : np.array
        Correlation between PC1 and Cld data
    P_value : np.array
        P_value between PC1 and Cld data
    """

    Correlation = np.empty((Cld_data.shape[1], Cld_data.shape[2]))
    P_value = np.empty((Cld_data.shape[1], Cld_data.shape[2]))

    for i in range(Cld_data.shape[1]):
        for j in range(Cld_data.shape[2]):
            # Only include indices where both PC_data and Cld_data are finite
            valid_indices = np.isfinite(
                PC_data[:, i, j]
            ) & np.isfinite(Cld_data[:, i, j])

            Correlation[i, j], P_value[i, j] = stats.pearsonr(
                PC_data[valid_indices, i, j],
                Cld_data[valid_indices, i, j],
            )

    return Correlation, P_value


def calc_correlation_PC1_Cld_all(
    PC_data,
    Cld_data,
):
    PC_data_flatten = PC_data.reshape(-1)
    Cld_data_flatten = Cld_data.reshape(-1)

    # Only include indices where both PC_data and Cld_data are finite
    valid_indices = np.isfinite(PC_data_flatten[:]) & np.isfinite(
        Cld_data_flatten[:]
    )

    Correlation, P_value = stats.pearsonr(
        PC_data_flatten[valid_indices],
        Cld_data_flatten[valid_indices],
    )

    print("Correlation: ", Correlation)
    print("P_value: ", P_value)

    return Correlation, P_value


def plot_full_hemisphere_self_cmap(
    data,
    min,
    max,
    var_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
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
        The path of the color map file, by default "/RAID01/data/muqy/color/test.txt"
    """
    # set the font
    plt.rcParams.update({"font.family": "Times New Roman"})

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    # set color using RGB values
    cmap.set_under((27 / 255, 44 / 255, 98 / 255))

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(11, 7),
        constrained_layout=True,
    )

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
    # cb = fig.colorbar(
    #     b,
    #     ax=ax1,
    #     location="right",
    #     shrink=0.65,
    #     extend="both",
    # )
    # cb.set_label(label=cb_label, size=24)
    # cb.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    plt.savefig(
        "/RAID01/data/python_fig/MODIS_CM_250hPa_"
        + var_name
        + "_11years.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
        edgecolor="w",
    )


def plot_full_hemisphere_no_antarctica(
    data,
    min,
    max,
    var_name,
    cmap_file="/RAID01/data/muqy/color/test.txt",
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
        The path of the color map file, by default "/RAID01/data/muqy/color/test.txt"
    """
    # set the font
    plt.rcParams.update({"font.family": "Times New Roman"})

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-60, 89, 150)

    cmap = dcmap(cmap_file)
    cmap.set_bad("gray", alpha=0)
    cmap.set_over("#800000")
    # set color using RGB values
    cmap.set_under((27 / 255, 44 / 255, 98 / 255))

    fig, (ax1) = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(11, 7),
        constrained_layout=True,
    )

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
    # cb = fig.colorbar(
    #     b,
    #     ax=ax1,
    #     location="right",
    #     shrink=0.65,
    #     extend="both",
    # )
    # cb.set_label(label=cb_label, size=24)
    # cb.ax.tick_params(labelsize=24)

    # cbar.ax.tick_params(labelsize=24)
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}

    plt.savefig(
        "/RAID01/data/python_fig/MODIS_CM_250hPa_"
        + var_name
        + "_11years.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
        edgecolor="w",
    )


def plot_full_hemisphere_no_antarctica_with_cbar(
    data,
    min,
    max,
    cmap_file="/RAID01/data/muqy/color/test.txt",
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
        The path of the color map file, by default "/RAID01/data/muqy/color/test.txt"
    """
    # set the font
    plt.rcParams.update({"font.family": "Times New Roman"})

    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-60, 89, 150)

    cmap = dcmap(cmap_file)
    cmap.set_over("#800000")
    # set color using RGB values
    cmap.set_under((27 / 255, 44 / 255, 98 / 255))

    fig, (ax1) = plt.subplots(
        figsize=(11, 7), constrained_layout=True, dpi=400
    )

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

    # gl = ax1.gridlines(
    #     linestyle="-.", lw=0.1, alpha=0.1, draw_labels=True
    # )
    # gl.top_labels = False
    # gl.right_labels = False
    fig.colorbar(
        b,
        ax=ax1,
        location="right",
        shrink=0.65,
        extend="both",
    )

    # cbar.ax.tick_params(labelsize=24)
    # gl.xlabel_style = {"size": 18}
    # gl.ylabel_style = {"size": 18}


Instab_250_300 = xr.open_dataset(
    "/RAID01/data/ERA5_Var_data/2010_2020_250hPa_4_paras.nc"
)
Instab_250_300 = (
    Instab_250_300["Stability_250"]
    .values.reshape(31, 12, 28, 180, 360)[-11:]
    .reshape(-1, 180, 360)
)

Instab_225_250 = xr.open_dataset(
    "/RAID01/data/ERA5_Var_data/2010_2020_225hPa_4_paras.nc"
)
Instab_225_250 = (
    Instab_225_250["Stability_225_250"]
    .values.reshape(31, 12, 28, 180, 360)[-11:]
    .reshape(-1, 180, 360)
)
Instab_225_250 = Instab_225_250[:, 30:, :]

Corr_Instab_cld_225_250, _ = get_correlation(
    HCF_data, Instab_225_250
)
Corr_Instab_cld_225_250, _ = get_correlation_spatial(
    HCF_data, Instab_225_250
)

# Calculate the correlation between PC1 and HCF
Corr_PC_cld_all, _ = calc_correlation_pvalue_PC1_Cld(
    HCF_data, PC_best_corr
)

Corr_PC_Cld_all_flatten, P_value = calc_correlation_PC1_Cld_all(
    HCF_data, PC_best_corr
)

# ---------------------------------------------------------------------------- #
# ------------------- Plot the correlation ------------------- #
# ---------------------------------------------------------------------------- #

# pressure_levels = [
#     "150",
#     "175",
#     "200",
#     "225",
#     "250",
#     "300",
#     "350",
#     "400",
# ]

plot_full_hemisphere_no_antarctica(
    data=corr_data_spatial["200"]["Corr_RH_HCF"],
    min=-1,
    max=1,
    var_name="RH",
)

plot_full_hemisphere_no_antarctica(
    data=corr_data_spatial["350"]["Corr_T_HCF"],
    min=-1,
    max=1,
    var_name="T",
)

plot_full_hemisphere_no_antarctica(
    data=corr_data_spatial["250"]["Corr_W_HCF"],
    min=-1,
    max=1,
    var_name="W",
)

plot_full_hemisphere_no_antarctica(
    data=Corr_Instab_cld_225_250,
    min=-1,
    max=1,
    var_name="Instab",
)

plot_full_hemisphere_no_antarctica(
    data=corr_PC_HCF,
    min=-1,
    max=1,
    var_name="PC best corr",
)

plot_full_hemisphere_no_antarctica(
    data=corr_PC_HCF_best_corr,
    min=-1,
    max=1,
    var_name="PC best corr spatial",
)

plot_full_hemisphere_no_antarctica_with_cbar(
    data=corr_PC_HCF_best_corr_spatial,
    min=-1,
    max=1,
)

plot_full_hemisphere_no_antarctica(
    data=corr_PC_HCF_og,
    min=-1,
    max=1,
    var_name="PC best corr spatial",
)

plot_full_hemisphere_no_antarctica_with_cbar(
    data=corr_PC_HCF_spatial - corr_PC_HCF,
    min=-0.2,
    max=0.2,
    var_name="PC D corr",
)

plot_full_hemisphere_no_antarctica_with_cbar(
    data=corr_PC_HCF_og - corr_PC_HCF,
    min=-0.1,
    max=0.1,
    var_name="PC D corr",
)

plot_full_hemisphere_no_antarctica_with_cbar(
    data=corr_PC_HCF_spatial_og - corr_PC_HCF_spatial,
    min=-0.1,
    max=0.1,
    var_name="PC D corr",
)

##############################################################################
#### Plot atms variables vary with PC1 ######################################
##############################################################################

# index of level in reshape data
"150",
"175",
"200",
"225",
"250",
"300",
"350",
"400",
# variables = ["RelativeH", "Temperature", "Wvelocity"]

RelativeH_175 = reshaped_data_no_antarctica["200"][0]
Temperature_350 = reshaped_data_no_antarctica["150"][1]
Wvelocity_250 = reshaped_data_no_antarctica["250"][2]

# Set the start, end, and gap values
start = -2.5
end = 5.5
gap = 0.05

# Initialize the FilterAtmosDataFitPCgap class with lat and lon
filter_atmos_fit_PC1 = FilterAtmosDataFitPCgap(
    start, end, gap, lat=np.arange(150), lon=np.arange(360)
)

# Apply the Filter_data_fit_PC1_gap_new method to your data
RelativeH_filtered = (
    filter_atmos_fit_PC1.Filter_data_fit_PC1_gap_new(
        Atms_data=RelativeH_175,
        PC_data=PC_best_corr_spatial,
    )
)
Temperature_filtered = (
    filter_atmos_fit_PC1.Filter_data_fit_PC1_gap_new(
        Atms_data=Temperature_350,
        PC_data=PC_best_corr_spatial,
    )
)
Wvelocity_filtered = (
    filter_atmos_fit_PC1.Filter_data_fit_PC1_gap_new(
        Atms_data=Wvelocity_250,
        PC_data=PC_best_corr_spatial,
    )
)
Stability_filtered = (
    filter_atmos_fit_PC1.Filter_data_fit_PC1_gap_new(
        Atms_data=Instab_225_250,
        PC_data=PC_best_corr_spatial,
    )
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
        (line,) = ax.plot(
            PC1_bin, mean, label=var_name, color=color
        )
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
        (line,) = ax.plot(
            PC1_bin, mean, label=var_name, color=color
        )
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
