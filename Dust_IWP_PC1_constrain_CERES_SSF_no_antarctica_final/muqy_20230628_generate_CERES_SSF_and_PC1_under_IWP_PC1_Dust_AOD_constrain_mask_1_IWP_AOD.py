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
        version 1.1: 2023-05-05
        
        This time we mask polar region and filter extreme 2.5% data for IWP and AOD only
        
"""

# import modules
import gc
from typing import Union

import matplotlib as mpl
import numpy as np
from muqy_20220628_uti_PCA_CLD_analysis_function import *
from muqy_20221026_func_filter_hcf_anormal_data import (
    filter_data_PC1_gap_lowermost_highermost_error as filter_data_PC1_gap_lowermost_highermost_error,
)

# --------- import done ------------
# --------- Plot style -------------
mpl.rc("font", family="Times New Roman")
# Set parameter to avoid warning
mpl.rcParams["agg.path.chunksize"] = 10000
mpl.style.use("seaborn-v0_8-ticks")
mpl.rc("font", family="Times New Roman")

# ---------- Read PCA&CLD data from netcdf file --------

# --------------------------------------------------------------- #
# now we read IWP and other cld data (not IWP) from netcdf file

(
    PC_data_og,
    _,
) = read_PC1_CERES_clean(
    PC_path="/RAID01/data/PC_data/1990_2020_best_spatialcorr_var_PC1_no_antarctica.nc",
    CERES_Cld_dataset_name="Cldicerad",
)

# --------------------------------------------------------------- #
# ------ Read in CERES SSF data and filter anormal data --------- #
# --------------------------------------------------------------- #
# read in CERES SSF data

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
CERES_SSF_COT = CERES_SSF_28day[
    "cldtau_high_daynight_daily"
].values.reshape(-1, 180, 360)

# use the 2010-2020 PC1 only
PC_data = PC_data_og.reshape(31, 12, 28, 150, 360)[-11:, :, :, :, :]
PC_data = PC_data.reshape(-1, 150, 360)


# use the 2010-2020 PC1 only
PC_data = PC_data.astype(np.float32).reshape(3696, 150, 360)
HCF_data = CERES_SSF_HCF.astype(np.float32).reshape(3696, 180, 360)
HCF_ice_data = CERES_SSF_ice_HCF.astype(np.float32).reshape(
    3696, 180, 360
)
IWP_data = CERES_SSF_IWP.astype(np.float32).reshape(3696, 180, 360)
IPR_data = CERES_SSF_IPR_37.astype(np.float32).reshape(
    3696, 180, 360
)
CTP_data = CERES_SSF_CTP.astype(np.float32).reshape(3696, 180, 360)
COD_data = CERES_SSF_COT.astype(np.float32).reshape(3696, 180, 360)

# --------------------------------------------------------------- #
# ------------------ Read the MERRA2 dust data ------------------ #
# --------------------------------------------------------------- #
# Implementation for MERRA2 dust AOD
# extract the data from 2010 to 2014 like above
data_merra2_2010_2020_new_lon = xr.open_dataset(
    "/RAID01/data/merra2/merra_2_daily_2010_2020_new_lon.nc"
)

# # Extract Dust aerosol data from all data
Dust_AOD = data_merra2_2010_2020_new_lon["DUEXTTAU"].values.reshape(
    3696, 180, 360
)

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


# -------------------------------------------------------------------------------------------
# ------ Segmentation of cloud data within each PC interval ---------------------------------
# -------------------------------------------------------------------------------------------
#### triout for IWP constrain the same time with PC1 gap constrain ####
# first we need to divide IWP data and PC1 data into n intervals
# this step is aimed to create pcolormesh plot for PC1 and IWP data
# Divide 1, IWP data
def generate_filtered_data_for_all_years(
    AOD_data: np.ndarray,
    IWP_data: np.ndarray,
    PC_all: np.ndarray,
    Cld_all: np.ndarray,
    IWP_n: int = 50,
    PC_n: int = 50,
):
    """
    Generate filtered data for all years based on
    input AOD, IWP, PC, and cloud data.

    Parameters
    ----------
    AOD_data : np.ndarray
    Array of Dust Aerosol Optical Depth (AOD) data.
    IWP_data : np.ndarray
    Array of Ice Water Path (IWP) data.
    PC_all : np.ndarray
    Array of principal component (PC) data.
    Cld_data_all : np.ndarray
    Array of cloud data.
    IWP_n : int, optional
    Number of IWP data bins to divide the data into. Default is 50.
    PC_n : int, optional
    Number of PC data bins to divide the data into. Default is 50.

    Returns:
    tuple
    A tuple of the following elements:
    - Cld_match_PC_gap_IWP_AOD_constrain_mean : np.ndarray
    Array of filtered cloud data matched with PC, IWP, AOD data.
    - PC_match_PC_gap_IWP_AOD_constrain_mean : np.ndarray
    Array of filtered PC data matched with PC, IWP, AOD data.
    - AOD_gap : np.ndarray
    Array of AOD data bins.
    - IWP_gap : np.ndarray
    Array of IWP data bins.
    - PC_gap : np.ndarray
    Array of PC data bins.
    """
    divide_IWP = DividePCByDataVolume(
        dataarray_main=IWP_data,
        n=IWP_n,
    )
    IWP_gap = divide_IWP.main_gap()

    # Divide 2, PC1 data
    divide_PC = DividePCByDataVolume(
        dataarray_main=PC_all,
        n=PC_n,
    )
    PC_gap = divide_PC.main_gap()

    # Use the PC1, IWP data to filter the cloud data and AOD data
    filter_cld_under_AOD_IWP_PC_constrain = (
        Filter_data_fit_PC1_IWP_AOD_constrain(
            lat=[i for i in range(150)],
            lon=[i for i in range(360)],
        )
    )

    # Now we can filter the CLd and PC1 data into pieces
    # Based on AOD, IWP, PC1 gap we just created
    # Shape is (AOD_bin, IWP_bin, PC_bin, lat, lon)
    (
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        PC1_match_PC_gap_IWP_AOD_constrain_mean,
    ) = filter_cld_under_AOD_IWP_PC_constrain.Filter_data_fit_gap_PC1(
        Cld_data=Cld_all.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        IWP_data=IWP_data.reshape(-1, 150, 360),
        AOD_data=AOD_data.reshape(-1, 150, 360),
        PC_gap=PC_gap,
        IWP_gap=IWP_gap,
    )

    return (
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        PC1_match_PC_gap_IWP_AOD_constrain_mean,
        IWP_gap,
        PC_gap,
    )


def save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean: np.ndarray,
    PC1_match_PC_gap_IWP_AOD_constrain_mean: np.ndarray,
    IWP_gap: np.ndarray,
    PC_gap: np.ndarray,
    AOD_name: str = "Dust_AOD",
    save_path: str = "/RAID01/data/Filtered_data/",
    save_PC1: bool = True,
):
    """
    Save the fitted data as netcdf file.

    Parameters
    ----------
    Cld_match_PC_gap_IWP_AOD_constrain_mean : np.ndarray
        Mean cloud data with matching PC1 gap, IWP, and AOD constraint.
    AOD_match_PC_gap_IWP_AOD_constrain_mean : np.ndarray
        Mean AOD data with matching PC1 gap, IWP, and AOD constraint.
    AOD_gap : np.ndarray
        AOD gap.
    IWP_gap : np.ndarray
        IWP gap.
    PC_gap : np.ndarray
        PC gap.
    AOD_name : str, optional
        Name of AOD data, by default "Dust_AOD".
    save_path : str, optional
        Path to save netcdf file, by default "/RAID01/data/Filtered_data/".
    save_AOD : bool, optional
        Flag to control whether to save AOD data, by default True.
    """
    # Save the fitted data as netcdf file
    Cld_match_PC_gap_IWP_AOD_constrain_mean = xr.DataArray(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        dims=["IWP_bin", "PC_bin", "lat", "lon"],
        coords={
            "IWP_bin": IWP_gap[1:],
            "PC_bin": PC_gap[1:],
            "lat": np.arange(150),
            "lon": np.arange(360),
        },
    )
    Cld_match_PC_gap_IWP_AOD_constrain_mean.to_netcdf(
        save_path
        + "_match_PC_gap_IWP_constrain_mean_2010_2020_"
        + AOD_name
        + "_mask_1_IWP_AOD_no_antarc.nc"
    )
    if save_PC1:
        PC1_match_PC_gap_IWP_AOD_constrain_mean = xr.DataArray(
            PC1_match_PC_gap_IWP_AOD_constrain_mean,
            dims=["IWP_bin", "PC_bin", "lat", "lon"],
            coords={
                "IWP_bin": IWP_gap[1:],
                "PC_bin": PC_gap[1:],
                "lat": np.arange(150),
                "lon": np.arange(360),
            },
        )
        PC1_match_PC_gap_IWP_AOD_constrain_mean.to_netcdf(
            "/RAID01/data/Filtered_data/PC1_match_PC_gap_IWP_constrain_mean_2010_2020_"
            + AOD_name
            + "_mask_1_IWP_AOD_no_antarc.nc"
        )


# HCF data
# Load the data
(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC1_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    IWP_gap,
    PC_gap,
) = generate_filtered_data_for_all_years(
    AOD_data=Dust_AOD_filtered,
    IWP_data=IWP_data_filtered,
    PC_all=PC_data,
    Cld_all=HCF_data,
    IWP_n=40,
    PC_n=40,
)

# Save the filtered data
save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC1_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    IWP_gap,
    PC_gap,
    AOD_name="Dust_AOD",
    save_path="/RAID01/data/Filtered_data/CERES_SSF_HCF",
)

del (
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC1_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    IWP_gap,
    PC_gap,
)
gc.collect()

# IPR data
# Load the data
(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC1_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    IWP_gap,
    PC_gap,
) = generate_filtered_data_for_all_years(
    AOD_data=Dust_AOD_filtered,
    IWP_data=IWP_data_filtered,
    PC_all=PC_data,
    Cld_all=IPR_data,
    IWP_n=40,
    PC_n=40,
)

# Save the filtered data
save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC1_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    IWP_gap,
    PC_gap,
    AOD_name="Dust_AOD",
    save_path="/RAID01/data/Filtered_data/CERES_SSF_IPR",
    save_PC1=False,
)

del (
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC1_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    IWP_gap,
    PC_gap,
)
gc.collect()

# CTP data
# Load the data
(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC1_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    IWP_gap,
    PC_gap,
) = generate_filtered_data_for_all_years(
    AOD_data=Dust_AOD_filtered,
    IWP_data=IWP_data_filtered,
    PC_all=PC_data,
    Cld_all=CTP_data,
    IWP_n=40,
    PC_n=40,
)

# Save the filtered data
save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC1_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    IWP_gap,
    PC_gap,
    AOD_name="Dust_AOD",
    save_path="/RAID01/data/Filtered_data/CERES_SSF_CTP",
    save_PC1=False,
)

del (
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC1_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    IWP_gap,
    PC_gap,
)
gc.collect()

# COD data
# Load the data
(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC1_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    IWP_gap,
    PC_gap,
) = generate_filtered_data_for_all_years(
    AOD_data=Dust_AOD_filtered,
    IWP_data=IWP_data_filtered,
    PC_all=PC_data,
    Cld_all=COD_data,
    IWP_n=40,
    PC_n=40,
)

# Save the filtered data
save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC1_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    IWP_gap,
    PC_gap,
    AOD_name="Dust_AOD",
    save_path="/RAID01/data/Filtered_data/CERES_SSF_COT",
    save_PC1=False,
)

del (
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC1_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    IWP_gap,
    PC_gap,
)
gc.collect()
