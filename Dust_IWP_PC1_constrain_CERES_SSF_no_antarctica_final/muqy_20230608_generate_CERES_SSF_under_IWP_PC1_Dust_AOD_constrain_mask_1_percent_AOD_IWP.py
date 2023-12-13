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


def read_PC1_MODIS_from_netcdf(PC_file_name):
    """
    Read the PC1 and MODIS data from the netcdf file
    """
    # Read data from netcdf file
    print("Reading data from netcdf file...")

    # Read in the MODIS cirrus data
    MODIS_cld_data = xr.open_dataset(
        "/RAID01/data/MCD06COSP_D3_2010_2022_filled_time_new_lon.nc"
    )

    MODIS_cld_data_2010_2020 = MODIS_cld_data.sel(
        time=slice("2010-01-01", "2020-12-31")
    )

    # Extract the cloud data in the 28 days of each month
    ds_MODIS_28day = MODIS_cld_data_2010_2020.isel(
        time=(MODIS_cld_data_2010_2020.time.dt.day >= 1)
        & (MODIS_cld_data_2010_2020.time.dt.day <= 28)
    )

    # Extract each variable
    MODIS_CM = ds_MODIS_28day["Cloud_Mask"].values.reshape(
        -1, 180, 360
    )
    MODIS_IPR = ds_MODIS_28day["Cloud_Size"].values.reshape(
        -1, 180, 360
    )
    MODIS_IWP = ds_MODIS_28day["Water_Path"].values.reshape(
        -1, 180, 360
    )
    MODIS_CTP = ds_MODIS_28day["Top_Pressure"].values.reshape(
        -1, 180, 360
    )
    MODIS_COT = ds_MODIS_28day["Cloud_Thickness"].values.reshape(
        -1, 180, 360
    )

    # -------------------------------------------------
    data_pc = xr.open_dataset(PC_file_name)

    PC_all = np.array(data_pc.PC1)

    # Arrange data from all years
    PC_all = PC_all.reshape(31, 12, 28, 180, 360)

    # -------------------------------------------------

    HCF_data = MODIS_CM
    IWP_data = MODIS_IWP
    IPR_data = MODIS_IPR
    CTP_data = MODIS_CTP
    COD_data = MODIS_COT

    print("Done loading netcdf file.")

    HCF_data[HCF_data == -999] = np.nan
    IWP_data[IWP_data == -999] = np.nan
    IPR_data[IPR_data == -999] = np.nan
    CTP_data[CTP_data == -999] = np.nan
    COD_data[COD_data == -999] = np.nan

    return (
        # pc
        PC_all,
        # cld
        HCF_data,
        IWP_data,
        IPR_data,
        CTP_data,
        COD_data,
    )


# --------------------------------------------------------------- #
# ------------------ Read the MODIS cloud data ------------------ #
# --------------------------------------------------------------- #

# now we read IWP and other cld data (not IWP) from netcdf file
(
    # pc
    PC_data,
    # cld
    HCF_data,
    IWP_data,
    IPR_data,
    CTP_data,
    COD_data,
) = read_PC1_MODIS_from_netcdf(
    PC_file_name="/RAID01/data/PC_data/1990_2020_4_parameters_300hPa_PC1.nc"
)

# use the 2010-2020 PC1 only
PC_data = PC_data[-11:]


# use the 2010-2020 PC1 only
PC_data = PC_data[-11:].astype(np.float32).reshape(3696, 180, 360)
HCF_data = HCF_data.astype(np.float32).reshape(3696, 180, 360)
IWP_data = IWP_data.astype(np.float32).reshape(3696, 180, 360)
IPR_data = IPR_data.astype(np.float32).reshape(3696, 180, 360)
CTP_data = CTP_data.astype(np.float32).reshape(3696, 180, 360)
COD_data = COD_data.astype(np.float32).reshape(3696, 180, 360)

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
        upper_threshold = np.nanpercentile(data_array, 100 - threshold)

    # Set the largest and smallest 5% of the data array to nan
    data_array_filtered[data_array_filtered < lower_threshold] = np.nan
    data_array_filtered[data_array_filtered > upper_threshold] = np.nan

    return data_array_filtered


# Filter out extreme 2.5% AOD data to avoid the influence of dust origin
Dust_AOD_filtered = set_extreme_percent_to_nan(
    threshold=1, data_array=Dust_AOD
)

# Filter out extreme 1% IWP data to avoid extreme large IWP values
IWP_data_filtered = set_extreme_percent_to_nan(
    threshold=1, data_array=IWP_data
)

del Dust_AOD, IWP_data
gc.collect()


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
    AOD_n: int = 5,
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
    AOD_n : int, optional
    Number of AOD data bins to divide the data into. Default is 5.
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

    # Divide 3, Dust AOD data
    # Divide AOD data as well
    divide_AOD = DividePCByDataVolume(
        dataarray_main=AOD_data,
        n=AOD_n,
    )
    AOD_gap = divide_AOD.main_gap()

    filter_cld_under_AOD_IWP_PC_constrain = (
        Filter_data_fit_PC1_gap_IWP_AOD_constrain(
            lat=[i for i in range(180)],
            lon=[i for i in range(360)],
        )
    )

    # Now we can filter the CLd and PC1 data into pieces
    # Based on AOD, IWP, PC1 gap we just created
    # Shape is (AOD_bin, IWP_bin, PC_bin, lat, lon)
    (
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        PC_match_PC_gap_IWP_AOD_constrain_mean,
    ) = filter_cld_under_AOD_IWP_PC_constrain.Filter_data_fit_gap(
        Cld_data=Cld_all.reshape(-1, 180, 360),
        PC_data=PC_all.reshape(-1, 180, 360),
        IWP_data=IWP_data.reshape(-1, 180, 360),
        AOD_data=AOD_data.reshape(-1, 180, 360),
        PC_gap=PC_gap,
        IWP_gap=IWP_gap,
        AOD_gap=AOD_gap,
    )

    return (
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        PC_match_PC_gap_IWP_AOD_constrain_mean,
        AOD_gap,
        IWP_gap,
        PC_gap,
    )


def save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean: np.ndarray,
    AOD_gap: np.ndarray,
    IWP_gap: np.ndarray,
    PC_gap: np.ndarray,
    AOD_name: str = "Dust_AOD",
    save_path: str = "/RAID01/data/Filtered_data/",
):
    """
    Save the fitted data as netcdf file.

    Parameters
    ----------
    Cld_match_PC_gap_IWP_AOD_constrain_mean : np.ndarray
        Mean cloud data with matching PC1 gap, IWP, and AOD constraint.
    PC_match_PC_gap_IWP_AOD_constrain_mean : np.ndarray
        Mean PC1 data with matching PC1 gap, IWP, and AOD constraint.
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
    """
    # Save the fitted data as netcdf file
    Cld_match_PC_gap_IWP_AOD_constrain_mean = xr.DataArray(
        Cld_match_PC_gap_IWP_AOD_constrain_mean,
        dims=["AOD_bin", "IWP_bin", "PC_bin", "lat", "lon"],
        coords={
            "AOD_bin": AOD_gap[1:],
            "IWP_bin": IWP_gap[1:],
            "PC_bin": PC_gap[1:],
            "lat": np.arange(180),
            "lon": np.arange(360),
        },
    )
    Cld_match_PC_gap_IWP_AOD_constrain_mean.to_netcdf(
        save_path
        + "_match_PC_gap_IWP_AOD_constrain_mean_2010_2020_"
        + AOD_name
        + "_mask_1_percent_AOD_IWP.nc"
    )


# HCF data
# Load the data
(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
) = generate_filtered_data_for_all_years(
    AOD_data=Dust_AOD_filtered,
    IWP_data=IWP_data_filtered,
    PC_all=PC_data,
    Cld_all=HCF_data,
    AOD_n=6,
    IWP_n=40,
    PC_n=40,
)

del PC_match_PC_gap_IWP_AOD_constrain_mean_Dust
gc.collect()

# Save the filtered data
save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
    AOD_name="Dust_AOD",
    save_path="/RAID01/data/Filtered_data/MODIS_HCF",
)

del (
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
)
gc.collect()

# IPR data
# Load the data
(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
) = generate_filtered_data_for_all_years(
    AOD_data=Dust_AOD_filtered,
    IWP_data=IWP_data_filtered,
    PC_all=PC_data,
    Cld_all=IPR_data,
    AOD_n=6,
    IWP_n=40,
    PC_n=40,
)

del PC_match_PC_gap_IWP_AOD_constrain_mean_Dust
gc.collect()

# Save the filtered data
save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
    AOD_name="Dust_AOD",
    save_path="/RAID01/data/Filtered_data/MODIS_IPR",
)

del (
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
)
gc.collect()

# CTP data
# Load the data
(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
) = generate_filtered_data_for_all_years(
    AOD_data=Dust_AOD_filtered,
    IWP_data=IWP_data_filtered,
    PC_all=PC_data,
    Cld_all=CTP_data,
    AOD_n=6,
    IWP_n=40,
    PC_n=40,
)

del PC_match_PC_gap_IWP_AOD_constrain_mean_Dust
gc.collect()

# Save the filtered data
save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
    AOD_name="Dust_AOD",
    save_path="/RAID01/data/Filtered_data/MODIS_CTP",
)

del (
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
)
gc.collect()

# COD data
# Load the data
(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    PC_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
) = generate_filtered_data_for_all_years(
    AOD_data=Dust_AOD_filtered,
    IWP_data=IWP_data_filtered,
    PC_all=PC_data,
    Cld_all=COD_data,
    AOD_n=6,
    IWP_n=40,
    PC_n=40,
)

del PC_match_PC_gap_IWP_AOD_constrain_mean_Dust
gc.collect()

# Save the filtered data
save_filtered_data_as_nc(
    Cld_match_PC_gap_IWP_AOD_constrain_mean_Dust,
    AOD_gap,
    IWP_gap,
    PC_gap,
    AOD_name="Dust_AOD",
    save_path="/RAID01/data/Filtered_data/MODIS_COT",
)
