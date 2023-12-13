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
    PC_30_years_250hpa,
    _,
) = read_PC1_CERES_clean(
    PC_path="/RAID01/data/PC_data/1990_2020_4_parameters_250hPa_PC1(250_300hPa_unstab).nc",
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
PC_data = PC_30_years_250hpa.reshape(31, 12, 28, 180, 360)[
    -11:, :, :, :, :
]
PC_data = PC_data.reshape(-1, 180, 360)

# use the 2010-2020 PC1 only
PC_data = PC_data.astype(np.float32).reshape(3696, 180, 360)
HCF_data = CERES_SSF_HCF.astype(np.float32).reshape(3696, 180, 360)
HCF_ice_data = CERES_SSF_ice_HCF.astype(np.float32).reshape(
    3696, 180, 360
)
IWP_data = CERES_SSF_IWP.astype(np.float32).reshape(3696, 180, 360)
IPR_data = CERES_SSF_IPR_37.astype(np.float32).reshape(3696, 180, 360)
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
        + "_pristine.nc"
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
    AOD_data=Dust_AOD,
    IWP_data=IWP_data,
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
    save_path="/RAID01/data/Filtered_data/CERES_SSF_HCF",
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
    AOD_data=Dust_AOD,
    IWP_data=IWP_data,
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
    save_path="/RAID01/data/Filtered_data/CERES_SSF_IPR",
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
    AOD_data=Dust_AOD,
    IWP_data=IWP_data,
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
    save_path="/RAID01/data/Filtered_data/CERES_SSF_CTP",
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
    AOD_data=Dust_AOD,
    IWP_data=IWP_data,
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
    save_path="/RAID01/data/Filtered_data/CERES_SSF_COT",
)
