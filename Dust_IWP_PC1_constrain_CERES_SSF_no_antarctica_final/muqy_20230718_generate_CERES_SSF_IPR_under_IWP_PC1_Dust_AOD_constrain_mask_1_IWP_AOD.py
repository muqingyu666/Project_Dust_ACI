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
from joblib import Parallel, delayed

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

# region

# Use the joblib library for parallelization
def read_parallel(path: str, var_name: str) -> np.ndarray:
    return (
        xr.open_dataset(path)[var_name]
        .values.astype(np.float32)
        .reshape(-1, 180, 360)
    )


def read_PC1_clean(PC_path: str):
    print("Reading data from netcdf file...")
    data1 = xr.open_dataset(PC_path)
    print("Done loading netcdf file.")
    return (
        np.array(data1.PC1)
        .astype(np.float32)
        .reshape(-1, 150, 360)
    )


def set_extreme_percent_to_nan(
    threshold: float, data_array: np.ndarray
) -> Union[np.ndarray, None]:
    data_array_filtered = np.copy(data_array)
    try:
        lower_threshold = np.nanpercentile(data_array, threshold)
    except IndexError:
        print("ERROR: Data array is empty.")
        return None
    upper_threshold = np.nanpercentile(
        data_array, 100 - threshold
    )
    data_array_filtered[
        data_array_filtered < lower_threshold
    ] = np.nan
    data_array_filtered[
        data_array_filtered > upper_threshold
    ] = np.nan
    return data_array_filtered


def divide_intervals(
    data: np.ndarray, intervals: np.ndarray, n: int
):
    data = data.flatten()
    all_gaps = []
    for i in range(len(intervals) - 1):
        start, end = intervals[i], intervals[i + 1]
        mask = (data >= start) & (data <= end)
        data_in_interval = data[mask]
        gaps = np.percentile(
            data_in_interval, np.linspace(0, 100, n + 1)
        )
        all_gaps.extend(gaps[:-1])
    all_gaps.append(intervals[-1])
    return np.array(all_gaps)


PC_data = read_PC1_clean(
    "/RAID01/data/PC_data/1990_2020_250_hPa_vars_250_300_Instab_PC1_no_antarc.nc"
)

# Read multiple datasets in parallel
variables_to_read = [
    "cldarea_high_daynight_daily",
    "cldarea_ice_high_daynight_daily",
    "cldicerad37_high_daynight_daily",
    "iwp37_high_daynight_daily",
    "cldpress_top_high_daynight_daily",
    "cldtau_high_daynight_daily",
    "cldpress_eff_high_daynight_daily",
]
path = "/RAID01/data/Cld_data/CERES_SSF_data_2005_2020_28_days.nc"
CERES_SSF_data = Parallel(n_jobs=len(variables_to_read))(
    delayed(read_parallel)(path, var) for var in variables_to_read
)

IPR_data = CERES_SSF_data[2]


Dust_AOD = read_parallel(
    "/RAID01/data/merra2/merra2_2005_2020_new_lon.nc",
    "DUEXTTAU",
)

# Filter data
Dust_AOD_filtered = set_extreme_percent_to_nan(1, Dust_AOD)
IWP_data_filtered = set_extreme_percent_to_nan(
    1, CERES_SSF_data[3]
)

# Delete Antarctica region (not done in parallel to reduce code complexity)
sliced_arrays = [
    arr[:, 30:, :]
    for arr in [Dust_AOD_filtered, IWP_data_filtered]
    + CERES_SSF_data
]

# Calculate new intervals
IWP_new_intervals = divide_intervals(
    sliced_arrays[1], np.array([0.5, 1, 5, 20, 50, 100, 500]), 5
)
PC_new_intervals = divide_intervals(
    PC_data, np.arange(-3, 7, 1.5), 5
)

AOD_gap = divide_intervals(
    sliced_arrays[0],
    np.linspace(
        np.nanmin(sliced_arrays[0]),
        np.nanmax(sliced_arrays[0]),
        5,
    ),
    4,
)

# endregion

# -------------------------------------------------------------------------------------------
# ------ Segmentation of cloud data within each PC interval ---------------------------------
# -------------------------------------------------------------------------------------------
#### triout for IWP constrain the same time with PC1 gap constrain ####
# first we need to divide IWP data and PC1 data into n intervals
# this step is aimed to create pcolormesh plot for PC1 and IWP data
# Divide 1, IWP data


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
    data=IWP_data, n=4
)  # Modify 'n' as needed
IWP_gap = np.array([0.5, 1, 5, 20, 50, 100, 500])
IWP_new_intervals = divisor_IWP.divide_intervals(IWP_gap)

divisor_PC = DivideDataGivingGapByVolumeToNewGap(
    data=PC_data, n=4
)  # Modify 'n' as needed
PC_gap = np.arange(-3, 7, 1.5)
PC_new_intervals = divisor_PC.divide_intervals(PC_gap)


def generate_filtered_data_for_all_years(
    AOD_data: np.ndarray,
    IWP_data: np.ndarray,
    PC_all: np.ndarray,
    Cld_all: np.ndarray,
    AOD_n: int = 5,
    IWP_gaps: np.ndarray = IWP_new_intervals,
    PC_gaps: np.ndarray = PC_new_intervals,
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

    IWP_gap = IWP_gaps
    PC_gap = PC_gaps

    # Divide 3, Dust AOD data
    # Divide AOD data as well
    divide_AOD = DividePCByDataVolume(
        dataarray_main=AOD_data,
        n=AOD_n,
    )
    AOD_gap = divide_AOD.main_gap()

    filter_cld_under_AOD_IWP_PC_constrain = (
        Filter_data_fit_PC1_gap_IWP_AOD_constrain(
            lat=[i for i in range(150)],
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
        Cld_data=Cld_all.reshape(-1, 150, 360),
        PC_data=PC_all.reshape(-1, 150, 360),
        IWP_data=IWP_data.reshape(-1, 150, 360),
        AOD_data=AOD_data.reshape(-1, 150, 360),
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
            "lat": np.arange(150),
            "lon": np.arange(360),
        },
    )
    Cld_match_PC_gap_IWP_AOD_constrain_mean.to_netcdf(
        save_path
        + "_match_PC_gap_IWP_AOD_constrain_mean_2005_2020_"
        + AOD_name
        + "_mask_1_IWP_AOD_no_antarc_new_PC_IWP_gaps_3_AOD_gaps_for_FIE_ME.nc"
    )


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
    AOD_n=3,
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
