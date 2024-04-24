# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2023-10-16 13:36:03
# @Last Modified by:   Muqy
# @Last Modified time: 2023-12-24 16:42:14
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

    Function for combining all CERES SSF data together
    
    Owner: Mu Qingyu
    version 1.0
    Created: 2023-07-04
    
    Including the following parts:

        1) Read in CERES SSF data in mfdataset
        
        2) Combine all CERES SSF data together

"""
import pandas as pd
import xarray as xr

# Read in CERES SSF data in mfdataset
CERES_SSF_all_files = xr.open_mfdataset(
    "../Data_python/Cld_data/CERES_SSF_data/*.nc"
)


# Combine all CERES SSF data together
# Extract the first 28 days of each month
def extract_first_28_days(ds):
    """
    Extracts the first 28 days of each month from an xarray dataset.

    Parameters:
        ds (xr.Dataset): The input xarray dataset.

    Returns:
        xr.Dataset: The output xarray dataset containing only the first 28 days of each month.
    """

    # Initialize a list to store the data for each month
    monthly_data = []

    # Loop over each year and month
    for year in pd.unique(ds["time.year"]):
        for month in range(1, 13):
            # Select the data for this month
            monthly_ds = ds.sel(
                time=(ds["time.year"] == year)
                & (ds["time.month"] == month)
            )

            # If the month has at least 28 days, select the first 28 days and append to the list
            if len(monthly_ds["time"]) >= 28:
                monthly_data.append(monthly_ds.isel(time=slice(0, 28)))

    # Concatenate all the monthly data along the time dimension
    ds_28days = xr.concat(monthly_data, dim="time")

    return ds_28days


# Use the function
CERES_SSF_first_28_days = extract_first_28_days(CERES_SSF_all_files)

# Save to a netCDF file
CERES_SSF_first_28_days.to_netcdf(
    "../Data_python/Cld_data/CERES_SSF_Terra_data_2005_2020_28_days.nc",
)
