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

    Function for combining ERA5 data to daily mean data
    
    Owner: Mu Qingyu
    version 1.0
    Created: 2023-07-03
    
    Including the following parts:

        1) Read in ERA5 data
        
        2) Combine ERA5 data to daily mean data

"""

import os
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr



def load_file(file_path):
    """
    Loads an ERA5 data file from the specified file path using xarray.

    Parameters:
        file_path (str): The file path of the ERA5 data file to load.

    Returns:
        xr.Dataset: The loaded ERA5 data file as an xarray dataset.
    """
    with xr.open_dataset(file_path) as ds:
        return ds


def process_file(ds):
    """
    Calculates the daily average of the ERA5 data for the given dataset.

    Parameters:
        ds (xr.Dataset): The ERA5 data as an xarray dataset.

    Returns:
        tuple: A tuple containing the latitude, longitude, pressure levels, and daily averages of the ERA5 data.
    """
    lat = ds.latitude.values
    lon = ds.longitude.values
    pressure_levels = ds.level.values

    # Calculate daily average
    variables = ["r", "z", "t", "w", "u", "v"]
    daily_averages = [
        ds[var].mean(dim="time").values for var in variables
    ]

    return lat, lon, pressure_levels, daily_averages


def save_dataset(
    year: int,
    month: int,
    lat: np.ndarray,
    lon: np.ndarray,
    pressure_levels: np.ndarray,
    monthly_averages: np.ndarray,
    file_path: str,
) -> None:
    """
    Saves the daily mean ERA5 data to a netCDF file.

    Parameters:
        year (int): The year of the ERA5 data.
        month (int): The month of the ERA5 data.
        lat (np.ndarray): The latitude values of the ERA5 data.
        lon (np.ndarray): The longitude values of the ERA5 data.
        pressure_levels (np.ndarray): The pressure levels of the ERA5 data.
        monthly_averages (np.ndarray): A list of daily averages for each day in the month of the ERA5 data.
        file_path (str): The file path to save the ERA5 data to.

    Returns:
        None
    """
    ds = xr.Dataset(
        {
            "RH": (
                ("Time", "Level", "Latitude", "Longitude"),
                monthly_averages[:, 0],
            ),
            "Geo": (
                ("Time", "Level", "Latitude", "Longitude"),
                monthly_averages[:, 1],
            ),
            "T": (
                ("Time", "Level", "Latitude", "Longitude"),
                monthly_averages[:, 2],
            ),
            "W": (
                ("Time", "Level", "Latitude", "Longitude"),
                monthly_averages[:, 3],
            ),
            "U": (
                ("Time", "Level", "Latitude", "Longitude"),
                monthly_averages[:, 4],
            ),
            "V": (
                ("Time", "Level", "Latitude", "Longitude"),
                monthly_averages[:, 5],
            ),
        },
        coords={
            "lat": lat,
            "lon": lon,
            "time": pd.date_range(
                f"{year}-{month:02d}-01",
                periods=monthly_averages.shape[0],
            ),
            "level": pressure_levels,
        },
    )

    ds.to_netcdf(file_path)


def create_ERA5_dataset_daily(year, month):
    """
    Creates a daily mean ERA5 dataset for a given year and month.

    Parameters:
        year (int): The year of the ERA5 data.
        month (int): The month of the ERA5 data.

    Returns:
        None
    """
    # function code here
    year_str = str(year).zfill(4)
    month_str = str(month).zfill(2)
    time_str = year_str + month_str

    directory = Path(
        f"/RAID01/data/ERA5_og_Z_RH_T_UV_W/{year_str}/{month_str}"
    )
    files = sorted(directory.iterdir())

    daily_data = []
    for i, file in enumerate(files[:28]):
        ds = load_file(file)
        lat, lon, pressure_levels, daily_averages = process_file(ds)
        daily_data.append(daily_averages)

    # Stack all daily averages together along a new 'Time' dimension
    monthly_averages = np.stack(daily_data, axis=0)

    file_path = f"/RAID01/data/Pre_Data_PCA/ERA5_daily_mean/ERA5_daily_{time_str}.nc"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    save_dataset(
        year,
        month,
        lat,
        lon,
        pressure_levels,
        monthly_averages,
        file_path,
    )


def process_month(args):
    """
    Calls the create_ERA5_dataset_daily function for a given year and month.

    Parameters:
        args (tuple): A tuple containing the year and month.

    Returns:
        None
    """
    year, month = args
    create_ERA5_dataset_daily(year, month)


def main():
    """
    Main function that creates a pool of processes and maps the process_month function to each year-month combination
    from 1990-01 to 2021-12.

    Parameters:
        None

    Returns:
        None
    """
    with Pool() as pool:
        year_month_combinations = [
            (year, month)
            for year in range(1990, 2022)
            for month in range(1, 13)
        ]
        pool.map(process_month, year_month_combinations)


if __name__ == "__main__":
    main()
