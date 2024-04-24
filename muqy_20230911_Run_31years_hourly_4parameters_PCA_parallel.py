# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2023-10-16 13:36:00
# @Last Modified by:   Muqy
# @Last Modified time: 2024-02-16 09:40:56

"""

    Code for PCA method using pre-combined ERA5 data
    This code is suitable for 4 parameters: RH, W, T, Sita
    
    Owner: Mu Qingyu
    version 1.0
    Created: 2023-04-11
    
    Including the following parts:
        
        1) Read the Pre-combined ERA5 atmospheric data 
        
        2) Calculate convective instability from those data
        
        3) Form the 1 demensional array from VERTICAL VELOCITY, TEMPERATURE, 
        RELATIVE HUMIDITY, CONVECTIVE INSTABILITY
        
        4) Run PCA procedure to get PC1
        
        5) Write PC1 in the nc file
        
"""

import glob
import os

import dask.array as da
import metpy.calc as mpcalc
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from dask_ml.decomposition import PCA as daskPCA
from metpy.units import units
from scipy import stats


class ERA5_parameters_preproccess_PCA(object):
    """
    Read ERA5 parameters from netcdf file and preproccess them in order to generate
    the PC1 array in shape of [month, day, lat, lon]

    No input arguments needed, all function runs in the __call__ method automatically
    after the initialization of the class
    """

    def __init__(
        self,
    ):
        self.ERA_FILE_PATH = "/RAID01/data/ERA5_data_hourly/"
        self.ERA_FILE_NAME = "ERA5_hourly_"

    def __call__(self):
        # Make auxiliary arrays for atmospheric variables
        RelativeH_1d = da.empty(
            64800 * 372 * 28 * 24, chunks=(1000000,)
        )
        Temperature_1d = da.empty_like(
            RelativeH_1d, chunks=(1000000,)
        )
        Wvelocity_1d = da.empty_like(RelativeH_1d, chunks=(1000000,))
        Stability_1d = da.empty_like(RelativeH_1d, chunks=(1000000,))

        # Extract the data from netcdf files and preproccess them
        dataset = xr.open_mfdataset(
            self.ERA_FILE_PATH + self.ERA_FILE_NAME + "*.nc",
            chunks={"lat": "auto", "lon": "auto", "time": "auto"},
            parallel=True,
        )

        # read the data from netcdf file
        lat = dataset["lat"].astype(np.float32)
        lon = dataset["lon"].astype(np.float32)
        P = dataset["level"].astype(np.float32)
        T = dataset["T"].astype(np.float32)
        W = dataset["W"].astype(np.float32)
        RH = dataset["RH"].astype(np.float32)
        z = dataset["Geo"].astype(np.float32)

        # Extract the atmospheric variables at 300 hPa & 250 hPa
        T250 = T.sel(level=250)
        T300 = T.sel(level=300)
        RH250 = RH.sel(level=250)
        RH300 = RH.sel(level=300)
        W300 = W.sel(level=300)
        Z300 = z.sel(level=300)
        Z250 = z.sel(level=250)

        # Loop through all the hours in the dataset
        index = 0
        for hour in range(0, 28 * 24):
            (
                stab_1,
                RH300_1,
                RH250_1,
                T250_1,
                T300_1,
                W300_1,
                Z300_1,
                Z250_1,
            ) = self.prepariton_for_PCA(
                hour,
                RH300,
                RH250,
                T250,
                T300,
                W300,
                Z300,
                Z250,
            )

            # Concatenate all 11 years data into 1D array
            slice_index = slice(index, index + 64800)
            RelativeH_1d[slice_index] = RH300_1
            Temperature_1d[slice_index] = T300_1
            Wvelocity_1d[slice_index] = W300_1
            Stability_1d[slice_index] = stab_1

            index += 64800

        # Convert the data type to float32
        RelativeH_1d_N = da.map_blocks(
            stats.zscore, RelativeH_1d, dtype=float
        )
        Temperature_1d_N = da.map_blocks(
            stats.zscore, Temperature_1d, dtype=float
        )
        Wvelocity_1d_N = da.map_blocks(
            stats.zscore, Wvelocity_1d, dtype=float
        )
        Stability_1d_N = da.map_blocks(
            stats.zscore, Stability_1d, dtype=float
        )

        # start PCA procedure
        PC_all = self.Principle_Component_Analysis(
            RelativeH_1d_N,
            Temperature_1d_N,
            Wvelocity_1d_N,
            Stability_1d_N,
        )
        return PC_all

    def prepariton_for_PCA(
        self,
        hour,
        RH300,
        RH250,
        T250,
        T300,
        W300,
        Z300,
        Z250,
    ):
        # delete the variables that are not needed
        # (era5 resolution is 181 x 360 --> 180 x 360)
        T250 = T250.isel(lat=slice(1, None))
        T300 = T300.isel(lat=slice(1, None))
        RH300 = RH300.isel(lat=slice(1, None))
        RH250 = RH250.isel(lat=slice(1, None))
        W300 = W300.isel(lat=slice(1, None))
        Z300 = Z300.isel(lat=slice(1, None))
        Z250 = Z250.isel(lat=slice(1, None))

        # flip the variables to have the same orientation as the CERES data
        # and reshape the variables to 1D arrays
        T250_reshaped = (
            T250.sel(time=hour)
            .isel(lat=slice(None, None, -1))
            .data.reshape(-1)
        )
        T300_reshaped = (
            T300.sel(time=hour)
            .isel(lat=slice(None, None, -1))
            .data.reshape(-1)
        )
        RH300_reshaped = (
            RH300.sel(time=hour)
            .isel(lat=slice(None, None, -1))
            .data.reshape(-1)
        )
        RH250_reshaped = (
            RH250.sel(time=hour)
            .isel(lat=slice(None, None, -1))
            .data.reshape(-1)
        )
        W300_reshaped = (
            W300.sel(time=hour)
            .isel(lat=slice(None, None, -1))
            .data.reshape(-1)
        )
        Z300_reshaped = (
            Z300.sel(time=hour)
            .isel(lat=slice(None, None, -1))
            .data.reshape(-1)
        )
        Z250_reshaped = (
            Z250.sel(time=hour)
            .isel(lat=slice(None, None, -1))
            .data.reshape(-1)
        )

        # Reshape the variables to 1D arrays
        RH300_reshaped = da.where(
            RH300_reshaped <= 0, 0.01, RH300_reshaped
        )
        RH250_reshaped = da.where(
            RH250_reshaped <= 0, 0.01, RH250_reshaped
        )

        # calculate the convective instability
        stability_index = self.unstability_calculator(
            RH300_reshaped,
            RH250_reshaped,
            T250_reshaped,
            T300_reshaped,
            Z300_reshaped,
            Z250_reshaped,
        )

        return (
            stability_index,
            RH300_reshaped,
            RH250_reshaped,
            T250_reshaped,
            T300_reshaped,
            W300_reshaped,
            Z300_reshaped,
            Z250_reshaped,
        )

    def unstability_calculator(
        self,
        RH300_reshaped,
        RH250_reshaped,
        T250_reshaped,
        T300_reshaped,
        Z300_reshaped,
        Z250_reshaped,
    ):
        dewpoint300 = np.array(
            mpcalc.dewpoint_from_relative_humidity(
                T300_reshaped * units.kelvin,
                RH300_reshaped * units.dimensionless,
            )
        )
        dewpoint250 = np.array(
            mpcalc.dewpoint_from_relative_humidity(
                T250_reshaped * units.kelvin,
                RH250_reshaped * units.dimensionless,
            )
        )
        thetaE300 = np.array(
            mpcalc.equivalent_potential_temperature(
                300.0 * units.mbar,
                T300_reshaped * units.kelvin,
                dewpoint300 * units.degree_Celsius,
            )
        )
        thetaE250 = np.array(
            mpcalc.equivalent_potential_temperature(
                250.0 * units.mbar,
                T250_reshaped * units.kelvin,
                dewpoint250 * units.degree_Celsius,
            )
        )
        stability_index = (thetaE300 - thetaE250) / (
            Z300_reshaped - Z250_reshaped
        )

        return stability_index

    def Principle_Component_Analysis(
        self,
        RelativeH_1d_N,
        Temperature_1d_N,
        Wvelocity_1d_N,
        Stability_1d_N,
    ):
        """
        _summary_

        Parameters
        ----------
        RelativeH_1d_N : dask.array
            Dask array of relative humidity values with shape of 1D
        Temperature_1d_N : dask.array
            Dask array of temperature values with shape of 1D
        Wvelocity_1d_N : dask.array
            Dask array of vertical velocity values with shape of 1D
        Stability_1d_N : dask.array
            Dask array of stability values with shape of 1D

        Returns
        -------
        PC1 : dask.array
            Dask array of first principle components with shape of 4D: [month, day, lat, lon]
        """

        # Stack arrays along the last axis
        stacked_data = da.stack(
            [
                RelativeH_1d_N,
                Temperature_1d_N,
                Wvelocity_1d_N,
                Stability_1d_N,
            ],
            axis=-1,
        )

        # Utilize Dask's implementation of PCA for better memory management and parallelization
        pca = daskPCA(n_components=1, whiten=True, copy=False)

        # Perform PCA
        PC1 = pca.fit_transform(stacked_data)

        # Normalize
        PC1_N = da.map_blocks(stats.zscore, PC1, dtype=float)

        # Reshape the array to 4D
        PC1_reshaped = PC1_N.reshape((372, 28 * 24, 180, 360))

        return PC1_reshaped


def save_PCA_data_as_netcdf(PC_all):
    """
    save principle components as netcdf file

    Parameters
    ----------
    PC_all : array
        array of principle components with shape of 4D: [month, day, lat, lon]
    """
    lon = np.linspace(0, 359, 360)
    lat = np.linspace(-90, 89, 180)

    PC_all = PC_all.reshape(372, 28 * 24, 180, 360)  # PCA

    ds = xr.Dataset(
        {
            "PC1": (
                ("Month", "Hour", "Latitude", "Longitude"),
                PC_all[:, :, :, :],
            ),
        },
        coords={
            "lat": ("Latitude", np.linspace(-90, 89, 180)),
            "lon": ("Longitude", np.linspace(0, 359, 360)),
            "month": ("months", np.linspace(0, 371, 372)),
            "hour": (
                "Hour",
                np.linspace(0, 28 * 24 - 1, 28 * 24),
            ),
        },
    )

    os.makedirs("/RAID01/data/PC_data/", exist_ok=True)
    ds.to_netcdf(
        "/RAID01/data/PC_data/1990_2020_4_parameters_hourly_300hPa_PC1.nc"
    )


# Run this script directly, perform PCA and save data
if __name__ == "__main__":
    # Initialize the class and run all functions
    ERA_PCA = ERA5_parameters_preproccess_PCA()
    PC_all = ERA_PCA()

    # Save data as netcdf
    save_PCA_data_as_netcdf(PC_all)
