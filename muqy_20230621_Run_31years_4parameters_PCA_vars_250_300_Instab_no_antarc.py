# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2023-10-16 20:41:54
# @Last Modified by:   Muqy
# @Last Modified time: 2023-11-26 10:06:10
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from metpy import calc as mpcalc
from metpy.units import units
from scipy import stats
from sklearn.decomposition import PCA


class ERA5PCAProcessor:
    """
    A class to handle PCA processing of ERA5 parameters.
    Reads parameters from a netcdf file, preprocesses them, and calculates the first principal component.
    """

    def __init__(
        self,
        file_path="/RAID01/data/Pre_Data_PCA/ERA5_daily_mean/",
        # Path to the directory containing input netcdf files
        file_name="ERA5_daily_",
        # Prefix of the netcdf file names
        level_specs=None,
        # Dict with format {'Variable_Name': [levels]}
        # Specifying the levels to extract for each variable
        instability_levels=None,
        # Dict with format {'level1': , 'level2': }
        # Levels to use for calculating instability
    ):
        self.file_path = file_path
        self.file_name = file_name
        self.level_specs = level_specs if level_specs else {}
        self.instability_levels = (
            instability_levels
            if instability_levels
            else {"level1": 7, "level2": 6}
        )
        self.level_to_pressure = {
            1: 125,
            2: 150,
            3: 175,
            4: 200,
            5: 225,
            6: 250,
            7: 300,
            8: 350,
            9: 400,
        }

    def load_and_process_files(self):
        files = sorted(
            glob.glob(self.file_path + self.file_name + "*.nc")
        )
        results = []
        for file in files:
            result = self.process_file(file)
            results.extend(result)
        return np.concatenate(results, axis=0)

    def process_file(self, file):
        """Preprocesses the data from a single netcdf file.

        Parameters
        ----------
        file : str
            Path to the netcdf file.

        Returns
        -------
        data : list
            A list of preprocessed data from the file.
        """
        ds = xr.open_dataset(file)
        data = []
        for day in range(28):
            day_data = self.preprocess_data(ds, day)
            day_data = self.flatten_and_normalize_data(day_data)
            data.append(day_data)

        return data

    def preprocess_data(self, ds, day):
        """Preprocesses the data for a single day from the dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset containing the data.
        day : int
            The day number (0-27).

        Returns
        -------
        processed_data : list
            A list of preprocessed variables.
        """
        ds = ds.sel(Time=day)

        processed_data = []
        for var_name, levels in self.level_specs.items():
            for level in levels:
                variable = ds[var_name].sel(Level=level)
                variable = (
                    variable.clip(min=0.0001)
                    if var_name == "RH"
                    else variable
                )

                # Delete the first row of the data to make the resolution 180x360
                variable = np.delete(variable.values, 0, axis=0)

                # Flip the variable data to have the same orientation as the CERES data
                variable = np.flipud(variable)

                # Reshape the variable to 1D array
                variable = variable.reshape(64800)

                processed_data.append(variable)

        temp_300 = ds["T"].sel(
            Level=self.instability_levels["level1"]
        )
        temp_250 = ds["T"].sel(
            Level=self.instability_levels["level2"]
        )
        z_300 = ds["Geo"].sel(
            Level=self.instability_levels["level1"]
        )
        z_250 = ds["Geo"].sel(
            Level=self.instability_levels["level2"]
        )
        rh_300 = (
            ds["RH"]
            .sel(Level=self.instability_levels["level1"])
            .clip(min=0.01)
        )
        rh_250 = (
            ds["RH"]
            .sel(Level=self.instability_levels["level2"])
            .clip(min=0.01)
        )

        temp_300 = self.prepare_variable(temp_300, False)
        temp_250 = self.prepare_variable(temp_250, False)
        z_300 = self.prepare_variable(z_300, False)
        z_250 = self.prepare_variable(z_250, False)
        rh_300 = self.prepare_variable(rh_300, True)
        rh_250 = self.prepare_variable(rh_250, True)

        dp_300 = mpcalc.dewpoint_from_relative_humidity(
            temp_300 * units.kelvin,
            rh_300 * units.dimensionless,
        ).magnitude
        dp_250 = mpcalc.dewpoint_from_relative_humidity(
            temp_250 * units.kelvin,
            rh_250 * units.dimensionless,
        ).magnitude
        theta_e_300 = mpcalc.equivalent_potential_temperature(
            self.level_to_pressure[
                self.instability_levels["level1"]
            ]
            * units.mbar,
            temp_300 * units.kelvin,
            dp_300 * units.celsius,
        ).magnitude
        theta_e_250 = mpcalc.equivalent_potential_temperature(
            self.level_to_pressure[
                self.instability_levels["level2"]
            ]
            * units.mbar,
            temp_250 * units.kelvin,
            dp_250 * units.celsius,
        ).magnitude
        stability = (theta_e_300 - theta_e_250) / (z_300 - z_250)

        processed_data.append(stability)

        processed_data_no_antarctica = []

        for variable in processed_data:
            # Reshape the 1D array back to 2D
            var_2D = variable.reshape(180, 360)

            # Determine the index corresponding to 60S
            ny = var_2D.shape[0]  # number of latitude points
            sixty_s_index = int(
                ny * (30 / 180)
            )  # index for 60S assuming latitude ranges from -90 to 90

            # Slice array to exclude values south of 60S
            var_2D = var_2D[sixty_s_index:, :]

            # Reshape the 2D array back to 1D and append to the list
            processed_data_no_antarctica.append(
                var_2D.reshape(54000)
            )  # 150 * 360 = 54000

        return processed_data_no_antarctica

        # return processed_data

    def prepare_variable(self, variable, is_rh):
        """Prepares a single variable by clipping, reshaping and flipping.

        Parameters
        ----------
        variable : np.array
            The raw variable data.
        is_rh : bool
            Whether the variable is relative humidity or not. RH needs clipping.

        Returns
        -------
        prepared_variable : np.array
            The prepared variable data.
        """
        variable = variable.clip(min=0.01) if is_rh else variable
        variable = np.delete(
            variable.values, 0, axis=0
        )  # 180x360 resolution
        variable = np.flipud(variable)  # Flip the variable data
        return variable.reshape(64800)  # 1D array

    def flatten_and_normalize_data(self, data):
        """Flattens the list of variables into a 2D array and normalizes it.

        Parameters
        ----------
        data : list
            A list of variable data.

        Returns
        -------
        flattened_data : np.array
            A 2D array (samples x variables) containing the flattened and
            normalized data.
        """

        flattened_data = np.stack(
            [arr.ravel() for arr in data], axis=-1
        )
        return stats.zscore(flattened_data, axis=0)

    def perform_pca(self, data):
        """Performs PCA on the data and returns the first principal component.

        Parameters
        ----------
        data : np.array
            A 2D array (samples x variables) containing the data.

        Returns
        -------
        pca_results : np.array
            A 4D array (time x lat x lon x PC1) containing the first principal
            component.
        """

        PC_all_unnaned = self.replace_with_mean(data)

        # Convert the data type to float32
        PC_all_unnaned = PC_all_unnaned.astype(np.float32)

        pca = PCA(n_components=2, whiten=True)
        pca_results = pca.fit_transform(PC_all_unnaned)

        return pca_results

    def save_to_netcdf(self, data, output_path):
        """Saves the PCA results to a netcdf file.

        Parameters
        ----------
        data : np.array
            A 4D array (time x lat x lon x PC1) containing the PCA results.
        output_path : str
            The path to save the netcdf file.
        """
        lat = np.linspace(-60, 89, 150)
        lon = np.linspace(0, 359, 360)
        years = np.arange(1990, 2021)
        months = np.arange(1, 13)
        days = np.arange(1, 29)

        # Reshape data to desired shape
        data = data.reshape(31, 12, 28, 150, 360)

        xr.DataArray(
            data,
            coords={
                "year": ("year", years),
                "month": ("month", months),
                "day": ("day", days),
                "lat": ("lat", lat),
                "lon": ("lon", lon),
            },
            dims=["year", "month", "day", "lat", "lon"],
            name="PC1",
        ).to_netcdf(output_path)

    def replace_with_mean(self, data):
        finite_mask = np.isfinite(
            data
        )  # mask that selects only finite (not nan and not inf) values
        if np.nanmean(data[finite_mask]) == np.nan:
            mean_value = np.nanmean(
                data[finite_mask]
            )  # calculate mean of finite values
            data[
                ~finite_mask
            ] = mean_value  # replace non-finite values with mean
        else:
            data[
                ~finite_mask
            ] = 0  # replace non-finite values with mean

        return data

    def run(self):
        """The main method that runs the full processing pipeline.

        Parameters
        ----------
        output_path : str
            The path to save the final netcdf file containing the PCA results.
        """
        data = self.load_and_process_files()

        data_unnaned = self.replace_with_mean(data)
        pca_results = self.perform_pca(data_unnaned)

        return pca_results


if __name__ == "__main__":
    "1 means 125 hPa"
    "2 means 150 hPa"
    "3 means 175 hPa"
    "4 means 200 hPa"
    "5 means 225 hPa"
    "6 means 250 hPa"
    "7 means 300 hPa"
    "8 means 350 hPa"
    "9 means 400 hPa"

    # atmos levels
    level_specs = {
        "T": [6],  # Temperature levels
        "RH": [6],  # Relative Humidity levels
        # "Geo": [7],  # Potential height levels
        "W": [6],  # Vertical velocity levels
    }

    # Instability levels
    instability_levels = {
        "level1": 7,  # Level1 for instability calculation
        "level2": 6,  # Level2 for instability calculation
    }

    # run the processor
    processor = ERA5PCAProcessor(
        level_specs=level_specs,
        instability_levels=instability_levels,
    )
    pca_results = processor.run()

    pca_results = pca_results.reshape(31, 12, 28, 150, 360, 2)

    # output path
    output_path = "/RAID01/data/PC_data/PC1_2_1990_2020_250_hPa_vars_250_300_Instab_PC1_no_antarc.nc"

    # save the results to a netcdf file
    processor.save_to_netcdf(pca_results, output_path)

    save_to_netcdf(pca_results, output_path)


def save_to_netcdf(data, output_path):
    """Saves the PCA results to a netcdf file.

    Parameters
    ----------
    data : np.array
        A 4D array (time x lat x lon x PC1) containing the PCA results.
    output_path : str
        The path to save the netcdf file.
    """
    lat = np.linspace(-60, 89, 150)
    lon = np.linspace(0, 359, 360)
    years = np.arange(1990, 2021)
    months = np.arange(1, 13)
    days = np.arange(1, 29)

    # Reshape data to desired shape
    pc1 = data[:, :, :, :, :, 0]
    pc2 = data[:, :, :, :, :, 1]

    # Create DataArrays for PC1 and PC2
    pc1_da = xr.DataArray(
        pc1,
        coords={
            "year": ("year", years),
            "month": ("month", months),
            "day": ("day", days),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
        dims=["year", "month", "day", "lat", "lon"],
        name="PC1",
    )

    pc2_da = xr.DataArray(
        pc2,
        coords={
            "year": ("year", years),
            "month": ("month", months),
            "day": ("day", days),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
        dims=["year", "month", "day", "lat", "lon"],
        name="PC2",
    )

    # Combine into a Dataset
    ds = xr.Dataset({"PC1": pc1_da, "PC2": pc2_da})

    # Save to NetCDF
    ds.to_netcdf(output_path)