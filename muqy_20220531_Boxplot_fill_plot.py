# -*- coding: utf-8 -*-
# @Author: Muqy
# @Date:   2023-03-23 15:09:27
# @Last Modified by:   Muqy
# @Last Modified time: 2023-11-22 20:10:29
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

    This code consists of two main components: a BoxPlot class and an error_fill_plot 
    function. The primary purpose of this code is to process and visualize data related
    to cloud cover (Cld) matched with each Principal Component 1 (PC1) interval.

    The BoxPlot class takes an input array of cloud data with a shape of 
    (PC1_gap, lat, lon) and generates a box plot for the given data. 
    It first reshapes the input data into a pandas DataFrame using the 
    _convert_pandas method. This DataFrame has columns representing 
    each PC1 interval and rows representing reshaped latitude and longitude values. 
    The plot_box_plot method then generates a box plot using this DataFrame, 
    with whiskers representing the 10th and 90th percentiles, and a notch 
    representing the median. The x-axis represents the PC1 values, and the 
    y-axis shows the High Cloud Fraction (HCF) percentages.

    The error_fill_plot function takes an input array of cloud data with a 
    shape of (PC1_gap, lat, lon) and generates an error bar plot. The data 
    is reshaped to (PC1_gap, lat*lon), and the mean and standard deviation 
    are calculated for each PC1 interval. The function then creates a line plot 
    with the mean values, and an error bar (filled region) representing +/- one 
    standard deviation. The x-axis represents the PC1 values, and the y-axis 
    shows the variable of interest. The function also saves the generated 
    plot to a specified file path.

    In summary, this code provides tools to visualize the relationship 
    between PC1 intervals and cloud cover data in two different ways: 
    box plots and error bar plots. These visualizations can help researchers 
    better understand the patterns and trends in the given data.

    
    Owner: Mu Qingyu
    version 1.0
    Created: 2023-03-16
            
"""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BoxPlot:
    """
    Plot boxplot of Cld data match each PC1 interval
    """

    def __init__(self, cld_match_pc_gap: np.ndarray):
        """
        Initialize the class

        Parameters
        ----------
        cld_match_pc_gap : Cld data fit in PC1 interval
            array shape in (PC1_gap, lat, lon)
        """
        # Input array must be in shape of (PC1_gap, lat, lon)
        self.cld_match_pc_gap = cld_match_pc_gap

    def _convert_pandas(self) -> pd.DataFrame:
        gap_num = self.cld_match_pc_gap.shape[0]
        rows, cols = self.cld_match_pc_gap.shape[1:]

        box = np.zeros((rows * cols, gap_num))

        for i in range(gap_num):
            box[:, i] = self.cld_match_pc_gap[i].flatten()

        box = pd.DataFrame(box)
        # Number of PC1 interval is set in np.arange(start, end, interval gap)
        box.columns = np.round(np.arange(-1.5, 4.5, 0.05), 3)

        return box

    def plot_box_plot(self) -> None:
        """
        Plot boxplot of Cld data match each PC1 interval
        Main plot function
        """
        box = self._convert_pandas()

        fig, ax = plt.subplots(figsize=(18, 10))
        flierprops = dict(
            marker="o",
            markersize=7,
            markeredgecolor="grey",
        )
        box.boxplot(
            flierprops=flierprops,
            whis=[10, 90],
            meanline=None,
            showmeans=True,
            notch=True,
        )
        plt.xlabel("PC1", size=26, weight="bold")
        plt.ylabel("HCF (%)", size=26, weight="bold")
        plt.yticks(fontsize=26, weight="bold")
        plt.show()


def error_fill_plot(data: np.ndarray, xlabel: str, savefig_str: str) -> None:
    """
    Plot error bar of Cld data match each PC1 interval

    Args:
        data (array): Cld data fit in PC1 interval
            array shape in (PC1_gap, lat, lon)
    """
    # Input array must be in shape of (PC1_gap, lat, lon)
    # reshape data to (PC1_gap, lat*lon)
    data = data.reshape(data.shape[0], -1)

    # Calculate mean and std of each PC1 interval
    data_y = np.round(np.nanmean(data, axis=1), 3)
    data_x = np.round(np.arange(-2.5, 5.5, 0.05), 3)
    data_std = np.nanstd(data, axis=1)

    # Create up and down limit of error bar
    data_up = data_y + data_std
    data_down = data_y - data_std

    # Create a figure instance
    fig, ax = plt.subplots(figsize=(7, 5))

    plt.plot(data_x, data_y, linewidth=3, color="#A3AECC")
    plt.fill_between(data_x, data_up, data_down, facecolor="#A3AECC", alpha=0.5)

    # Add labels and title
    plt.xlabel("PC1")
    plt.ylabel(xlabel)
    
    # Save figure
    output_path = "/RAID01/data/python_fig/fill_between_plot_cld_var/"
    os.makedirs(output_path, exist_ok=True)

    plt.savefig(
        os.path.join(output_path, savefig_str),
        dpi=300,
        facecolor="w",
        edgecolor="w",
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    pass
