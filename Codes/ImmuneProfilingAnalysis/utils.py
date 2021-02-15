# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################

"""
Functions needed to run the notebooks
"""

"""
Import python packages
"""

import numpy as np

import sys
import os

import ipywidgets as widgets
import ipyfilechooser
from ipyfilechooser import FileChooser
from ipywidgets import HBox, Label, Layout

"""
Interfaces
"""

def define_parameters_interface():
    
    parameters = []

    print('\x1b[1m'+"Input directory for data")
    input_data_dir = FileChooser('./')
    display(input_data_dir)
    print('\x1b[1m'+"Output directory for files")
    output_files_dir = FileChooser('./')
    display(output_files_dir)
    print('\x1b[1m'+"Output directory for figures")
    output_figures_dir = FileChooser('./')
    display(output_figures_dir)
    
    
    label_layout = Layout(width='300px',height='30px')

    data_name = HBox([Label('Data name:', layout=label_layout), widgets.Text(
    value='', placeholder='Type something', description='', disabled=False)])
    display(data_name)
    
    pixel_width_micron = HBox([Label('Pixel width in microns:', layout=label_layout), widgets.FloatText(
        value=1, description='', disabled=False)])
    display(pixel_width_micron)
    x_coordinate = HBox([Label('Nucleus center x coordinate:', layout=label_layout), widgets.IntText(
        value=1, description='', disabled=False)])
    display(x_coordinate)
    y_coordinate = HBox([Label('Nucleus center y coordinate:', layout=label_layout), widgets.IntText(
        value=2, description='', disabled=False)])
    display(y_coordinate)
    tissue_column = HBox([Label('Column for tissue:', layout=label_layout), widgets.IntText(
        value=3, description='', disabled=False)])
    display(tissue_column)
    epithelium_id = HBox([Label('Epithelium id in tissue column:', layout=label_layout), widgets.IntText(
        value=1, description='', disabled=False)])
    display(epithelium_id)
    stroma_id = HBox([Label('Stroma id in tissue column:', layout=label_layout), widgets.IntText(
        value=2, description='', disabled=False)])
    display(stroma_id)
    phenotype_first_column = HBox([Label('First column for cell type:', layout=label_layout), widgets.IntText(
        value=4, description='', disabled=False)])
    display(phenotype_first_column)
    phenotype_last_column = HBox([Label('Last column for cell type:', layout=label_layout), widgets.IntText(
        value=5, description='', disabled=False)])
    display(phenotype_last_column)
    search_radius = HBox([Label('Radius search (um) for spatial distribution analysis:', layout=label_layout), widgets.FloatText(
        value=25, description='', disabled=False)])
    display(search_radius)


    
    parameters.append(input_data_dir)
    parameters.append(output_files_dir)
    parameters.append(output_figures_dir)
    parameters.append(data_name)
    parameters.append(pixel_width_micron)
    parameters.append(x_coordinate)
    parameters.append(y_coordinate)
    parameters.append(tissue_column)
    parameters.append(epithelium_id)
    parameters.append(stroma_id)
    parameters.append(phenotype_first_column)
    parameters.append(phenotype_last_column)
    parameters.append(search_radius)
    
    return parameters

def create_parameters_file(parameters):
    f = open("parameters.txt", "w+")
    f.write("input_data_dir\t"+parameters[0].selected+"\n")
    f.write("output_files_dir\t"+parameters[1].selected+"\n")
    f.write("output_figures_dir\t"+parameters[2].selected+"\n")
    f.write("data_name\t"+str(parameters[3].children[1].value)+"\n")
    f.write("pixel_width_micron\t"+str(parameters[4].children[1].value)+"\n")
    f.write("x_coordinate\t"+str(parameters[5].children[1].value)+"\n")
    f.write("y_coordinate\t"+str(parameters[6].children[1].value)+"\n")
    f.write("tissue_column\t"+str(parameters[7].children[1].value)+"\n")
    f.write("epithelium_id\t"+str(parameters[8].children[1].value)+"\n")
    f.write("stroma_id\t"+str(parameters[9].children[1].value)+"\n")
    f.write("phenotype_first_column\t"+str(parameters[10].children[1].value)+"\n")
    f.write("phenotype_last_column\t"+str(parameters[11].children[1].value)+"\n")
    f.write("search_radius\t"+str(parameters[12].children[1].value)+"\n")
    f.close()