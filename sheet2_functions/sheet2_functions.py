# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:06:46 2016

@author: Bert Coerver (b.coerver[at]unesco-ihe.org)
"""
from __future__ import print_function

from builtins import str
from builtins import zip
from builtins import range
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import xml.etree.ElementTree as ET
import datetime
import cairosvg

from WA_Hyperloop import hyperloop as hl
import WA_Hyperloop.becgis as becgis
import WA_Hyperloop.get_dictionaries as gd
from WA_Hyperloop.paths import get_path

def create_sheet2(complete_data, metadata, output_dir):
    
    if not np.all(['i' in list(complete_data.keys()), 't' in list(complete_data.keys())]):
        t_files, t_dates, i_files, i_dates = splitET_ITE(metadata['lu'],
                                                         complete_data['et'][0], 
                                                         complete_data['et'][1], 
                                                         complete_data['lai'][0], 
                                                         complete_data['lai'][1], 
                                                         complete_data['p'][0], 
                                                         complete_data['p'][1], 
                                                         complete_data['n'][0], 
                                                         complete_data['n'][1], 
                                                         complete_data['ndm'][0], 
                                                         complete_data['ndm'][1], 
                                                         os.path.join(output_dir, metadata['name'], 'data'), 
                                                         ndm_max_original = metadata['ndm_max_original'], 
                                                         plot_graph = True, 
                                                         save_e = False)
    
        complete_data['i'] = (i_files, i_dates)
        complete_data['t'] = (t_files, t_dates)
    
    output_dir = os.path.join(output_dir, metadata['name'], 'sheet2')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lulc_dict = gd.get_lulcs(lulc_version = '4.0')
    classes_dict = gd.get_sheet2_classes(version = '1.0')
    
    monthly_csvs, yearly_csvs = create_sheet2_csv(lulc_dict, classes_dict, metadata['lu'], 
                                                  metadata['water_year_start_month'],
                                                  complete_data['et'][0], complete_data['et'][1], 
                                                  complete_data['t'][0], complete_data['t'][1], 
                                                  complete_data['i'][0], complete_data['i'][1], 
                                                  output_dir, catchment_name = metadata['name'], 
                                                  full_years = True)

    for fh in yearly_csvs:
        output_fh = fh.replace('csv', 'png')
        year = str(fh[-8:-4])
        create_sheet2_png(metadata['name'], year, 'km3/year', fh, output_fh, template = get_path('sheet2_svg'), smart_unit = True)
        
    for fh in monthly_csvs:
        output_fh = fh.replace('csv', 'pdf')
        month = str(fh[-6:-4])
        year = str(fh[-11:-7])
        create_sheet2_png(metadata['name'], '{0}-{1}'.format(year, month), 'km3/month', fh, output_fh, template = get_path('sheet2_svg'), smart_unit = True)
        
    return complete_data

def create_sheet2_csv(lulc_dict, classes_dict, lu_fh, start_month, et_fhs, et_dates, t_fhs,
                      t_dates, i_fhs, i_dates, output_dir, catchment_name = None,
                      full_years = True):
    """
    Create sheet 2 csv-files.
    
    Parameters
    ----------
    lulc_dict : dict 
        Describing the different land use classes, import using 'get_dictionaries'.
    classes_dict   : dict   
        Describing the sheet 2 specific aggregation of classes from lulc_dict.
    lu_fh : str        
        Filehandle pointing to a landuse map.
    et_fhs : ndarray       
        Array of filehandles pointing to ET maps.
    et_dates : ndarray     
        Array with datetime.date objects specifying the date of the ET maps.
    t_fhs : ndarray      
        Array of filehandles pointing to T maps.
    t_dates : ndarray        
        Array with datetime.date objects specifying the date of the T maps.
    i_fhs : ndarray
        Array of filehandles pointing to I maps.
    i_dates : ndarray
        Array with datetime.date objects specifying the date of the I maps.    
    output_dir : str  
        Filehandle to folder to save output.
    catchment_name : str, optional
        Name of the catchment, default is None.
    full_years : boolean, optional  
        Choose to also create yearly csv-files, default is True.
    
    Returns
    -------
    csv_fhs : ndarray      
        Array of filehandles pointing to monthly csv-files.
    csv_fhs_yearly : ndarray    
        Array of filehandles pointing to yearly csv-files.
    """
    
    # Check if all maps have the same projection, resolution and No-Data-Value.
    becgis.assert_proj_res_ndv([lu_fh, et_fhs, t_fhs, i_fhs])
    
    # Calculate the size of each pixel in km2.
    MapArea = becgis.map_pixel_area_km(lu_fh)
    
    # Create some constants.
    month_labels = {1:'01',2:'02',3:'03',4:'04',5:'05',6:'06',7:'07',8:'08',9:'09',10:'10',11:'11',12:'12'}
    first_row = ['LAND_USE', 'CLASS', 'TRANSPIRATION', 'WATER', 'SOIL', 'INTERCEPTION', 'AGRICULTURE', 'ENVIRONMENT', 'ECONOMY', 'ENERGY', 'LEISURE', 'NON_BENEFICIAL']
    
    # Check if output folder exists.
    directory_months = os.path.join(output_dir, "sheet2_monthly")
    if not os.path.exists(directory_months):
        os.makedirs(directory_months)
    
    # Check if the name of the catchment is known.
    if catchment_name is None:
        catchment_name = 'Unknown'
    
    # Check for which dates calculations can be made.
    common_dates = becgis.common_dates([et_dates, t_dates, i_dates])
    water_dates = np.copy(common_dates)
    for w in water_dates:
        if w.month < start_month:
            water_dates[water_dates == w] = datetime.date(w.year-1, w.month, w.day)
    
    # Open the landuse-map.
    LULC = becgis.open_as_array(lu_fh)
    
    # Create some variables needed for yearly sheets.
    complete_years = [None]
    year_count = [None]
    if full_years:
        # Check if output folder for yearly csv-files exists.
        directory_years = os.path.join(output_dir, "sheet2_yearly")
        if not os.path.exists(directory_years):
            os.makedirs(directory_years)
        # Check for which years data for 12 months is available.
#        yrs, counts = np.unique([date.year for date in common_dates], return_counts = True)
        yrs, counts = np.unique([date.year for date in water_dates], return_counts = True)
        complete_years = [int(year) for year, count in zip(yrs, counts) if count == 12]
        year_count = 1
    
    # Start calculations.
    for (date, w_date) in zip(common_dates, water_dates):
        
        # Create csv-file.
        csv_filename = os.path.join(directory_months, '{0}_{1}_{2}.csv'.format(catchment_name, date.year, month_labels[date.month]))
        csv_file = open(csv_filename, 'w')
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(first_row)
        
        # Open the T, ET and I maps and set NDV pixels to NaN.
        T = becgis.open_as_array(t_fhs[t_dates == date][0], nan_values = True)
        ET = becgis.open_as_array(et_fhs[et_dates == date][0], nan_values = True)
        I = becgis.open_as_array(i_fhs[i_dates == date][0], nan_values = True)
                
        # Convert units from [mm/month] to [km3/month].
        I = I * MapArea / 1000000
        T = T * MapArea / 1000000
        ET = ET * MapArea / 1000000        
        
        # Add monthly values to yearly totals.
        if np.all([full_years, (w_date.year in complete_years), (year_count is 1)]):
            Tyear = T
            ETyear = ET
            Iyear = I
            year_count += 1
        
        # Add monthly values to yearly totals.
        elif np.all([full_years, (w_date.year in complete_years), (year_count is not 1)]):
            Tyear += T
            ETyear += ET
            Iyear += I
            year_count += 1
        
        # Calculate evaporation.
        E = ET - T - I
        
        # Write data to csv-file.
        for LAND_USE in list(classes_dict.keys()):
            for CLASS in list(classes_dict[LAND_USE].keys()):
                write_sheet2_row(LAND_USE, CLASS, lulc_dict, classes_dict, LULC, T, I, E, writer)
        
        # Close the csv-file.
        csv_file.close()

        # Start creating a yearly csv-file.
        if np.all([full_years, (w_date.year in complete_years), (year_count is 13)]):
            # Calcultate evaporation for a whole year.
            Eyear = ETyear - Tyear - Iyear
            
            # Create csv-file for yearly data.
            csv_filename = os.path.join(directory_years, '{0}_{1}.csv'.format(catchment_name, w_date.year))
            csv_file_year = open(csv_filename, 'w')
            writer_year = csv.writer(csv_file_year, delimiter=';')
            writer_year.writerow(first_row)
            
            # Write data to yearly csv-file.
            for LAND_USE in list(classes_dict.keys()):
                for CLASS in list(classes_dict[LAND_USE].keys()):
                    write_sheet2_row(LAND_USE, CLASS, lulc_dict, classes_dict, LULC, Tyear, Iyear, Eyear, writer_year)
            
            # Close csv-file.
            csv_file_year.close()
            
            # Set counter back to one.
            year_count = 1

    # Create list of created files.
    csv_fhs = becgis.list_files_in_folder(directory_months, extension = 'csv')
    
    # Return list of filehandles.
    if full_years:
        csv_fhs_yearly = becgis.list_files_in_folder(directory_years, extension = 'csv')
        return csv_fhs, csv_fhs_yearly
    else:
        return csv_fhs

def splitET_ITE(lu_fh, et_fhs, et_dates, lai_fhs, lai_dates, p_fhs, p_dates, n_fhs, n_dates, ndm_fhs, ndm_dates, output_dir, ndm_max_original = True, plot_graph = True, save_e = False):
    """
    Split evapotranspiration into transpiration and interception.
    
    Parameters
    ----------
    et_fhs : ndarray
        Array of filehandles pointing to ET maps.
    et_dates : ndarray
        Array with datetime.date objects specifying the date of the ET maps.
    lai_fhs : ndarray
        Array of filehandles pointing to LAI maps.
    lai_dates : ndarray
        Array with datetime.date objects specifying the date of the LAI maps.
    p_fhs : ndarray
        Array of filehandles pointing to P maps.
    p_dates : ndarray
        Array with datetime.date objects specifying the date of the P maps. 
    n_fhs : ndarray
        Array of filehandles pointing to rainy days maps.
    n_dates : ndarray
        Array with datetime.date objects specifying the date of the rainy days maps.
    ndm_fhs : ndarray
        Array of filehandles pointing to NDM maps.
    ndm_dates : ndarray
        Array with datetime.date objects specifying the date of the NDM maps. 
    output_dir : str
        Filehandle specifying the folder to save output.
    plot_graph : boolean, optional
        Plot a graph of the spatially averaged ET and the fractions of E, T and I, default is True.
    
    Returns
    -------
    i_fhs : ndarray
        Array of filehandles pointing to I maps.
    i_dates : ndarray
        Array with datetime.date objects specifying the date of the I maps.
    t_fhs : ndarray
        Array of filehandles pointing to T maps.
    t_dates : ndarray
        Array with datetime.date objects specifying the date of the T maps.
    """
    # Check if all maps have the same projection, resolution and No-Data-Value.
    becgis.assert_proj_res_ndv([et_fhs, lai_fhs, p_fhs, n_fhs, ndm_fhs])
    LU = becgis.open_as_array(lu_fh)
    # Create some constants.
    month_labels = {1:'01',2:'02',3:'03',4:'04',5:'05',6:'06',7:'07',8:'08',9:'09',10:'10',11:'11',12:'12'}
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.get_geoinfo(et_fhs[0])
    
    # Check for which dates calculations can be made.
    common_dates = becgis.common_dates([et_dates, lai_dates, p_dates, n_dates, ndm_dates])
    
    if not ndm_max_original:
        ndm_months = np.array([date.month for date in ndm_dates])
        
        ndm_max_folder = os.path.join(output_dir, "ndm_max")
        if not os.path.exists(ndm_max_folder):
                os.makedirs(ndm_max_folder)
                
        ndm_max_fhs = dict()

        footprint = np.ones((10,10), dtype = np.bool)
        
        for month in np.unique(ndm_months):
            std, mean = becgis.calc_mean_std(ndm_fhs[ndm_months == month])
            ndm_temporal_mean = mean #+ 2 * std
            ndm_temporal_mean [np.isnan(ndm_temporal_mean )] = 0.
            ndm_spatial_max = ndm_temporal_mean * 0.0
            for lu in np.unique(LU):
                ndm_lu = ndm_temporal_mean * 0.0
                ndm_lu[LU == lu] = ndm_temporal_mean[LU == lu]
                intermediate = ndimage.maximum_filter(ndm_lu, footprint = footprint)
                ndm_spatial_max[LU==lu] = intermediate[LU==lu]
            output_fh = os.path.join(ndm_max_folder, 'ndm_max_{0}.tif'.format(month_labels[month]))
            becgis.create_geotiff(output_fh, ndm_spatial_max, driver, NDV, xsize, ysize, GeoT, Projection)
            ndm_max_fhs[month] = output_fh
            del std, mean, ndm_spatial_max
    
    if ndm_max_original:
        # Create some variables to calculate the monthly maximum NDM.
        ndm_months = np.array([date.month for date in ndm_dates])
        unique_ndm_months, counts = np.unique(ndm_months, return_counts = True)
        NDMmax = dict()
        
        # Calculate the maximum average NDM value for each month.
        for month in unique_ndm_months:
            ndm_monthly_mean = np.zeros((ysize,xsize))
            for date in ndm_dates[ndm_months == month]:
                data = becgis.open_as_array(ndm_fhs[ndm_dates == date][0], nan_values = True)
                ndm_monthly_mean[:,:] += data
            ndm_monthly_mean[:,:] /= counts[unique_ndm_months == month]
            NDMmax[month] = np.nanmax(ndm_monthly_mean)
    
    # Create some variables needed to plot graphs.
    if plot_graph:
        et = np.array([])
        i = np.array([])
        t = np.array([])
        e = np.array([])
    
    # Start iterating over dates.
    for date in common_dates:
        # Open data to calculate I and set NDV pixels to NaN.
        LAI = becgis.open_as_array(lai_fhs[lai_dates == date][0], nan_values = True)
        P = becgis.open_as_array(p_fhs[p_dates == date][0], nan_values = True)
        n = becgis.open_as_array(n_fhs[n_dates == date][0], nan_values = True)
        
        # Calculate I.
        I = LAI * (1 - (1 + (P/n) * (1 - np.exp(-0.5 * LAI)) * (1/LAI))**-1) * n
        
        # Set boundary conditions.
        I[LAI == 0] = 0.
        I[n == 0] = 0.
        I[np.isnan(LAI)] = 0.
        
        # Open ET and NDM maps and set NDV pixels to NaN.
        ET = becgis.open_as_array(et_fhs[et_dates == date][0], nan_values = True)
        
        I = np.nanmin((I, ET), axis = 0)
        
        NDM = becgis.open_as_array(ndm_fhs[ndm_dates == date][0], nan_values = True)
        
        if ndm_max_original:
            NDMMAX = 0.95 / NDMmax[date.month]
        
        if not ndm_max_original:
            NDMMAX = 1.00 / becgis.open_as_array(ndm_max_fhs[date.month], nan_values = True)
    
        # Calculate T.
        T = np.nanmin(((NDM * NDMMAX),np.ones(np.shape(NDM)) * 0.95), axis = 0) * (ET - I)
            
        # Create folder to store maps.
        directory_t = os.path.join(output_dir, "t")
        if not os.path.exists(directory_t):
            os.makedirs(directory_t)
        
        directory_i = os.path.join(output_dir, "i")
        if not os.path.exists(directory_i):
            os.makedirs(directory_i)

        if save_e:
            directory_e = os.path.join(output_dir, "e")
            if not os.path.exists(directory_e):
                os.makedirs(directory_e)
            E = ET - I - T
            output_fh = os.path.join(directory_e, 'E_{0}{1}.tif'.format(date.year,month_labels[date.month]))
            becgis.create_geotiff(output_fh, E, driver, NDV, xsize, ysize, GeoT, Projection)
        
        # Store values to plot a graph.
        if plot_graph:
            et = np.append(et, np.nanmean(ET))
            i = np.append(i, np.nanmean(I))
            t = np.append(t, np.nanmean(T))
            e = np.append(e, np.nanmean(ET - I - T))
        
        # Save I map.
        output_fh = os.path.join(directory_i, 'I_{0}{1}.tif'.format(date.year,month_labels[date.month]))
        becgis.create_geotiff(output_fh, I, driver, NDV, xsize, ysize, GeoT, Projection)

        # Save T map.
        output_fh = os.path.join(directory_t, 'T_{0}{1}.tif'.format(date.year,month_labels[date.month]))
        becgis.create_geotiff(output_fh, T, driver, NDV, xsize, ysize, GeoT, Projection)
        
        print("Finished E,T,I for {0}".format(date))
    
    # Plot graph of ET and E, T and I fractions.
    if plot_graph:
        fig = plt.figure(figsize = (10,10))
        plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
        ax = fig.add_subplot(111)
        ax.plot(common_dates, et, color = 'k')
        ax.patch.set_visible(False)
        ax.set_title('Average ET and E, T and I fractions')
        ax.set_ylabel('ET [mm/month]')
        ax.patch.set_visible(True)
        ax.fill_between(common_dates, et, color = '#a3db76', label = 'Evaporation')
        ax.fill_between(common_dates, i + t , color = '#6bb8cc', label = 'Transpiration')
        ax.fill_between(common_dates, i , color = '#497e7c', label = 'Interception')
        ax.scatter(common_dates, et, color = 'k')
        ax.legend(loc = 'upper left',fancybox=True, shadow=True)
        fig.autofmt_xdate()
        ax.set_xlim([common_dates[0], common_dates[-1]])
        ax.set_ylim([0, max(et) *1.2])
        ax.set_xlabel('Time')
        [r.set_zorder(10) for r in ax.spines.values()]
        plt.savefig(os.path.join(output_dir,'ETfractions_ITE.png'))

    # Create arrays with filehandles and datetime.date objects on the created maps.
    t_fhs, t_dates, t_years, t_months, t_days = becgis.sort_files(directory_t, [-10,-6], month_position = [-6,-4])
    i_fhs, i_dates, i_years, i_months, i_days = becgis.sort_files(directory_i, [-10,-6], month_position = [-6,-4])
    
    return t_fhs, t_dates, i_fhs, i_dates

def calc_footprint(max_radius, pixel_size):
    """
    Calculate a footprint to select local maxima.
    
    Parameters
    ----------
    max_radius : float
        Radius of circle around each pixel in which maximum should be found.
    pixel_size : float
        Size of one pixel.
    
    Return
    ------
    footprint : ndarray
        Boolean array with circle of max_radius.
    
    """
    pixels = int(max_radius / pixel_size) + 1
    X, Y = np.meshgrid(list(range(pixels)), list(range(pixels)))
    radius = np.sqrt(X**2 + Y**2)
    half_circle = np.concatenate((np.flipud(radius), radius[1:,:]))
    half_circle_2 = np.fliplr(np.concatenate((np.flipud(radius), radius[1:,:])))[:,:-1]
    test = np.hstack((half_circle_2, half_circle))
    footprint = test <= pixels
    return footprint
    
def write_sheet2_row(LAND_USE, CLASS, lulc_dict, classes_dict, lulc, T, I, E, writer):
    """
    Write a row with spatial aggregates to a sheet2 csv-file.
    
    Parameters
    ----------
    LAND_USE : str
        The landuse category of the row.
    CLASS : str
        The class of the landuse category of the row.
    lulc_dict : dict 
        Describing the different land use classes, import using 'get_dictionaries'.
    classes_dict   : dict   
        Describing the sheet 2 specific aggregation of classes from lulc_dict.
    T : ndarray
        The spatial transpiration data.
    I : ndarray
        The spatial interception data.        
    E : ndarray
        The spatial evaporation data.
    writer : object
        csv.writer object.
    """
    # Get a list of the different landuse classes to be aggregated.
    lulcs = classes_dict[LAND_USE][CLASS]
    
    # Create a mask to ignore non relevant pixels.
    mask=np.logical_or.reduce([lulc == value for value in lulcs])
    
    # Calculate the spatial sum of the different parameters.
    transpiration = np.nansum(T[mask])
    interception = np.nansum(I[mask])
    evaporation = np.nansum(E[mask])
    
    # Set special cases.
    if np.any([CLASS == 'Natural water bodies', CLASS == 'Managed water bodies']):
        soil_evaporation = 0
        water_evaporation = evaporation
    else:
        soil_evaporation = evaporation
        water_evaporation = 0            
        
    # Create some necessary variables.
    agriculture = 0
    environment = 0
    economy = 0
    energy = 0
    leisure = 0
    non_beneficial = 0
    
    # Calculate several landuse type specific variables.
    for lu_type in lulcs:
        
        # Get some constants for the landuse type.
        beneficial_percentages = np.array(lulc_dict[lu_type][3:6]) / 100
        service_contributions = np.array(lulc_dict[lu_type][6:11]) / 100         
          
        # Calculate the beneficial ET.
        benef_et = np.nansum([np.nansum(T[lulc == lu_type]) * beneficial_percentages[0],
               np.nansum(E[lulc == lu_type]) * beneficial_percentages[1],
               np.nansum(I[lulc == lu_type]) * beneficial_percentages[2]])
               
        # Determine the service contributions.
        agriculture += benef_et * service_contributions[0] 
        environment += benef_et * service_contributions[1] 
        economy += benef_et * service_contributions[2]
        energy += benef_et * service_contributions[3]
        leisure += benef_et * service_contributions[4]
       
        # Determine non-beneficial ET.
        non_beneficial += (np.nansum([np.nansum(T[lulc == lu_type]) * (1 - beneficial_percentages[0]),
               np.nansum(E[lulc == lu_type]) * (1 - beneficial_percentages[1]),
               np.nansum(I[lulc == lu_type]) * (1 - beneficial_percentages[2])]))
    
    # Create the row to be written
    row = [LAND_USE, CLASS, "{0}".format(np.nansum([0, transpiration])), "{0}".format(np.nansum([0, water_evaporation])), "{0}".format(np.nansum([0, soil_evaporation])), "{0}".format(np.nansum([0, interception])), "{0}".format(np.nansum([0, agriculture])), "{0}".format(np.nansum([0, environment])), "{0}".format(np.nansum([0, economy])), "{0}".format(np.nansum([0, energy])), "{0}".format(np.nansum([0, leisure])), "{0}".format(np.nansum([0, non_beneficial]))]
    
    # Write the row.
    writer.writerow(row)

def create_sheet2_png(basin, period, units, data, output, template=False,
                  tolerance=0.2, smart_unit = False):
    """

    Keyword arguments:
    basin -- The name of the basin
    period -- The period of analysis
    units -- The units of the data
    data -- A csv file that contains the water data. The csv file has to
            follow an specific format. A sample csv is available in the link:
            https://github.com/wateraccounting/wa/tree/master/Sheets/csv
    output -- The output path of the jpg file for the sheet.
    template -- A svg file of the sheet. Use False (default) to use the
                standard svg file.
    tolerance -- Tolerance (in km3/year) of the difference in total ET
                 measured from (1) evaporation and transpiration and
                 (2) beneficial and non-beneficial ET.

    Example:
    from wa.Sheets import *
    create_sheet2(basin='Nile Basin', period='2010', units='km3/year',
                  data=r'C:\Sheets\csv\Sample_sheet2.csv',
                  output=r'C:\Sheets\sheet_2.jpg')
    """

    # Read table

    df = pd.read_csv(data, sep=';')

    scale = 0
    if smart_unit:
        scale_test = np.nansum(df['TRANSPIRATION'].values + 
                               df['WATER'].values + 
                               df['SOIL'].values + 
                               df['INTERCEPTION'].values)
        scale = hl.scale_factor(scale_test)
        
        list_of_vars = [key for key in list(df.keys())]
        
        for vari in ['LAND_USE', 'CLASS']:
            idx = list_of_vars.index(vari)           
            del list_of_vars[idx]
        
        for vari in list_of_vars:
            df[vari] *= 10**scale
        
    # Data frames

    df_Pr = df.loc[df.LAND_USE == "PROTECTED"]
    df_Ut = df.loc[df.LAND_USE == "UTILIZED"]
    df_Mo = df.loc[df.LAND_USE == "MODIFIED"]
    df_Mc = df.loc[df.LAND_USE == "MANAGED CONVENTIONAL"]
    df_Mn = df.loc[df.LAND_USE == "MANAGED NON_CONVENTIONAL"]

    # Column 1: Transpiration

    c1r1_t1 = float(df_Pr.loc[df_Pr.CLASS == "Forest"].TRANSPIRATION)
    c1r2_t1 = float(df_Pr.loc[df_Pr.CLASS == "Shrubland"].TRANSPIRATION)
    c1r3_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural grasslands"].TRANSPIRATION)
    c1r4_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural water bodies"].TRANSPIRATION)
    c1r5_t1 = float(df_Pr.loc[df_Pr.CLASS == "Wetlands"].TRANSPIRATION)
    c1r6_t1 = float(df_Pr.loc[df_Pr.CLASS == "Glaciers"].TRANSPIRATION)
    c1r7_t1 = float(df_Pr.loc[df_Pr.CLASS == "Others"].TRANSPIRATION)
    c1_t1_total = c1r1_t1 + c1r2_t1 + c1r3_t1 + c1r4_t1 + c1r5_t1 + \
        c1r6_t1 + c1r7_t1

    c1r1_t2 = float(df_Ut.loc[df_Ut.CLASS == "Forest"].TRANSPIRATION)
    c1r2_t2 = float(df_Ut.loc[df_Ut.CLASS == "Shrubland"].TRANSPIRATION)
    c1r3_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural grasslands"].TRANSPIRATION)
    c1r4_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural water bodies"].TRANSPIRATION)
    c1r5_t2 = float(df_Ut.loc[df_Ut.CLASS == "Wetlands"].TRANSPIRATION)
    c1r6_t2 = float(df_Ut.loc[df_Ut.CLASS == "Others"].TRANSPIRATION)
    c1_t2_total = c1r1_t2 + c1r2_t2 + c1r3_t2 + c1r4_t2 + c1r5_t2 + c1r6_t2

    c1r1_t3 = float(df_Mo.loc[df_Mo.CLASS == "Rainfed crops"].TRANSPIRATION)
    c1r2_t3 = float(df_Mo.loc[df_Mo.CLASS == "Forest plantations"].TRANSPIRATION)
    c1r3_t3 = float(df_Mo.loc[df_Mo.CLASS == "Settlements"].TRANSPIRATION)
    c1r4_t3 = float(df_Mo.loc[df_Mo.CLASS == "Others"].TRANSPIRATION)
    c1_t3_total = c1r1_t3 + c1r2_t3 + c1r3_t3 + c1r4_t3

    c1r1_t4 = float(df_Mc.loc[df_Mc.CLASS == "Irrigated crops"].TRANSPIRATION)
    c1r2_t4 = float(df_Mc.loc[df_Mc.CLASS == "Managed water bodies"].TRANSPIRATION)
    c1r3_t4 = float(df_Mc.loc[df_Mc.CLASS == "Residential"].TRANSPIRATION)
    c1r4_t4 = float(df_Mc.loc[df_Mc.CLASS == "Industry"].TRANSPIRATION)
    c1r5_t4 = float(df_Mc.loc[df_Mc.CLASS == "Others"].TRANSPIRATION)
    c1_t4_total = c1r1_t4 + c1r2_t4 + c1r3_t4 + c1r4_t4 + c1r5_t4

    c1r1_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor domestic"].TRANSPIRATION)
    c1r2_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor industry"].TRANSPIRATION)
    c1r3_t5 = float(df_Mn.loc[df_Mn.CLASS == "Greenhouses"].TRANSPIRATION)
    c1r4_t5 = float(df_Mn.loc[df_Mn.CLASS == "Livestock and husbandry"].TRANSPIRATION)
    c1r5_t5 = float(df_Mn.loc[df_Mn.CLASS == "Power and energy"].TRANSPIRATION)
    c1r6_t5 = float(df_Mn.loc[df_Mn.CLASS == "Others"].TRANSPIRATION)
    c1_t5_total = c1r1_t5 + c1r2_t5 + c1r3_t5 + c1r4_t5 + c1r5_t5 + c1r6_t5

    # Column 2: Water

    c2r1_t1 = float(df_Pr.loc[df_Pr.CLASS == "Forest"].WATER)
    c2r2_t1 = float(df_Pr.loc[df_Pr.CLASS == "Shrubland"].WATER)
    c2r3_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural grasslands"].WATER)
    c2r4_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural water bodies"].WATER)
    c2r5_t1 = float(df_Pr.loc[df_Pr.CLASS == "Wetlands"].WATER)
    c2r6_t1 = float(df_Pr.loc[df_Pr.CLASS == "Glaciers"].WATER)
    c2r7_t1 = float(df_Pr.loc[df_Pr.CLASS == "Others"].WATER)
    c2_t1_total = c2r1_t1 + c2r2_t1 + c2r3_t1 + c2r4_t1 + c2r5_t1 + \
        c2r6_t1 + c2r7_t1

    c2r1_t2 = float(df_Ut.loc[df_Ut.CLASS == "Forest"].WATER)
    c2r2_t2 = float(df_Ut.loc[df_Ut.CLASS == "Shrubland"].WATER)
    c2r3_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural grasslands"].WATER)
    c2r4_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural water bodies"].WATER)
    c2r5_t2 = float(df_Ut.loc[df_Ut.CLASS == "Wetlands"].WATER)
    c2r6_t2 = float(df_Ut.loc[df_Ut.CLASS == "Others"].WATER)
    c2_t2_total = c2r1_t2 + c2r2_t2 + c2r3_t2 + c2r4_t2 + c2r5_t2 + c2r6_t2

    c2r1_t3 = float(df_Mo.loc[df_Mo.CLASS == "Rainfed crops"].WATER)
    c2r2_t3 = float(df_Mo.loc[df_Mo.CLASS == "Forest plantations"].WATER)
    c2r3_t3 = float(df_Mo.loc[df_Mo.CLASS == "Settlements"].WATER)
    c2r4_t3 = float(df_Mo.loc[df_Mo.CLASS == "Others"].WATER)
    c2_t3_total = c2r1_t3 + c2r2_t3 + c2r3_t3 + c2r4_t3

    c2r1_t4 = float(df_Mc.loc[df_Mc.CLASS == "Irrigated crops"].WATER)
    c2r2_t4 = float(df_Mc.loc[df_Mc.CLASS == "Managed water bodies"].WATER)
    c2r3_t4 = float(df_Mc.loc[df_Mc.CLASS == "Residential"].WATER)
    c2r4_t4 = float(df_Mc.loc[df_Mc.CLASS == "Industry"].WATER)
    c2r5_t4 = float(df_Mc.loc[df_Mc.CLASS == "Others"].WATER)
    c2_t4_total = c2r1_t4 + c2r2_t4 + c2r3_t4 + c2r4_t4 + c2r5_t4

    c2r1_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor domestic"].WATER)
    c2r2_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor industry"].WATER)
    c2r3_t5 = float(df_Mn.loc[df_Mn.CLASS == "Greenhouses"].WATER)
    c2r4_t5 = float(df_Mn.loc[df_Mn.CLASS == "Livestock and husbandry"].WATER)
    c2r5_t5 = float(df_Mn.loc[df_Mn.CLASS == "Power and energy"].WATER)
    c2r6_t5 = float(df_Mn.loc[df_Mn.CLASS == "Others"].WATER)
    c2_t5_total = c2r1_t5 + c2r2_t5 + c2r3_t5 + c2r4_t5 + c2r5_t5 + c2r6_t5

    # Column 3: Soil

    c3r1_t1 = float(df_Pr.loc[df_Pr.CLASS == "Forest"].SOIL)
    c3r2_t1 = float(df_Pr.loc[df_Pr.CLASS == "Shrubland"].SOIL)
    c3r3_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural grasslands"].SOIL)
    c3r4_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural water bodies"].SOIL)
    c3r5_t1 = float(df_Pr.loc[df_Pr.CLASS == "Wetlands"].SOIL)
    c3r6_t1 = float(df_Pr.loc[df_Pr.CLASS == "Glaciers"].SOIL)
    c3r7_t1 = float(df_Pr.loc[df_Pr.CLASS == "Others"].SOIL)
    c3_t1_total = c3r1_t1 + c3r2_t1 + c3r3_t1 + c3r4_t1 + c3r5_t1 + \
        c3r6_t1 + c3r7_t1

    c3r1_t2 = float(df_Ut.loc[df_Ut.CLASS == "Forest"].SOIL)
    c3r2_t2 = float(df_Ut.loc[df_Ut.CLASS == "Shrubland"].SOIL)
    c3r3_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural grasslands"].SOIL)
    c3r4_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural water bodies"].SOIL)
    c3r5_t2 = float(df_Ut.loc[df_Ut.CLASS == "Wetlands"].SOIL)
    c3r6_t2 = float(df_Ut.loc[df_Ut.CLASS == "Others"].SOIL)
    c3_t2_total = c3r1_t2 + c3r2_t2 + c3r3_t2 + c3r4_t2 + c3r5_t2 + c3r6_t2

    c3r1_t3 = float(df_Mo.loc[df_Mo.CLASS == "Rainfed crops"].SOIL)
    c3r2_t3 = float(df_Mo.loc[df_Mo.CLASS == "Forest plantations"].SOIL)
    c3r3_t3 = float(df_Mo.loc[df_Mo.CLASS == "Settlements"].SOIL)
    c3r4_t3 = float(df_Mo.loc[df_Mo.CLASS == "Others"].SOIL)
    c3_t3_total = c3r1_t3 + c3r2_t3 + c3r3_t3 + c3r4_t3

    c3r1_t4 = float(df_Mc.loc[df_Mc.CLASS == "Irrigated crops"].SOIL)
    c3r2_t4 = float(df_Mc.loc[df_Mc.CLASS == "Managed water bodies"].SOIL)
    c3r3_t4 = float(df_Mc.loc[df_Mc.CLASS == "Residential"].SOIL)
    c3r4_t4 = float(df_Mc.loc[df_Mc.CLASS == "Industry"].SOIL)
    c3r5_t4 = float(df_Mc.loc[df_Mc.CLASS == "Others"].SOIL)
    c3_t4_total = c3r1_t4 + c3r2_t4 + c3r3_t4 + c3r4_t4 + c3r5_t4

    c3r1_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor domestic"].SOIL)
    c3r2_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor industry"].SOIL)
    c3r3_t5 = float(df_Mn.loc[df_Mn.CLASS == "Greenhouses"].SOIL)
    c3r4_t5 = float(df_Mn.loc[df_Mn.CLASS == "Livestock and husbandry"].SOIL)
    c3r5_t5 = float(df_Mn.loc[df_Mn.CLASS == "Power and energy"].SOIL)
    c3r6_t5 = float(df_Mn.loc[df_Mn.CLASS == "Others"].SOIL)
    c3_t5_total = c3r1_t5 + c3r2_t5 + c3r3_t5 + c3r4_t5 + c3r5_t5 + c3r6_t5

    # Column 4: INTERCEPTION

    c4r1_t1 = float(df_Pr.loc[df_Pr.CLASS == "Forest"].INTERCEPTION)
    c4r2_t1 = float(df_Pr.loc[df_Pr.CLASS == "Shrubland"].INTERCEPTION)
    c4r3_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural grasslands"].INTERCEPTION)
    c4r4_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural water bodies"].INTERCEPTION)
    c4r5_t1 = float(df_Pr.loc[df_Pr.CLASS == "Wetlands"].INTERCEPTION)
    c4r6_t1 = float(df_Pr.loc[df_Pr.CLASS == "Glaciers"].INTERCEPTION)
    c4r7_t1 = float(df_Pr.loc[df_Pr.CLASS == "Others"].INTERCEPTION)
    c4_t1_total = c4r1_t1 + c4r2_t1 + c4r3_t1 + c4r4_t1 + c4r5_t1 + \
        c4r6_t1 + c4r7_t1

    c4r1_t2 = float(df_Ut.loc[df_Ut.CLASS == "Forest"].INTERCEPTION)
    c4r2_t2 = float(df_Ut.loc[df_Ut.CLASS == "Shrubland"].INTERCEPTION)
    c4r3_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural grasslands"].INTERCEPTION)
    c4r4_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural water bodies"].INTERCEPTION)
    c4r5_t2 = float(df_Ut.loc[df_Ut.CLASS == "Wetlands"].INTERCEPTION)
    c4r6_t2 = float(df_Ut.loc[df_Ut.CLASS == "Others"].INTERCEPTION)
    c4_t2_total = c4r1_t2 + c4r2_t2 + c4r3_t2 + c4r4_t2 + c4r5_t2 + c4r6_t2

    c4r1_t3 = float(df_Mo.loc[df_Mo.CLASS == "Rainfed crops"].INTERCEPTION)
    c4r2_t3 = float(df_Mo.loc[df_Mo.CLASS == "Forest plantations"].INTERCEPTION)
    c4r3_t3 = float(df_Mo.loc[df_Mo.CLASS == "Settlements"].INTERCEPTION)
    c4r4_t3 = float(df_Mo.loc[df_Mo.CLASS == "Others"].INTERCEPTION)
    c4_t3_total = c4r1_t3 + c4r2_t3 + c4r3_t3 + c4r4_t3

    c4r1_t4 = float(df_Mc.loc[df_Mc.CLASS == "Irrigated crops"].INTERCEPTION)
    c4r2_t4 = float(df_Mc.loc[df_Mc.CLASS == "Managed water bodies"].INTERCEPTION)
    c4r3_t4 = float(df_Mc.loc[df_Mc.CLASS == "Residential"].INTERCEPTION)
    c4r4_t4 = float(df_Mc.loc[df_Mc.CLASS == "Industry"].INTERCEPTION)
    c4r5_t4 = float(df_Mc.loc[df_Mc.CLASS == "Others"].INTERCEPTION)
    c4_t4_total = c4r1_t4 + c4r2_t4 + c4r3_t4 + c4r4_t4 + c4r5_t4

    c4r1_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor domestic"].INTERCEPTION)
    c4r2_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor industry"].INTERCEPTION)
    c4r3_t5 = float(df_Mn.loc[df_Mn.CLASS == "Greenhouses"].INTERCEPTION)
    c4r4_t5 = float(df_Mn.loc[df_Mn.CLASS == "Livestock and husbandry"].INTERCEPTION)
    c4r5_t5 = float(df_Mn.loc[df_Mn.CLASS == "Power and energy"].INTERCEPTION)
    c4r6_t5 = float(df_Mn.loc[df_Mn.CLASS == "Others"].INTERCEPTION)
    c4_t5_total = c4r1_t5 + c4r2_t5 + c4r3_t5 + c4r4_t5 + c4r5_t5 + c4r6_t5

    # Column 6: AGRICULTURE

    c6r1_t1 = float(df_Pr.loc[df_Pr.CLASS == "Forest"].AGRICULTURE)
    c6r2_t1 = float(df_Pr.loc[df_Pr.CLASS == "Shrubland"].AGRICULTURE)
    c6r3_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural grasslands"].AGRICULTURE)
    c6r4_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural water bodies"].AGRICULTURE)
    c6r5_t1 = float(df_Pr.loc[df_Pr.CLASS == "Wetlands"].AGRICULTURE)
    c6r6_t1 = float(df_Pr.loc[df_Pr.CLASS == "Glaciers"].AGRICULTURE)
    c6r7_t1 = float(df_Pr.loc[df_Pr.CLASS == "Others"].AGRICULTURE)
    c6_t1_total = c6r1_t1 + c6r2_t1 + c6r3_t1 + c6r4_t1 + c6r5_t1 + \
        c6r6_t1 + c6r7_t1

    c6r1_t2 = float(df_Ut.loc[df_Ut.CLASS == "Forest"].AGRICULTURE)
    c6r2_t2 = float(df_Ut.loc[df_Ut.CLASS == "Shrubland"].AGRICULTURE)
    c6r3_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural grasslands"].AGRICULTURE)
    c6r4_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural water bodies"].AGRICULTURE)
    c6r5_t2 = float(df_Ut.loc[df_Ut.CLASS == "Wetlands"].AGRICULTURE)
    c6r6_t2 = float(df_Ut.loc[df_Ut.CLASS == "Others"].AGRICULTURE)
    c6_t2_total = c6r1_t2 + c6r2_t2 + c6r3_t2 + c6r4_t2 + c6r5_t2 + c6r6_t2

    c6r1_t3 = float(df_Mo.loc[df_Mo.CLASS == "Rainfed crops"].AGRICULTURE)
    c6r2_t3 = float(df_Mo.loc[df_Mo.CLASS == "Forest plantations"].AGRICULTURE)
    c6r3_t3 = float(df_Mo.loc[df_Mo.CLASS == "Settlements"].AGRICULTURE)
    c6r4_t3 = float(df_Mo.loc[df_Mo.CLASS == "Others"].AGRICULTURE)
    c6_t3_total = c6r1_t3 + c6r2_t3 + c6r3_t3 + c6r4_t3

    c6r1_t4 = float(df_Mc.loc[df_Mc.CLASS == "Irrigated crops"].AGRICULTURE)
    c6r2_t4 = float(df_Mc.loc[df_Mc.CLASS == "Managed water bodies"].AGRICULTURE)
    c6r3_t4 = float(df_Mc.loc[df_Mc.CLASS == "Residential"].AGRICULTURE)
    c6r4_t4 = float(df_Mc.loc[df_Mc.CLASS == "Industry"].AGRICULTURE)
    c6r5_t4 = float(df_Mc.loc[df_Mc.CLASS == "Others"].AGRICULTURE)
    c6_t4_total = c6r1_t4 + c6r2_t4 + c6r3_t4 + c6r4_t4 + c6r5_t4

    c6r1_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor domestic"].AGRICULTURE)
    c6r2_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor industry"].AGRICULTURE)
    c6r3_t5 = float(df_Mn.loc[df_Mn.CLASS == "Greenhouses"].AGRICULTURE)
    c6r4_t5 = float(df_Mn.loc[df_Mn.CLASS == "Livestock and husbandry"].AGRICULTURE)
    c6r5_t5 = float(df_Mn.loc[df_Mn.CLASS == "Power and energy"].AGRICULTURE)
    c6r6_t5 = float(df_Mn.loc[df_Mn.CLASS == "Others"].AGRICULTURE)
    c6_t5_total = c6r1_t5 + c6r2_t5 + c6r3_t5 + c6r4_t5 + c6r5_t5 + c6r6_t5

    # Column 7: ENVIRONMENT

    c7r1_t1 = float(df_Pr.loc[df_Pr.CLASS == "Forest"].ENVIRONMENT)
    c7r2_t1 = float(df_Pr.loc[df_Pr.CLASS == "Shrubland"].ENVIRONMENT)
    c7r3_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural grasslands"].ENVIRONMENT)
    c7r4_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural water bodies"].ENVIRONMENT)
    c7r5_t1 = float(df_Pr.loc[df_Pr.CLASS == "Wetlands"].ENVIRONMENT)
    c7r6_t1 = float(df_Pr.loc[df_Pr.CLASS == "Glaciers"].ENVIRONMENT)
    c7r7_t1 = float(df_Pr.loc[df_Pr.CLASS == "Others"].ENVIRONMENT)
    c7_t1_total = c7r1_t1 + c7r2_t1 + c7r3_t1 + c7r4_t1 + c7r5_t1 + \
        c7r6_t1 + c7r7_t1

    c7r1_t2 = float(df_Ut.loc[df_Ut.CLASS == "Forest"].ENVIRONMENT)
    c7r2_t2 = float(df_Ut.loc[df_Ut.CLASS == "Shrubland"].ENVIRONMENT)
    c7r3_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural grasslands"].ENVIRONMENT)
    c7r4_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural water bodies"].ENVIRONMENT)
    c7r5_t2 = float(df_Ut.loc[df_Ut.CLASS == "Wetlands"].ENVIRONMENT)
    c7r6_t2 = float(df_Ut.loc[df_Ut.CLASS == "Others"].ENVIRONMENT)
    c7_t2_total = c7r1_t2 + c7r2_t2 + c7r3_t2 + c7r4_t2 + c7r5_t2 + c7r6_t2

    c7r1_t3 = float(df_Mo.loc[df_Mo.CLASS == "Rainfed crops"].ENVIRONMENT)
    c7r2_t3 = float(df_Mo.loc[df_Mo.CLASS == "Forest plantations"].ENVIRONMENT)
    c7r3_t3 = float(df_Mo.loc[df_Mo.CLASS == "Settlements"].ENVIRONMENT)
    c7r4_t3 = float(df_Mo.loc[df_Mo.CLASS == "Others"].ENVIRONMENT)
    c7_t3_total = c7r1_t3 + c7r2_t3 + c7r3_t3 + c7r4_t3

    c7r1_t4 = float(df_Mc.loc[df_Mc.CLASS == "Irrigated crops"].ENVIRONMENT)
    c7r2_t4 = float(df_Mc.loc[df_Mc.CLASS == "Managed water bodies"].ENVIRONMENT)
    c7r3_t4 = float(df_Mc.loc[df_Mc.CLASS == "Residential"].ENVIRONMENT)
    c7r4_t4 = float(df_Mc.loc[df_Mc.CLASS == "Industry"].ENVIRONMENT)
    c7r5_t4 = float(df_Mc.loc[df_Mc.CLASS == "Others"].ENVIRONMENT)
    c7_t4_total = c7r1_t4 + c7r2_t4 + c7r3_t4 + c7r4_t4 + c7r5_t4

    c7r1_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor domestic"].ENVIRONMENT)
    c7r2_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor industry"].ENVIRONMENT)
    c7r3_t5 = float(df_Mn.loc[df_Mn.CLASS == "Greenhouses"].ENVIRONMENT)
    c7r4_t5 = float(df_Mn.loc[df_Mn.CLASS == "Livestock and husbandry"].ENVIRONMENT)
    c7r5_t5 = float(df_Mn.loc[df_Mn.CLASS == "Power and energy"].ENVIRONMENT)
    c7r6_t5 = float(df_Mn.loc[df_Mn.CLASS == "Others"].ENVIRONMENT)
    c7_t5_total = c7r1_t5 + c7r2_t5 + c7r3_t5 + c7r4_t5 + c7r5_t5 + c7r6_t5

    # Column 8: ECONOMY

    c8r1_t1 = float(df_Pr.loc[df_Pr.CLASS == "Forest"].ECONOMY)
    c8r2_t1 = float(df_Pr.loc[df_Pr.CLASS == "Shrubland"].ECONOMY)
    c8r3_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural grasslands"].ECONOMY)
    c8r4_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural water bodies"].ECONOMY)
    c8r5_t1 = float(df_Pr.loc[df_Pr.CLASS == "Wetlands"].ECONOMY)
    c8r6_t1 = float(df_Pr.loc[df_Pr.CLASS == "Glaciers"].ECONOMY)
    c8r7_t1 = float(df_Pr.loc[df_Pr.CLASS == "Others"].ECONOMY)
    c8_t1_total = c8r1_t1 + c8r2_t1 + c8r3_t1 + c8r4_t1 + c8r5_t1 + \
        c8r6_t1 + c8r7_t1

    c8r1_t2 = float(df_Ut.loc[df_Ut.CLASS == "Forest"].ECONOMY)
    c8r2_t2 = float(df_Ut.loc[df_Ut.CLASS == "Shrubland"].ECONOMY)
    c8r3_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural grasslands"].ECONOMY)
    c8r4_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural water bodies"].ECONOMY)
    c8r5_t2 = float(df_Ut.loc[df_Ut.CLASS == "Wetlands"].ECONOMY)
    c8r6_t2 = float(df_Ut.loc[df_Ut.CLASS == "Others"].ECONOMY)
    c8_t2_total = c8r1_t2 + c8r2_t2 + c8r3_t2 + c8r4_t2 + c8r5_t2 + c8r6_t2

    c8r1_t3 = float(df_Mo.loc[df_Mo.CLASS == "Rainfed crops"].ECONOMY)
    c8r2_t3 = float(df_Mo.loc[df_Mo.CLASS == "Forest plantations"].ECONOMY)
    c8r3_t3 = float(df_Mo.loc[df_Mo.CLASS == "Settlements"].ECONOMY)
    c8r4_t3 = float(df_Mo.loc[df_Mo.CLASS == "Others"].ECONOMY)
    c8_t3_total = c8r1_t3 + c8r2_t3 + c8r3_t3 + c8r4_t3

    c8r1_t4 = float(df_Mc.loc[df_Mc.CLASS == "Irrigated crops"].ECONOMY)
    c8r2_t4 = float(df_Mc.loc[df_Mc.CLASS == "Managed water bodies"].ECONOMY)
    c8r3_t4 = float(df_Mc.loc[df_Mc.CLASS == "Residential"].ECONOMY)
    c8r4_t4 = float(df_Mc.loc[df_Mc.CLASS == "Industry"].ECONOMY)
    c8r5_t4 = float(df_Mc.loc[df_Mc.CLASS == "Others"].ECONOMY)
    c8_t4_total = c8r1_t4 + c8r2_t4 + c8r3_t4 + c8r4_t4 + c8r5_t4

    c8r1_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor domestic"].ECONOMY)
    c8r2_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor industry"].ECONOMY)
    c8r3_t5 = float(df_Mn.loc[df_Mn.CLASS == "Greenhouses"].ECONOMY)
    c8r4_t5 = float(df_Mn.loc[df_Mn.CLASS == "Livestock and husbandry"].ECONOMY)
    c8r5_t5 = float(df_Mn.loc[df_Mn.CLASS == "Power and energy"].ECONOMY)
    c8r6_t5 = float(df_Mn.loc[df_Mn.CLASS == "Others"].ECONOMY)
    c8_t5_total = c8r1_t5 + c8r2_t5 + c8r3_t5 + c8r4_t5 + c8r5_t5 + c8r6_t5

    # Column 9: ENERGY

    c9r1_t1 = float(df_Pr.loc[df_Pr.CLASS == "Forest"].ENERGY)
    c9r2_t1 = float(df_Pr.loc[df_Pr.CLASS == "Shrubland"].ENERGY)
    c9r3_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural grasslands"].ENERGY)
    c9r4_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural water bodies"].ENERGY)
    c9r5_t1 = float(df_Pr.loc[df_Pr.CLASS == "Wetlands"].ENERGY)
    c9r6_t1 = float(df_Pr.loc[df_Pr.CLASS == "Glaciers"].ENERGY)
    c9r7_t1 = float(df_Pr.loc[df_Pr.CLASS == "Others"].ENERGY)
    c9_t1_total = c9r1_t1 + c9r2_t1 + c9r3_t1 + c9r4_t1 + c9r5_t1 + \
        c9r6_t1 + c9r7_t1

    c9r1_t2 = float(df_Ut.loc[df_Ut.CLASS == "Forest"].ENERGY)
    c9r2_t2 = float(df_Ut.loc[df_Ut.CLASS == "Shrubland"].ENERGY)
    c9r3_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural grasslands"].ENERGY)
    c9r4_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural water bodies"].ENERGY)
    c9r5_t2 = float(df_Ut.loc[df_Ut.CLASS == "Wetlands"].ENERGY)
    c9r6_t2 = float(df_Ut.loc[df_Ut.CLASS == "Others"].ENERGY)
    c9_t2_total = c9r1_t2 + c9r2_t2 + c9r3_t2 + c9r4_t2 + c9r5_t2 + c9r6_t2

    c9r1_t3 = float(df_Mo.loc[df_Mo.CLASS == "Rainfed crops"].ENERGY)
    c9r2_t3 = float(df_Mo.loc[df_Mo.CLASS == "Forest plantations"].ENERGY)
    c9r3_t3 = float(df_Mo.loc[df_Mo.CLASS == "Settlements"].ENERGY)
    c9r4_t3 = float(df_Mo.loc[df_Mo.CLASS == "Others"].ENERGY)
    c9_t3_total = c9r1_t3 + c9r2_t3 + c9r3_t3 + c9r4_t3

    c9r1_t4 = float(df_Mc.loc[df_Mc.CLASS == "Irrigated crops"].ENERGY)
    c9r2_t4 = float(df_Mc.loc[df_Mc.CLASS == "Managed water bodies"].ENERGY)
    c9r3_t4 = float(df_Mc.loc[df_Mc.CLASS == "Residential"].ENERGY)
    c9r4_t4 = float(df_Mc.loc[df_Mc.CLASS == "Industry"].ENERGY)
    c9r5_t4 = float(df_Mc.loc[df_Mc.CLASS == "Others"].ENERGY)
    c9_t4_total = c9r1_t4 + c9r2_t4 + c9r3_t4 + c9r4_t4 + c9r5_t4

    c9r1_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor domestic"].ENERGY)
    c9r2_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor industry"].ENERGY)
    c9r3_t5 = float(df_Mn.loc[df_Mn.CLASS == "Greenhouses"].ENERGY)
    c9r4_t5 = float(df_Mn.loc[df_Mn.CLASS == "Livestock and husbandry"].ENERGY)
    c9r5_t5 = float(df_Mn.loc[df_Mn.CLASS == "Power and energy"].ENERGY)
    c9r6_t5 = float(df_Mn.loc[df_Mn.CLASS == "Others"].ENERGY)
    c9_t5_total = c9r1_t5 + c9r2_t5 + c9r3_t5 + c9r4_t5 + c9r5_t5 + c9r6_t5

    # Column 10: LEISURE

    c10r1_t1 = float(df_Pr.loc[df_Pr.CLASS == "Forest"].LEISURE)
    c10r2_t1 = float(df_Pr.loc[df_Pr.CLASS == "Shrubland"].LEISURE)
    c10r3_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural grasslands"].LEISURE)
    c10r4_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural water bodies"].LEISURE)
    c10r5_t1 = float(df_Pr.loc[df_Pr.CLASS == "Wetlands"].LEISURE)
    c10r6_t1 = float(df_Pr.loc[df_Pr.CLASS == "Glaciers"].LEISURE)
    c10r7_t1 = float(df_Pr.loc[df_Pr.CLASS == "Others"].LEISURE)
    c10_t1_total = c10r1_t1 + c10r2_t1 + c10r3_t1 + c10r4_t1 + c10r5_t1 + \
        c10r6_t1 + c10r7_t1

    c10r1_t2 = float(df_Ut.loc[df_Ut.CLASS == "Forest"].LEISURE)
    c10r2_t2 = float(df_Ut.loc[df_Ut.CLASS == "Shrubland"].LEISURE)
    c10r3_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural grasslands"].LEISURE)
    c10r4_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural water bodies"].LEISURE)
    c10r5_t2 = float(df_Ut.loc[df_Ut.CLASS == "Wetlands"].LEISURE)
    c10r6_t2 = float(df_Ut.loc[df_Ut.CLASS == "Others"].LEISURE)
    c10_t2_total = c10r1_t2 + c10r2_t2 + c10r3_t2 + c10r4_t2 + \
        c10r5_t2 + c10r6_t2

    c10r1_t3 = float(df_Mo.loc[df_Mo.CLASS == "Rainfed crops"].LEISURE)
    c10r2_t3 = float(df_Mo.loc[df_Mo.CLASS == "Forest plantations"].LEISURE)
    c10r3_t3 = float(df_Mo.loc[df_Mo.CLASS == "Settlements"].LEISURE)
    c10r4_t3 = float(df_Mo.loc[df_Mo.CLASS == "Others"].LEISURE)
    c10_t3_total = c10r1_t3 + c10r2_t3 + c10r3_t3 + c10r4_t3

    c10r1_t4 = float(df_Mc.loc[df_Mc.CLASS == "Irrigated crops"].LEISURE)
    c10r2_t4 = float(df_Mc.loc[df_Mc.CLASS == "Managed water bodies"].LEISURE)
    c10r3_t4 = float(df_Mc.loc[df_Mc.CLASS == "Residential"].LEISURE)
    c10r4_t4 = float(df_Mc.loc[df_Mc.CLASS == "Industry"].LEISURE)
    c10r5_t4 = float(df_Mc.loc[df_Mc.CLASS == "Others"].LEISURE)
    c10_t4_total = c10r1_t4 + c10r2_t4 + c10r3_t4 + c10r4_t4 + c10r5_t4

    c10r1_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor domestic"].LEISURE)
    c10r2_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor industry"].LEISURE)
    c10r3_t5 = float(df_Mn.loc[df_Mn.CLASS == "Greenhouses"].LEISURE)
    c10r4_t5 = float(df_Mn.loc[df_Mn.CLASS == "Livestock and husbandry"].LEISURE)
    c10r5_t5 = float(df_Mn.loc[df_Mn.CLASS == "Power and energy"].LEISURE)
    c10r6_t5 = float(df_Mn.loc[df_Mn.CLASS == "Others"].LEISURE)
    c10_t5_total = c10r1_t5 + c10r2_t5 + c10r3_t5 + c10r4_t5 + \
        c10r5_t5 + c10r6_t5

    # Column 11: NON_BENEFICIAL

    c11r1_t1 = float(df_Pr.loc[df_Pr.CLASS == "Forest"].NON_BENEFICIAL)
    c11r2_t1 = float(df_Pr.loc[df_Pr.CLASS == "Shrubland"].NON_BENEFICIAL)
    c11r3_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural grasslands"].NON_BENEFICIAL)
    c11r4_t1 = float(df_Pr.loc[df_Pr.CLASS == "Natural water bodies"].NON_BENEFICIAL)
    c11r5_t1 = float(df_Pr.loc[df_Pr.CLASS == "Wetlands"].NON_BENEFICIAL)
    c11r6_t1 = float(df_Pr.loc[df_Pr.CLASS == "Glaciers"].NON_BENEFICIAL)
    c11r7_t1 = float(df_Pr.loc[df_Pr.CLASS == "Others"].NON_BENEFICIAL)
    c11_t1_total = c11r1_t1 + c11r2_t1 + c11r3_t1 + c11r4_t1 + c11r5_t1 + \
        c11r6_t1 + c11r7_t1

    c11r1_t2 = float(df_Ut.loc[df_Ut.CLASS == "Forest"].NON_BENEFICIAL)
    c11r2_t2 = float(df_Ut.loc[df_Ut.CLASS == "Shrubland"].NON_BENEFICIAL)
    c11r3_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural grasslands"].NON_BENEFICIAL)
    c11r4_t2 = float(df_Ut.loc[df_Ut.CLASS == "Natural water bodies"].NON_BENEFICIAL)
    c11r5_t2 = float(df_Ut.loc[df_Ut.CLASS == "Wetlands"].NON_BENEFICIAL)
    c11r6_t2 = float(df_Ut.loc[df_Ut.CLASS == "Others"].NON_BENEFICIAL)
    c11_t2_total = c11r1_t2 + c11r2_t2 + c11r3_t2 + c11r4_t2 + \
        c11r5_t2 + c11r6_t2

    c11r1_t3 = float(df_Mo.loc[df_Mo.CLASS == "Rainfed crops"].NON_BENEFICIAL)
    c11r2_t3 = float(df_Mo.loc[df_Mo.CLASS == "Forest plantations"].NON_BENEFICIAL)
    c11r3_t3 = float(df_Mo.loc[df_Mo.CLASS == "Settlements"].NON_BENEFICIAL)
    c11r4_t3 = float(df_Mo.loc[df_Mo.CLASS == "Others"].NON_BENEFICIAL)
    c11_t3_total = c11r1_t3 + c11r2_t3 + c11r3_t3 + c11r4_t3

    c11r1_t4 = float(df_Mc.loc[df_Mc.CLASS == "Irrigated crops"].NON_BENEFICIAL)
    c11r2_t4 = float(df_Mc.loc[df_Mc.CLASS == "Managed water bodies"].NON_BENEFICIAL)
    c11r3_t4 = float(df_Mc.loc[df_Mc.CLASS == "Residential"].NON_BENEFICIAL)
    c11r4_t4 = float(df_Mc.loc[df_Mc.CLASS == "Industry"].NON_BENEFICIAL)
    c11r5_t4 = float(df_Mc.loc[df_Mc.CLASS == "Others"].NON_BENEFICIAL)
    c11_t4_total = c11r1_t4 + c11r2_t4 + c11r3_t4 + c11r4_t4 + c11r5_t4

    c11r1_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor domestic"].NON_BENEFICIAL)
    c11r2_t5 = float(df_Mn.loc[df_Mn.CLASS == "Indoor industry"].NON_BENEFICIAL)
    c11r3_t5 = float(df_Mn.loc[df_Mn.CLASS == "Greenhouses"].NON_BENEFICIAL)
    c11r4_t5 = float(df_Mn.loc[df_Mn.CLASS == "Livestock and husbandry"].NON_BENEFICIAL)
    c11r5_t5 = float(df_Mn.loc[df_Mn.CLASS == "Power and energy"].NON_BENEFICIAL)
    c11r6_t5 = float(df_Mn.loc[df_Mn.CLASS == "Others"].NON_BENEFICIAL)
    c11_t5_total = c11r1_t5 + c11r2_t5 + c11r3_t5 + c11r4_t5 + \
        c11r5_t5 + c11r6_t5

    # Check if left and right side agree

    # Table 1
    r1_t1_bene = c6r1_t1 + c7r1_t1 + c8r1_t1 + c9r1_t1 + c10r1_t1
    r2_t1_bene = c6r2_t1 + c7r2_t1 + c8r2_t1 + c9r2_t1 + c10r2_t1
    r3_t1_bene = c6r3_t1 + c7r3_t1 + c8r3_t1 + c9r3_t1 + c10r3_t1
    r4_t1_bene = c6r4_t1 + c7r4_t1 + c8r4_t1 + c9r4_t1 + c10r4_t1
    r5_t1_bene = c6r5_t1 + c7r5_t1 + c8r5_t1 + c9r5_t1 + c10r5_t1
    r6_t1_bene = c6r6_t1 + c7r6_t1 + c8r6_t1 + c9r6_t1 + c10r6_t1
    r7_t1_bene = c6r7_t1 + c7r7_t1 + c8r7_t1 + c9r7_t1 + c10r7_t1

    c5r1_t1_left = c1r1_t1 + c2r1_t1 + c3r1_t1 + c4r1_t1
    c5r2_t1_left = c1r2_t1 + c2r2_t1 + c3r2_t1 + c4r2_t1
    c5r3_t1_left = c1r3_t1 + c2r3_t1 + c3r3_t1 + c4r3_t1
    c5r4_t1_left = c1r4_t1 + c2r4_t1 + c3r4_t1 + c4r4_t1
    c5r5_t1_left = c1r5_t1 + c2r5_t1 + c3r5_t1 + c4r5_t1
    c5r6_t1_left = c1r6_t1 + c2r6_t1 + c3r6_t1 + c4r6_t1
    c5r7_t1_left = c1r7_t1 + c2r7_t1 + c3r7_t1 + c4r7_t1

    c5r1_t1_right = r1_t1_bene + c11r1_t1
    c5r2_t1_right = r2_t1_bene + c11r2_t1
    c5r3_t1_right = r3_t1_bene + c11r3_t1
    c5r4_t1_right = r4_t1_bene + c11r4_t1
    c5r5_t1_right = r5_t1_bene + c11r5_t1
    c5r6_t1_right = r6_t1_bene + c11r6_t1
    c5r7_t1_right = r7_t1_bene + c11r7_t1

    # Table 2
    r1_t2_bene = c6r1_t2 + c7r1_t2 + c8r1_t2 + c9r1_t2 + c10r1_t2
    r2_t2_bene = c6r2_t2 + c7r2_t2 + c8r2_t2 + c9r2_t2 + c10r2_t2
    r3_t2_bene = c6r3_t2 + c7r3_t2 + c8r3_t2 + c9r3_t2 + c10r3_t2
    r4_t2_bene = c6r4_t2 + c7r4_t2 + c8r4_t2 + c9r4_t2 + c10r4_t2
    r5_t2_bene = c6r5_t2 + c7r5_t2 + c8r5_t2 + c9r5_t2 + c10r5_t2
    r6_t2_bene = c6r6_t2 + c7r6_t2 + c8r6_t2 + c9r6_t2 + c10r6_t2

    c5r1_t2_left = c1r1_t2 + c2r1_t2 + c3r1_t2 + c4r1_t2
    c5r2_t2_left = c1r2_t2 + c2r2_t2 + c3r2_t2 + c4r2_t2
    c5r3_t2_left = c1r3_t2 + c2r3_t2 + c3r3_t2 + c4r3_t2
    c5r4_t2_left = c1r4_t2 + c2r4_t2 + c3r4_t2 + c4r4_t2
    c5r5_t2_left = c1r5_t2 + c2r5_t2 + c3r5_t2 + c4r5_t2
    c5r6_t2_left = c1r6_t2 + c2r6_t2 + c3r6_t2 + c4r6_t2

    c5r1_t2_right = r1_t2_bene + c11r1_t2
    c5r2_t2_right = r2_t2_bene + c11r2_t2
    c5r3_t2_right = r3_t2_bene + c11r3_t2
    c5r4_t2_right = r4_t2_bene + c11r4_t2
    c5r5_t2_right = r5_t2_bene + c11r5_t2
    c5r6_t2_right = r6_t2_bene + c11r6_t2

    # Table 3
    r1_t3_bene = c6r1_t3 + c7r1_t3 + c8r1_t3 + c9r1_t3 + c10r1_t3
    r2_t3_bene = c6r2_t3 + c7r2_t3 + c8r2_t3 + c9r2_t3 + c10r2_t3
    r3_t3_bene = c6r3_t3 + c7r3_t3 + c8r3_t3 + c9r3_t3 + c10r3_t3
    r4_t3_bene = c6r4_t3 + c7r4_t3 + c8r4_t3 + c9r4_t3 + c10r4_t3

    c5r1_t3_left = c1r1_t3 + c2r1_t3 + c3r1_t3 + c4r1_t3
    c5r2_t3_left = c1r2_t3 + c2r2_t3 + c3r2_t3 + c4r2_t3
    c5r3_t3_left = c1r3_t3 + c2r3_t3 + c3r3_t3 + c4r3_t3
    c5r4_t3_left = c1r4_t3 + c2r4_t3 + c3r4_t3 + c4r4_t3

    c5r1_t3_right = r1_t3_bene + c11r1_t3
    c5r2_t3_right = r2_t3_bene + c11r2_t3
    c5r3_t3_right = r3_t3_bene + c11r3_t3
    c5r4_t3_right = r4_t3_bene + c11r4_t3

    # Table 4
    r1_t4_bene = c6r1_t4 + c7r1_t4 + c8r1_t4 + c9r1_t4 + c10r1_t4
    r2_t4_bene = c6r2_t4 + c7r2_t4 + c8r2_t4 + c9r2_t4 + c10r2_t4
    r3_t4_bene = c6r3_t4 + c7r3_t4 + c8r3_t4 + c9r3_t4 + c10r3_t4
    r4_t4_bene = c6r4_t4 + c7r4_t4 + c8r4_t4 + c9r4_t4 + c10r4_t4
    r5_t4_bene = c6r5_t4 + c7r5_t4 + c8r5_t4 + c9r5_t4 + c10r5_t4

    c5r1_t4_left = c1r1_t4 + c2r1_t4 + c3r1_t4 + c4r1_t4
    c5r2_t4_left = c1r2_t4 + c2r2_t4 + c3r2_t4 + c4r2_t4
    c5r3_t4_left = c1r3_t4 + c2r3_t4 + c3r3_t4 + c4r3_t4
    c5r4_t4_left = c1r4_t4 + c2r4_t4 + c3r4_t4 + c4r4_t4
    c5r5_t4_left = c1r5_t4 + c2r5_t4 + c3r5_t4 + c4r5_t4

    c5r1_t4_right = r1_t4_bene + c11r1_t4
    c5r2_t4_right = r2_t4_bene + c11r2_t4
    c5r3_t4_right = r3_t4_bene + c11r3_t4
    c5r4_t4_right = r4_t4_bene + c11r4_t4
    c5r5_t4_right = r5_t4_bene + c11r5_t4

    # Table 5
    r1_t5_bene = c6r1_t5 + c7r1_t5 + c8r1_t5 + c9r1_t5 + c10r1_t5
    r2_t5_bene = c6r2_t5 + c7r2_t5 + c8r2_t5 + c9r2_t5 + c10r2_t5
    r3_t5_bene = c6r3_t5 + c7r3_t5 + c8r3_t5 + c9r3_t5 + c10r3_t5
    r4_t5_bene = c6r4_t5 + c7r4_t5 + c8r4_t5 + c9r4_t5 + c10r4_t5
    r5_t5_bene = c6r5_t5 + c7r5_t5 + c8r5_t5 + c9r5_t5 + c10r5_t5
    r6_t5_bene = c6r6_t5 + c7r6_t5 + c8r6_t5 + c9r6_t5 + c10r6_t5

    c5r1_t5_left = c1r1_t5 + c2r1_t5 + c3r1_t5 + c4r1_t5
    c5r2_t5_left = c1r2_t5 + c2r2_t5 + c3r2_t5 + c4r2_t5
    c5r3_t5_left = c1r3_t5 + c2r3_t5 + c3r3_t5 + c4r3_t5
    c5r4_t5_left = c1r4_t5 + c2r4_t5 + c3r4_t5 + c4r4_t5
    c5r5_t5_left = c1r5_t5 + c2r5_t5 + c3r5_t5 + c4r5_t5
    c5r6_t5_left = c1r6_t5 + c2r6_t5 + c3r6_t5 + c4r6_t5

    c5r1_t5_right = r1_t5_bene + c11r1_t5
    c5r2_t5_right = r2_t5_bene + c11r2_t5
    c5r3_t5_right = r3_t5_bene + c11r3_t5
    c5r4_t5_right = r4_t5_bene + c11r4_t5
    c5r5_t5_right = r5_t5_bene + c11r5_t5
    c5r6_t5_right = r6_t5_bene + c11r6_t5

    # t1
    if abs(c5r1_t1_left - c5r1_t1_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('PROTECTED', 'Forest'))
    elif abs(c5r2_t1_left - c5r2_t1_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('PROTECTED', 'Shrubland'))
    elif abs(c5r3_t1_left - c5r3_t1_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('PROTECTED',
                                               'Natural grasslands'))
    elif abs(c5r4_t1_left - c5r4_t1_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('PROTECTED',
                                               'Natural water bodies'))
    elif abs(c5r5_t1_left - c5r5_t1_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('PROTECTED', 'Wetlands'))
    elif abs(c5r6_t1_left - c5r6_t1_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('PROTECTED', 'Glaciers'))
    elif abs(c5r7_t1_left - c5r7_t1_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('PROTECTED', 'Others'))

    # t2
    elif abs(c5r1_t2_left - c5r1_t2_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('UTILIZED', 'Forest'))
    elif abs(c5r2_t2_left - c5r2_t2_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('UTILIZED', 'Shrubland'))
    elif abs(c5r3_t2_left - c5r3_t2_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('UTILIZED',
                                               'Natural grasslands'))
    elif abs(c5r4_t2_left - c5r4_t2_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('UTILIZED',
                                               'Natural water bodies'))
    elif abs(c5r5_t2_left - c5r5_t2_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('UTILIZED', 'Wetlands'))
    elif abs(c5r6_t2_left - c5r6_t2_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('UTILIZED', 'Others'))

    # t3
    elif abs(c5r1_t3_left - c5r1_t3_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MODIFIED', 'Rainfed crops'))
    elif abs(c5r2_t3_left - c5r2_t3_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MODIFIED',
                                               'Forest plantations'))
    elif abs(c5r3_t3_left - c5r3_t3_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MODIFIED', 'Settlements'))
    elif abs(c5r4_t3_left - c5r4_t3_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MODIFIED', 'Others'))

    # t4
    elif abs(c5r1_t4_left - c5r1_t4_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MANAGED CONVENTIONAL',
                                               'Irrigated crops'))
    elif abs(c5r2_t4_left - c5r2_t4_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MANAGED CONVENTIONAL',
                                               'Managed water bodies'))
    elif abs(c5r3_t4_left - c5r3_t4_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MANAGED CONVENTIONAL',
                                               'Residential'))
    elif abs(c5r4_t4_left - c5r4_t4_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MANAGED CONVENTIONAL',
                                               'Industry'))
    elif abs(c5r5_t4_left - c5r5_t4_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MANAGED CONVENTIONAL',
                                               'Others'))

    # t5
    elif abs(c5r1_t5_left - c5r1_t5_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MANAGED NON_CONVENTIONAL',
                                               'Indoor domestic'))
    elif abs(c5r2_t5_left - c5r2_t5_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MANAGED NON_CONVENTIONAL',
                                               'Indoor industrial'))
    elif abs(c5r3_t5_left - c5r3_t5_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MANAGED NON_CONVENTIONAL',
                                               'Greenhouses'))
    elif abs(c5r4_t5_left - c5r4_t5_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MANAGED NON_CONVENTIONAL',
                                               'Livestock and husbandry'))
    elif abs(c5r5_t5_left - c5r5_t5_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MANAGED NON_CONVENTIONAL',
                                               'Power and energy'))
    elif abs(c5r6_t5_left - c5r6_t5_right) > tolerance:
        raise ValueError('The left and rigth sides \
                          do not add up ({0} table \
                          and {1} row)'.format('MANAGED NON_CONVENTIONAL',
                                               'Others'))

    # Calculations & modify svg
    if not template:
        path = os.path.dirname(os.path.abspath(__file__))
        svg_template_path = os.path.join(path, 'svg', 'sheet_2.svg')
    else:
        svg_template_path = os.path.abspath(template)

    tree = ET.parse(svg_template_path)

    # Titles

    xml_txt_box = tree.findall('''.//*[@id='basin']''')[0]
    list(xml_txt_box)[0].text = 'Basin: ' + basin

    xml_txt_box = tree.findall('''.//*[@id='period']''')[0]
    list(xml_txt_box)[0].text = 'Period: ' + period

    xml_txt_box = tree.findall('''.//*[@id='units']''')[0]
    
    
    if np.all([smart_unit, scale > 0]):
        list(xml_txt_box)[0].text = 'Sheet 2: Evapotranspiration ({0} {1})'.format(10**-scale, units)
    else:
        list(xml_txt_box)[0].text = 'Sheet 2: Evapotranspiration ({0})'.format(units)

    # Total ET
    total_et_t1 = c5r1_t1_left + c5r2_t1_left + c5r3_t1_left + c5r4_t1_left + \
        c5r5_t1_left + c5r6_t1_left + c5r7_t1_left
    total_et_t2 = c5r1_t2_left + c5r2_t2_left + c5r3_t2_left + c5r4_t2_left + \
        c5r5_t2_left + c5r6_t2_left
    total_et_t3 = c5r1_t3_left + c5r2_t3_left + c5r3_t3_left + c5r4_t3_left
    total_et_t4 = c5r1_t4_left + c5r2_t4_left + c5r3_t4_left + c5r4_t4_left + \
        c5r5_t4_left
    total_et_t5 = c5r1_t5_left + c5r2_t5_left + c5r3_t5_left + c5r4_t5_left + \
        c5r5_t5_left + c5r6_t5_left

    total_et = total_et_t1 + total_et_t2 + total_et_t3 + \
        total_et_t4 + total_et_t5
        
    

    et_total_managed_lu = total_et_t4 + total_et_t5
    et_total_managed = total_et_t3 + et_total_managed_lu

    t_total_managed_lu = c1_t4_total + c1_t5_total

    xml_txt_box = tree.findall('''.//*[@id='total_et']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_et

    xml_txt_box = tree.findall('''.//*[@id='non-manageble']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_et_t1

    xml_txt_box = tree.findall('''.//*[@id='manageble']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_et_t2

    xml_txt_box = tree.findall('''.//*[@id='managed']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % et_total_managed

    # Totals land use

    xml_txt_box = tree.findall('''.//*[@id='protected_lu_et']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_et_t1

    xml_txt_box = tree.findall('''.//*[@id='protected_lu_t']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1_t1_total

    xml_txt_box = tree.findall('''.//*[@id='utilized_lu_et']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_et_t2

    xml_txt_box = tree.findall('''.//*[@id='utilized_lu_t']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1_t2_total

    xml_txt_box = tree.findall('''.//*[@id='modified_lu_et']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_et_t3

    xml_txt_box = tree.findall('''.//*[@id='modified_lu_t']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1_t3_total

    xml_txt_box = tree.findall('''.//*[@id='managed_lu_et']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % et_total_managed_lu

    xml_txt_box = tree.findall('''.//*[@id='managed_lu_t']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % t_total_managed_lu

    # Table 1
    xml_txt_box = tree.findall('''.//*[@id='plu_et_forest']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r1_t1_left

    xml_txt_box = tree.findall('''.//*[@id='plu_t_forest']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r1_t1

    xml_txt_box = tree.findall('''.//*[@id='plu_et_shrubland']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r2_t1_left

    xml_txt_box = tree.findall('''.//*[@id='plu_t_shrubland']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r2_t1

    xml_txt_box = tree.findall('''.//*[@id='plu_et_grasslands']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r3_t1_left

    xml_txt_box = tree.findall('''.//*[@id='plu_t_grasslands']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r3_t1

    xml_txt_box = tree.findall('''.//*[@id='plu_et_waterbodies']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r4_t1_left

    xml_txt_box = tree.findall('''.//*[@id='plu_t_waterbodies']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r4_t1

    xml_txt_box = tree.findall('''.//*[@id='plu_et_wetlands']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r5_t1_left

    xml_txt_box = tree.findall('''.//*[@id='plu_t_wetlands']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r5_t1

    xml_txt_box = tree.findall('''.//*[@id='plu_et_glaciers']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r6_t1_left

    xml_txt_box = tree.findall('''.//*[@id='plu_t_glaciers']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r6_t1

    xml_txt_box = tree.findall('''.//*[@id='plu_et_others']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r7_t1_left

    xml_txt_box = tree.findall('''.//*[@id='plu_t_others']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r7_t1

    # Table 2
    xml_txt_box = tree.findall('''.//*[@id='ulu_et_forest']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r1_t2_left

    xml_txt_box = tree.findall('''.//*[@id='ulu_t_forest']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r1_t2

    xml_txt_box = tree.findall('''.//*[@id='ulu_et_shrubland']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r2_t2_left

    xml_txt_box = tree.findall('''.//*[@id='ulu_t_shrubland']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r2_t2

    xml_txt_box = tree.findall('''.//*[@id='ulu_et_grasslands']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r3_t2_left

    xml_txt_box = tree.findall('''.//*[@id='ulu_t_grasslands']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r3_t2

    xml_txt_box = tree.findall('''.//*[@id='ulu_et_waterbodies']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r4_t2_left

    xml_txt_box = tree.findall('''.//*[@id='ulu_t_waterbodies']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r4_t2

    xml_txt_box = tree.findall('''.//*[@id='ulu_et_wetlands']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r5_t2_left

    xml_txt_box = tree.findall('''.//*[@id='ulu_t_wetlands']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r5_t2

    xml_txt_box = tree.findall('''.//*[@id='ulu_et_others']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r6_t2_left

    xml_txt_box = tree.findall('''.//*[@id='ulu_t_others']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r6_t2

    # Table 3
    xml_txt_box = tree.findall('''.//*[@id='molu_et_rainfed']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r1_t3_left

    xml_txt_box = tree.findall('''.//*[@id='molu_t_rainfed']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r1_t3

    xml_txt_box = tree.findall('''.//*[@id='molu_et_forest']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r2_t3_left

    xml_txt_box = tree.findall('''.//*[@id='molu_t_forest']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r2_t3

    xml_txt_box = tree.findall('''.//*[@id='molu_et_settlements']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r3_t3_left

    xml_txt_box = tree.findall('''.//*[@id='molu_t_settlements']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r3_t3

    xml_txt_box = tree.findall('''.//*[@id='molu_et_others']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r4_t3_left

    xml_txt_box = tree.findall('''.//*[@id='molu_t_others']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r4_t3

    # Table 4
    xml_txt_box = tree.findall('''.//*[@id='malu_et_crops']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r1_t4_left

    xml_txt_box = tree.findall('''.//*[@id='malu_t_crops']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r1_t4

    xml_txt_box = tree.findall('''.//*[@id='malu_et_waterbodies']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r2_t4_left

    xml_txt_box = tree.findall('''.//*[@id='malu_t_waterbodies']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r2_t4

    xml_txt_box = tree.findall('''.//*[@id='malu_et_residential']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r3_t4_left

    xml_txt_box = tree.findall('''.//*[@id='malu_t_residential']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r3_t4

    xml_txt_box = tree.findall('''.//*[@id='malu_et_industry']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r4_t4_left

    xml_txt_box = tree.findall('''.//*[@id='malu_t_industry']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r4_t4

    xml_txt_box = tree.findall('''.//*[@id='malu_et_others1']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r5_t4_left

    xml_txt_box = tree.findall('''.//*[@id='malu_t_others1']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r5_t4

    # Table 5
    xml_txt_box = tree.findall('''.//*[@id='malu_et_idomestic']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r1_t5_left

    xml_txt_box = tree.findall('''.//*[@id='malu_t_idomestic']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r1_t5

    xml_txt_box = tree.findall('''.//*[@id='malu_et_iindustry']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r2_t5_left

    xml_txt_box = tree.findall('''.//*[@id='malu_t_iindustry']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r2_t5

    xml_txt_box = tree.findall('''.//*[@id='malu_et_greenhouses']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r3_t5_left

    xml_txt_box = tree.findall('''.//*[@id='malu_t_greenhouses']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r3_t5

    xml_txt_box = tree.findall('''.//*[@id='malu_et_livestock']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r4_t5_left

    xml_txt_box = tree.findall('''.//*[@id='malu_t_livestock']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r4_t5

    xml_txt_box = tree.findall('''.//*[@id='malu_et_powerandenergy']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r5_t5_left

    xml_txt_box = tree.findall('''.//*[@id='malu_t_powerandenergy']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r5_t5

    xml_txt_box = tree.findall('''.//*[@id='malu_et_others2']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c5r6_t5_left

    xml_txt_box = tree.findall('''.//*[@id='malu_t_others2']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % c1r6_t5

    # Right box
    total_t = c1_t1_total + c1_t2_total + c1_t3_total + \
        c1_t4_total + c1_t5_total
    total_e = total_et - total_t

    total_water = c2_t1_total + c2_t2_total + c2_t3_total + \
        c2_t4_total + c2_t5_total
    total_soil = c3_t1_total + c3_t2_total + c3_t3_total + \
        c3_t4_total + c3_t5_total
    total_interception = c4_t1_total + c4_t2_total + c4_t3_total + \
        c4_t4_total + c4_t5_total

    xml_txt_box = tree.findall('''.//*[@id='evaporation']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_e

    xml_txt_box = tree.findall('''.//*[@id='transpiration']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_t

    xml_txt_box = tree.findall('''.//*[@id='water']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_water

    xml_txt_box = tree.findall('''.//*[@id='soil']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_soil

    xml_txt_box = tree.findall('''.//*[@id='interception']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_interception

    total_agr = c6_t1_total + c6_t2_total + c6_t3_total + \
        c6_t4_total + c6_t5_total
    total_env = c7_t1_total + c7_t2_total + c7_t3_total + \
        c7_t4_total + c7_t5_total
    total_eco = c8_t1_total + c8_t2_total + c8_t3_total + \
        c8_t4_total + c8_t5_total
    total_ene = c9_t1_total + c9_t2_total + c9_t3_total + \
        c9_t4_total + c9_t5_total
    total_lei = c10_t1_total + c10_t2_total + c10_t3_total + \
        c10_t4_total + c10_t5_total

    total_bene = total_agr + total_env + total_eco + total_ene + total_lei
    total_non_bene = c11_t1_total + c11_t2_total + c11_t3_total + \
        c11_t4_total + c11_t5_total

    xml_txt_box = tree.findall('''.//*[@id='non-beneficial']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_non_bene

    xml_txt_box = tree.findall('''.//*[@id='beneficial']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_bene

    xml_txt_box = tree.findall('''.//*[@id='agriculture']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_agr

    xml_txt_box = tree.findall('''.//*[@id='environment']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_env

    xml_txt_box = tree.findall('''.//*[@id='economy']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_eco

    xml_txt_box = tree.findall('''.//*[@id='energy']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_ene

    xml_txt_box = tree.findall('''.//*[@id='leisure']''')[0]
    list(xml_txt_box)[0].text = '%.1f' % total_lei

    # Export svg to png
    tempout_path = output.replace('.pdf', '_temporary.svg')
    tree.write(tempout_path)    
    cairosvg.svg2pdf(url=tempout_path, write_to=output)    
    os.remove(tempout_path)


    # Return
    return output
