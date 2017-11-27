# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:55:04 2016

@author: bec
"""
import os
import becgis
import numpy as np
import matplotlib.pyplot as plt
import gdal
import tempfile
import shutil
from scipy import interpolate
import csv
import pandas as pd
import xml.etree.ElementTree as ET
import datetime
import calendar
import subprocess
import tempfile as tf
import get_dictionaries as gd
import sheet6_functions as sh6

def create_sheet4_6(complete_data, metadata, output_dir, global_data):
    
    output_dir = os.path.join(output_dir, metadata['name'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    lucs = gd.get_sheet4_6_classes()
    consumed_fractions, sw_supply_fractions, sw_return_fractions = gd.get_sheet4_6_fractions()

    equiped_sw_irrigation_tif = global_data["equiped_sw_irrigation"]
    wpl_tif = global_data["wpl_tif"]
    population_tif = global_data["population_tif"]
    
    sw_supply_fraction_tif = fractions(metadata['lu'], sw_supply_fractions, lucs, output_dir, filename = 'sw_supply_fraction.tif')
    sw_supply_fraction_tif = update_irrigation_fractions(metadata['lu'], sw_supply_fraction_tif, lucs, equiped_sw_irrigation_tif)

    non_recov_fraction_tif = non_recoverable_fractions(metadata['lu'], wpl_tif, lucs, output_dir)
    sw_return_fraction_tif = fractions(metadata['lu'], sw_return_fractions, lucs, output_dir, filename = 'sw_return_fraction.tif')
    
    supply_gw = supply_sw = return_flow_sw_sw = return_flow_sw_gw = return_flow_gw_sw = return_flow_gw_gw = np.array([])
    
    bf_ts = cr_ts = vgw_ts = vr_ts = rfg_ts = rfs_ts = ds_ts = np.array([])
    
    for date in complete_data['etb'][1]:
        
        conventional_et_tif = complete_data['etb'][0][complete_data['etb'][1] == date][0]

        other_consumed_tif = None
        non_conventional_et_tif = None
        
        ###
        # Calculate supply and split into GW and SW supply
        ###
        total_supply_tif = total_supply(conventional_et_tif, other_consumed_tif, metadata['lu'], lucs, consumed_fractions, output_dir, date)

        residential_demand = include_residential_supply(population_tif, metadata['lu'], total_supply_tif, date, lucs, 110, wcpc_minimal = 100)
        
        supply_sw_tif, supply_gw_tif = split_flows(total_supply_tif, sw_supply_fraction_tif, output_dir, date, flow_names = ['SUPPLYsw','SUPPLYgw'])
    
        ###
        # Calculate non-consumed supplies per source
        ###
        non_consumed_tif = calc_delta_flow(total_supply_tif, conventional_et_tif, output_dir, date)
        non_consumed_sw_tif, non_consumed_gw_tif = split_flows(non_consumed_tif, sw_supply_fraction_tif, output_dir, date, flow_names = ['NONCONSUMEDsw', 'NONCONSUMEDgw'])
    
        ###
        # Calculate (non-)recoverable return flows per source
        ###
        non_recov_tif, recov_tif = split_flows(non_consumed_tif, non_recov_fraction_tif, output_dir, date, flow_names = ['NONRECOV', 'RECOV'])
        recov_sw_tif, recov_gw_tif = split_flows(recov_tif, sw_return_fraction_tif, output_dir, date, flow_names = ['RECOVsw', 'RECOVgw'])
        non_recov_sw_tif, non_recov_gw_tif = split_flows(non_recov_tif, sw_return_fraction_tif, output_dir, date, flow_names = ['NONRECOVsw', 'NONRECOVgw'])
    
        ###
        # Caculate return flows to gw and sw
        ###
        return_flow_sw_sw_tif, return_flow_sw_gw_tif = split_flows(non_consumed_sw_tif, sw_return_fraction_tif, output_dir, date, flow_names = ['RETURNFLOW_swgw', 'RETURNFLOW_swsw'])
        return_flow_gw_sw_tif, return_flow_gw_gw_tif = split_flows(non_consumed_gw_tif, sw_return_fraction_tif, output_dir, date, flow_names = ['RETURNFLOW_gwsw', 'RETURNFLOW_gwgw'])
    
        ###
        # Calculate the blue water demand
        ###
        demand_tif = calc_demand(complete_data['lai'][0][complete_data['lai'][1] == date][0], complete_data['etref'][0][complete_data['etref'][1] == date][0], complete_data['p'][0][complete_data['p'][1] == date][0], metadata['lu'], date, output_dir)

        set_classes_to_value(demand_tif, metadata['lu'], lucs['Residential'], value = residential_demand)
        
        ###
        # Create sheet 4
        ###
        entries_sh4 = {'SUPPLY_SURFACEWATER' : supply_sw_tif,
                       'SUPPLY_GROUNDWATER' : supply_gw_tif,
                       'CONSUMED_ET' : conventional_et_tif,
                       'CONSUMED_OTHER' : other_consumed_tif,
                       'NON_CONVENTIONAL_ET' : non_conventional_et_tif,
                       'RECOVERABLE_SURFACEWATER' : recov_sw_tif,
                       'RECOVERABLE_GROUNDWATER' : recov_gw_tif,
                       'NON_RECOVERABLE_SURFACEWATER': non_recov_sw_tif,
                       'NON_RECOVERABLE_GROUNDWATER': non_recov_gw_tif,
                       'DEMAND': demand_tif}
        
        sheet4_csv =create_sheet4_csv(entries_sh4, metadata['lu'], lucs, date, os.path.join(output_dir, 'sheet4'), convert_unit = 1)
        
        create_sheet4(metadata['name'], '{0}-{1}'.format(date.year, str(date.month).zfill(2)), ['km3/month', 'km3/month'], [sheet4_csv, sheet4_csv], 
                          [sheet4_csv.replace('.csv','_a.png'), sheet4_csv.replace('.csv','_b.png')], template = [r"C:\Users\bec\Dropbox\UNESCO\Scripts\bert\sheet_svgs\sheet_4_part1.svg", r"C:\Users\bec\Dropbox\UNESCO\Scripts\bert\sheet_svgs\sheet_4_part2.svg"])
        
        supply_gw = np.append(supply_gw, supply_gw_tif)
        supply_sw = np.append(supply_sw, supply_sw_tif)
        return_flow_sw_sw = np.append(return_flow_sw_sw, return_flow_sw_sw_tif)
        return_flow_sw_gw = np.append(return_flow_sw_gw, return_flow_sw_gw_tif)
        return_flow_gw_sw = np.append(return_flow_gw_sw, return_flow_gw_sw_tif)
        return_flow_gw_gw = np.append(return_flow_gw_gw, return_flow_gw_gw_tif)
        
        print "sheet 4 finished for {0} (going to {1})".format(date, complete_data['etb'][1][-1])
        complete_data["bf"][0][complete_data["bf"][1] == date][0]
        recharge_tif = complete_data["r"][0][complete_data["r"][1] == date][0]
        baseflow = accumulate_per_classes(metadata['lu'], complete_data["bf"][0][complete_data["bf"][1] == date][0], range(1,81), scale = 1e-6)
        capillaryrise = 0.01 * accumulate_per_classes(metadata['lu'], supply_gw_tif, range(1,81), scale = 1e-6)
    
        entries_sh6 = {'VERTICAL_RECHARGE': recharge_tif,
                       'VERTICAL_GROUNDWATER_WITHDRAWALS': supply_gw_tif,
                       'RETURN_FLOW_GROUNDWATER': return_flow_gw_gw_tif,
                       'RETURN_FLOW_SURFACEWATER':return_flow_sw_gw_tif}
    
        entries_2_sh6 = {'CapillaryRise': capillaryrise,
                         'DeltaS': 'nan',
                         'ManagedAquiferRecharge': 'nan',
                         'Baseflow': baseflow,
                         'GWInflow': 'nan',
                         'GWOutflow': 'nan'}
    
        sheet6_csv = sh6.create_sheet6_csv(entries_sh6, entries_2_sh6, metadata['lu'], lucs, date, os.path.join(output_dir,'sheet6'), convert_unit = 1)
        
        p1 = sh6.create_sheet6(metadata['name'], '{0}-{1}'.format(date.year, str(date.month).zfill(2)), 'km3/month', sheet6_csv, sheet6_csv.replace('.csv', '.png'), template = r"C:\Users\bec\Dropbox\UNESCO\Scripts\bert\sheet_svgs\sheet_6.svg")
 
        ds_ts = np.append(ds_ts, p1['delta_S'])
        bf_ts = np.append(bf_ts, p1['baseflow'])   
        cr_ts = np.append(cr_ts, p1['CRtotal'])
        vgw_ts = np.append(vgw_ts, p1['VGWtotal'])
        vr_ts = np.append(vr_ts, p1['VRtotal'])   
        rfg_ts = np.append(rfg_ts, p1['RFGtotal'])   
        rfs_ts = np.append(rfs_ts, p1['RFStotal'])
        
        print "sheet 6 finished"
        
    csv4_folder = os.path.join(output_dir, 'sheet4')
    csv4_yearly_folder = os.path.join(output_dir, 'sheet4_yearly')
    sheet4_csv_yearly = create_csv_yearly(csv4_folder, csv4_yearly_folder, year_position = [-11,-7], month_position = [-6,-4], header_rows = 1, header_columns = 1)

    sh6.plot_storages(ds_ts, bf_ts, cr_ts, vgw_ts, vr_ts, rfg_ts, rfs_ts, complete_data['etb'][1], output_dir, metadata['name'])
    
    csv6_folder = os.path.join(output_dir, 'sheet6')
    csv6_yearly_folder = os.path.join(output_dir, 'sheet6_yearly')
    csv6 = create_csv_yearly(csv6_folder, csv6_yearly_folder, year_position = [-11,-7], month_position = [-6,-4], header_rows = 1, header_columns = 2)
    
    for csv_file in csv6:
        year = csv_file[-8:-4]
        p1 = sh6.create_sheet6(metadata['name'], year, 'km3/year', csv_file, csv_file.replace('.csv', '.png'), template = r"C:\Users\bec\Dropbox\UNESCO\Scripts\bert\sheet_svgs\sheet_6.svg")
    
    for cv in sheet4_csv_yearly:
        year = int(cv[-8:-4])
        create_sheet4(metadata['name'], '{0}'.format(year), ['km3/month', 'km3/month'], [cv, cv], 
                          [cv.replace('.csv','_a.png'), cv.replace('.csv','_b.png')], template = [r"C:\Users\bec\Dropbox\UNESCO\Scripts\bert\sheet_svgs\sheet_4_part1.svg", r"C:\Users\bec\Dropbox\UNESCO\Scripts\bert\sheet_svgs\sheet_4_part2.svg"])

        
    complete_data['supply_gw'] = (supply_gw, complete_data['etb'][1])
    complete_data['supply_sw'] = (supply_sw, complete_data['etb'][1])
    complete_data['return_flow_sw_sw'] = (return_flow_sw_sw, complete_data['etb'][1])
    complete_data['return_flow_sw_gw'] = (return_flow_sw_gw, complete_data['etb'][1])
    complete_data['return_flow_gw_sw'] = (return_flow_gw_sw, complete_data['etb'][1])
    complete_data['return_flow_gw_gw'] = (return_flow_gw_gw, complete_data['etb'][1])
    
    ####
    ## Remove some datasets
    ####   
    shutil.rmtree(os.path.split(total_supply_tif)[0])
    shutil.rmtree(os.path.split(non_consumed_tif)[0])
    shutil.rmtree(os.path.split(non_consumed_sw_tif)[0])
    shutil.rmtree(os.path.split(non_consumed_gw_tif)[0])
    shutil.rmtree(os.path.split(non_recov_tif)[0])
    shutil.rmtree(os.path.split(recov_tif)[0])
    
    return complete_data 

def update_irrigation_fractions(lu_tif, fraction_tif, lucs, equiped_sw_irrigation_tif):
    """
    Update a fractions map used to split total supply into supply_sw and supply_gw with values for the irrigated
    pixels.
    
    Parameters
    ----------
    lu_tif : str
        Landuse map.
    fraction_tif : str
        Map to be updated.
    lucs: dict
        Dictionary describing the different landuse categories.
    equiped_sw_irrigation_tif : str
        Map with values to fill in the irrigated pixels.
        
    Returns
    -------
    fraction_tif : str
        Updated map.
    """
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(fraction_tif)
    
    sw_tif = becgis.MatchProjResNDV(lu_tif, np.array([equiped_sw_irrigation_tif]), tf.mkdtemp())[0]
    
    SW = becgis.OpenAsArray(sw_tif, nan_values = True) / 100
    LULC = becgis.OpenAsArray(lu_tif, nan_values = True)
    FRACTIONS = becgis.OpenAsArray(fraction_tif, nan_values = True)
    
    mask = np.logical_or.reduce([LULC == value for value in lucs['Irrigated crops']])
    
    SW[np.isnan(SW)] = np.nanmean(SW)    
    FRACTIONS[mask] = SW[mask]
    
    becgis.CreateGeoTiff(fraction_tif, FRACTIONS, driver, NDV, xsize, ysize, GeoT, Projection)
    
    return fraction_tif

def non_recoverable_fractions(lu_tif, wpl_tif, lucs, output_folder):
    """
    Create a map non recoverable fractions. For manmade landuseclasses, the grey water
    footprint map is used. Non-recoverable for residential areas is set to 1. Natural
    landusecategories are 0.
    
    Parameters
    ----------
    lu_tif : str
        Landuse map.
    wpl_tif : str
        Map with Grey Water Footprint fractions.
    lucs : dict
        Dictionary describing the different landuse categories.
    output_folder : str
        Folder to save results.
    
    Returns
    -------
    tif : str
        String pointing to file with results.
    """
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_tif)
    
    wpl_tif = becgis.MatchProjResNDV(lu_tif, np.array([wpl_tif]), tf.mkdtemp())[0]
    
    WPL = becgis.OpenAsArray(wpl_tif, nan_values = True)
    LULC = becgis.OpenAsArray(lu_tif, nan_values = True)
    
    manmade_categories = ['Irrigated crops','Managed water bodies','Aquaculture','Residential','Greenhouses','Other']
    
    mask = np.zeros(np.shape(LULC)).astype(np.bool)
    
    for category in manmade_categories:
        mask = np.any([mask, np.logical_or.reduce([LULC == value for value in lucs[category]])], axis = 0)
        
    FRACTIONS = np.zeros(np.shape(LULC))
    FRACTIONS[mask] = WPL[mask]
    
    mask = np.logical_or.reduce([LULC == value for value in lucs['Residential']])
    FRACTIONS[mask] = 1.0
    
    tif = os.path.join(output_folder, 'non_recov_fraction.tif')
    becgis.CreateGeoTiff(tif, FRACTIONS, driver, NDV, xsize, ysize, GeoT, Projection)
    return tif
    
def calc_demand(lai_tif, etref_tif, p_tif, lu_tif, date, output_folder):
    """
    Calculate the blue-water demand for a pixel as the Potential ET minus the effective
    rainfall. The PET is calculated by multiplying the ETref with a Kc factor, the Kc is
    a function of the LAI. The effective fraction of the total rainfall is determined
    using a budyko curve.
    
    Parameters
    ----------
    lai_tif : str
        Map of LAI.
    etref_tif : str
        Map of ETref.
    p_tif : str
        Map of P.
    lu_tif : str
        Map of landuse
    date : object
        Datetime.date object of the time under consideration.
    output_folder : str
        Folder to store results.
    
    Returns
    -------
    fh : str
        Map of demands.
    """
    month_labels = becgis.GetMonthLabels()
       
    LAI = becgis.OpenAsArray(lai_tif, nan_values = True)
    ETREF = becgis.OpenAsArray(etref_tif, nan_values = True)
    P = becgis.OpenAsArray(p_tif, nan_values = True)
    LULC = becgis.OpenAsArray(lu_tif, nan_values= True)
    
    water_mask = np.logical_or.reduce([LULC == value for value in [4, 5, 30, 23, 24, 63]])
    
    KC = np.where(water_mask, 1.4, (1 - np.exp(-0.5 * LAI)) / 0.76)    
    
    PET = KC * ETREF
    
    PHI = PET / P
    
    EFFECTIVE_RAINFALL = np.sqrt(PHI*np.tanh(1/PHI)*(1-np.exp(-PHI))) * P
    
    DEMAND = PET - EFFECTIVE_RAINFALL
    
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_tif)
    
    DEMAND[LULC == NDV] = NDV
    DEMAND[np.isnan(DEMAND)] = NDV
    
    output_folder = os.path.join(output_folder, 'DEMAND')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    fh = os.path.join(output_folder, 'demand_{0}_{1}.tif'.format(date.year, month_labels[date.month]))
    
    becgis.CreateGeoTiff(fh, DEMAND, driver, NDV, xsize, ysize, GeoT, Projection)
    
    return fh  
    
def create_csv_yearly(input_folder, output_folder, year_position = [-11,-7], month_position = [-6,-4], header_rows = 1, header_columns = 1, minus_header_colums = None):
    """
    Calculate yearly csvs from monthly csvs for complete years (i.e. with 12
    months of data available).
    
    Parameters
    ----------
    input_folder : str
        Folder containing monthly csv-files.
    output_folder : str
        Folder to store the yearly csv-files.
    year_position : list, optional
        The indices where the year is positioned in the filenames. Default connects to
        the filenames generated by create_sheet4_csv.
    month_position : list, optional
        The indices where the month is positioned in the filenames. Default connects to
        the filenames generated by create_sheet4_csv.
    header_rows : int
        The number of fixed rows at the top of the csv without any numerical data.
    header_columns : int
        The number of fixed columns at the left side of the csv without any numerical data.
        
    Returns
    -------
    output_fhs : ndarray
        Array with filehandles pointing to the generated yearly csv-files.
    """
    fhs, dates = becgis.SortFiles(input_folder, year_position, month_position = month_position, extension = 'csv')[0:2]
    years, years_counts = np.unique([date.year for date in dates], return_counts = True)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    reader = csv.reader(open(fhs[0], 'r'), delimiter=';')
    template = np.array(list(reader))
    shape = np.shape(template)
    
    output_fhs = np.array([])
    
    data = list()
    for date in dates:
        if date.year in years[years_counts == 12]:
            
            reader = csv.reader(open(fhs[dates == date], 'r'), delimiter=';')
            data.append(np.array(list(reader))[header_rows:,header_columns:minus_header_colums].astype(np.float))
            
            if len(data) == 12:
                data_stack = np.stack(data)
                yearly_data = np.sum(data_stack, axis=0)
                data = list()
                template[header_rows:,header_columns:minus_header_colums] = yearly_data.astype(np.str)
                fh = os.path.join(output_folder, 'sheet_{0}.csv'.format(date.year))
                csv_file = open(fh, 'wb')
                writer = csv.writer(csv_file, delimiter=';')
                for row_index in range(shape[0]):
                    writer.writerow(template[row_index,:])
                output_fhs = np.append(output_fhs, fh)
                csv_file.close()
    
    return output_fhs

def include_residential_supply(population_fh, lu_fh, total_supply_fh, date, sheet4_lucs, wcpc, wcpc_minimal = None):
    """
    Changes the pixel values in a provided map based on the population and the
    per capita daily water consumption.
    
    Parameters
    ----------
    population_fh : str
        Filehandle pointing to a map with population numbers in [pop/ha].
    lu_fh : str
        Filehandle pointing to a landusemap.
    total_supply_fh : str
        Filehandle pointing to a map with water supplies in [mm/month] to be adjusted
        by adding the water supply at Residential landuse classes.
    date : object or str
        Datetime.date object for the date to be selected or str for year to select.
    sheet4_lucs : dict
        Dictionary with the different landuseclasses per category.
    wcpc : float or int
        Water consumption per capita per day in [liter/person/day].
    wcpc_minimal : float or int, optional
        The minimal required water consumption per capita per day in [liter/person/day],
        default is None.
        
    Returns
    -------
    demand : float
        The accumulated minimal required water supply converted into [km3], only 
        returned if wcpc_minimal is not None.    
    """
    temp_folder = tempfile.mkdtemp()
    
    population_fh = becgis.MatchProjResNDV(lu_fh, np.array([population_fh]), temp_folder)
    
    POP = becgis.OpenAsArray(population_fh[0], nan_values = True)
    LULC = becgis.OpenAsArray(lu_fh, nan_values = True)
    
    classes = sheet4_lucs['Residential']
    mask = np.logical_or.reduce([LULC == value for value in classes])
    
    # Get area of each pixel.
    AREA = becgis.MapPixelAreakm(lu_fh)
    
    # Convert [pop/ha] to [pop/pixel]
    POP *= AREA * 100
    
    monthlength = calendar.monthrange(date.year, date.month)[1]
    
    # Calculate WC per pixel in [mm/month]
    SUPPLY_new = wcpc * POP * monthlength * 10**-6 / AREA
    
    SUPPLY_new[np.isnan(SUPPLY_new)] = np.nanmean(SUPPLY_new[mask])
    
    SUPPLY = becgis.OpenAsArray(total_supply_fh, nan_values = True)
    
    SUPPLY[mask] += SUPPLY_new[mask]
    
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_fh)
    
    becgis.CreateGeoTiff(total_supply_fh, SUPPLY, driver, NDV, xsize, ysize, GeoT, Projection)
    
    if wcpc_minimal is not None:
        SUPPLY_demand = wcpc_minimal * POP * monthlength * 10**-6 / AREA
        demand = accumulate_per_classes(lu_fh, SUPPLY_demand, classes, scale = 1e-6)
        return demand
    
def correct_conventional_et(total_supply_fh, conventional_et_fh, lu_fh, sheet4_lucs, sheet4_cfs, category = 'Residential'):
    """
    Adjust the conventional_et map to match between total supply and consumed fractions.
    
    Parameters
    ----------
    total_supply_fh : str
        Filehandle pointing to a total_supply map in [mm/month].
    conventional_et_fh : str
        Filehandle pointing to a conventional_et map map in [mm/month] to be adjusted
        in pixels with landuseclasses belonging to the provided category.
    lu_fh : str
        Filehandle pointing to a landusemap.
    sheet4_lucs : dict
        Dictionary with the different landuseclasses per category.
    sheet4_cfs : dict
        Dictionary with the different consumed fractions per catergory.
    category : str, optional
        Category to be adjusted, this category should also be a key in sheet4_lucs and
        sheet4_cfs.
    """
    SUPPLY = becgis.OpenAsArray(total_supply_fh, nan_values = True)
    CONV_ET = becgis.OpenAsArray(conventional_et_fh, nan_values = True)
    LULC = becgis.OpenAsArray(lu_fh, nan_values = True)
    
    classes = sheet4_lucs[category]
    mask = np.logical_or.reduce([LULC == value for value in classes])
    
    CONV_ET[mask] = SUPPLY[mask] * sheet4_cfs[category]
    
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_fh)
    becgis.CreateGeoTiff(conventional_et_fh, CONV_ET, driver, NDV, xsize, ysize, GeoT, Projection)
    
def accumulate_per_classes(lu_fh, fh, classes, scale = 1e-6):
    """
    Accumulate values on a rastermap masked by a landusemap. Function can also
    convert the values to volumetric units.
    
    Parameters
    ----------
    lu_fh : str
        Filehandle pointing to a landusemap.
    classes : list
        List with values corresponding to values on the landusemap which should
        be accumulated.
    fh : str or ndarray
        Filehandle to the map which should be accumulated, should have same 
        dimensions as the landusemap.
    scale : float or None, optional
        Accum is the sum of the masked values in fh multiplied with scale and the area-extend in km2. 
        If scale is None, accum is the mean of the values in fh. Default is 1e-6, which converts
        mm to km3.
        
    Returns
    -------
    accum : float
        The sum or mean (depending on scale) of the masked values in fh.
    
    """
    LULC = becgis.OpenAsArray(lu_fh, nan_values = True)
    mask = np.logical_or.reduce([LULC == value for value in classes])
    if np.any([type(fh) is str, type(fh) is np.string_]):
        data = becgis.OpenAsArray(fh, nan_values = True)
    else:
        data = fh
        
    if scale == None:
        accum = np.nanmean(data[mask])
    else:
        areas = becgis.MapPixelAreakm(lu_fh)
        accum = np.nansum(data[mask] * scale * areas[mask])
    return accum

def accumulate_per_categories(lu_fh, fh, dictionary, scale = 1e-6):
    """
    Accumulate values on a rastermap for different categories defined in a
    dictionary.
    
    Parameters
    ----------
    lu_fh : str
        Filehandle pointing to a landusemap.
    fh : str
        Filehandle pointing to a map with data to be accumulated.
    dictionary : dict
        Dictionary with the different landuseclasses per category, also see 
        examples.
    scale : float or None, optional
        Value used to scale the values in the et maps. Fill in None
        to use original unit of map. Fill in 1e-6 to convert mm to km (area of
        pixels is also calculated). When scale is not None, the values
        are also multiplied by the area of each pixel. Default is 1e-6.
        
    Returns
    -------
    accumulated : dict
        Dictionary with the total ET in km3 per category.
        
    Examples
    --------
    >>> dictionary = {'Forests': [1, 8, 9, 10, 11, 17],
                      'Shrubland': [2, 12, 14, 15]}
    
    """
    accumulated = dict()
    for category in dictionary.keys():
        classes = dictionary[category]
        if len(classes) is 0:
            classes = [-99]
        accumulated[category] = accumulate_per_classes(lu_fh, fh, classes, scale = scale)
    return accumulated

def plot_per_category(fhs, dates, lu_fh, dictionary, output_fh, scale = 1e-6, gradient_steepness = 2, quantity_unit = ['ET', 'mm/month']):
    """
    Plot the total data per landuse categories as defined in dictionary. Categories
    that provide less than 1% of the total stock/flux are omitted.
    
    Parameters
    ----------
    fhs : ndarray
        Array containing strings with filehandles pointing to georeferenced rastermaps.
    dates : ndarray
        Array containing datetime.date objects corresponding to the filehandles in fhs. Length should be equal
        to fhs.
    lu_fh : str
        Filehandle pointing to a landusemap.
    dictionary : dict
        Dictionary with the different landuseclasses per category.    
    output_fh : str
        Filehandle pointing to a file to save the plotted graph.
    scale : float or None, optional
        Value used to scale the values in the et maps. Fill in None
        to use original unit of map. Fill in 1e-6 to convert mm to km3 (area of
        pixels is also calculated). Default is 1e-6.
    gradient_steepness : int, optional
        Value to indicate how fast the colors in the graph should change per entry.
        Default is 2.
    quantity_unit : list or tuple, optional
        First entry in list or tuple should be string defining the quantity of the data
        in the maps in fhs. Second entry should be the unit. Note that if scale is not
        None, the original unit might change. Default is ['ET', 'mm/month'].
    """
    ets_accumulated = dict()
    for key in dictionary.keys():
        ets_accumulated[key] = np.array([])
        
    for et_fh in fhs:
        et_accumulated = accumulate_per_categories(lu_fh, et_fh, dictionary, scale = scale)
        for key in et_accumulated.keys():
            ets_accumulated[key] = np.append(ets_accumulated[key], et_accumulated[key])
            
    colors =  ['#6bb8cc', '#7bbebd', '#87c5ad', '#91cb9d', '#9ad28d', '#a1d97c', '#acd27a', '#b9c47f', '#c3b683',
               '#cca787', '#d4988b', '#d18d8d', '#b98b89', '#a08886', '#868583', '#6a827f', '#497e7c']
    
    j = 0
    k = 0
    baseline = np.zeros(len(fhs))
    fig = plt.figure(figsize = (13,13))
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
    ax = fig.add_subplot(111)
    for key in ets_accumulated.keys():
        if np.any([np.nansum(ets_accumulated[key]) <= 0.01 * np.nansum(ets_accumulated.values()), np.isnan(np.nansum(ets_accumulated[key]))]):
            continue
        else:
            baseline += ets_accumulated[key]
            try:
                colors[j]
            except:
                j += len(colors)
            ax.fill_between(dates, baseline, label = key, zorder = k, color = colors[j])
            ax.plot(dates, baseline, ':k', zorder = k)
            j -= gradient_steepness
            k -= 1
    
    ax.plot(dates, baseline, color = 'k')
    ax.scatter(dates, baseline, color = 'k')
    fig.autofmt_xdate()
    ax.set_xlabel('Time')
    ax.set_ylabel('{0} [{1}]'.format(quantity_unit[0], quantity_unit[1]))
    ax.set_title('Accumulated {0} per Landuse Category'.format(quantity_unit[0]))
    ax.set_xlim([dates[0], dates[-1]])
    ax.set_ylim([0, max(baseline) * 1.1])
    [i.set_zorder(10) for i in ax.spines.itervalues()]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)
    plt.savefig(output_fh)
    
def distance_to_class(lu_fh, output_folder, proximity_to_values = 23, approximate_kms = False):
    """
    Calculates the distance for each pixel to the closest pixel with a specified
    value.
    
    Parameters
    ----------
    lu_fh : str
        Filehandle pointing to a geotiff for which the distances will be
        calculated.
    output_folder : str
        Folder to store the map 'distance.tif'.
    proximity_to_values : int or float or list, optional
        Value for which the shortest distance will be computer for each pixel,
        default value is 23. Can also be list of multiple values.
    approximate_kms : boolean, optional
        When False, the unit of the output is in degrees. When True, the degrees
        are converted to estimated kilometres. For each pixel a conversion
        rate is approximated by dividing the square root of the area in kilometers
        by the square root of the area in degrees. Default is False.
        
    Returns
    -------
    distance_fh : str
        Filehandle pointing to the 'distance.tif' file.
    
    """
    distance_fh = os.path.join(output_folder, 'distance.tif')
    
    src_ds = gdal.Open(lu_fh)
    srcband = src_ds.GetRasterBand(1)
    try:
        driver = gdal.IdentifyDriver(distance_fh)
        if driver is not None:
            dst_ds = gdal.Open(distance_fh, gdal.GA_Update)
            dstband = dst_ds.GetRasterBand(1)
        else:
            dst_ds = None
    except:
        dst_ds = None
    if dst_ds is None:
        drv = gdal.GetDriverByName('GTiff')
        dst_ds = drv.Create(distance_fh, src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GetDataTypeByName('Float32'))
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjectionRef())
        dstband = dst_ds.GetRasterBand(1)
    if type(proximity_to_values) == list:
        proximity_to_values = str(proximity_to_values)[1:-1]
    options = ['VALUES={0}'.format(proximity_to_values), 'DISTUNITS=GEO']
    gdal.ComputeProximity(srcband, dstband, options, callback = gdal.TermProgress)
    srcband = None
    dstband = None
    src_ds = None
    dst_ds = None
    
    if approximate_kms:
        lengths = becgis.MapPixelAreakm(lu_fh, approximate_lengths = True)
        distance = becgis.OpenAsArray(distance_fh)
        array = distance * lengths
        driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(distance_fh)
        becgis.CreateGeoTiff(distance_fh, array, driver, NDV, xsize, ysize, GeoT, Projection)
    
    print('Finished calculating distances to waterbodies.')
    return distance_fh
    
def total_supply(conventional_et_fh, other_consumed_fh, lu_fh, lu_categories, sheet4_cfs, output_folder, date, greenhouse_et_factor = 0.5):
    """
    Apply a consumed fraction to groups of landuse classes (categories) to aqcuire 
    maps of blue water supplies.
    
    Parameters
    ----------
    conventional_et_fh : str
        Filehandle pointing to a map with conventional or incremental ET data (blue
        ET). Projection, Resolution and NDV should be equal to other_consumed_fh
        and lu_fh.
    other_consumed_fh :str or None
        Filehandle pointing to a map with other water consumptions. When None,
        other is taken to be zero. Projection, Resolution and NDV should be equal 
        to conventional_et_fh and lu_fh.
    lu_fh : str
        Filehandle pointing to a landusemap. Projection, Resolution and NDV should 
        be equal to other_consumed_fh and conventional_et_fh.
    lu_categories : dict
        Dictionary with the different landuseclasses per category.  
    sheet4_cfs : dict
        Dictionary with the different consumed fractions per catergory. The
        keys in this dictionary should also be in lu_categories.
    output_folder : str
        Folder to store result, 'total_supply.tif'.
    date : object or str
        Datetime.date object corresponding to the other inputs.
    greenhouse_et_factor : float, optional
        The conventional ET is scaled with this factor before calculating the supply. When
        greenhouses are present in the basin, it is important to use correct_conventional_et
        after this function to make sure the conventional ET and cfs match with the supply maps.
        Default is 0.5.
        
    Returns
    -------
    output_fh : str
        Filehandle pointing to the total supply map.
        
    Examples
    --------
    >>> conventional_et_fh = r'D:\path\et_blue\ETblue_2006_03.tif'
    >>> other_consumed_fh = None
    >>> lu_fh = r"D:\path\LULC_map_250m.tif"
    >>> output_folder = r"D:\path\"
    
    >>> lu_categories = {'Forests':          [1, 8, 9, 10, 11, 17],
                     'Shrubland':            [2, 12, 14, 15],
                     'Rainfed Crops':        [34, 35, 36, 37, 38, 39, 40, 41, 
                                              42, 43],
                     'Forest Plantations':   [33, 44],
                     'Natural Water Bodies': [4, 19, 23, 24],
                     'Wetlands':             [5, 25, 30, 31],
                     'Natural Grasslands':   [3, 13, 16, 20],
                     'Other (Non-Manmade)':  [6, 7, 18, 21, 22, 26, 27, 28, 29,
                                              32, 45, 46, 47, 48, 49, 50, 51]}
                                              
    >>> sheet4b_cfs = {'Forests':              1.00,
                          'Shrubland':            1.00,
                          'Rainfed Crops':        1.00,
                          'Forest Plantations':   1.00,
                          'Natural Water Bodies': 1.00,
                          'Wetlands':             0.70,
                          'Natural Grasslands':   0.70,
                          'Other (Non-Manmade)':  0.40}
                          
    >>> total_supply(conventional_et_fh, other_consumed_fh, lu_fh, 
                     lu_categories, sheet4b_cfs, output_fh)
    "D:\path/total_supply.tif"
    """
    output_folder = os.path.join(output_folder, 'total_supply')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if isinstance(date, datetime.date):
        output_fh = os.path.join(output_folder, 'total_supply_{0}_{1}.tif'.format(date.year,str(date.month).zfill(2)))
    else:
        output_fh = os.path.join(output_folder, 'total_supply_{0}.tif'.format(date))
    
    list_of_maps = [np.array([lu_fh]), np.array([conventional_et_fh])]
    if other_consumed_fh != None:
        list_of_maps.append(np.array([other_consumed_fh]))
    becgis.AssertProjResNDV(list_of_maps)
    becgis.AssertPresentKeys(sheet4_cfs, lu_categories)
    
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_fh)
    CONSUMED = becgis.OpenAsArray(conventional_et_fh, nan_values = True)
    if other_consumed_fh != None:
        OTHER = becgis.OpenAsArray(other_consumed_fh, nan_values = True)
        CONSUMED = np.nansum([CONSUMED, OTHER], axis = 0)
    LULC = becgis.OpenAsArray(lu_fh, nan_values = True)
    
    for key in sheet4_cfs.keys():
        classes = lu_categories[key]
        mask = np.logical_or.reduce([LULC == value for value in classes])
        consumed_fraction = sheet4_cfs[key]
        if key is 'Greenhouses':
            CONSUMED[mask] /= consumed_fraction * (1 / greenhouse_et_factor)
        else:
            CONSUMED[mask] /= consumed_fraction
        
    all_classes = becgis.Flatten(lu_categories.values())
    mask = np.logical_or.reduce([LULC == value for value in all_classes])
    CONSUMED[~mask] = NDV
    becgis.CreateGeoTiff(output_fh, CONSUMED, driver, NDV, xsize, ysize, GeoT, Projection)
    
    return output_fh

def upstream_of_lu_class(dem_fh, lu_fh, output_folder, clss = 63):
    """
    Calculate which pixels are upstream of a certain landuseclass.
    
    Parameters
    ----------
    dem_fh : str
        Filehandle pointing to a Digital Elevation Model.
    lu_fh : str
        Filehandle pointing to a landuse classification map.
    clss : int, optional
        Landuse identifier for which the upstream pixels will be determined.
        Default is 63 (Managed Water Bodies)
    output_folder : str
        Folder to store the map 'upstream.tif', contains value 1 for
        pixels upstream of waterbodies, 0 for pixels downstream.
        
    Returns
    -------
    upstream_fh : str
        Filehandle pointing to 'upstream.tif' map.
    """
    upstream_fh = os.path.join(output_folder, 'upstream.tif')
    
    if clss is not None:
        import pcraster as pcr
        
        temp_folder = tempfile.mkdtemp()
        extra_temp_folder = os.path.join(temp_folder, "out")
        os.makedirs(extra_temp_folder)
        
        temp_dem_fh = os.path.join(temp_folder, "dem.map")
        output1 = os.path.join(temp_folder, "catchments.map")
        output2 = os.path.join(temp_folder, "catchments.tif")
        temp2_lu_fh = os.path.join(temp_folder, "lu.map")
        
        srs_lu, ts_lu, te_lu, ndv_lu = becgis.GetGdalWarpInfo(lu_fh)
        te_lu_new = ' '.join([te_lu.split(' ')[0], te_lu.split(' ')[3], te_lu.split(' ')[2], te_lu.split(' ')[1]])
        
        GeoT = becgis.GetGeoInfo(dem_fh)[4]
        
        assert abs(GeoT[1]) == abs(GeoT[5]), "Please provide a DEM with square pixels. Unfortunately, PCRaster does not support rectangular pixels."
        
        temp1_lu_fh = becgis.MatchProjResNDV(dem_fh, np.array([lu_fh]), temp_folder)
    
        os.system("gdal_translate -projwin {0} -of PCRaster {1} {2}".format(te_lu_new, dem_fh, temp_dem_fh)) 
        os.system("gdal_translate -projwin {0} -of PCRaster {1} {2}".format(te_lu_new, temp1_lu_fh[0], temp2_lu_fh)) 
        
        dem = pcr.readmap(temp_dem_fh)
        ldd = pcr.lddcreate(dem,9999999,9999999,9999999,9999999)
        lulc = pcr.nominal(pcr.readmap(temp2_lu_fh))
        waterbodies = (lulc == clss)
        catch = pcr.catchment(ldd, waterbodies)
        pcr.report(catch, output1)
        
        os.system("gdal_translate -of GTiff {0} {1}".format(output1, output2))
                
        output3 = becgis.MatchProjResNDV(lu_fh, np.array([output2]), extra_temp_folder)
        
        upstream = becgis.OpenAsArray(output3[0], nan_values = True)
        upstream[np.isnan(upstream)] = 0.
        upstream = upstream.astype(np.bool)
        
        shutil.rmtree(temp_folder)
        
        if upstream_fh is not None:
            driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_fh)
            becgis.CreateGeoTiff(upstream_fh, upstream.astype(np.int8), driver, NDV, xsize, ysize, GeoT, Projection)
    else:
        dummy = becgis.OpenAsArray(lu_fh, nan_values = False) * 0.
        dummy = dummy.astype(np.bool)
        driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_fh)
        becgis.CreateGeoTiff(upstream_fh, dummy.astype(np.int8), driver, NDV, xsize, ysize, GeoT, Projection)
        
    print("Finished calculating up and downstream areas.")
    return upstream_fh

def linear_fractions(lu_fh, upstream_fh, proxy_fh, output_fh, xs, unit = 'km', quantity = 'Distance to water', gw_only_classes = None, plot_graph = True):
    """
    Determine the fractions to split water supply or recoverable water into
    ground and surface water per pixel.
    
    Parameters
    ----------
    lu_fh : str
        Filehandle pointing to a landuse classification map.
    upstream_fh : str
        Filehandle pointing to map indicating areas upstream and downstream
        of managed water bodies.
    proxy_fh : str
        Filehandle pointing to a map with values used to determine a fraction based on
        a linear function.
    output_fh : str
        Filehandle to store results.
    xs : list
        List with 4 floats or integers, like [x1, x2, x3, x4]. The first two 
        numbers refer to the linear relation used to determine alpha or beta upstream
        of waterbodies. The last two numbers refer to the linear relation used
        to determine alpha or beta downstream. x1 and x3 are the distances to water 
        in kilometers up to which pixels will depend fully on surfacewater. 
        x2 and x4 are the distances from which pixels will depend fully on groundwater. 
        Pixels with a distance to surfacewater between x1 x2 and x3 x4 will depend on a mixture
        of surface and groundwater. Choose x1=x3 and x2=x4 to make no distinction
        between upstream or downstream.
    unit : str, optional
        Unit of the proxy, default is 'km'.
    quantity :str, optional
        Quantity of the proxy, default is 'Distance to water'.
    gw_only_classes : dict or None, optional
        Dictionary with the landuseclasses per category for sheet4b, i.e. lu_categories
        from the total_supply function. When this dictionary is provided, the pixel values for
        either beta or alpha for the landuseclasses 'Forests', Shrubland',
        'Rainfed Crops' and 'Forest Plantations' are set to zero. Use this to
        set beta to zero for these classes, since they are likely to only use 
        groundwater. Default is None.
    plot_graph : boolean, optional
        True, plot a graph. False, dont plot a graph. Default is True.

    Returns
    -------
    alpha_fh : str
        Filehandle pointing to the rasterfile containing the values for alpha
        or beta.
    """
       
    upstream = becgis.OpenAsArray(upstream_fh).astype(np.bool)
    distances = becgis.OpenAsArray(proxy_fh, nan_values = True)

    f1 = interpolate.interp1d([xs[0], xs[1]], [1,0], kind = 'linear',bounds_error = False, fill_value=(1,0))
    f2 = interpolate.interp1d([xs[2], xs[3]], [1,0], kind = 'linear',bounds_error = False, fill_value=(1,0))
    
    LULC = becgis.OpenAsArray(lu_fh, nan_values = True)
    distances[np.isnan(LULC)] = np.nan
    
    alpha = np.zeros(np.shape(distances))
    alpha[upstream] = f1(distances[upstream])
    alpha[~upstream] = f2(distances[~upstream])
    
    if plot_graph:
        graph_fh = output_fh.replace('.tif', '.jpg')
        x = np.arange(np.nanmin(distances), np.nanmax(distances), 1)
        fig = plt.figure(figsize=(10,10))
        plt.clf()
        ax = fig.add_subplot(111)
        ax.plot(x, f1(x), '--k', label = 'Fraction Upstream')
        ax.plot(x, f2(x),  '-k', label = 'Fraction Downstream')
        ax.set_ylabel('Fraction [-]')
        ax.set_xlabel('{1} [{0}]'.format(unit, quantity))
        ax.set_zorder(2)
        ax.patch.set_visible(False)
        ax.set_ylim([0,1])
        plt.legend(loc = 'lower right')
        plt.title('Fractions from {0}'.format(quantity))
        plt.suptitle('x1 = {0:.0f}{4}, x2 = {1:.0f}{4}, x3 = {2:.0f}{4}, x4 = {3:.0f}{4}'.format(xs[0], xs[1], xs[2], xs[3], unit))
        
        bins = np.arange(np.nanmin(distances), np.nanmax(distances), (np.nanmax(distances) - np.nanmin(distances)) / 35)
        hist_up, bins_up = np.histogram(distances[upstream & ~np.isnan(distances)], bins = bins)
        hist_down, bins = np.histogram(distances[~upstream & ~np.isnan(distances)], bins = bins_up)
        width = bins[1] - bins[0]
        ax2 = ax.twinx()
        ax2.bar(bins[:-1], hist_up, width, color='#6bb8cc', label = 'Upstream')
        ax2.bar(bins[:-1], hist_down, width, color='#a3db76',bottom=hist_up, label = 'Downstream')
        ax2.set_ylabel('Frequency [-]')
        ax2.set_zorder(1)
        ax2.patch.set_visible(True)
        ax2.set_xlim([x[0],x[-1]])
        plt.legend(loc = 'upper right')
        
        ax.legend(loc='upper right', fancybox = True, shadow = True)
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11),fancybox=True, shadow=True, ncol=5)
        plt.savefig(graph_fh)
          
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_fh)
    becgis.CreateGeoTiff(output_fh, alpha, driver, NDV, xsize, ysize, GeoT, Projection)
    
    if gw_only_classes is not None:
        try:
            ls = [gw_only_classes['Forests'],
                  gw_only_classes['Shrubland'],
                  gw_only_classes['Rainfed Crops'],
                  gw_only_classes['Forest Plantations']]
        except KeyError:
            print('Please provide a dictionary with at least the following keys: Forests, Shrubland, Rainfed Crops and Forest plantations')
        classes = list(becgis.Flatten(ls)) 
        set_classes_to_value(output_fh, lu_fh, classes, value = 0)  

def split_flows(flow_fh, fraction_fh, output_folder, date, flow_names = ['sw','gw']):
    """
    Split a map with watersupplies into water supplied from groundwater and from
    surfacewater. Also calculates the recoverable flow and splits the flow into
    groun and surfacewater.
    
    total_supply_fh : str
        Filehandle pointing to map with total water supplies.
    conventional_et_fh : str
        Filehandle pointing to map with incremental ET water consumption, ETblue.
    other_consumed_fh : str or None
        Filehandle pointing to map with other water consumptions. When None, this 
        parameter is ignored.
    alpha_recoverable_fh : str
        Filehandle pointing to map with the alpha values to split recoverable 
        into surface and groundwater.
    beta_supply_fh : str
        Filehandle pointing to map with the beta values to split supply into
        supply from surface and groundwater.
    output_folder : str
        Folder to save results. Four maps are stored, 'recoverable_gw.tif',
        'recoverable_sw.tif', 'supply_sw.tif' and 'supply_gw.tif'.
    date : object or str
        Datetime.date object corresponding to the other inputs.
        
    Returns
    -------
    rec_gw_fh : str
        Filehandle pointing to the recoverable groundwater map.
    rec_sw_fh : str
        Filehandle pointing to the recoverable surfacewater map.     
    """
    output_folder_one = os.path.join(output_folder, flow_names[0])
    if not os.path.exists(output_folder_one):
        os.makedirs(output_folder_one)
    
    output_folder_two = os.path.join(output_folder, flow_names[1])
    if not os.path.exists(output_folder_two):
        os.makedirs(output_folder_two)
        
    if isinstance(date, datetime.date):
        flow_one_fh = os.path.join(output_folder_one, '{0}_{1}_{2}.tif'.format(flow_names[0],date.year,str(date.month).zfill(2)))
        flow_two_fh = os.path.join(output_folder_two, '{0}_{1}_{2}.tif'.format(flow_names[1],date.year,str(date.month).zfill(2)))
    else:
        flow_one_fh = os.path.join(output_folder_one, '{0}_{1}.tif'.format(flow_names[0],date))
        flow_two_fh = os.path.join(output_folder_two, '{0}_{1}.tif'.format(flow_names[1],date))        
    
    list_of_maps = [np.array([flow_fh]), np.array([fraction_fh])]
    becgis.AssertProjResNDV(list_of_maps)
    
    FLOW = becgis.OpenAsArray(flow_fh, nan_values = True)
    
    FRACTION = becgis.OpenAsArray(fraction_fh, nan_values = True)
    
    FLOW_one = FRACTION * FLOW
    FLOW_two = (1. - FRACTION) * FLOW
    
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(flow_fh)
    becgis.CreateGeoTiff(flow_one_fh, FLOW_one, driver, NDV, xsize, ysize, GeoT, Projection)
    becgis.CreateGeoTiff(flow_two_fh, FLOW_two, driver, NDV, xsize, ysize, GeoT, Projection)
    
    return flow_one_fh, flow_two_fh

def set_classes_to_value(fh, lu_fh, classes, value = 0):
    """
    Open a rasterfile and change certain pixels to a new value. Classes and
    lu_fh is used to create a mask. The mask is then used to the pixel values
    in fh to value.
    
    Parameters
    ----------
    fh : str
        Filehandle pointing to georeferenced tiff raster map.
    lu_fh : str
        Filehandle pointing to georeferenced tiff raster map. Should have same
        dimensions as fh.
    classes : list
        List with values, the values are looked up in lu_fh, the corresponding 
        pixels in fh are then changed.
    value : float or int, optional
        Value to change the pixelvalues in fh into.
    """
    ALPHA = becgis.OpenAsArray(fh, nan_values = True)
    LULC = becgis.OpenAsArray(lu_fh)
    mask = np.logical_or.reduce([LULC == x for x in classes])
    ALPHA[mask] = value
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_fh)
    becgis.CreateGeoTiff(fh, ALPHA, driver, NDV, xsize, ysize, GeoT, Projection)

def insert_values(results, test, lu_category):
    """
    Insert values into dictionaries nested inside anther dictionary.
    
    Parameters
    ----------
    results : dict
        Dictionary containing more dictionaries.
    test : dict
        Dictionary.
    lu_category : str
        Name for the new key.
        
    Returns
    -------
    results : dict
        Dictionary with added keys.
    """
    becgis.AssertSameKeys([results, test])
    for key in test.keys():
        results[key][lu_category] = test[key]
    return results
    
def create_results_dict(entries, lu_fh, sheet4_lucs, aquaculture = None, power = None, industry = None):
    """
    Create a dictionary with values to be stored in a csv-file.
    
    Parameters
    ----------
    entries : dict
        Dictionary with strings pointing to different tif-files, see example below.
    lu_fh : str
        Landusemap.
    sheet4_lucs : dict
        Dictionary describing the sheet 4 and 6 landuse categories.
    aquaculture : dict
        Non-spatial values for aquaculture to be added to the sheet. Default is None.
    power : dict
        Non-spatial values for power to be added to the sheet. Default is None.
    industry : dict
        Non-spatial values for industry to be added to the sheet. Default is None.
        
    Returns
    -------
    results : dict
        Dictionary with values to be saved by create_sheet4_csv in a csv-file.
    """
    list_of_maps = [np.array(value) for value in entries.values() if not np.any([value is None, type(value) is dict])]
    becgis.AssertProjResNDV(list_of_maps)
    
    results = dict()
        
    for key in entries.keys():
        if entries[key] is None:
            results[key] = becgis.ZeroesDictionary(sheet4_lucs)
        if np.any([type(entries[key]) is str, type(entries[key]) is np.string_]):
            results[key] = accumulate_per_categories(lu_fh, entries[key], sheet4_lucs, scale = 1e-6)
        if type(entries[key]) is dict:
            results[key] = entries[key]
    
    if aquaculture is not None:
        results = insert_values(results, aquaculture, 'Aquaculture')
    if power is not None:
        results = insert_values(results, power, 'Power and Energy')
    if industry is not None:
        results = insert_values(results, industry, 'Industry')
        
    return results

def create_sheet4_csv(entries, lu_fh, sheet4_lucs, date, output_folder, aquaculture = None, power = None, industry = None, convert_unit = 1):
    """
    Create a csv-file used to generate sheet 4.
    
    Parameters
    ----------
    entries : dict
        Dictionary with strings pointing to different tif-files, see example below.
    lu_fh : str
        Landusemap.
    sheet4_lucs : dict
        Dictionary describing the sheet 4 and 6 landuse categories.
    date : object
        Datetime.date object describing the date for which csv should be generated.
    output_folder : str
        Folder to store results.
    aquaculture : dict
        Non-spatial values for aquaculture to be added to the sheet. Default is None.
    power : dict
        Non-spatial values for power to be added to the sheet. Default is None.
    industry : dict
        Non-spatial values for industry to be added to the sheet. Default is None.
    convert_unit : int or float
        Multiply all values in the csv-file with convert_unit to change the unit of the data. Default is 1.
        
    Returns
    -------
    output_csv_file : str
        newly created csv-file.
        
    Examples
    --------
    >>> entries_sh4 = {'SUPPLY_SURFACEWATER' : supply_sw_tif,
    >>>           'SUPPLY_GROUNDWATER' : supply_gw_tif,
    >>>           'CONSUMED_ET' : conventional_et_tif,
    >>>           'CONSUMED_OTHER' : other_consumed_tif,
    >>>           'NON_CONVENTIONAL_ET' : non_conventional_et_tif,
    >>>           'RECOVERABLE_SURFACEWATER' : recov_sw_tif,
    >>>           'RECOVERABLE_GROUNDWATER' : recov_gw_tif,
    >>>           'NON_RECOVERABLE_SURFACEWATER': non_recov_sw_tif,
    >>>           'NON_RECOVERABLE_GROUNDWATER': non_recov_gw_tif,
    >>>           'DEMAND': demand_tif}
    """
    required_landuse_types = ['Wetlands','Greenhouses','Rainfed Crops','Residential','Industry','Natural Grasslands',
                              'Forests','Shrubland','Managed water bodies','Other (Non-Manmade)','Aquaculture','Power and Energy','Forest Plantations',
                              'Irrigated crops','Other','Natural Water Bodies']
                              
    results = create_results_dict(entries, lu_fh, sheet4_lucs, aquaculture = aquaculture, power = power, industry = industry)
    
    month_labels = {1:'01',2:'02',3:'03',4:'04',5:'05',6:'06',7:'07',8:'08',9:'09',10:'10',11:'11',12:'12'}
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if isinstance(date, datetime.date):
        output_csv_fh = os.path.join(output_folder, 'sheet4_{0}_{1}.csv'.format(date.year,month_labels[date.month]))
    else:
        output_csv_fh = os.path.join(output_folder, 'sheet4_{0}.csv'.format(date))
                
    first_row = ['LANDUSE_TYPE'] + results.keys()
    
    csv_file = open(output_csv_fh, 'wb')
    writer = csv.writer(csv_file, delimiter=';')
    writer.writerow(first_row)
    
    for lu_type in results.values()[0].keys():
        row = list()
        row.append(lu_type)
        for flow in results.keys():
            row.append(results[flow][lu_type] * convert_unit)
        writer.writerow(row)
        if lu_type in required_landuse_types: 
            required_landuse_types.remove(lu_type)
            
    for missing_lu_type in required_landuse_types:
        writer.writerow([missing_lu_type, 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'])
    
    csv_file.close()
    
    return output_csv_fh

def create_sheet4(basin, period, units, data, output, template=False, margin = 0.01):
    """
    Create sheet 4 of the Water Accounting Plus framework.
    
    Parameters
    ----------
    basin : str
        The name of the basin.
    period : str
        The period of analysis.
    units : list
        A list with strings of the units of the data on sheet 4a and 4b
        respectively.
    data : list
        List with two values pointing to csv files that contains the water data. The csv file has to
        follow an specific format. A sample csv is available here:
        https://github.com/wateraccounting/wa/tree/master/Sheets/csv
    output : list
        Filehandles pointing to the jpg files to be created.
    template : list or boolean, optional
        A list with two entries of the svg files of the sheet. False
        uses the standard svg files. Default is False.

    Examples
    --------
    >>> from wa.Sheets import *
    >>> create_sheet3(basin='Helmand', period='2007-2011',
                  units = ['km3/yr', 'km3/yr'],
                  data = [r'C:\Sheets\csv\Sample_sheet4_part1.csv',
                          r'C:\Sheets\csv\Sample_sheet4_part2.csv'],
                  output = [r'C:\Sheets\sheet_4_part1.jpg',
                            r'C:\Sheets\sheet_4_part2.jpg'])
    """
    if data[0] is not None:
        df1 = pd.read_csv(data[0], sep=';')
    if data[1] is not None:
        df2 = pd.read_csv(data[1], sep=';')
    
    # Read csv part 1
    if data[0] is not None:
        p1 = dict()
        p1['sp_r01_c01'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].SUPPLY_SURFACEWATER),
                                   float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].SUPPLY_GROUNDWATER)])
        p1['sp_r02_c01'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].SUPPLY_SURFACEWATER),
                                   float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].SUPPLY_GROUNDWATER)])
        p1['sp_r03_c01'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].SUPPLY_SURFACEWATER),
                                   float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].SUPPLY_GROUNDWATER)])
        p1['sp_r04_c01'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].SUPPLY_SURFACEWATER),
                                   float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].SUPPLY_GROUNDWATER)])
        p1['sp_r05_c01'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].SUPPLY_SURFACEWATER),
                                   float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].SUPPLY_GROUNDWATER)])
        p1['sp_r06_c01'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].SUPPLY_SURFACEWATER),
                                   float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].SUPPLY_GROUNDWATER)])
        p1['sp_r07_c01'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].SUPPLY_SURFACEWATER),
                                   float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].SUPPLY_GROUNDWATER)])
        p1['sp_r08_c01'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Other")].SUPPLY_SURFACEWATER),
                                   float(df1.loc[(df1.LANDUSE_TYPE == "Other")].SUPPLY_GROUNDWATER)])
        
        p1['dm_r01_c01'] = float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].DEMAND)
        p1['dm_r02_c01'] = float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].DEMAND) 
        p1['dm_r03_c01'] = float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].DEMAND) 
        p1['dm_r04_c01'] = float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].DEMAND) 
        p1['dm_r05_c01'] = float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].DEMAND) 
        p1['dm_r06_c01'] = float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].DEMAND) 
        p1['dm_r07_c01'] = float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].DEMAND) 
        p1['dm_r08_c01'] = float(df1.loc[(df1.LANDUSE_TYPE == "Other")].DEMAND)
        
        p1['sp_r01_c02'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].CONSUMED_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].CONSUMED_OTHER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].NON_CONVENTIONAL_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].NON_RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].NON_RECOVERABLE_SURFACEWATER)])
        p1['sp_r02_c02'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].CONSUMED_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].CONSUMED_OTHER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].NON_CONVENTIONAL_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].NON_RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].NON_RECOVERABLE_SURFACEWATER)]) 
        p1['sp_r03_c02'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].CONSUMED_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].CONSUMED_OTHER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].NON_CONVENTIONAL_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].NON_RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].NON_RECOVERABLE_SURFACEWATER)]) 
        p1['sp_r04_c02'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].CONSUMED_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].CONSUMED_OTHER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].NON_CONVENTIONAL_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].NON_RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].NON_RECOVERABLE_SURFACEWATER)]) 
        p1['sp_r05_c02'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].CONSUMED_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].CONSUMED_OTHER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].NON_CONVENTIONAL_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].NON_RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].NON_RECOVERABLE_SURFACEWATER)]) 
        p1['sp_r06_c02'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].CONSUMED_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].CONSUMED_OTHER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].NON_CONVENTIONAL_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].NON_RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].NON_RECOVERABLE_SURFACEWATER)]) 
        p1['sp_r07_c02'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].CONSUMED_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].CONSUMED_OTHER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].NON_CONVENTIONAL_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].NON_RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].NON_RECOVERABLE_SURFACEWATER)]) 
        p1['sp_r08_c02'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Other")].CONSUMED_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Other")].CONSUMED_OTHER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Other")].NON_CONVENTIONAL_ET),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Other")].NON_RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Other")].NON_RECOVERABLE_SURFACEWATER)]) 
    
        p1['sp_r01_c03'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].RECOVERABLE_SURFACEWATER)])
        p1['sp_r02_c03'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].RECOVERABLE_SURFACEWATER)]) 
        p1['sp_r03_c03'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].RECOVERABLE_SURFACEWATER)]) 
        p1['sp_r04_c03'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].RECOVERABLE_SURFACEWATER)]) 
        p1['sp_r05_c03'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].RECOVERABLE_SURFACEWATER)]) 
        p1['sp_r06_c03'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].RECOVERABLE_SURFACEWATER)]) 
        p1['sp_r07_c03'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].RECOVERABLE_SURFACEWATER)]) 
        p1['sp_r08_c03'] = pd.np.sum([float(df1.loc[(df1.LANDUSE_TYPE == "Other")].RECOVERABLE_GROUNDWATER),
                                         float(df1.loc[(df1.LANDUSE_TYPE == "Other")].RECOVERABLE_SURFACEWATER)])
        
        assert pd.np.any([np.isnan(p1['sp_r01_c01']), pd.np.all([p1['sp_r01_c01'] <= (1 + margin) * (p1['sp_r01_c02'] + p1['sp_r01_c03']), 
                          p1['sp_r01_c01'] >= (1 - margin) * (p1['sp_r01_c02'] + p1['sp_r01_c03'])])])
        assert pd.np.any([np.isnan(p1['sp_r02_c01']), pd.np.all([p1['sp_r02_c01'] <= (1 + margin) * (p1['sp_r02_c02'] + p1['sp_r02_c03']), 
                          p1['sp_r02_c01'] >= (1 - margin) * (p1['sp_r02_c02'] + p1['sp_r02_c03'])])])
        assert pd.np.any([np.isnan(p1['sp_r03_c01']), pd.np.all([p1['sp_r03_c01'] <= (1 + margin) * (p1['sp_r03_c02'] + p1['sp_r03_c03']), 
                          p1['sp_r03_c01'] >= (1 - margin) * (p1['sp_r03_c02'] + p1['sp_r03_c03'])])])
        assert pd.np.any([np.isnan(p1['sp_r04_c01']), pd.np.all([p1['sp_r04_c01'] <= (1 + margin) * (p1['sp_r04_c02'] + p1['sp_r04_c03']), 
                          p1['sp_r04_c01'] >= (1 - margin) * (p1['sp_r04_c02'] + p1['sp_r04_c03'])])])
        assert pd.np.any([np.isnan(p1['sp_r05_c01']), pd.np.all([p1['sp_r05_c01'] <= (1 + margin) * (p1['sp_r05_c02'] + p1['sp_r05_c03']), 
                          p1['sp_r05_c01'] >= (1 - margin) * (p1['sp_r05_c02'] + p1['sp_r05_c03'])])])
        assert pd.np.any([np.isnan(p1['sp_r06_c01']), pd.np.all([p1['sp_r06_c01'] <= (1 + margin) * (p1['sp_r06_c02'] + p1['sp_r06_c03']), 
                          p1['sp_r06_c01'] >= (1 - margin) * (p1['sp_r06_c02'] + p1['sp_r06_c03'])])])
        assert pd.np.any([np.isnan(p1['sp_r07_c01']), pd.np.all([p1['sp_r07_c01'] <= (1 + margin) * (p1['sp_r07_c02'] + p1['sp_r07_c03']), 
                          p1['sp_r07_c01'] >= (1 - margin) * (p1['sp_r07_c02'] + p1['sp_r07_c03'])])])
        assert pd.np.any([np.isnan(p1['sp_r08_c01']), pd.np.all([p1['sp_r08_c01'] <= (1 + margin) * (p1['sp_r08_c02'] + p1['sp_r08_c03']), 
                          p1['sp_r08_c01'] >= (1 - margin) * (p1['sp_r08_c02'] + p1['sp_r08_c03'])])])
        
        p1['wd_r01_c01'] = pd.np.nansum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Other")].SUPPLY_GROUNDWATER)])
        
        p1['wd_r02_c01'] = pd.np.nansum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Other")].SUPPLY_SURFACEWATER)])
                               
        p1['wd_r03_c01'] = pd.np.nansum([p1['wd_r01_c01'],p1['wd_r02_c01']])
        
        p1['sp_r01_c04'] = pd.np.nansum([p1['sp_r01_c02'],p1['sp_r02_c02'],p1['sp_r03_c02'],p1['sp_r04_c02'],p1['sp_r05_c02'],p1['sp_r06_c02'],p1['sp_r07_c02'],p1['sp_r08_c02']])
        
        p1['of_r03_c02'] = pd.np.nansum([p1['sp_r01_c03'],p1['sp_r02_c03'],p1['sp_r03_c03'],p1['sp_r04_c03'],p1['sp_r05_c03'],p1['sp_r06_c03'],p1['sp_r07_c03'],p1['sp_r08_c03']])
        
        p1['of_r02_c01'] = pd.np.nansum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Other")].RECOVERABLE_SURFACEWATER)])
                               
        p1['of_r04_c01'] = pd.np.nansum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Other")].RECOVERABLE_GROUNDWATER)])
                               
        p1['of_r03_c01'] = pd.np.nansum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].NON_RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].NON_RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].NON_RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].NON_RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].NON_RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].NON_RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].NON_RECOVERABLE_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Other")].NON_RECOVERABLE_SURFACEWATER)])
                               
        p1['of_r05_c01'] = pd.np.nansum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].NON_RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].NON_RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].NON_RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].NON_RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].NON_RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].NON_RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].NON_RECOVERABLE_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Other")].NON_RECOVERABLE_GROUNDWATER)])
                               
        p1['of_r04_c02'] = pd.np.nansum([p1['of_r05_c01'],p1['of_r03_c01']])
        
        p1['sp_r02_c04'] = pd.np.nansum([p1['of_r02_c01'],p1['of_r04_c01']])
        
        p1['of_r09_c02'] = pd.np.nansum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].CONSUMED_OTHER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].CONSUMED_OTHER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].CONSUMED_OTHER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].CONSUMED_OTHER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].CONSUMED_OTHER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].CONSUMED_OTHER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].CONSUMED_OTHER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Other")].CONSUMED_OTHER)])
    
        p1['of_r02_c02'] = pd.np.nansum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].NON_CONVENTIONAL_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].NON_CONVENTIONAL_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].NON_CONVENTIONAL_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].NON_CONVENTIONAL_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].NON_CONVENTIONAL_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].NON_CONVENTIONAL_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].NON_CONVENTIONAL_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Other")].NON_CONVENTIONAL_ET)])
                               
        p1['of_r01_c02'] = pd.np.nansum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].CONSUMED_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].CONSUMED_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].CONSUMED_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].CONSUMED_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].CONSUMED_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].CONSUMED_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].CONSUMED_ET),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Other")].CONSUMED_ET)])
                               
        p1['of_r01_c01'] = pd.np.nansum([p1['of_r02_c02'],p1['of_r01_c02']])
    
    # Read csv part 2
    if data[1] is not None:
        p2 = dict()
        p2['sp_r01_c02'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].CONSUMED_ET),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].CONSUMED_OTHER)])
        p2['sp_r02_c02'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].CONSUMED_ET),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].CONSUMED_OTHER)])
        p2['sp_r03_c02'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].CONSUMED_ET),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].CONSUMED_OTHER)])
        p2['sp_r04_c02'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].CONSUMED_ET),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].CONSUMED_OTHER)])
        p2['sp_r05_c02'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].CONSUMED_ET),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].CONSUMED_OTHER)])
        p2['sp_r06_c02'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].CONSUMED_ET),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].CONSUMED_OTHER)])
        p2['sp_r07_c02'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].CONSUMED_ET),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].CONSUMED_OTHER)])
        p2['sp_r08_c02'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].CONSUMED_ET),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].CONSUMED_OTHER)])
        
        p2['sp_r01_c03'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].RECOVERABLE_GROUNDWATER)])
        p2['sp_r02_c03'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].RECOVERABLE_GROUNDWATER)])
        p2['sp_r03_c03'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].RECOVERABLE_GROUNDWATER)])
        p2['sp_r04_c03'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].RECOVERABLE_GROUNDWATER)])
        p2['sp_r05_c03'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].RECOVERABLE_GROUNDWATER)])
        p2['sp_r06_c03'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].RECOVERABLE_GROUNDWATER)])
        p2['sp_r07_c03'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].RECOVERABLE_GROUNDWATER)])
        p2['sp_r08_c03'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].RECOVERABLE_GROUNDWATER)])
        
        p2['sp_r01_c01'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].SUPPLY_GROUNDWATER)])
        p2['sp_r02_c01'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].SUPPLY_GROUNDWATER)])
        p2['sp_r03_c01'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].SUPPLY_GROUNDWATER)])
        p2['sp_r04_c01'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].SUPPLY_GROUNDWATER)])
        p2['sp_r05_c01'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].SUPPLY_GROUNDWATER)])
        p2['sp_r06_c01'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].SUPPLY_GROUNDWATER)])
        p2['sp_r07_c01'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].SUPPLY_GROUNDWATER)])
        p2['sp_r08_c01'] = pd.np.sum([float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].SUPPLY_GROUNDWATER)])
        
        assert pd.np.any([np.isnan(p2['sp_r01_c01']), pd.np.all([p2['sp_r01_c01'] <= (1 + margin) * (p2['sp_r01_c02'] + p2['sp_r01_c03']), 
                          p2['sp_r01_c01'] >= (1 - margin) * (p2['sp_r01_c02'] + p2['sp_r01_c03'])])])
        assert pd.np.any([np.isnan(p2['sp_r02_c01']), pd.np.all([p2['sp_r02_c01'] <= (1 + margin) * (p2['sp_r02_c02'] + p2['sp_r02_c03']), 
                          p2['sp_r02_c01'] >= (1 - margin) * (p2['sp_r02_c02'] + p2['sp_r02_c03'])])])
        assert pd.np.any([np.isnan(p2['sp_r03_c01']), pd.np.all([p2['sp_r03_c01'] <= (1 + margin) * (p2['sp_r03_c02'] + p2['sp_r03_c03']), 
                          p2['sp_r03_c01'] >= (1 - margin) * (p2['sp_r03_c02'] + p2['sp_r03_c03'])])])
        assert pd.np.any([np.isnan(p2['sp_r04_c01']), pd.np.all([p2['sp_r04_c01'] <= (1 + margin) * (p2['sp_r04_c02'] + p2['sp_r04_c03']), 
                          p2['sp_r04_c01'] >= (1 - margin) * (p2['sp_r04_c02'] + p2['sp_r04_c03'])])])
        assert pd.np.any([np.isnan(p2['sp_r05_c01']), pd.np.all([p2['sp_r05_c01'] <= (1 + margin) * (p2['sp_r05_c02'] + p2['sp_r05_c03']), 
                          p2['sp_r05_c01'] >= (1 - margin) * (p2['sp_r05_c02'] + p2['sp_r05_c03'])])])
        assert pd.np.any([np.isnan(p2['sp_r06_c01']), pd.np.all([p2['sp_r06_c01'] <= (1 + margin) * (p2['sp_r06_c02'] + p2['sp_r06_c03']), 
                          p2['sp_r06_c01'] >= (1 - margin) * (p2['sp_r06_c02'] + p2['sp_r06_c03'])])])
        assert pd.np.any([np.isnan(p2['sp_r07_c01']), pd.np.all([p2['sp_r07_c01'] <= (1 + margin) * (p2['sp_r07_c02'] + p2['sp_r07_c03']), 
                          p2['sp_r07_c01'] >= (1 - margin) * (p2['sp_r07_c02'] + p2['sp_r07_c03'])])])
        assert pd.np.any([np.isnan(p2['sp_r08_c01']), pd.np.all([p2['sp_r08_c01'] <= (1 + margin) * (p2['sp_r08_c02'] + p2['sp_r08_c03']), 
                          p2['sp_r08_c01'] >= (1 - margin) * (p2['sp_r08_c02'] + p2['sp_r08_c03'])])])
        
        
        p2['dm_r01_c01'] = float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].DEMAND)
        p2['dm_r02_c01'] = float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].DEMAND)
        p2['dm_r03_c01'] = float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].DEMAND)
        p2['dm_r04_c01'] = float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].DEMAND)
        p2['dm_r05_c01'] = float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].DEMAND)
        p2['dm_r06_c01'] = float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].DEMAND)
        p2['dm_r07_c01'] = float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].DEMAND)
        p2['dm_r08_c01'] = float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].DEMAND)
        
        p2['wd_r01_c01'] = pd.np.nansum([float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].SUPPLY_GROUNDWATER)])
        
        p2['wd_r03_c01'] = pd.np.nansum([float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].SUPPLY_SURFACEWATER)])
        
        p2['wd_r02_c01'] = pd.np.nansum([p2['wd_r01_c01'],p2['wd_r03_c01']])
        
        p2['sp_r01_c04'] = pd.np.nansum([p2['sp_r01_c02'],
                                   p2['sp_r02_c02'],
                                   p2['sp_r03_c02'],
                                   p2['sp_r04_c02'],
                                   p2['sp_r05_c02'],
                                   p2['sp_r06_c02'],
                                   p2['sp_r07_c02'],
                                   p2['sp_r08_c02']])
                                   
        p2['of_r03_c02'] = p2['sp_r02_c04'] = pd.np.nansum([p2['sp_r01_c03'],
                                   p2['sp_r02_c03'],
                                   p2['sp_r03_c03'],
                                   p2['sp_r04_c03'],
                                   p2['sp_r05_c03'],
                                   p2['sp_r06_c03'],
                                   p2['sp_r07_c03'],
                                   p2['sp_r08_c03']])
                                   
        p2['of_r01_c01'] = p2['of_r01_c02'] = pd.np.nansum([float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].CONSUMED_ET),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].CONSUMED_ET),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].CONSUMED_ET),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].CONSUMED_ET),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].CONSUMED_ET),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].CONSUMED_ET),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].CONSUMED_ET),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].CONSUMED_ET)])
        
        p2['of_r02_c02'] = pd.np.nansum([float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].CONSUMED_OTHER),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].CONSUMED_OTHER),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].CONSUMED_OTHER),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].CONSUMED_OTHER),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].CONSUMED_OTHER),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].CONSUMED_OTHER),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].CONSUMED_OTHER),
                                                float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].CONSUMED_OTHER)])
        
        
        p2['of_r03_c01'] = pd.np.nansum([float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].RECOVERABLE_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].RECOVERABLE_SURFACEWATER)]) 
        
        p2['of_r02_c01'] = pd.np.nansum([float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].RECOVERABLE_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].RECOVERABLE_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].RECOVERABLE_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].RECOVERABLE_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].RECOVERABLE_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].RECOVERABLE_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].RECOVERABLE_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].RECOVERABLE_GROUNDWATER)]) 

    # Calculations & modify svgs
    if not template:
        path = os.path.dirname(os.path.abspath(__file__))
        svg_template_path_1 = os.path.join(path, 'svg', 'sheet_4_part1.svg')
        svg_template_path_2 = os.path.join(path, 'svg', 'sheet_4_part2.svg')
    else:
        svg_template_path_1 = os.path.abspath(template[0])
        svg_template_path_2 = os.path.abspath(template[1])
    
    if data[0] is not None:
        tree1 = ET.parse(svg_template_path_1)
        xml_txt_box = tree1.findall('''.//*[@id='basin1']''')[0]
        xml_txt_box.getchildren()[0].text = 'Basin: ' + basin
        
        xml_txt_box = tree1.findall('''.//*[@id='period1']''')[0]
        xml_txt_box.getchildren()[0].text = 'Period: ' + period
        
        xml_txt_box = tree1.findall('''.//*[@id='units1']''')[0]
        xml_txt_box.getchildren()[0].text = 'Part 1: Manmade ({0})'.format(units[0])
        
        for key in p1.keys():
            xml_txt_box = tree1.findall(".//*[@id='{0}']".format(key))[0]
            if not pd.isnull(p1[key]):
                xml_txt_box.getchildren()[0].text = '%.2f' % p1[key]
            else:
                xml_txt_box.getchildren()[0].text = '-'
                
    if data[1] is not None:
        tree2 = ET.parse(svg_template_path_2)
        xml_txt_box = tree2.findall('''.//*[@id='basin2']''')[0]
        xml_txt_box.getchildren()[0].text = 'Basin: ' + basin
        
        xml_txt_box = tree2.findall('''.//*[@id='period2']''')[0]
        xml_txt_box.getchildren()[0].text = 'Period: ' + period
        
        xml_txt_box = tree2.findall('''.//*[@id='units2']''')[0]
        xml_txt_box.getchildren()[0].text = 'Part 2: Natural Landuse ({0})'.format(units[1])
        
        for key in p2.keys():
            xml_txt_box = tree2.findall(".//*[@id='{0}']".format(key))[0]
            if not pd.isnull(p2[key]):
                xml_txt_box.getchildren()[0].text = '%.2f' % p2[key]
            else:
                xml_txt_box.getchildren()[0].text = '-'    

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    
    if data[0] is not None:
        tempout_path = output[0].replace('.png', '_temporary.svg')
        tree1.write(tempout_path)
        subprocess.call(['C:\Program Files\Inkscape\inkscape.exe',tempout_path,'--export-png='+output[0], '-d 300'])
        os.remove(tempout_path)
        
    if data[1] is not None:
        tempout_path = output[1].replace('.png', '_temporary.svg')
        tree2.write(tempout_path)
        subprocess.call(['C:\Program Files\Inkscape\inkscape.exe',tempout_path,'--export-png='+output[1], '-d 300'])
        os.remove(tempout_path)

def fractions(lu_fh, fractions, lucs, output_folder, filename = 'fractions.tif'):
    """
    Create a map with fractions provided by a dictionary.
    
    Parameters
    ----------
    lu_fh : str
        Filehandle pointing to a landusemap.
    fractions : dict
        Dictionary with non-recoverable fractions per landuse category.
    lucs : dict
        Dictionary with landuseclasses per landuse category.
    output_folder : str
        Folder to store results.
        
    Returns
    -------
    fraction_fh : str
        Filehandle pointing to the map with fractions.
    """
    fraction_fh = os.path.join(output_folder, filename)
    LULC = becgis.OpenAsArray(lu_fh, nan_values = True)
    FRACTION = np.zeros(np.shape(LULC)) * np.nan
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_fh)
    for key in fractions.keys():
        classes = lucs[key]
        mask = np.logical_or.reduce([LULC == value for value in classes])
        fraction = fractions[key]
        FRACTION[mask] = fraction
    becgis.CreateGeoTiff(fraction_fh, FRACTION, driver, NDV, xsize, ysize, GeoT, Projection)
    return fraction_fh
  
def calc_delta_flow(supply_fh, conventional_et_fh, output_folder, date, non_conventional_et_fh = None, other_consumed_fh = None):
    """
    Calculate the difference between supply and the sum of conventional ET, non-conventional ET and
    other consumptions.
    
    Parameters
    ----------
    supply_fh : str
        Filehandle pointing to a map with water supply values.
    conventional_et_fh : str
        Filehandle pointing to a map with conventional ET values.
    output_folder : str
        Folder to store results.
    date : object or str
        Date used to name the output file.
    non_conventional_et_fh : str
        Filehandle pointing to a non-conventional ET map. Default is None.
    other_consumed_fh : str
        Filehandle pointing to a other consumed map. Default is None.
        
    Returns
    -------
    delta_fh : str
        Filehandle pointing to a map with the per pixel difference between
        the supply and consumption.
    """
    SUPPLY = becgis.OpenAsArray(supply_fh, nan_values = True)
    
    CONSUMED = becgis.OpenAsArray(conventional_et_fh, nan_values = True)
    if other_consumed_fh != None:
        OTHER = becgis.OpenAsArray(other_consumed_fh, nan_values = True)
        CONSUMED = np.nansum([CONSUMED, OTHER], axis = 0)
    if non_conventional_et_fh != None:
        NON_CONV = becgis.OpenAsArray(non_conventional_et_fh, nan_values = True)
        CONSUMED = np.nansum([CONSUMED, NON_CONV], axis = 0)
    
    DELTA = SUPPLY - CONSUMED

    output_folder = os.path.join(output_folder, 'delta')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if isinstance(date, datetime.date):    
        delta_fh = os.path.join(output_folder,'delta_{0}_{1}.tif'.format(date.year, date.month))
    else:
        delta_fh = os.path.join(output_folder,'delta_{0}.tif'.format(date))
    
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(supply_fh)
    becgis.CreateGeoTiff(delta_fh, DELTA, driver, NDV, xsize, ysize, GeoT, Projection)
    
    return delta_fh