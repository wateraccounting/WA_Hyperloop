# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:55:04 2016

@author: bec
"""
import os
import gdal
import shutil
import csv
import datetime
import calendar
import subprocess
from scipy import interpolate
import tempfile as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET

import WA_Hyperloop.becgis as becgis
from WA_Hyperloop import hyperloop as hl
import WA_Hyperloop.get_dictionaries as gd
from WA_Hyperloop.paths import get_path
from WA_Hyperloop.grace_tr_correction import correct_var
from WA_Hyperloop.sup_is_etb_in_natural_lu import bf_reduction_with_gwsup

def sw_ret_wpix(non_consumed_dsro, non_consumed_dperc, lu, ouput_dir_ret_frac):
    DSRO = becgis.OpenAsArray(non_consumed_dsro, nan_values = True)
    DPERC = becgis.OpenAsArray(non_consumed_dperc, nan_values = True)
    DSRO[np.isnan(DSRO)] = 0
    DPERC[np.isnan(DPERC)] = 0
    LU = becgis.OpenAsArray(lu, nan_values = True)
    DTOT = DSRO + DPERC
    SWRETFRAC = LU * 0
    SWRETFRAC[DTOT > 0] = (DSRO/(DTOT))[DTOT > 0]
    
    geo_info = becgis.GetGeoInfo(non_consumed_dsro)
    fh = os.path.join(ouput_dir_ret_frac, 'sw_return_fraction' + os.path.basename(non_consumed_dsro)[-13:])
    becgis.CreateGeoTiff(fh, SWRETFRAC, *geo_info)
    return fh    

def multiply_raster_by_c(sw_supply_fraction_tif, alpha):
    geo_info = becgis.GetGeoInfo(sw_supply_fraction_tif)
    SW_FRAC = becgis.OpenAsArray(sw_supply_fraction_tif, nan_values = True)
    SW_FRAC = SW_FRAC * alpha
    becgis.CreateGeoTiff(sw_supply_fraction_tif, SW_FRAC, *geo_info)
    return
    
def calc_difference(ds1, ds2, output_folder):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    common_dates = becgis.CommonDates([ds1[1], ds2[1]])
    
    for dt in common_dates:
    
        DS1 = becgis.OpenAsArray(ds1[0][ds1[1] == dt][0], nan_values = True)
        DS2 = becgis.OpenAsArray(ds2[0][ds2[1] == dt][0], nan_values = True)
        
        DS2[np.isnan(DS2)] = 0.0
        
        DIFF = DS1 - DS2
        
        fn = 'temp_{0}{1}.tif'.format(dt.year, str(dt.month).zfill(2))
        fh = os.path.join(output_folder, fn)
        
        geo_info = becgis.GetGeoInfo(ds1[0][0])
        
        becgis.CreateGeoTiff(fh, DIFF, *geo_info)
        
    diff = becgis.SortFiles(output_folder, [-10,-6], month_position = [-6,-4])[0:2]

    return diff


def calc_recharge(perc, dperc):
    
    output_folder_pcs = os.path.normpath(perc[0][0]).split(os.path.sep)[:-2]
    
    output_folder1 = output_folder_pcs + ['diff']
    output_folder1.insert(1, os.path.sep)
    output_folder1 = os.path.join(*output_folder1)
    
    diff = calc_difference(perc, dperc, output_folder1)

    output_folder2 = output_folder_pcs + ['r']
    output_folder2.insert(1, os.path.sep)
    output_folder2 = os.path.join(*output_folder2)
    
    rchrg = becgis.AverageSeries(diff[0], diff[1], 1, 
                                 output_folder2, para_name = 'rchrg')
#    shutil.rmtree(output_folder1)
    
    return rchrg

def create_gw_supply(metadata, complete_data, output_dir):
    
    common_dates = becgis.CommonDates([complete_data['supply_total'][1], complete_data['supply_sw'][1]])
    for date in common_dates:
        
        total_supply_tif = complete_data['supply_total'][0][complete_data['supply_total'][1] == date][0]
        supply_sw_tif = complete_data['supply_sw'][0][complete_data['supply_sw'][1] == date][0]
        
        SUP = becgis.OpenAsArray(total_supply_tif, nan_values = True)
        SW = becgis.OpenAsArray(supply_sw_tif, nan_values = True)
        
        GW = SUP - SW
        
        geo_info = becgis.GetGeoInfo(supply_sw_tif)
        
        folder = os.path.join(output_dir, 'data', 'supply_gw')
        if not os.path.exists(folder):
            os.makedirs(folder)

        supply_gw_tif = os.path.join(folder, 'supply_gw_{0}_{1}.tif'.format(date.year, str(date.month).zfill(2)))
        
        becgis.CreateGeoTiff(supply_gw_tif, GW, *geo_info)
        
    meta = becgis.SortFiles(folder, [-11,-7], month_position = [-6,-4])[0:2]
    
    return meta
   
    
def create_sheet4_6(complete_data, metadata, output_dir, global_data):

    output_dir = os.path.join(output_dir, metadata['name'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_dir2 = os.path.join(output_dir, 'sheet4')
    if not os.path.exists(output_dir2):
        os.makedirs(output_dir2)
        
    output_dir3 = os.path.join(output_dir, 'sheet6')
    if not os.path.exists(output_dir3):
        os.makedirs(output_dir3)
        
    lucs = gd.get_sheet4_6_classes() 
    sw_supply_fractions = gd.get_sheet4_6_fractions()

    lu_based_supply_split = metadata['lu_based_supply_split'] 
    grace_supply_split = metadata['grace_supply_split']

    equiped_sw_irrigation_tif = global_data["equiped_sw_irrigation"]
    wpl_tif = global_data["wpl_tif"]
    

    non_recov_fraction_tif = non_recoverable_fractions(metadata['lu'], wpl_tif, lucs, output_dir2)

    supply_swa = return_flow_sw_sw = return_flow_sw_gw = return_flow_gw_sw = return_flow_gw_gw = np.array([])

    complete_data['recharge'] = calc_recharge(complete_data['perc'], complete_data['dperc'])

    common_dates = becgis.CommonDates([complete_data['recharge'][1],
                                       complete_data['etb'][1],
                                       complete_data['lai'][1],
                                       complete_data['etref'][1],
                                       complete_data['p'][1],
                                       complete_data['bf'][1]])

    other_consumed_tif = None
    non_conventional_et_tif = None
    if lu_based_supply_split:
        sw_supply_fraction_tif = fractions(metadata['lu'], sw_supply_fractions, lucs, output_dir2, filename = 'sw_supply_fraction.tif')
        sw_supply_fraction_tif = update_irrigation_fractions(metadata['lu'], sw_supply_fraction_tif, lucs, equiped_sw_irrigation_tif)
        print 'max supply frac:', np.nanmax(becgis.OpenAsArray(sw_supply_fraction_tif, nan_values = True))
    else:
        driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(metadata['lu'])
        mask = (~np.isnan(becgis.OpenAsArray(metadata['lu'], nan_values=True))).astype(int)
        mask[mask==0] = -9999
        sw_supply_fraction_tif = os.path.join(output_dir2, 'sw_supply_fraction.tif')
        becgis.CreateGeoTiff(sw_supply_fraction_tif, mask, driver, NDV, xsize, ysize, GeoT, Projection)
        
    for date in common_dates:
        conventional_et_tif = complete_data['etb'][0][complete_data['etb'][1] == date][0]
        ###
        # Calculate supply and split into GW and SW supply
        ###
        total_supply_tif = complete_data['supply_total'][0][complete_data['supply_total'][1] == date][0]
        if lu_based_supply_split:
            supply_sw_tif, supply_gw_tif = split_flows(total_supply_tif, sw_supply_fraction_tif, 
                                                       os.path.join(output_dir, 'data'), date, 
                                                       flow_names = ['supply_swa','supply_gw'])
            os.remove(supply_gw_tif)
        
            supply_swa = np.append(supply_swa, supply_sw_tif)
        else:
            supply_swa = np.append(supply_swa, total_supply_tif)
        complete_data['supply_swa'] = (supply_swa, common_dates)
    
    # Correct sw/gw split to match with GRACE storage
    if grace_supply_split: 
        inp = metadata['grace_split_alpha_bounds']
        assert np.all(inp[0] <= inp[1]), "invalid bounds"
        bounds = (np.clip(inp[0], [0.,0.,0.], [1.,1.,12.]), np.clip(inp[1], [0.,0.,0.], [1.,1.,12.]))

        a, complete_data['supply_sw'] = correct_var(metadata, complete_data,
                        os.path.split(output_dir)[0], 'p-et-tr+supply_swa',
                        slope = True, new_var = 'supply_sw', bounds = bounds)

        print '-----> alpha = {0}'.format(a[0])

        multiply_raster_by_c(sw_supply_fraction_tif, a[0]) #claire - update sw_fraction_tif


    else: 
        complete_data['supply_sw'] = complete_data['supply_swa']
        
    complete_data['supply_gw'] = create_gw_supply(metadata, complete_data, output_dir)
    
#    complete_data = bf_reduction_with_gwsup(metadata, complete_data)
    
    for date in common_dates:    
        total_supply_tif = complete_data['supply_total'][0][complete_data['supply_total'][1] == date][0]
        supply_sw_tif = complete_data['supply_sw'][0][complete_data['supply_sw'][1] == date][0]
        supply_gw_tif = complete_data['supply_gw'][0][complete_data['supply_gw'][1] == date][0]
        conventional_et_tif = complete_data['etb'][0][complete_data['etb'][1] == date][0]

        non_consumed_dsro = complete_data['dro'][0][complete_data['dro'][1] == date][0] 
        non_consumed_dperc = complete_data['dperc'][0][complete_data['dperc'][1] == date][0]     
        ouput_dir_ret_frac = os.path.join(output_dir,'data','return_fractions')
        if not os.path.exists(ouput_dir_ret_frac):
            os.makedirs(ouput_dir_ret_frac)
        sw_return_fraction_tif = sw_ret_wpix(non_consumed_dsro, non_consumed_dperc, metadata['lu'], ouput_dir_ret_frac)

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
        return_flow_sw_sw_tif, return_flow_sw_gw_tif = split_flows(non_consumed_sw_tif, sw_return_fraction_tif, os.path.join(output_dir, 'data'), date, flow_names = ['return_swsw', 'return_swgw'])
        return_flow_gw_sw_tif, return_flow_gw_gw_tif = split_flows(non_consumed_gw_tif, sw_return_fraction_tif, os.path.join(output_dir, 'data'), date, flow_names = ['return_gwsw', 'return_gwgw'])

        ###
        # Calculate the blue water demand
        ###
        demand_tif = calc_demand(complete_data['lai'][0][complete_data['lai'][1] == date][0], complete_data['etref'][0][complete_data['etref'][1] == date][0], complete_data['p'][0][complete_data['p'][1] == date][0], metadata['lu'], date, os.path.join(output_dir, 'data'))
        if "population_tif" in global_data.keys():
            population_tif = global_data["population_tif"]
            residential_demand = include_residential_supply(population_tif, metadata['lu'], total_supply_tif, date, lucs, 110, wcpc_minimal = 100)
            becgis.set_classes_to_value(demand_tif, metadata['lu'], lucs['Residential'], value = residential_demand)

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
        
        sheet4_csv =create_sheet4_csv(entries_sh4, metadata['lu'], lucs, date, os.path.join(output_dir2, 'sheet4_monthly'), convert_unit = 1)
        
        create_sheet4(metadata['name'], '{0}-{1}'.format(date.year, str(date.month).zfill(2)), ['km3/month', 'km3/month'], [sheet4_csv, sheet4_csv], 
                          [sheet4_csv.replace('.csv','_a.png'), sheet4_csv.replace('.csv','_b.png')], template = [get_path('sheet4_1_svg'), get_path('sheet4_2_svg')], smart_unit = True)
        
        return_flow_sw_sw = np.append(return_flow_sw_sw, return_flow_sw_sw_tif)
        return_flow_sw_gw = np.append(return_flow_sw_gw, return_flow_sw_gw_tif)
        return_flow_gw_sw = np.append(return_flow_gw_sw, return_flow_gw_sw_tif)
        return_flow_gw_gw = np.append(return_flow_gw_gw, return_flow_gw_gw_tif)
        
        print "sheet 4 finished for {0} (going to {1})".format(date, complete_data['etb'][1][-1])
        
        recharge_tif = complete_data["recharge"][0][complete_data["recharge"][1] == date][0]
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
    
        sheet6_csv = create_sheet6_csv(entries_sh6, entries_2_sh6, metadata['lu'], lucs, date, os.path.join(output_dir3,'sheet6_monthly'), convert_unit = 1)
        
        create_sheet6(metadata['name'], '{0}-{1}'.format(date.year, str(date.month).zfill(2)), 'km3/month', sheet6_csv, sheet6_csv.replace('.csv', '.png'), template = get_path('sheet6_svg'), smart_unit = True)
        
        print "sheet 6 finished"
        
    csv4_folder = os.path.join(output_dir2, 'sheet4_monthly')
    csv4_yearly_folder = os.path.join(output_dir2, 'sheet4_yearly')
    sheet4_csv_yearly = hl.create_csv_yearly(csv4_folder, csv4_yearly_folder, 4, 
                                             metadata['water_year_start_month'], year_position = [-11,-7], month_position = [-6,-4],
                                             header_rows = 1, header_columns = 1)
    
    csv6_folder = os.path.join(output_dir3, 'sheet6_monthly')
    csv6_yearly_folder = os.path.join(output_dir3, 'sheet6_yearly')
    csv6 = hl.create_csv_yearly(csv6_folder, csv6_yearly_folder, 6, metadata['water_year_start_month'], year_position = [-11,-7], month_position = [-6,-4], header_rows = 1, header_columns = 2)
    
    for csv_file in csv6:
        year = csv_file[-8:-4]
        create_sheet6(metadata['name'], year, 'km3/year', csv_file, csv_file.replace('.csv', '.png'), template = get_path('sheet6_svg'), smart_unit = True)
    
    for cv in sheet4_csv_yearly:
        year = int(cv[-8:-4])
        create_sheet4(metadata['name'], '{0}'.format(year), ['km3/year', 'km3/year'], [cv, cv], 
                          [cv.replace('.csv','_a.png'), cv.replace('.csv','_b.png')], template = [get_path('sheet4_1_svg'), get_path('sheet4_2_svg')], smart_unit = True)

    complete_data['return_flow_sw_sw'] = (return_flow_sw_sw, common_dates)
    complete_data['return_flow_sw_gw'] = (return_flow_sw_gw, common_dates)
    complete_data['return_flow_gw_sw'] = (return_flow_gw_sw, common_dates)
    complete_data['return_flow_gw_gw'] = (return_flow_gw_gw, common_dates)
    
    ####
    ## Remove some datasets
    ####   
    shutil.rmtree(os.path.split(non_consumed_tif)[0])
    shutil.rmtree(os.path.split(non_consumed_sw_tif)[0])
    shutil.rmtree(os.path.split(non_consumed_gw_tif)[0])
    shutil.rmtree(os.path.split(non_recov_tif)[0])
    shutil.rmtree(os.path.split(recov_tif)[0])
    shutil.rmtree(os.path.split(non_recov_sw_tif)[0])
    shutil.rmtree(os.path.split(non_recov_gw_tif)[0])
    shutil.rmtree(os.path.split(recov_sw_tif)[0])
    shutil.rmtree(os.path.split(recov_gw_tif)[0])
    
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
    
    output_folder = os.path.join(output_folder, 'demand')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    fh = os.path.join(output_folder, 'demand_{0}_{1}.tif'.format(date.year, month_labels[date.month]))
    
    becgis.CreateGeoTiff(fh, DEMAND, driver, NDV, xsize, ysize, GeoT, Projection)
    
    return fh  
    

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
    temp_folder = tf.mkdtemp()
    
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
        flow_one_fh = os.path.join(output_folder_one, '{0}_{1}{2}.tif'.format(flow_names[0],date.year,str(date.month).zfill(2)))
        flow_two_fh = os.path.join(output_folder_two, '{0}_{1}{2}.tif'.format(flow_names[1],date.year,str(date.month).zfill(2)))
    else:
        flow_one_fh = os.path.join(output_folder_one, '{0}_{1}.tif'.format(flow_names[0],date))
        flow_two_fh = os.path.join(output_folder_two, '{0}_{1}.tif'.format(flow_names[1],date))        
    
    if type(fraction_fh) == np.float64:
        FLOW = becgis.OpenAsArray(flow_fh, nan_values = True)

        FLOW_one = fraction_fh * FLOW
        FLOW_two = (1. - fraction_fh) * FLOW     
    
    else:
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


def create_sheet4(basin, period, units, data, output, template=False, margin = 0.01, smart_unit = True):
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
    
    scale = 0.
    if smart_unit:
        scale_test = pd.np.nanmax([
        
        pd.np.nansum([pd.np.nansum([float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].SUPPLY_GROUNDWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].SUPPLY_GROUNDWATER)]),
                        pd.np.nansum([float(df2.loc[(df2.LANDUSE_TYPE == "Forests")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Shrubland")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Rainfed Crops")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Forest Plantations")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Water Bodies")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Wetlands")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Natural Grasslands")].SUPPLY_SURFACEWATER),
                                   float(df2.loc[(df2.LANDUSE_TYPE == "Other (Non-Manmade)")].SUPPLY_SURFACEWATER)])]),
        
        pd.np.nansum([pd.np.nansum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].SUPPLY_GROUNDWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Other")].SUPPLY_GROUNDWATER)]),
                    pd.np.nansum([float(df1.loc[(df1.LANDUSE_TYPE == "Irrigated crops")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Managed water bodies")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Industry")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Aquaculture")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Residential")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Greenhouses")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Power and Energy")].SUPPLY_SURFACEWATER),
                               float(df1.loc[(df1.LANDUSE_TYPE == "Other")].SUPPLY_SURFACEWATER)])])
        ])
        
        scale = hl.scale_factor(scale_test)
        
        for df in [df1, df2]:
            for column in ['SUPPLY_GROUNDWATER', 'NON_RECOVERABLE_GROUNDWATER', 'SUPPLY_SURFACEWATER',
                           'NON_CONVENTIONAL_ET', 'RECOVERABLE_GROUNDWATER', 'CONSUMED_OTHER', 'CONSUMED_ET',
                           'DEMAND', 'RECOVERABLE_SURFACEWATER', 'NON_RECOVERABLE_SURFACEWATER']:
                
                df[column] *= 10**scale

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
        assert pd.np.any([np.isnan(p1['sp_r07_c01']), pd.np.all([p1['sp_r07_c01'] <= (1 + margin) * (p1['sp_r07_c02'] + p1['sp_r07_c03']), 
                          p1['sp_r07_c01'] >= (1 - margin) * (p1['sp_r07_c02'] + p1['sp_r07_c03'])])])
        assert pd.np.any([np.isnan(p1['sp_r06_c01']), pd.np.all([p1['sp_r06_c01'] <= (1 + margin) * (p1['sp_r06_c02'] + p1['sp_r06_c03']), 
                          p1['sp_r06_c01'] >= (1 - margin) * (p1['sp_r06_c02'] + p1['sp_r06_c03'])])])
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
        #xml_txt_box.getchildren()[0].text = 'Part 1: Manmade ({0})'.format(units[0])

        if np.all([smart_unit, scale > 0]):
            xml_txt_box.getchildren()[0].text = 'Part 1: Manmade ({0} {1})'.format(10.**-scale, units[1])
        else:
            xml_txt_box.getchildren()[0].text = 'Part 1: Manmade ({0})'.format(units[1])

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
        #xml_txt_box.getchildren()[0].text = 'Part 2: Natural Landuse ({0})'.format(units[1])

        if np.all([smart_unit, scale > 0]):
            xml_txt_box.getchildren()[0].text = 'Part 2: Natural Landuse ({0} {1})'.format(10**-scale, units[1])
        else:
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
        subprocess.call([get_path('inkscape'),tempout_path,'--export-png='+output[0], '-d 300'])
        os.remove(tempout_path)
        
    if data[1] is not None:
        tempout_path = output[1].replace('.png', '_temporary.svg')
        tree2.write(tempout_path)
        subprocess.call([get_path('inkscape'),tempout_path,'--export-png='+output[1], '-d 300'])
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

    output_folder = os.path.join(output_folder, 'DELTA')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if isinstance(date, datetime.date):    
        delta_fh = os.path.join(output_folder,'delta_{0}_{1}.tif'.format(date.year, date.month))
    else:
        delta_fh = os.path.join(output_folder,'delta_{0}.tif'.format(date))
    
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(supply_fh)
    becgis.CreateGeoTiff(delta_fh, DELTA, driver, NDV, xsize, ysize, GeoT, Projection)
    
    return delta_fh

def create_sheet6_csv(entries, entries_2, lu_fh, lucs, date, output_folder, convert_unit = 1):
    """
    Create a csv-file with all necessary values for Sheet 6.
    
    Parameters
    ----------
    entries : dict
        Dictionary with 'VERTICAL_RECHARGE', 'VERTICAL_GROUNDWATER_WITHDRAWALS',
        'RETURN_FLOW_GROUNDWATER' and 'RETURN_FLOW_SURFACEWATER' keys. Values are strings pointing to
        files of maps.
    entries_2 : dict
        Dictionary with 'CapillaryRise', 'DeltaS', 'ManagedAquiferRecharge', 'Baseflow',
        'GWInflow' and 'GWOutflow' as keys. Values are floats or 'nan.
    lu_fh : str
        String pointing to landusemap.
    lucs : dict
        Dictionary describing the landuse categories
    date : object
        Datetime.date object describing for which date to create the csv file.
    output_folder : str
        Folder to store results.
    convert_unit : int
        Value with which all results are multiplied before saving the csv-file.
        
    Returns
    -------
    output_csv_fh : str
        String pointing to the newly created csv-file.
    """
  
    required_landuse_types = ['Wetlands','Greenhouses','Rainfed Crops','Residential','Industry','Natural Grasslands',
                              'Forests','Shrubland','Managed water bodies','Other (Non-Manmade)','Aquaculture','Forest Plantations',
                              'Irrigated crops','Other','Natural Water Bodies', 'Glaciers']
                
    results_sh6 = create_results_dict(entries, lu_fh, lucs)      
    
    month_labels = becgis.GetMonthLabels()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if isinstance(date, datetime.date):
        output_csv_fh = os.path.join(output_folder, 'sheet6_{0}_{1}.csv'.format(date.year,month_labels[date.month]))
    else:
        output_csv_fh = os.path.join(output_folder, 'sheet6_{0}.csv'.format(date))
                
    first_row = ['TYPE', 'SUBTYPE', 'VALUE']
    
    csv_file = open(output_csv_fh, 'wb')
    writer = csv.writer(csv_file, delimiter=';')
    writer.writerow(first_row)
    
    for SUBTYPE in results_sh6.keys():
        for TYPE in results_sh6[SUBTYPE].keys():
            row = [TYPE, SUBTYPE, results_sh6[SUBTYPE][TYPE] * convert_unit]
            writer.writerow(row)
            if TYPE in required_landuse_types:
                required_landuse_types.remove(TYPE)
    
    for missing_landuse_type in required_landuse_types:
        writer.writerow([missing_landuse_type, 'VERTICAL_RECHARGE', 'nan'])
        writer.writerow([missing_landuse_type, 'VERTICAL_GROUNDWATER_WITHDRAWALS', 'nan'])
        writer.writerow([missing_landuse_type, 'RETURN_FLOW_GROUNDWATER', 'nan'])
        writer.writerow([missing_landuse_type, 'RETURN_FLOW_SURFACEWATER', 'nan'])
                   
    for key in entries_2.keys():
        row = ['NON_LU_SPECIFIC', key, entries_2[key]]
        writer.writerow(row)
            
    csv_file.close()
    
    return output_csv_fh
    
def create_sheet6(basin, period, unit, data, output, template=False, decimal = 1, smart_unit = False):
    """
    Create sheet 6 of the Water Accounting Plus framework.
    
    Parameters
    ----------
    basin : str
        The name of the basin.
    period : str
        The period of analysis.
    units : str
        the unit of the data on sheet 6.
    data : str
        csv file that contains the water data. The csv file has to
        follow an specific format. A sample csv is available here:
        https://github.com/wateraccounting/wa/tree/master/Sheets/csv
    output : list
        Filehandles pointing to the jpg files to be created.
    template : str or boolean, optional
        the svg file of the sheet. False
        uses the standard svg files. Default is False.
        
    Returns
    -------
    p1 : dict
        Dictionary with all values present on sheet 6.

    Examples
    --------
    >>> from wa.Sheets import *
    >>> create_sheet6(basin='Helmand', period='2007-2011',
                  units = 'km3/yr',
                  data = r'C:\Sheets\csv\Sample_sheet6.csv',
                  output = r'C:\Sheets\sheet_6.png')
    """
    df1 = pd.read_csv(data, sep=';')
    
    scale = 0
    if smart_unit:
        scale_test = np.nanmax([pd.np.nansum(df1.loc[(df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE),
                  pd.np.nansum([pd.np.nansum(df1.loc[(df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE), 
                                float(df1.loc[(df1.SUBTYPE == 'ManagedAquiferRecharge')].VALUE)])])
        
        scale = hl.scale_factor(scale_test)
        
        df1['VALUE'] *= 10**scale
    
    p1 = dict()
    
    p1['VR_forest'] = float(df1.loc[(df1.TYPE == 'Forests') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VR_shrubland'] = float(df1.loc[(df1.TYPE == 'Shrubland') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VR_naturalgrassland'] = float(df1.loc[(df1.TYPE == 'Natural Grasslands') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VR_naturalwaterbodies'] = float(df1.loc[(df1.TYPE == 'Natural Water Bodies') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VR_wetlands'] = float(df1.loc[(df1.TYPE == 'Wetlands') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VR_rainfedcrops'] = float(df1.loc[(df1.TYPE == 'Rainfed Crops') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VR_forestplantations'] = float(df1.loc[(df1.TYPE == 'Forest Plantations') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VR_irrigatedcrops'] = float(df1.loc[(df1.TYPE == 'Irrigated crops') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VR_managedwaterbodies'] = float(df1.loc[(df1.TYPE == 'Managed water bodies') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VR_residential'] = float(df1.loc[(df1.TYPE == 'Residential') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VR_industry'] = float(df1.loc[(df1.TYPE == 'Industry') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VR_other'] = float(df1.loc[(df1.TYPE == 'Other (Non-Manmade)') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE) 
    + float(df1.loc[(df1.TYPE == 'Other') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VR_managedaquiferrecharge'] = float(df1.loc[(df1.TYPE == 'NON_LU_SPECIFIC') & (df1.SUBTYPE == 'ManagedAquiferRecharge')].VALUE)
    p1['VR_glaciers'] = float(df1.loc[(df1.TYPE == 'Glaciers') & (df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    
    p1['VGW_forest'] = float(df1.loc[(df1.TYPE == 'Forests') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    p1['VGW_shrubland'] = float(df1.loc[(df1.TYPE == 'Shrubland') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    p1['VGW_rainfedcrops'] = float(df1.loc[(df1.TYPE == 'Rainfed Crops') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    p1['VGW_forestplantations'] = float(df1.loc[(df1.TYPE == 'Forest Plantations') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    p1['VGW_wetlands'] = float(df1.loc[(df1.TYPE == 'Wetlands') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    p1['VGW_naturalgrassland'] = float(df1.loc[(df1.TYPE == 'Natural Grasslands') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    p1['VGW_othernatural'] = float(df1.loc[(df1.TYPE == 'Other (Non-Manmade)') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    p1['VGW_irrigatedcrops'] = float(df1.loc[(df1.TYPE == 'Irrigated crops') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    p1['VGW_industry'] = float(df1.loc[(df1.TYPE == 'Industry') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    p1['VGW_aquaculture'] = float(df1.loc[(df1.TYPE == 'Aquaculture') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    p1['VGW_residential'] = float(df1.loc[(df1.TYPE == 'Residential') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    p1['VGW_greenhouses'] = float(df1.loc[(df1.TYPE == 'Greenhouses') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    p1['VGW_othermanmade'] = float(df1.loc[(df1.TYPE == 'Other') & (df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    
    p1['RFG_irrigatedcrops'] = float(df1.loc[(df1.TYPE == 'Irrigated crops') & (df1.SUBTYPE == 'RETURN_FLOW_GROUNDWATER')].VALUE)
    p1['RFG_industry'] = float(df1.loc[(df1.TYPE == 'Industry') & (df1.SUBTYPE == 'RETURN_FLOW_GROUNDWATER')].VALUE)
    p1['RFG_aquaculture'] = float(df1.loc[(df1.TYPE == 'Aquaculture') & (df1.SUBTYPE == 'RETURN_FLOW_GROUNDWATER')].VALUE)
    p1['RFG_residential'] = float(df1.loc[(df1.TYPE == 'Residential') & (df1.SUBTYPE == 'RETURN_FLOW_GROUNDWATER')].VALUE)
    p1['RFG_greenhouses'] = float(df1.loc[(df1.TYPE == 'Greenhouses') & (df1.SUBTYPE == 'RETURN_FLOW_GROUNDWATER')].VALUE)
    p1['RFG_other'] = float(df1.loc[(df1.TYPE == 'Other') & (df1.SUBTYPE == 'RETURN_FLOW_GROUNDWATER')].VALUE)
    
    p1['RFS_forest'] = float(df1.loc[(df1.TYPE == 'Forests') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    p1['RFS_shrubland'] = float(df1.loc[(df1.TYPE == 'Shrubland') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    p1['RFS_rainfedcrops'] = float(df1.loc[(df1.TYPE == 'Rainfed Crops') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    p1['RFS_forestplantations'] = float(df1.loc[(df1.TYPE == 'Forest Plantations') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    p1['RFS_wetlands'] = float(df1.loc[(df1.TYPE == 'Wetlands') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    p1['RFS_naturalgrassland'] = float(df1.loc[(df1.TYPE == 'Natural Grasslands') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    p1['RFS_othernatural'] = float(df1.loc[(df1.TYPE == 'Other (Non-Manmade)') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    p1['RFS_irrigatedcrops'] = float(df1.loc[(df1.TYPE == 'Irrigated crops') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    p1['RFS_industry'] = float(df1.loc[(df1.TYPE == 'Industry') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    p1['RFS_aquaculture'] = float(df1.loc[(df1.TYPE == 'Aquaculture') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    p1['RFS_residential'] = float(df1.loc[(df1.TYPE == 'Residential') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    p1['RFS_greenhouses'] = float(df1.loc[(df1.TYPE == 'Greenhouses') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    p1['RFS_othermanmade'] = float(df1.loc[(df1.TYPE == 'Other') & (df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    
    for key, value in p1.items():
        p1[key] = np.round(value, decimals = decimal)
    
    p1['VRtotal_natural'] = pd.np.nansum(df1.loc[(df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VRtotal_manmade'] = float(df1.loc[(df1.SUBTYPE == 'ManagedAquiferRecharge')].VALUE)
    p1['VRtotal'] = pd.np.nansum([p1['VRtotal_natural'], p1['VRtotal_manmade']])
    
    p1['CRtotal'] = float(df1.loc[(df1.SUBTYPE == 'CapillaryRise')].VALUE)
    #p1['delta_S'] = float(df1.loc[(df1.SUBTYPE == 'DeltaS')].VALUE)
    
    p1['VGWtotal_natural'] = pd.np.nansum([p1['VGW_forest'], p1['VGW_shrubland'], p1['VGW_rainfedcrops'], p1['VGW_forestplantations'], p1['VGW_wetlands'], p1['VGW_naturalgrassland'], p1['VGW_othernatural']])
    p1['VGWtotal_manmade'] = pd.np.nansum([p1['VGW_irrigatedcrops'],p1['VGW_industry'],p1['VGW_aquaculture'],p1['VGW_residential'],p1['VGW_greenhouses'],p1['VGW_othermanmade']])
    p1['VGWtotal'] = pd.np.nansum(df1.loc[(df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    
    p1['RFGtotal_manmade'] = p1['RFGtotal'] = pd.np.nansum([p1['RFG_irrigatedcrops'],
                                                        p1['RFG_industry'], 
                                                        p1['RFG_aquaculture'], 
                                                        p1['RFG_residential'], 
                                                        p1['RFG_greenhouses'], 
                                                        p1['RFG_other']])
 
    p1['RFStotal_natural'] = pd.np.nansum([p1['RFS_forest'], p1['RFS_shrubland'], p1['RFS_rainfedcrops'], p1['RFS_forestplantations'], p1['RFS_wetlands'], p1['RFS_naturalgrassland'], p1['RFS_othernatural']])
    
    p1['RFStotal_manmade'] = pd.np.nansum([p1['RFS_irrigatedcrops'],p1['RFS_industry'],p1['RFS_aquaculture'],p1['RFS_residential'],p1['RFS_greenhouses'],p1['RFS_othermanmade']])
    
    p1['RFStotal'] = pd.np.nansum([p1['RFStotal_natural'], p1['RFStotal_manmade']])
    
    p1['HGI'] = float(df1.loc[(df1.TYPE == 'NON_LU_SPECIFIC') & (df1.SUBTYPE == 'GWInflow')].VALUE)
    p1['HGO'] = float(df1.loc[(df1.TYPE == 'NON_LU_SPECIFIC') & (df1.SUBTYPE == 'GWOutflow')].VALUE)
    p1['baseflow'] = float(df1.loc[(df1.TYPE == 'NON_LU_SPECIFIC') & (df1.SUBTYPE == 'Baseflow')].VALUE)
    
    p1['delta_S'] = p1['VRtotal'] - p1['CRtotal'] - p1['VGWtotal'] + p1['RFGtotal_manmade'] + p1['RFStotal'] - p1['baseflow']
    #p1['CRtotal'] = p1['VRtotal'] - p1['VGWtotal'] + p1['RFGtotal_manmade'] + p1['RFStotal'] - p1['baseflow'] - p1['delta_S']

    

    for key, value in p1.items():
        p1[key] = np.round(value, decimals = decimal)
        
    if not template:
        path = os.path.dirname(os.path.abspath(__file__))
        svg_template_path_1 = os.path.join(path, 'svg', 'sheet_6.svg')
    else:
        svg_template_path_1 = os.path.abspath(template)
    
    tree1 = ET.parse(svg_template_path_1)
    xml_txt_box = tree1.findall('''.//*[@id='basin']''')[0]
    xml_txt_box.getchildren()[0].text = 'Basin: ' + basin
    
    xml_txt_box = tree1.findall('''.//*[@id='period']''')[0]
    xml_txt_box.getchildren()[0].text = 'Period: ' + period
    
    xml_txt_box = tree1.findall('''.//*[@id='unit']''')[0]
    xml_txt_box.getchildren()[0].text = 'Sheet 6: Groundwater ({0})'.format(unit)

    if np.all([smart_unit, scale > 0]):
        xml_txt_box.getchildren()[0].text = 'Sheet 6: Groundwater ({0} {1})'.format(10**-scale, unit)
    else:
        xml_txt_box.getchildren()[0].text = 'Sheet 6: Groundwater ({0})'.format(unit)
        
    for key in p1.keys():
        xml_txt_box = tree1.findall(".//*[@id='{0}']".format(key))[0]
        if not pd.isnull(p1[key]):
            xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimal, p1[key])
        else:
            xml_txt_box.getchildren()[0].text = '-'
    
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    tempout_path = output.replace('.png', '_temporary.svg')
    tree1.write(tempout_path)
    
    subprocess.call([get_path('inkscape'),tempout_path,'--export-png='+output, '-d 300'])
    
    os.remove(tempout_path)

def plot_storages(ds_ts, bf_ts, cr_ts, vgw_ts, vr_ts, rfg_ts, rfs_ts, dates, output_folder, catchment_name, extension = 'png'):
    
    fig  = plt.figure(figsize = (10,10))
    plt.clf()
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder=0)
    ax = plt.subplot(111)
    dScum = np.cumsum(np.append(0., ds_ts))[1:]
    ordinal_dates = [date.toordinal() for date in dates]
    dScum = interpolate.interp1d(ordinal_dates, dScum)
    x = np.arange(min(ordinal_dates), max(ordinal_dates), 1)
    dScum = dScum(x)
    dtes = [datetime.date.fromordinal(ordinal) for ordinal in x]
    zeroes = np.zeros(np.shape(dScum))
    ax.plot(dtes, dScum, 'k',label = 'Cum. dS')
    ax.fill_between(dtes, dScum, y2 = zeroes, where = dScum <= zeroes, color = '#d98d8e', label = 'Storage decrease')
    ax.fill_between(dtes, dScum, y2 = zeroes, where = dScum >= zeroes, color = '#6bb8cc', label = 'Storage increase')
    ax.scatter(dates, np.cumsum(np.append(0., ds_ts))[1:] * -1, color = 'k')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative dS [0.1 km3]')
    ax.set_title('Cumulative dS, {0}'.format(catchment_name))
    ax.set_xlim([dtes[0], dtes[-1]])
    fig.autofmt_xdate()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.21),fancybox=True, shadow=True, ncol=5)
    [i.set_zorder(10) for i in ax.spines.itervalues()]
    plt.savefig(os.path.join(output_folder, 'sheet6_water_storage_{1}.{0}'.format(extension, dates[0].year)))
    
    fig = plt.figure(figsize = (10,10))
    plt.clf()
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
    ax = plt.subplot(111)
    ordinal_dates = [date.toordinal() for date in dates]
    outflow = interpolate.interp1d(ordinal_dates, bf_ts + cr_ts + vgw_ts)
    inflow = interpolate.interp1d(ordinal_dates, vr_ts + rfg_ts + rfs_ts)
    x = np.arange(min(ordinal_dates), max(ordinal_dates), 1)
    outflow = outflow(x)
    inflow = inflow(x)
    dtes = [datetime.date.fromordinal(ordinal) for ordinal in x]
    ax.plot(dtes, inflow, label = 'Inflow (VR + RFG + RFS)', color = 'k')
    ax.plot(dtes, outflow,  '--k', label = 'Outflow (BF + CR + VGW)')
    ax.fill_between(dtes, outflow, y2 = inflow, where = outflow >= inflow ,color = '#d98d8e', label = 'dS decrease')
    ax.fill_between(dtes, outflow, y2 = inflow, where = outflow <= inflow ,color = '#6bb8cc', label = 'dS increase')
    ax.set_xlabel('Time')
    ax.set_ylabel('Flows [km3/month]')
    ax.set_title('Water Balance, {0}'.format(catchment_name))
    fig.autofmt_xdate()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.21),fancybox=True, shadow=True, ncol=5)
    [i.set_zorder(10) for i in ax.spines.itervalues()]
    plt.savefig(os.path.join(output_folder, 'sheet6_water_balance_{1}.{0}'.format(extension, dates[0].year)))



