# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:54:24 2017

@author: bec
"""
import os
import pandas as pd
import xml.etree.ElementTree as ET
import subprocess
import csv
import WA_Hyperloop.sheet4_functions as sh4
import datetime
import WA_Hyperloop.becgis as becgis
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

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
                
    results_sh6 = sh4.create_results_dict(entries, lu_fh, lucs)      
    
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
    
def create_sheet6(basin, period, unit, data, output, template=False):
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
    
    p1['VRtotal_natural'] = pd.np.nansum(df1.loc[(df1.SUBTYPE == 'VERTICAL_RECHARGE')].VALUE)
    p1['VRtotal_manmade'] = float(df1.loc[(df1.SUBTYPE == 'ManagedAquiferRecharge')].VALUE)
    p1['VRtotal'] = pd.np.nansum([p1['VRtotal_natural'], p1['VRtotal_manmade']])
    
    p1['CRtotal'] = float(df1.loc[(df1.SUBTYPE == 'CapillaryRise')].VALUE)
    #p1['delta_S'] = float(df1.loc[(df1.SUBTYPE == 'DeltaS')].VALUE)
    
    p1['VGWtotal_natural'] = pd.np.nansum([p1['VGW_forest'], p1['VGW_shrubland'], p1['VGW_rainfedcrops'], p1['VGW_forestplantations'], p1['VGW_wetlands'], p1['VGW_naturalgrassland'], p1['VGW_othernatural']])
    p1['VGWtotal_manmade'] = pd.np.nansum([p1['VGW_irrigatedcrops'],p1['VGW_industry'],p1['VGW_aquaculture'],p1['VGW_residential'],p1['VGW_greenhouses'],p1['VGW_othermanmade']])
    p1['VGWtotal'] = pd.np.nansum(df1.loc[(df1.SUBTYPE == 'VERTICAL_GROUNDWATER_WITHDRAWALS')].VALUE)
    
    p1['RFGtotal_manmade'] = p1['RFGtotal'] = pd.np.nansum(df1.loc[(df1.SUBTYPE == 'RETURN_FLOW_GROUNDWATER')].VALUE)
    
    p1['RFStotal_natural'] = pd.np.nansum([p1['RFS_forest'], p1['RFS_shrubland'], p1['RFS_rainfedcrops'], p1['RFS_forestplantations'], p1['RFS_wetlands'], p1['RFS_naturalgrassland'], p1['RFS_othernatural']])
    
    p1['RFStotal_manmade'] = pd.np.nansum([p1['RFS_irrigatedcrops'],p1['RFS_industry'],p1['RFS_aquaculture'],p1['RFS_residential'],p1['RFS_greenhouses'],p1['RFS_othermanmade']])
    
    p1['RFStotal'] = pd.np.nansum(df1.loc[(df1.SUBTYPE == 'RETURN_FLOW_SURFACEWATER')].VALUE)
    
    p1['HGI'] = float(df1.loc[(df1.TYPE == 'NON_LU_SPECIFIC') & (df1.SUBTYPE == 'GWInflow')].VALUE)
    p1['HGO'] = float(df1.loc[(df1.TYPE == 'NON_LU_SPECIFIC') & (df1.SUBTYPE == 'GWOutflow')].VALUE)
    p1['baseflow'] = float(df1.loc[(df1.TYPE == 'NON_LU_SPECIFIC') & (df1.SUBTYPE == 'Baseflow')].VALUE)
    
    p1['delta_S'] = p1['VRtotal'] - p1['CRtotal'] - p1['VGWtotal'] + p1['RFGtotal_manmade'] + p1['RFStotal'] - p1['baseflow']
    #p1['CRtotal'] = p1['VRtotal'] - p1['VGWtotal'] + p1['RFGtotal_manmade'] + p1['RFStotal'] - p1['baseflow'] - p1['delta_S']

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
    
    for key in p1.keys():
        xml_txt_box = tree1.findall(".//*[@id='{0}']".format(key))[0]
        if not pd.isnull(p1[key]):
            xml_txt_box.getchildren()[0].text = '%.1f' % p1[key]
        else:
            xml_txt_box.getchildren()[0].text = '-'
    
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    tempout_path = output.replace('.png', '_temporary.svg')
    tree1.write(tempout_path)
    
    subprocess.call(['C:\Program Files\Inkscape\inkscape.exe',tempout_path,'--export-png='+output, '-d 300'])
    
    os.remove(tempout_path)
    
    return p1

def plot_storages(ds_ts, bf_ts, cr_ts, vgw_ts, vr_ts, rfg_ts, rfs_ts, dates, output_folder, catchment_name, extension = 'png'):
    
    fig  = plt.figure(figsize = (10,10))
    plt.clf()
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder=0)
    ax = plt.subplot(111)
    dScum = np.cumsum(np.append(0., ds_ts))[1:] * -1
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
    ax.set_ylabel('Cumulative dS [0.1 km3/month]')
    ax.set_title('Cumulative dS, {0}'.format(catchment_name))
    ax.set_xlim([dtes[0], dtes[-1]])
    fig.autofmt_xdate()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.21),fancybox=True, shadow=True, ncol=5)
    [i.set_zorder(10) for i in ax.spines.itervalues()]
    plt.savefig(os.path.join(output_folder, 'water_storage_{1}.{0}'.format(extension, dates[0].year)))
    
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
    plt.savefig(os.path.join(output_folder, 'water_balance_{1}.{0}'.format(extension, dates[0].year)))



























