# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:11:04 2016

@author: bec
"""

import os
import calendar
import csv
import datetime
import glob
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import xml.etree.ElementTree as ET

import WA_Hyperloop.becgis as becgis
import WA_Hyperloop.get_dictionaries as gd
from WA_Hyperloop import hyperloop as hl
import WA_Hyperloop.pairwise_validation as pwv
from WA_Hyperloop.paths import get_path

def sum_ts(flow_csvs):
    
    flows = list()
    dates = list()
    
    for cv in flow_csvs:
        
        coordinates, flow_ts, station_name, unit = pwv.create_dict_entry(cv)
        flow_dates, flow_values = becgis.Unzip(flow_ts)
        flow_dates = becgis.ConvertDatetimeDate(flow_dates)
        if unit == 'm3/s':
            flow_values = np.array([flow_values[flow_dates == date] * 60 * 60 * 24 * calendar.monthrange(date.year, date.month)[1] / 1000**3 for date in flow_dates])[:,0]
        flows.append(flow_values)
        dates.append(flow_dates)
       
    common_dates = becgis.CommonDates(dates)
    
    data = np.zeros(np.shape(common_dates))
    for flow_values, flow_dates in zip(flows, dates):
        add_data = np.array([flow_values[flow_dates == date][0] for date in common_dates])
        data += add_data

    return data, common_dates

def create_sheet1(complete_data, metadata, output_dir, global_data):
        
    output_folder = os.path.join(output_dir, metadata['name'], 'sheet1')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)    
    
    output_fh_in = False
    
    output_fh_in, output_fh_out = create_sheet1_in_outflows(os.path.join(output_dir, metadata['name'], "sheet5", "sheet5_monthly"), metadata, output_folder) 
    outflow_values, outflow_dates = sum_ts(np.array([output_fh_out]))
    transfer_values, transfer_dates = get_transfers(os.path.join(output_dir, metadata['name'], "sheet5", "sheet5_monthly"))
    
    if output_fh_in:
        inflow_values, inflow_dates = sum_ts(np.array([output_fh_in]))
    
    # Calculate the average longterm outflow.
    q_out_avg = np.nanmean(outflow_values)

    # Open a dictionary specyfing the landuseclasses.
    sheet1_lucs = gd.get_sheet1_classes()
    
    # Determine for what dates all the required data is available.
    if output_fh_in:
        common_dates = becgis.CommonDates([complete_data['p'][1], complete_data['etb'][1], complete_data['etg'][1], outflow_dates, inflow_dates])
    else:
        common_dates = becgis.CommonDates([complete_data['tr'][1],complete_data['p'][1], complete_data['etb'][1], complete_data['etg'][1]]) #, outflow_dates])
    
    # Create list to store results.
    all_results = list()
    
    for date in common_dates:
        # Summurize some data in a dictionary.
        entries = {'Fractions': complete_data['fractions'][0][complete_data['fractions'][1] == date][0],
                    'WPL': global_data["wpl_tif"],
                    'EWR': global_data["environ_water_req"],
                    'P': complete_data['p'][0][complete_data['p'][1] == date][0],
                    'ETblue': complete_data['etb'][0][complete_data['etb'][1] == date][0],
                    'ETgreen': complete_data['etg'][0][complete_data['etg'][1] == date][0]}
        
        # Select the required outflow value.
        q_outflow = outflow_values[outflow_dates == date][0]
        q_transfer = np.array(transfer_values)[np.array(transfer_dates) == date][0]
        if output_fh_in:
            q_inflow = inflow_values[inflow_dates == date][0]
        else:
            q_inflow = 0.0
        
        # Calculate the sheet values.
        results = calc_sheet1(entries, metadata['lu'], sheet1_lucs, metadata['recycling_ratio'], q_outflow, q_out_avg, output_folder,
                              q_in_sw=q_inflow, q_out_sw=q_transfer)
        
        # Save the results of the current month.
        all_results.append(results)    
        
        # Create the csv-file.
        output_fh = os.path.join(output_folder,'sheet1_monthly','sheet1_{0}_{1}.csv'.format(date.year, str(date.month).zfill(2)))
        create_csv(results, output_fh)
    
        # Plot the actual sheet.
        create_sheet1_png(metadata['name'], '{0}-{1}'.format(date.year, str(date.month).zfill(2)), 'km3/month', output_fh, output_fh.replace('.csv','.png'), template = get_path('sheet1_svg'), smart_unit = True)
    
    # Create some graphs.
    plot_storages(all_results, common_dates, metadata['name'], output_folder)
    plot_parameter(all_results, common_dates, metadata['name'], output_folder, 'utilizable_outflow')
    
    # Create yearly csv-files.
    yearly_csv_fhs = hl.create_csv_yearly(os.path.split(output_fh)[0], 
                                          os.path.join(output_folder, "sheet1_yearly"), 
                                          1, metadata['water_year_start_month'], 
                                          year_position = [-11,-7], month_position = [-6,-4], 
                                          header_rows = 1, header_columns = 3)
    
    # Plot yearly sheets.
    for csv_fh in yearly_csv_fhs:
        create_sheet1_png(metadata['name'], csv_fh[-8:-4], 'km3/year', csv_fh, csv_fh.replace('.csv','.png'), template = get_path('sheet1_svg'), smart_unit = True)
        
    return complete_data, all_results


def create_sheet1_png(basin, period, units, data, output, template=False , smart_unit = False):
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

    Example:
    from wa.Sheets import *
    create_sheet1(basin='Incomati', period='2005-2010', units='km3/year',
                  data=r'C:\Sheets\csv\Sample_sheet1.csv',
                  output=r'C:\Sheets\sheet_1.jpg')
    """
    decimals = 1
    # Read table

    df = pd.read_csv(data, sep=';')
    
    scale = 0
    if smart_unit:
        scale_test = np.nanmax(df['VALUE'].values)
        scale = hl.scale_factor(scale_test)
        
        df['VALUE'] *= 10**scale
    
    # Data frames

    df_i = df.loc[df.CLASS == "INFLOW"]
    df_s = df.loc[df.CLASS == "STORAGE"]
    df_o = df.loc[df.CLASS == "OUTFLOW"]

    # Inflow data

    rainfall = float(df_i.loc[(df_i.SUBCLASS == "PRECIPITATION") &
                              (df_i.VARIABLE == "Rainfall")].VALUE)
    snowfall = float(df_i.loc[(df_i.SUBCLASS == "PRECIPITATION") &
                              (df_i.VARIABLE == "Snowfall")].VALUE)
    p_recy = float(df_i.loc[(df_i.SUBCLASS == "PRECIPITATION") &
                   (df_i.VARIABLE == "Precipitation recycling")].VALUE)

    sw_mrs_i = float(df_i.loc[(df_i.SUBCLASS == "SURFACE WATER") &
                              (df_i.VARIABLE == "Main riverstem")].VALUE)
    sw_tri_i = float(df_i.loc[(df_i.SUBCLASS == "SURFACE WATER") &
                              (df_i.VARIABLE == "Tributaries")].VALUE)
    sw_usw_i = float(df_i.loc[(df_i.SUBCLASS == "SURFACE WATER") &
                     (df_i.VARIABLE == "Utilized surface water")].VALUE)
    sw_flo_i = float(df_i.loc[(df_i.SUBCLASS == "SURFACE WATER") &
                              (df_i.VARIABLE == "Flood")].VALUE)

    gw_nat_i = float(df_i.loc[(df_i.SUBCLASS == "GROUNDWATER") &
                              (df_i.VARIABLE == "Natural")].VALUE)
    gw_uti_i = float(df_i.loc[(df_i.SUBCLASS == "GROUNDWATER") &
                              (df_i.VARIABLE == "Utilized")].VALUE)

    q_desal = float(df_i.loc[(df_i.SUBCLASS == "OTHER") &
                             (df_i.VARIABLE == "Desalinized")].VALUE)

    # Storage data

    surf_sto = float(df_s.loc[(df_s.SUBCLASS == "CHANGE") &
                              (df_s.VARIABLE == "Surface storage")].VALUE)
    sto_sink = float(df_s.loc[(df_s.SUBCLASS == "CHANGE") &
                              (df_s.VARIABLE == "Storage in sinks")].VALUE)

    # Outflow data

    et_l_pr = float(df_o.loc[(df_o.SUBCLASS == "ET LANDSCAPE") &
                             (df_o.VARIABLE == "Protected")].VALUE)
    et_l_ut = float(df_o.loc[(df_o.SUBCLASS == "ET LANDSCAPE") &
                             (df_o.VARIABLE == "Utilized")].VALUE)
    et_l_mo = float(df_o.loc[(df_o.SUBCLASS == "ET LANDSCAPE") &
                             (df_o.VARIABLE == "Modified")].VALUE)
    et_l_ma = float(df_o.loc[(df_o.SUBCLASS == "ET LANDSCAPE") &
                             (df_o.VARIABLE == "Managed")].VALUE)

    et_u_pr = float(df_o.loc[(df_o.SUBCLASS == "ET UTILIZED FLOW") &
                             (df_o.VARIABLE == "Protected")].VALUE)
    et_u_ut = float(df_o.loc[(df_o.SUBCLASS == "ET UTILIZED FLOW") &
                             (df_o.VARIABLE == "Utilized")].VALUE)
    et_u_mo = float(df_o.loc[(df_o.SUBCLASS == "ET UTILIZED FLOW") &
                             (df_o.VARIABLE == "Modified")].VALUE)
    et_u_ma = float(df_o.loc[(df_o.SUBCLASS == "ET UTILIZED FLOW") &
                             (df_o.VARIABLE == "Managed")].VALUE)

    et_manmade = float(df_o.loc[(df_o.SUBCLASS == "ET INCREMENTAL") &
                                (df_o.VARIABLE == "Manmade")].VALUE)
    et_natural = float(df_o.loc[(df_o.SUBCLASS == "ET INCREMENTAL") &
                                (df_o.VARIABLE == "Natural")].VALUE)

    sw_mrs_o = float(df_o.loc[(df_o.SUBCLASS == "SURFACE WATER") &
                              (df_o.VARIABLE == "Main riverstem")].VALUE)
    sw_tri_o = float(df_o.loc[(df_o.SUBCLASS == "SURFACE WATER") &
                              (df_o.VARIABLE == "Tributaries")].VALUE)
    sw_usw_o = float(df_o.loc[(df_o.SUBCLASS == "SURFACE WATER") &
                     (df_o.VARIABLE == "Utilized surface water")].VALUE)
    sw_flo_o = float(df_o.loc[(df_o.SUBCLASS == "SURFACE WATER") &
                              (df_o.VARIABLE == "Flood")].VALUE)

    gw_nat_o = float(df_o.loc[(df_o.SUBCLASS == "GROUNDWATER") &
                              (df_o.VARIABLE == "Natural")].VALUE)
    gw_uti_o = float(df_o.loc[(df_o.SUBCLASS == "GROUNDWATER") &
                              (df_o.VARIABLE == "Utilized")].VALUE)

    basin_transfers = float(df_o.loc[(df_o.SUBCLASS == "SURFACE WATER") &
                            (df_o.VARIABLE == "Interbasin transfer")].VALUE)
    non_uti = float(df_o.loc[(df_o.SUBCLASS == "OTHER") &
                             (df_o.VARIABLE == "Non-utilizable")].VALUE)
    other_o = float(df_o.loc[(df_o.SUBCLASS == "OTHER") &
                             (df_o.VARIABLE == "Other")].VALUE)

    com_o = float(df_o.loc[(df_o.SUBCLASS == "RESERVED") &
                           (df_o.VARIABLE == "Commited")].VALUE)
    nav_o = float(df_o.loc[(df_o.SUBCLASS == "RESERVED") &
                           (df_o.VARIABLE == "Navigational")].VALUE)
    env_o = float(df_o.loc[(df_o.SUBCLASS == "RESERVED") &
                           (df_o.VARIABLE == "Environmental")].VALUE)

    # Calculations & modify svg
    if not template:
        path = os.path.dirname(os.path.abspath(__file__))
        svg_template_path = os.path.join(path, 'svg', 'sheet_1.svg')
    else:
        svg_template_path = os.path.abspath(template)

    tree = ET.parse(svg_template_path)

    # Titles

    xml_txt_box = tree.findall('''.//*[@id='basin']''')[0]
    xml_txt_box.getchildren()[0].text = 'Basin: ' + basin

    xml_txt_box = tree.findall('''.//*[@id='period']''')[0]
    xml_txt_box.getchildren()[0].text = 'Period: ' + period

    xml_txt_box = tree.findall('''.//*[@id='units']''')[0]
    
    if np.all([smart_unit, scale > 0]):
        xml_txt_box.getchildren()[0].text = 'Sheet 1: Resource Base ({0} {1})'.format(10**-scale, units)
    else:
        xml_txt_box.getchildren()[0].text = 'Sheet 1: Resource Base ({0})'.format(units)

    # Grey box

    p_advec = rainfall + snowfall
    q_sw_in = sw_mrs_i + sw_tri_i + sw_usw_i + sw_flo_i
    q_gw_in = gw_nat_i + gw_uti_i

    external_in = p_advec + q_desal + q_sw_in + q_gw_in
    gross_inflow = external_in + p_recy

    delta_s = surf_sto + sto_sink

    xml_txt_box = tree.findall('''.//*[@id='external_in']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, external_in)

    xml_txt_box = tree.findall('''.//*[@id='p_advec']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, p_advec)

    xml_txt_box = tree.findall('''.//*[@id='q_desal']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, q_desal)

    xml_txt_box = tree.findall('''.//*[@id='q_sw_in']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, q_sw_in)

    xml_txt_box = tree.findall('''.//*[@id='q_gw_in']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, q_gw_in)

    xml_txt_box = tree.findall('''.//*[@id='p_recycled']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, p_recy)

    xml_txt_box = tree.findall('''.//*[@id='gross_inflow']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, gross_inflow)

    if delta_s > 0:
        xml_txt_box = tree.findall('''.//*[@id='pos_delta_s']''')[0]
        xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, delta_s)

        xml_txt_box = tree.findall('''.//*[@id='neg_delta_s']''')[0]
        xml_txt_box.getchildren()[0].text = '0.0'
    else:
        xml_txt_box = tree.findall('''.//*[@id='pos_delta_s']''')[0]
        xml_txt_box.getchildren()[0].text = '0.0'

        xml_txt_box = tree.findall('''.//*[@id='neg_delta_s']''')[0]
        xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, -delta_s)

    # Pink box

    net_inflow = gross_inflow + delta_s

    xml_txt_box = tree.findall('''.//*[@id='net_inflow']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, net_inflow)

    # Light-green box

    land_et = et_l_pr + et_l_ut + et_l_mo + et_l_ma

    xml_txt_box = tree.findall('''.//*[@id='landscape_et']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, land_et)

    xml_txt_box = tree.findall('''.//*[@id='green_protected']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, et_l_pr)

    xml_txt_box = tree.findall('''.//*[@id='green_utilized']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, et_l_ut)

    xml_txt_box = tree.findall('''.//*[@id='green_modified']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, et_l_mo)

    xml_txt_box = tree.findall('''.//*[@id='green_managed']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, et_l_ma)

    xml_txt_box = tree.findall('''.//*[@id='et_rainfall']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, land_et)

    # Blue box (center)

    exploitable_water = net_inflow - land_et
    reserved_outflow = max(com_o, nav_o, env_o)

    available_water = exploitable_water - non_uti - reserved_outflow

    utilized_flow = et_u_pr + et_u_ut + et_u_mo + et_u_ma
    utilizable_outflow = available_water - utilized_flow

    inc_et = et_manmade + et_natural

    non_cons_water = utilizable_outflow + non_uti + reserved_outflow

    non_rec_flow = et_u_pr + et_u_ut + et_u_mo + et_u_ma - inc_et - other_o

    xml_txt_box = tree.findall('''.//*[@id='incremental_et']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, inc_et)

    xml_txt_box = tree.findall('''.//*[@id='exploitable_water']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, exploitable_water)

    xml_txt_box = tree.findall('''.//*[@id='available_water']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, available_water)

    xml_txt_box = tree.findall('''.//*[@id='utilized_flow']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, utilized_flow)

    xml_txt_box = tree.findall('''.//*[@id='blue_protected']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, et_u_pr)

    xml_txt_box = tree.findall('''.//*[@id='blue_utilized']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, et_u_ut)

    xml_txt_box = tree.findall('''.//*[@id='blue_modified']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, et_u_mo)

    xml_txt_box = tree.findall('''.//*[@id='blue_managed']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, et_u_ma)

    xml_txt_box = tree.findall('''.//*[@id='utilizable_outflow']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, utilizable_outflow)

    xml_txt_box = tree.findall('''.//*[@id='non-utilizable_outflow']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, non_uti)

    xml_txt_box = tree.findall('''.//*[@id='reserved_outflow_max']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, reserved_outflow)

    xml_txt_box = tree.findall('''.//*[@id='non-consumed_water']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, non_cons_water)

    xml_txt_box = tree.findall('''.//*[@id='manmade']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, et_manmade)

    xml_txt_box = tree.findall('''.//*[@id='natural']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, et_natural)

    xml_txt_box = tree.findall('''.//*[@id='other']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, other_o)

    xml_txt_box = tree.findall('''.//*[@id='non-recoverable_flow']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, non_rec_flow)

    # Blue box (right)

    outflow = non_cons_water + non_rec_flow 

    q_sw_out = sw_mrs_o + sw_tri_o + sw_usw_o + sw_flo_o 
    q_gw_out = gw_nat_o + gw_uti_o

    xml_txt_box = tree.findall('''.//*[@id='outflow']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, outflow)

    xml_txt_box = tree.findall('''.//*[@id='q_sw_outlet']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, q_sw_out)

    xml_txt_box = tree.findall('''.//*[@id='q_sw_out']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, basin_transfers)

    xml_txt_box = tree.findall('''.//*[@id='q_gw_out']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, q_gw_out)

    # Dark-green box

    consumed_water = land_et + utilized_flow
    depleted_water = consumed_water - p_recy - non_rec_flow
    external_out = depleted_water + outflow

    xml_txt_box = tree.findall('''.//*[@id='et_recycled']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, p_recy)

    xml_txt_box = tree.findall('''.//*[@id='consumed_water']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, consumed_water)

    xml_txt_box = tree.findall('''.//*[@id='depleted_water']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, depleted_water)

    xml_txt_box = tree.findall('''.//*[@id='external_out']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, external_out)

    xml_txt_box = tree.findall('''.//*[@id='et_out']''')[0]
    xml_txt_box.getchildren()[0].text = '{1:.{0}f}'.format(decimals, depleted_water)

    # svg to string
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    Path_Inkscape = get_path('inkscape')

    # Export svg to png
    tempout_path = output.replace('.png', '_temporary.svg')
    tree.write(tempout_path)
    subprocess.call([Path_Inkscape,tempout_path,'--export-png='+output, '-d 300'])
#    os.remove(tempout_path)

    # Return
    return output

def create_sheet1_in_outflows(sheet5_csv_folder, metadata, output_dir):
    
    output_fh_in = os.path.join(output_dir,"sheet1_inflow_km3_"+metadata['name']+'.csv')
    output_fh_out = os.path.join(output_dir,"sheet1_outflow_km3_"+metadata['name']+'.csv')
       
    csv_file_in = open(output_fh_in, 'wb')
    csv_file_out = open(output_fh_out, 'wb')

    writer_in = csv.writer(csv_file_in, delimiter=';')
    writer_in.writerow(['lat:',0,'lon:',0,'km3 (from Sheet5)'])
    writer_in.writerow(['datetime','year','month','day','data'])
    
    writer_out = csv.writer(csv_file_out, delimiter=';')
    writer_out.writerow(['lat:',0,'lon:',0,'km3 (from Sheet5)'])
    writer_out.writerow(['datetime','year','month','day','data'])

    csv_files = glob.glob(sheet5_csv_folder+'\\sheet5_*.csv')
    total_inflow = 0.0
    for f in csv_files:
        fn = os.path.split(f)[1]
        year = int(fn.split('sheet5_')[1].split('.csv')[0][:4])
        month = int(fn.split('sheet5_')[1].split('.csv')[0][-2:])
        date = datetime.datetime(year,month,1)
        df = pd.read_csv(f,sep=';')
        df_basin = df.loc[df.SUBBASIN == 'basin'] 
        df_inf = df_basin.loc[df_basin.VARIABLE == 'Inflow']
#        df_tran = df_basin.loc[df_basin.VARIABLE == 'Interbasin Transfer']
        df_outf = df_basin.loc[df_basin.VARIABLE == 'Outflow: Total']
#        df_nonrecov = df_basin.loc[df_basin.VARIABLE == 'Outflow: Non Recoverable']
        
#        writer_in.writerow([date.strftime("%Y-%m-%d %H:%M:%S"),year,month,1,'%s' %(float(df_inf.VALUE) + (float(df_tran.VALUE)))])
        writer_in.writerow([date.strftime("%Y-%m-%d %H:%M:%S"),year,month,1,'%s' %(float(df_inf.VALUE))])
        writer_out.writerow([date.strftime("%Y-%m-%d %H:%M:%S"),year,month,1,'%s' %float(df_outf.VALUE)])
#        total_inflow += float(df_inf.VALUE)+ (float(df_tran.VALUE))
        total_inflow += float(df_inf.VALUE)
    csv_file_in.close()
    csv_file_out.close()
    
    if total_inflow == 0.0:
        output_fh_in = None
    
    return output_fh_in, output_fh_out
    
def get_transfers(sheet5_csv_folder):
    csv_files = glob.glob(sheet5_csv_folder+'\\sheet5_*.csv')
    transfer_data = []
    transfer_date = []
    for f in csv_files:
        fn = os.path.split(f)[1]
        year = int(fn.split('sheet5_')[1].split('.csv')[0][:4])
        month = int(fn.split('sheet5_')[1].split('.csv')[0][-2:])
        date = datetime.date(year,month,1)
        df = pd.read_csv(f,sep=';')
        df_basin = df.loc[df.SUBBASIN == 'basin'] 
        df_tran = df_basin.loc[df_basin.VARIABLE == 'Interbasin Transfer'].VALUE
        transfer_data.append(-float(df_tran))
        transfer_date.append(date)
    return transfer_data, transfer_date

def get_ts(all_results, key):
    """
    Subtract a array with values from a list containing multiple dictionaries.
    
    Parameters
    ----------
    all_results : list
        List of dictionaries.
    key : str
        Key to be extracted out of the dictionaries.
        
    Returns
    -------
    ts : ndarray
        Array with the extracted values.
    """
    ts = np.array([dct[key] for dct in all_results])
    return ts

def plot_parameter(all_results, dates, catchment_name, output_dir, parameter, extension = 'png', color = '#6bb8cc'):
    """
    Plot a graph of a specified parameter calculated by calc_sheet1.
    
    Parameters
    ----------
    all_results : list
        List of dictionaries.
    datas : ndarray
        Array with datetime.date object coresponding to the dictionaries in all_results.
    catchment_name : str
        Name of the catchment, used a title of the graph.
    output_dir : str
        Directory to save the graph.
    parameter : str
        Parameter to plot, should be a key in the dictionaries in all_results.
    extension : str, optional
        Choose to save the graph as a png or pdf file. Default is 'png'.
    color : str, optional
        Choose the fill color of the graph. Default is '#6bb8cc'.
    """
    ts = get_ts(all_results, parameter)
    fig = plt.figure(figsize = (10,10))
    plt.clf()
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
    ax = plt.subplot(111)
    ax.fill_between(dates, ts, color = color)
    ax.plot(dates, ts,  '-k')
    ax.scatter(dates, ts)
    ax.set_xlabel('Time')
    ax.set_ylabel('Flow [km3/month]')
    ax.set_title('{0}, {1}'.format(parameter, catchment_name))
    fig.autofmt_xdate()
    ax.set_ylim([np.nanmin(ts), np.nanmax(ts)*1.1])
    ax.set_xlim([np.min(dates), np.max(dates)])
    [i.set_zorder(10) for i in ax.spines.itervalues()]
    plt.savefig(os.path.join(output_dir, '{0}.{1}'.format(parameter, extension)))
   
def plot_storages(all_results, dates, catchment_name, output_dir, extension = 'png'):
    """
    Plot two graphs regarding the waterbalanc.
    
    Parameters
    ----------
    all_results : list
        List of dictionaries.
    dates : ndarray
        Array with datetime.date objects corresponding to all_results.
    catchment_name : str
        Name of the catchment under consideration.
    output_dir : str
        Folder to store results.
    extension : str, optional
        Choose to save the graphs as a png or a pdf file. Default is 'png'.
    """
    et_ts = get_ts(all_results, 'et_advection')
    q_ts = get_ts(all_results, 'q_outflow')
    p_ts = get_ts(all_results, 'p_advection')
    qi_ts = get_ts(all_results, 'q_in_sw')
    ds_ts = get_ts(all_results, 'dS')
    
    fig = plt.figure(figsize = (10,10))
    plt.clf()
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
    ax = plt.subplot(111)
    ordinal_dates = [date.toordinal() for date in dates]
    outflow = interpolate.interp1d(ordinal_dates, et_ts + q_ts)
    inflow = interpolate.interp1d(ordinal_dates, p_ts + qi_ts)
    x = np.arange(min(ordinal_dates), max(ordinal_dates), 1)
    outflow = outflow(x)
    inflow = inflow(x)
    dtes = [datetime.date.fromordinal(ordinal) for ordinal in x]
    ax.plot(dtes, inflow, label = 'Inflow (P + Qin)', color = 'k')
    ax.plot(dtes, outflow,  '--k', label = 'Outflow (ET + Qout)')
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
    plt.savefig(os.path.join(output_dir, 'water_balance_fullbasin.{0}'.format(extension)))
    
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
    ax.set_ylabel('Cumulative dS [km3/month]')
    ax.set_title('Cumulative dS, {0}'.format(catchment_name))
    ax.set_xlim([dtes[0], dtes[-1]])
    fig.autofmt_xdate()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.21),fancybox=True, shadow=True, ncol=5)
    [i.set_zorder(10) for i in ax.spines.itervalues()]
    plt.savefig(os.path.join(output_dir, 'water_storage_fullbasin.{0}'.format(extension)))



def calc_ETs(ET, lu_fh, sheet1_lucs):
    """
    Calculates the sums of the values within a specified landuse category.
    
    Parameters
    ----------
    ET : ndarray
        Array of the data for which the sum needs to be calculated.
    lu_fh : str
        Filehandle pointing to landusemap.
    sheet1_lucs : dict
        Dictionary with landuseclasses per category.
    
    Returns
    -------
    et : dict
        Dictionary with the totals per landuse category.
    """
    LULC = becgis.OpenAsArray(lu_fh, nan_values = True)
    et = dict()
    for key in sheet1_lucs:
        classes = sheet1_lucs[key]
        mask = np.logical_or.reduce([LULC == value for value in classes])
        et[key] = np.nansum(ET[mask])
    return et

def calc_utilizedflow(incremental_et, other, non_recoverable, other_fractions, non_recoverable_fractions):
    """
    Calculate the utilized flows per landuse category from total incremental ET, non_recoverable water and other.
    
    Parameters
    ----------
    incremental_et : dict
        Incremental ET per landuse category (i.e. Protected, Utilized, Modified and Managed).
    other : dict
        Other water consumptions per landuse category (i.e. Protected, Utilized, Modified and Managed).
    non_recoverable : dict
        Non recoverable water consumption per landuse category (i.e. Protected, Utilized, Modified and Managed).
    other_fractions : dict
        Fractions describing how much of other water consumption should be assigned to each category.
    non_recoverable_fractions : dict
        Fractions describing how much of non_recoverable water consumption should be assigned to each category.
        
    Returns
    -------
    uf_plu : float
        Utilized Flow for Protected LU.
    uf_ulu : float
        Utilized Flow for Utilized LU.
    uf_mlu : float
        Utilized Flow for Modified LU.
    uf_mwu : float
        Utilized Flow for Managed Water Use.
    """
    assert np.sum(other_fractions.values()) == 1.00, "Fractions for other should sum to 1.00."
    assert np.sum(non_recoverable_fractions.values()) == 1.00, "Fractions for non_recoverable should sum to 1.00."   
    
    np.array(incremental_et.values()) + np.array(other_fractions.values()) * other + np.array(non_recoverable_fractions.values()) * non_recoverable
    
    uf_plu = incremental_et['Protected'] + other_fractions['Protected'] * other + non_recoverable_fractions['Protected'] * non_recoverable
    uf_ulu = incremental_et['Utilized'] + other_fractions['Utilized'] * other + non_recoverable_fractions['Utilized'] * non_recoverable
    uf_mlu = incremental_et['Modified'] + other_fractions['Modified'] * other + non_recoverable_fractions['Modified'] * non_recoverable
    uf_mwu = incremental_et['Managed']  + other_fractions['Managed'] * other + non_recoverable_fractions['Managed'] * non_recoverable
    
    return uf_plu, uf_ulu, uf_mlu, uf_mwu

def calc_sheet1(entries, lu_fh, sheet1_lucs, recycling_ratio, q_outflow, q_out_avg, 
                output_folder, q_in_sw, q_in_gw = 0., q_in_desal = 0., q_out_sw = 0., q_out_gw = 0.):
    """
    Calculate the required values to plot Water Accounting Plus Sheet 1.
    
    Parameters
    ----------
    entries : dict
        Dictionary with several filehandles, also see examples below.
    lu_fh : str
        Filehandle pointing to the landuse map.
    sheet1_lucs : dict
        Dictionary sorting different landuse classes into categories.
    recycling_ratio : float
        Value indicating the recycling ratio.
    q_outflow : float
        The outflow of the basin.
    q_out_avg : float
        The longterm average outflow.
    output_folder : str
        Folder to store results.
    q_in_sw : float, optional
        Surfacewater inflow into the basin. Default is 0.0.
    q_in_gw : float, optional
        Groundwater inflow into the basin. Default is 0.0.
    q_in_desal : float, optional
        Desalinised water inflow into the basin. Default is 0.0.
    q_out_sw : float, optional
        Additional surfacewater outflow from basin. Default is 0.0.
    q_out_gw : float, optional
        Groundwater outflow from the basin. Default is 0.0.
        
    Returns
    -------
    results : dict
        Dictionary containing necessary variables for Sheet 1.
    """
    results = dict()
    
    LULC = becgis.OpenAsArray(lu_fh, nan_values = True)
    P = becgis.OpenAsArray(entries['P'], nan_values = True)
    ETgreen = becgis.OpenAsArray(entries['ETgreen'], nan_values = True)
    ETblue = becgis.OpenAsArray(entries['ETblue'], nan_values = True)
    
    pixel_area = becgis.MapPixelAreakm(lu_fh)

    gray_water_fraction = becgis.calc_basinmean(entries['WPL'], lu_fh)
    ewr_percentage = becgis.calc_basinmean(entries['EWR'], lu_fh)
    
    P[np.isnan(LULC)] = ETgreen[np.isnan(LULC)] = ETblue[np.isnan(LULC)] = np.nan
    P, ETgreen, ETblue = np.array([P, ETgreen, ETblue]) * 0.000001 * pixel_area
    
    ET = np.nansum([ETblue, ETgreen], axis = 0)
    
    results['et_advection'], results['p_advection'], results['p_recycled'], results['dS'] = calc_wb(P, ET, q_outflow, recycling_ratio, 
           q_in_sw = q_in_sw, q_in_gw = q_in_gw, q_in_desal = q_in_desal, q_out_sw = q_out_sw, q_out_gw = q_out_gw)

    results['non_recoverable'] = gray_water_fraction * (q_outflow + q_out_sw) # Mekonnen and Hoekstra (2015), Global Gray Water Footprint and Water Pollution Levels Related to Anthropogenic Nitrogen Loads to Fresh Water
    results['reserved_outflow_demand'] = q_out_avg * ewr_percentage
    
    results['other'] = 0.0
    
    landscape_et = calc_ETs(ETgreen, lu_fh, sheet1_lucs)
    incremental_et = calc_ETs(ETblue, lu_fh, sheet1_lucs)
    
    results['manmade'] = incremental_et['Managed']
    results['natural'] = incremental_et['Modified'] + incremental_et['Protected'] + incremental_et['Utilized']    
    
    other_fractions = {'Modified': 0.00,
                       'Managed':  1.00,
                       'Protected':0.00,
                       'Utilized': 0.00}    
                       
    non_recoverable_fractions = {'Modified': 0.00,
                                 'Managed':  1.00,
                                 'Protected':0.00,
                                 'Utilized': 0.00}  
                                 
    results['uf_plu'], results['uf_ulu'], results['uf_mlu'], results['uf_mwu'] = calc_utilizedflow(incremental_et, results['other'], results['non_recoverable'], other_fractions, non_recoverable_fractions)
   
    net_inflow = results['p_recycled'] + results['p_advection'] + q_in_sw + q_in_gw + q_in_desal + results['dS'] 
    consumed_water = np.nansum(landscape_et.values()) + np.nansum(incremental_et.values()) + results['other'] + results['non_recoverable']
    non_consumed_water = net_inflow - consumed_water
    
    results['non_utilizable_outflow'] = min(non_consumed_water, max(0.0, calc_non_utilizable(P, ET, entries['Fractions'])))
    results['reserved_outflow_actual'] = min(non_consumed_water - results['non_utilizable_outflow'], results['reserved_outflow_demand'])
    results['utilizable_outflow'] = max(0.0, non_consumed_water - results['non_utilizable_outflow'] - results['reserved_outflow_actual'])
    
    results['landscape_et_mwu'] = landscape_et['Managed']
    results['landscape_et_mlu'] = landscape_et['Modified']
    results['landscape_et_ulu'] = landscape_et['Utilized']
    results['landscape_et_plu'] = landscape_et['Protected']
    results['q_outflow'] = q_outflow
    results['q_in_sw'] = q_in_sw
    results['q_in_gw'] = q_in_gw
    results['q_in_desal'] = q_in_desal
    results['q_out_sw'] = q_out_sw
    results['q_out_gw'] = q_out_gw
    
    return results
    
def calc_wb(P, ET, q_outflow, recycling_ratio, q_in_sw = 0, q_in_gw = 0, q_in_desal = 0, q_out_sw = 0, q_out_gw = 0):
    """
    Calculate the water balance given the relevant terms.
    
    Parameters
    ----------
    P : ndarray
        Array with the precipitation in volume per pixel.
    ET : ndarray
        Array with the evapotranspiration in volume per pixel.
    q_outflow : float
        Outflow volume.
    recycling_ratio : float
        Value indicating the recycling ratio.
    q_in_sw : float, optional
        Surfacewater inflow into the basin. Default is 0.0.
    q_in_gw : float, optional
        Groundwater inflow into the basin. Default is 0.0.
    q_in_desal : float, optional
        Desalinised water inflow into the basin. Default is 0.0.
    q_out_sw : float, optional
        Additional surfacewater outflow from basin. Default is 0.0.
    q_out_gw : float, optional
        Groundwater outflow from the basin. Default is 0.0.
        
    Returns
    -------
    et_advection : float
        The volume of ET advected out of the basin.
    p_advection : float
        The volume of precipitation advected into the basin.
    p_recycled : float
        The volume of ET recycled as P within the basin.
    dS : float
        The change in storage.        
    """
    et_advection = (1 - recycling_ratio) * np.nansum(ET)
    p_total = np.nansum(P)
    p_recycled = min(recycling_ratio * np.nansum(ET), p_total)
    et_advection = np.nansum(ET) - p_recycled
    p_advection = p_total - p_recycled
    dS = q_outflow + et_advection + q_out_sw + q_out_gw - p_advection - q_in_sw - q_in_gw - q_in_desal
    return et_advection, p_advection, p_recycled, dS

def create_csv(results, output_fh):
    """
    Create the csv-file needed to plot sheet 1.
    
    Parameters
    ----------
    results : dict
        Dictionary generated by calc_sheet1.
    output_fh : str
        Filehandle to store the csv-file.
    """
    first_row = ['CLASS', 'SUBCLASS', 'VARIABLE', 'VALUE']
    
    if not os.path.exists(os.path.split(output_fh)[0]):
        os.makedirs(os.path.split(output_fh)[0])
    
    csv_file = open(output_fh, 'wb')
    writer = csv.writer(csv_file, delimiter=';')
    writer.writerow(first_row)

    writer.writerow(['INFLOW', 'PRECIPITATION', 'Rainfall', '{0}'.format(results['p_advection'])])
    writer.writerow(['INFLOW', 'PRECIPITATION', 'Snowfall', 0.])
    writer.writerow(['INFLOW', 'PRECIPITATION', 'Precipitation recycling', '{0}'.format(results['p_recycled'])])
    writer.writerow(['INFLOW', 'SURFACE WATER', 'Main riverstem', '{0}'.format(results['q_in_sw'])])
    writer.writerow(['INFLOW', 'SURFACE WATER', 'Tributaries', 0.])
    writer.writerow(['INFLOW', 'SURFACE WATER', 'Utilized surface water', 0.])
    writer.writerow(['INFLOW', 'SURFACE WATER', 'Flood', 0.])
    writer.writerow(['INFLOW', 'GROUNDWATER', 'Natural', '{0}'.format(results['q_in_gw'])])
    writer.writerow(['INFLOW', 'GROUNDWATER', 'Utilized', 0.])
    writer.writerow(['INFLOW', 'OTHER', 'Desalinized', '{0}'.format(results['q_in_desal'])])
    writer.writerow(['STORAGE', 'CHANGE', 'Surface storage', '{0}'.format(results['dS'])])
    writer.writerow(['STORAGE', 'CHANGE', 'Storage in sinks', 0.])
    writer.writerow(['OUTFLOW', 'ET LANDSCAPE', 'Protected', '{0}'.format(results['landscape_et_plu'])])
    writer.writerow(['OUTFLOW', 'ET LANDSCAPE', 'Utilized', '{0}'.format(results['landscape_et_ulu'])])
    writer.writerow(['OUTFLOW', 'ET LANDSCAPE', 'Modified', '{0}'.format(results['landscape_et_mlu'])])
    writer.writerow(['OUTFLOW', 'ET LANDSCAPE', 'Managed', '{0}'.format(results['landscape_et_mwu'])])
    writer.writerow(['OUTFLOW', 'ET UTILIZED FLOW', 'Protected', '{0}'.format(results['uf_plu'])])
    writer.writerow(['OUTFLOW', 'ET UTILIZED FLOW', 'Utilized', '{0}'.format(results['uf_ulu'])])
    writer.writerow(['OUTFLOW', 'ET UTILIZED FLOW', 'Modified', '{0}'.format(results['uf_mlu'])])
    writer.writerow(['OUTFLOW', 'ET UTILIZED FLOW', 'Managed', '{0}'.format(results['uf_mwu'])])        
    writer.writerow(['OUTFLOW', 'ET INCREMENTAL', 'Manmade', '{0}'.format(results['manmade'])])  
    writer.writerow(['OUTFLOW', 'ET INCREMENTAL', 'Natural', '{0}'.format(results['natural'])])
    writer.writerow(['OUTFLOW', 'SURFACE WATER', 'Main riverstem', '{0}'.format(results['q_outflow'])])
    writer.writerow(['OUTFLOW', 'SURFACE WATER', 'Tributaries',  0.])  
    writer.writerow(['OUTFLOW', 'SURFACE WATER', 'Utilized surface water', 0.])  
    writer.writerow(['OUTFLOW', 'SURFACE WATER', 'Flood', 0.])
    writer.writerow(['OUTFLOW', 'SURFACE WATER', 'Interbasin transfer', '{0}'.format(results['q_out_sw'])])
    writer.writerow(['OUTFLOW', 'GROUNDWATER', 'Natural', '{0}'.format(results['q_out_gw'])]) 
    writer.writerow(['OUTFLOW', 'GROUNDWATER', 'Utilized', 0.])
    writer.writerow(['OUTFLOW', 'OTHER', 'Non-utilizable', '{0}'.format(results['non_utilizable_outflow'])])
    writer.writerow(['OUTFLOW', 'OTHER', 'Other', '{0}'.format(results['other'])])
    writer.writerow(['OUTFLOW', 'RESERVED', 'Commited', '{0}'.format(results['reserved_outflow_actual'])]) 
    writer.writerow(['OUTFLOW', 'RESERVED', 'Navigational', 0.]) 
    writer.writerow(['OUTFLOW', 'RESERVED', 'Environmental', 0.]) 
    
    csv_file.close()

def calc_non_utilizable(P, ET, fractions_fh):
    """
    Calculate non utilizable outflow.
    
    Parameters
    ----------
    P : ndarray
        Array with the volumes of precipitation per pixel.
    ET : ndarray
        Array with the volumes of evapotranspiration per pixel.
    fractions_fh : str
        Filehandle pointing to a map with fractions indicating how much of the
        (P-ET) difference is non-utilizable.
    
    Returns
    -------
    non_utilizable_runoff : float
        The total volume of non_utilizable runoff.
    """
    fractions = becgis.OpenAsArray(fractions_fh, nan_values = True)
    non_utilizable_runoff = np.nansum((P - ET) * fractions)
    return non_utilizable_runoff

