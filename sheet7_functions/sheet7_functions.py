# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:45:59 2017

@author: cmi001
"""
from builtins import range
import os
import csv
import cairosvg
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import netCDF4 as nc
import tempfile as tf
from WA_Hyperloop.sheet3_functions import sheet3_functions as sh3
import watools.General.raster_conversions as RC
import watools.General.data_conversions as DC

import WA_Hyperloop.becgis as becgis
from WA_Hyperloop.paths import get_path
from WA_Hyperloop import hyperloop as hl

#%%

def create_sheet7(complete_data, metadata, output_dir, global_data, data):
    template_m = get_path('sheet7m_svg')
    template_y = get_path('sheet7y_svg')
    lu_fh = metadata['lu']
    output_folder = os.path.join(output_dir, metadata['name'], 'sheet7')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    recy_ratio = metadata['recycling_ratio']

    date_list2 = becgis.common_dates([complete_data['etb'][1],
                                     complete_data['tr'][1],
                                     complete_data['recharge'][1]])
    date_list = becgis.convert_datetime_date(date_list2, out='datetime')

    live_feed, feed_dict, abv_grnd_biomass_ratio, fuel_dict, sheet7_lulc_classes, c_fractions = get_sheet7_classes()
    # year:fish production
    #avg 2003-2012: 375375 2013:528000 2014:505005 <http://www.fao.org/3/a-i5555e.pdf>
    #http://www.fao.org/fishery/statistics/global-production/en
#    fish_production = {'2000':245600,
#                       '2001':385000,
#                       '2002':360300,
#                       '2003':308750,
#                       '2004':250000,
#                       '2005':324000,
#                       '2006':422000,
#                       '2007':395000,
#                       '2008':365000,
#                       '2009':390000}

    # Select and project population to LULC map
    pop_fh = global_data['population_tif']
#    pop_temp = os.path.join(output_folder, 'temp_pop')
#    pop_fh = becgis.match_proj_res_ndv(lu_fh, pop_fh, pop_temp)

    # Select and project cattle to LULC map
    cattle_fh = [global_data["cattle"]]
    cattle_temp = os.path.join(output_folder, 'temp_cattle')
    cattle_fh = becgis.match_proj_res_ndv(lu_fh, cattle_fh, cattle_temp)[0]

    ndm_fhs = []
    ro_fhs = []
    et_blue_fhs = []
    et_green_fhs = []
    p_fhs = []
    dry_bf_fhs = []
    gw_rchg_fhs = []
    for d in date_list2:
        ndm_fhs.extend(complete_data['ndm'][0][complete_data['ndm'][1] == d])
        ro_fhs.extend(complete_data['tr'][0][complete_data['tr'][1] == d])
        et_blue_fhs.extend(complete_data['etb'][0][complete_data['etb'][1] == d])
        et_green_fhs.extend(complete_data['etg'][0][complete_data['etg'][1] == d])
        p_fhs.extend(complete_data['p'][0][complete_data['p'][1] == d])
        dry_bf_fhs.extend(complete_data['bf'][0][complete_data['bf'][1] == d])
        gw_rchg_fhs.extend(complete_data['recharge'][0][complete_data['recharge'][1] == d])

    # Make fraction maps to split feed and fuel yields in landscape and incremental ET
    fraction_fhs = split_yield(output_folder, p_fhs, et_blue_fhs, et_green_fhs,
                               ab=(1.0, 1.0))

    # calculate feed production and return filehandles of saved tif files
    feed_fhs_landscape, feed_fhs_incremental = livestock_feed(output_folder, lu_fh,
                                                              ndm_fhs, feed_dict,
                                                              live_feed, cattle_fh,
                                                              fraction_fhs, date_list2)

    # calculate fuel production and return filehandles of saved tif files
    fuel_fhs_landscape, fuel_fhs_incremental = fuel_wood(output_folder, lu_fh,
                                                         ndm_fhs, fraction_fhs,
                                                         date_list2)

    # calculate root_storage and return filehandles of saved tif files
    rz_depth_fh = global_data['root_depth']
    rz_depth_tif = becgis.match_proj_res_ndv(lu_fh, np.array([rz_depth_fh]), tf.mkdtemp())[0]
    rz_sm_fhs = complete_data['rzsm'][0]

    root_storage_fhs = root_zone_storage_Wpx(output_folder, rz_sm_fhs, rz_depth_tif)

    atm_recy_landscape_fhs = recycle(output_folder, et_green_fhs, recy_ratio,
                                     lu_fh, 'landscape')
    atm_recy_incremental_fhs = recycle(output_folder, et_blue_fhs, recy_ratio,
                                       lu_fh, 'incremental')

    class Vividict(dict):
        def __missing__(self, key):
            value = self[key] = type(self)()
            return value

    results = Vividict()
    for d in date_list:
        datestr1 = "%04d_%02d" %(d.year, d.month)
        datestr2 = "%04d%02d" %(d.year, d.month)
        ystr = "%04d" %(d.year)
        mstr = "%02d" %(d.month)

        ro_fh = ro_fhs[np.where([datestr2 in ro_fhs[i] for i in range(len(ro_fhs))])[0][0]]
        feed_fh_landscape = feed_fhs_landscape[np.where([datestr1 in feed_fhs_landscape[i] for i in range(len(feed_fhs_landscape))])[0][0]]
        feed_fh_incremental = feed_fhs_incremental[np.where([datestr1 in feed_fhs_incremental[i] for i in range(len(feed_fhs_incremental))])[0][0]]
        fuel_fh_landscape = fuel_fhs_landscape[np.where([datestr1 in fuel_fhs_landscape[i] for i in range(len(fuel_fhs_landscape))])[0][0]]
        fuel_fh_incremental = fuel_fhs_incremental[np.where([datestr1 in fuel_fhs_incremental[i] for i in range(len(fuel_fhs_incremental))])[0][0]]

        baseflow_fh = dry_bf_fhs[np.where([datestr2 in dry_bf_fhs[i] for i in range(len(dry_bf_fhs))])[0][0]]
        gw_recharge_fh = gw_rchg_fhs[np.where([datestr2 in gw_rchg_fhs[i] for i in range(len(gw_rchg_fhs))])[0][0]]

        root_storage_fh = root_storage_fhs[np.where([datestr2 in root_storage_fhs[i] for i in range(len(root_storage_fhs))])[0][0]]
        atm_recy_landscape_fh = atm_recy_landscape_fhs[np.where([datestr2 in atm_recy_landscape_fhs[i] for i in range(len(atm_recy_landscape_fhs))])[0][0]]
        atm_recy_incremental_fh = atm_recy_incremental_fhs[np.where([datestr2 in atm_recy_incremental_fhs[i] for i in range(len(atm_recy_incremental_fhs))])[0][0]]

        results[ystr][mstr]['tot_runoff'] = lu_type_sum(ro_fh, lu_fh, sheet7_lulc_classes, convert='mm_to_km3')
      #  results['fish'] =
        results[ystr][mstr]['feed_incremental'] = lu_type_sum(feed_fh_incremental, lu_fh, sheet7_lulc_classes)
        results[ystr][mstr]['feed_landscape'] = lu_type_sum(feed_fh_landscape, lu_fh, sheet7_lulc_classes)
        results[ystr][mstr]['fuel_incremental'] = lu_type_sum(fuel_fh_incremental, lu_fh, sheet7_lulc_classes)
        results[ystr][mstr]['fuel_landscape'] = lu_type_sum(fuel_fh_landscape, lu_fh, sheet7_lulc_classes)

        results[ystr][mstr]['baseflow'] = lu_type_sum(baseflow_fh, lu_fh, sheet7_lulc_classes, convert='mm_to_km3')
        results[ystr][mstr]['gw_rech'] = lu_type_sum(gw_recharge_fh, lu_fh, sheet7_lulc_classes, convert='mm_to_km3')
        results[ystr][mstr]['root_storage'] = lu_type_sum(root_storage_fh, lu_fh, sheet7_lulc_classes, convert='mm_to_km3')
        results[ystr][mstr]['atm_recycl_landscape'] = lu_type_sum(atm_recy_landscape_fh, lu_fh, sheet7_lulc_classes, convert='mm_to_km3')
        results[ystr][mstr]['atm_recycl_incremental'] = lu_type_sum(atm_recy_incremental_fh, lu_fh, sheet7_lulc_classes, convert='mm_to_km3')

        output_fh = output_folder +"\\sheet7_monthly\\sheet7_"+datestr1+".csv"
        create_csv(results[ystr][mstr], output_fh)
        output = output_folder + '\\sheet7_monthly\\sheet7_'+datestr1+'.pdf'
        create_sheet7_svg(metadata['name'], datestr1, output_fh, output, 
                          template=template_m)

    fhs = hl.create_csv_yearly(os.path.join(output_folder, "sheet7_monthly"),
                               os.path.join(output_folder, "sheet7_yearly"), 7,
                               metadata['water_year_start_month'],
                               year_position=[-11, -7], month_position=[-6, -4],
                               header_rows=1, header_columns=3,
                               minus_header_colums=-1)
    for csv_fh in fhs:
        year = csv_fh[-8:-4] 
        create_sheet7_svg(metadata['name'], year, 
                          csv_fh, csv_fh.replace('.csv','.pdf'), template=template_y)


## PROVISIONING SERVICES
def livestock_feed(output_folder, lu_fh, ndm_fhs, feed_dict, live_feed, cattle_fh, fraction_fhs, ndmdates):
    """
    Calculate natural livestock feed production

    INPUTS
    ----------
    lu_fh : str
        filehandle for land use map
    ndm_fhs: nd array
        array of filehandles of NDM maps
    ndm_dates: nd array
        array of dates for NDM maps
    feed_dict: dict
        dictionnary 'pasture class':[list of LULC]
    feed_pct: dict
        dictionnary 'pasture class':[percent available as feed]
    cattle_fh : str
        filehandle for cattle map
    """
    Data_Path_Feed = "Feed"
    out_folder = os.path.join(output_folder, Data_Path_Feed)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    area_ha = becgis.map_pixel_area_km(lu_fh) * 100
    LULC = RC.Open_tiff_array(lu_fh)
  #  cattle = RC.Open_tiff_array(cattle_fh)
    geo_out, proj, size_X, size_Y = RC.Open_array_info(lu_fh)

    f_pct = np.zeros(LULC.shape)
    for lu_type in list(feed_dict.keys()):
        classes = feed_dict[lu_type]
        mask = np.logical_or.reduce([LULC == value for value in classes])
        f_pct[mask] = live_feed[lu_type]
    feed_fhs_landscape = []
    feed_fhs_incremental = []
    for d in range(len(ndm_fhs)):
        ndm_fh = ndm_fhs[d]
        fraction_fh = fraction_fhs[d]
        date1 = ndmdates[d]
        year = '%d' %date1.year
        month = '%02d' %date1.month

        yield_fract = RC.Open_tiff_array(fraction_fh)

        out_fh_l = out_folder+'\\feed_prod_landscape_%s_%s.tif' %(year, month)
        out_fh_i = out_folder+'\\feed_prod_incremental_%s_%s.tif' %(year, month)
#        out_fh2 = out_folder+'\\Feed_prod_pH_%s_%s.tif' %(year, month)
        NDM = becgis.open_as_array(ndm_fh, nan_values=True)
        NDM_feed = NDM * f_pct
        NDM_feed_incremental = NDM_feed * yield_fract * area_ha/1e6
        NDM_feed_landscape = (NDM_feed *(1-yield_fract)) * area_ha/1e6
        DC.Save_as_tiff(out_fh_l, NDM_feed_landscape, geo_out)
        DC.Save_as_tiff(out_fh_i, NDM_feed_incremental, geo_out)
#        NDM_feed_perHead = NDM_feed / cattle
#        DC.Save_as_tiff(out_fh2, NDM_feed, geo_out)
        feed_fhs_landscape.append(out_fh_l)
        feed_fhs_incremental.append(out_fh_i)
    return feed_fhs_landscape, feed_fhs_incremental

def fuel_wood(output_folder, lu_fh, ndm_fhs, fraction_fhs, ndmdates):
    """
    Calculate natural livestock feed production

    INPUTS
    ----------
    lu_fh : str
        filehandle for land use map
    ndm_fhs: nd array
        array of filehandles of NDM maps
    abv_grnd_biomass_ratio: dict
        dictionnary 'LULC':[above ground biomass]
    """
    Data_Path_Fuel = "Fuel"
    out_folder = os.path.join(output_folder, Data_Path_Fuel)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    area_ha = becgis.map_pixel_area_km(lu_fh) * 100
    LULC = RC.Open_tiff_array(lu_fh)
    geo_out, proj, size_X, size_Y = RC.Open_array_info(lu_fh)

    fuel_classes = [1, 8, 9, 10, 11, 12, 13]
    fuel_mask = np.zeros(LULC.shape)
    for fc in fuel_classes:
        fuel_mask[np.where(LULC == fc)] = 1

    fuel_fhs_landscape = []
    fuel_fhs_incremental = []

    for d in range(len(ndm_fhs)):
        ndm_fh = ndm_fhs[d]
        fraction_fh = fraction_fhs[d]
        yield_fract = RC.Open_tiff_array(fraction_fh)
        date1 = ndmdates[d]
        year = '%d' %date1.year
        month = '%02d' %date1.month
#        year = ndm_fh[-14:-10]
#        month = ndm_fh[-9:-7]
        out_fh_l = out_folder+'\\fuel_prod_landscape_%s_%s.tif' %(year, month)
        out_fh_i = out_folder+'\\fuel_prod_incremental_%s_%s.tif' %(year, month)
        NDM = becgis.open_as_array(ndm_fh, nan_values=True)

        NDM_fuel_incremental = NDM * .05 * fuel_mask * yield_fract * area_ha/1e6
        NDM_fuel_landscape = NDM  * .05 * fuel_mask *(1-yield_fract) * area_ha/1e6
        DC.Save_as_tiff(out_fh_i, NDM_fuel_incremental, geo_out)
        DC.Save_as_tiff(out_fh_l, NDM_fuel_landscape, geo_out)
        fuel_fhs_landscape.append(out_fh_l)
        fuel_fhs_incremental.append(out_fh_i)

    return fuel_fhs_landscape, fuel_fhs_incremental

## REGULATING SERVICES

def dry_season_bf(output_folder, WPixOutFile, dry_months):
    data_path = "dry_bf"
    out_folder = os.path.join(output_folder, data_path)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    WPOut = nc.Dataset(WPixOutFile)
    lat = WPOut.variables['latitude']
    lon = WPOut.variables['longitude']
    px_size_lat = np.mean([lat[i+1]-lat[i] for i in range(len(lat)-1)])
    px_size_lon = np.mean([lon[i+1]-lon[i] for i in range(len(lon)-1)])
    geo_out = (np.min(lon)-px_size_lon/2, px_size_lon,
               0, np.max(lat)-px_size_lat/2, 0, px_size_lat)

    times = WPOut.variables['time_yyyymm'][:]
    years = WPOut.variables['time_yyyy'][:]
    months = np.array(times) % 100 #might need this later if not a complete year etc.
    dry_months2 = np.where([months[i] in dry_months for i in range(len(months))])[0]
    bf = WPOut.variables['Baseflow_M'][:].data
    dry_bf = bf[:, :, dry_months2]
    dry_bf = np.nanmean(dry_bf, axis=2)
    dry_bf_fh = out_folder + '\\dry_bf_%d.tif' %years
    DC.Save_as_tiff(dry_bf_fh, dry_bf, geo_out)
    return  dry_bf_fh

def root_zone_storage_Wpx(output_folder, rz_sm_fhs, rz_depth_fh):
    Data_Path_RZ = "RZstor"
    out_folder = os.path.join(output_folder, Data_Path_RZ)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    root_depth = becgis.open_as_array(rz_depth_fh, nan_values=True)
    geo = becgis.get_geoinfo(rz_depth_fh)
    root_storage_fhs = []
    for rz_sm_fh in rz_sm_fhs:
        root_depth_sm = becgis.open_as_array(rz_sm_fh, nan_values=True)
        root_storage = root_depth * root_depth_sm
        out_fh = os.path.join(out_folder, 'RZ_storage_mm_%s' %(rz_sm_fh[-10:]))
        becgis.create_geotiff(out_fh, root_storage, *geo)
        root_storage_fhs.append(out_fh)
    return root_storage_fhs

#def carbon_seq(output_folder,lu_fh,ndm_fhs,abv_grnd_biomass_ratio,c_fraction,abv_grnd_biomass_ratio):
#    Data_Path_Carbon = "carbon"
#    out_folder = os.path.join(output_folder, Data_Path_Carbon)
#    if not os.path.exists(out_folder):
#        os.mkdir(out_folder)
#
#    area_ha =  Area_converter.Degrees_to_m2(lu_fh)/10000
#    LULC = RC.Open_tiff_array(lu_fh)
#    geo_out, proj, size_X, size_Y = RC.Open_array_info(lu_fh)
#
#    abv_pct = np.zeros(LULC.shape)
#    for fuel_cat in abv_grnd_biomass_ratio.keys():
#        lu_class = int(fuel_cat)
#        mask = [LULC == lu_class]
#        abv_pct[mask] =  1/(1+abv_grnd_biomass_ratio[fuel_cat])
#    return carbon_fhs


def recycle(output_folder, et_bg_fhs, recy_ratio, lu_fh, et_type):
    Data_Path_rec = "temp_et_recycle"
    out_folder = os.path.join(output_folder, Data_Path_rec)
    geo_out, proj, size_X, size_Y = RC.Open_array_info(lu_fh)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    recycle_fhs = []
    for et_fh in et_bg_fhs:
        out_fh = out_folder + "\\recycled_et_"+et_type+et_fh[-11:-4]+".tif"
        et = becgis.open_as_array(et_fh, nan_values=True)
        et_recy = et*recy_ratio
        DC.Save_as_tiff(out_fh, et_recy, geo_out)
        recycle_fhs.append(out_fh)
    return recycle_fhs

### Other functions
def lu_type_average(data_fh, lu_fh, lu_dict):
    LULC = RC.Open_tiff_array(lu_fh)
    in_data = RC.Open_tiff_array(data_fh)
    out_data = {}
    for lu_class in list(lu_dict.keys()):
        mask = [LULC == value for value in lu_dict[lu_class]]
        mask = (np.sum(mask, axis=0)).astype(bool)
        out_data[lu_class] = np.nanmean(in_data[mask])
    return out_data

def lu_type_sum(data_fh, lu_fh, lu_dict, convert=None):
    LULC = RC.Open_tiff_array(lu_fh)
    in_data = becgis.open_as_array(data_fh, nan_values=True)
#    in_data = RC.Open_tiff_array(data_fh)
    if convert == 'mm_to_km3':
        AREA = becgis.map_pixel_area_km(data_fh)
        in_data *= AREA / 1e6
    out_data = {}
    for lu_class in list(lu_dict.keys()):
        mask = [LULC == value for value in lu_dict[lu_class]]
        mask = (np.sum(mask, axis=0)).astype(bool)
        out_data[lu_class] = np.nansum(in_data[mask])
    return out_data

def split_yield(output_folder, p_fhs, et_blue_fhs, et_green_fhs, ab=(1.0, 1.0)):
    Data_Path_split = "split_y"
    out_folder = os.path.join(output_folder, Data_Path_split)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    sp_yield_fhs = []
    geo_out, proj, size_X, size_Y = RC.Open_array_info(p_fhs[0])
    for m in range(len(p_fhs)):
        out_fh = out_folder+'\\split_yield'+et_blue_fhs[m][-12:]
        P = RC.Open_tiff_array(p_fhs[m])
        ETBLUE = RC.Open_tiff_array(et_blue_fhs[m])
        ETGREEN = RC.Open_tiff_array(et_green_fhs[m])
        etbfraction = ETBLUE / (ETBLUE + ETGREEN)
        pfraction = P / np.nanmax(P)
        fraction = sh3.split_Yield(pfraction, etbfraction, ab[0], ab[1])
        DC.Save_as_tiff(out_fh, fraction, geo_out)
        sp_yield_fhs.append(out_fh)
    return sp_yield_fhs

def get_sheet7_classes():
    live_feed = {'Pasture':.5,
                 'Crop':.25}
    # LULC class to Pasture or Crop
    feed_dict = {'Pasture':[2, 3, 12, 13, 14, 15, 16, 17, 20, 29, 34],
                 'Crop':[35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 54, 55, 56, 57, 58, 59, 60, 61, 62]}
    # above ground biomass for tree and bush classes
    abv_grnd_biomass_ratio = {'1':.26, # protected forest
                              '8':.24, # closed deciduous forest
                              '9':.43, # open deciduous forest
                              '10':.23, # closed evergreen forest
                              '11':.46, # open evergreen forest
                              '12':.48, # closed savanna
                              '13':.48} # open savanna
    # C content as fraction dry matter for Carbon sequestration calculations
    c_fraction = {'default':.47}

    # 5% of above ground biomass for now
    fuel_dict = {'all':.05}
    # dict: class - lulc
    sheet7_lulc = {
        'PROTECTED':    {'Forest': [1],
                         'Shrubland': [2],
                         'Natural grasslands':[3],
                         'Natural water bodies':[4],
                         'Wetlands':[5],
                         'Glaciers':[6],
                         'Others':[7]
                        },
        'UTILIZED':     {'Forest':[8, 9, 10, 11],
                         'Shrubland':[14],
                         'Natural grasslands':[12, 13, 15, 16, 20],
                         'Natural water bodies':[23, 24],
                         'Wetlands':[17, 19, 25, 30, 31],
                         'Others':[18, 21, 22, 26, 27, 28, 29, 32]
                        },
        'MODIFIED':     {'Rainfed crops': [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
                         'Forest plantations':[33],
                         'Settlements':[47, 48, 49, 50, 51],
                         'Others':[45, 46]
                        },
        'MANAGED': {'Irrigated crops':[52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
                    'Managed water bodies':[63, 65, 74],
                    'Residential':[68, 69, 71, 72],
                    'Industry':[67, 70, 76],
                    'Others':[75, 78, 77],
                    'Indoor domestic':[66],
                    'Indoor industry':[0],
                    'Greenhouses':[64],
                    'Livestock and husbandry':[73],
                    'Power and energy':[79, 80],
                   }
        }

    sheet7_lulc_classes = dict()
    for k in list(sheet7_lulc.keys()):
        l = []
        for k2 in list(sheet7_lulc[k].keys()):
            l.append(sheet7_lulc[k][k2])
        sheet7_lulc_classes[k] = [item for sublist in l for item in sublist]
    return live_feed, feed_dict, abv_grnd_biomass_ratio, fuel_dict, sheet7_lulc_classes, c_fraction


def create_csv(results, output_fh):
    """
    Create the csv-file needed to plot sheet 7.

    Parameters
    ----------
    results : dict
        Dictionary generated by calc_sheet7.
    output_fh : str
        Filehandle to store the csv-file.
    """
    first_row = ['LAND_USE', 'VARIABLE', 'SERVICE', 'VALUE', 'UNITS']
    if not os.path.exists(os.path.split(output_fh)[0]):
        os.makedirs(os.path.split(output_fh)[0])

    csv_file = open(output_fh, 'w')
    writer = csv.writer(csv_file, delimiter=';', lineterminator = '\n')
    
    writer.writerow(first_row)
    lu_classes = ['PROTECTED', 'UTILIZED', 'MODIFIED', 'MANAGED']
    for lu_class in lu_classes:
        writer.writerow([lu_class, 'Total Runoff', 'Non-consumptive',
                         '{0:.3f}'.format(results['tot_runoff'][lu_class]), 'km3'])
        writer.writerow([lu_class, 'Groundwater Recharge', 'Non-consumptive',
                         '{0:.3f}'.format(results['gw_rech'][lu_class]), 'km3'])
    #    writer.writerow(['PROTECTED','Natural water storage in lakes', 'Non-consumptive',  '{0:.2f}'.format(results['nat_stor']),'km3'])
    #    writer.writerow(['PROTECTED','Inland Capture Fishery', 'Non-consumptive', '{0:.2f}'.format(results['fish']),'t'])
        writer.writerow([lu_class, 'Natural Feed Production', 'Incremental ET natural',
                         '{0:.3f}'.format(results['feed_incremental'][lu_class]), 't'])
        writer.writerow([lu_class, 'Natural Feed Production', 'Landscape ET',
                         '{0:.3f}'.format(results['feed_landscape'][lu_class]), 't'])
        writer.writerow([lu_class, 'Natural Fuel Wood Production', 'Incremental ET natural',
                         '{0:.3f}'.format(results['fuel_incremental'][lu_class]), 't'])
        writer.writerow([lu_class, 'Natural Fuel Wood Production', 'Landscape ET',
                         '{0:.3f}'.format(results['fuel_landscape'][lu_class]), 't'])
        writer.writerow([lu_class, 'Dry Season Baseflow', 'Non-consumptive',
                         '{0:.3f}'.format(results['baseflow'][lu_class]), 'km3'])
    #    writer.writerow(['PROTECTED','Groundwater Recharge', 'SURFACE WATER', 'Flood', 0.])
        writer.writerow([lu_class, 'Root Zone Water Storage', 'Non-consumptive',
                         '{0:.3f}'.format(results['root_storage'][lu_class]), 'km3'])
        writer.writerow([lu_class, 'Atmospheric Water Recycling', 'Incremental ET natural',
                         '{0:.3f}'.format(results['atm_recycl_incremental'][lu_class]),'km3'])
        writer.writerow([lu_class, 'Atmospheric Water Recycling', 'Landscape ET',
                         '{0:.3f}'.format(results['atm_recycl_landscape'][lu_class]),'km3'])
    csv_file.close()

def create_sheet7_svg(basin, period, data, output, template=False):

    df = pd.read_csv(data, sep=';')
    if not template:
        svg_template_path = get_path('sheet7m_svg')
    else:
        svg_template_path = os.path.abspath(template)

    tree = ET.parse(svg_template_path)

    xml_txt_box = tree.findall('''.//*[@id='basin']''')[0]
    list(xml_txt_box)[0].text = 'Basin: ' + basin

    xml_txt_box = tree.findall('''.//*[@id='period']''')[0]
    list(xml_txt_box)[0].text = 'Period: ' + period

    # Assuming all variables in .csv are found in .svg. If not, need to adjust the sets
    #variables = set(df.VARIABLE)
    #land_use = set(df.LAND_USE)
    #services = set(df.SERVICE)

    # Dictionaries to match .csv file to .svg cell names
    dict1 = {'Total Runoff':'tot_runoff',
             'Natural water storage in lakes':'nat_stor',
             'Inland Capture Fishery':'fish',
             'Natural Feed Production':'feed',
             'Natural Fuel Wood Production':'fuel',
             'Dry Season Baseflow':'baseflow',
             'Groundwater Recharge':'GW_rech',
             'Root Zone Water Storage':'RZ',
             'Atmospheric Water Recycling':'Recy'
            }

    dict2 = {'MANAGED':'MWU', 'MODIFIED':'MLU', 'PROTECTED':'PLU', 'UTILIZED':'ULU'}
  #  dict3 = {'-':'0', 'Incremental ET natural':'3', 'Landscape ET':'2', 'Non-consumptive':'1'}
    dict3 = {'Incremental ET natural':'2', 'Landscape ET':'3', 'Non-consumptive':'1'}

    variables = list(dict1.keys())
    land_use = list(dict2.keys())
    services = list(dict3.keys())

    for v in variables:
        df_1 = df.loc[df.VARIABLE == v]
        if df_1.shape[0] > 0:
            var = dict1[v]
            cell_id = var+'_basin'
            basin_wide = np.sum(df_1.VALUE)
            xml_txt_box = tree.findall('''.//*[@id='{0}']'''.format(cell_id))[0]
            list(xml_txt_box)[0].text = '%.1f' % basin_wide
            for l in land_use:
                df_2 = df_1.loc[df_1.LAND_USE == l]
                if df_2.shape[0] > 0:
                    for s in services:
                        df_3 = df_2.loc[df_2.SERVICE == s]
                        if df_3.shape[0] > 0:
                            var = dict1[v]
                            lu = dict2[l]
                            serv = dict3[s]
                            cell_id = var+'_'+lu+'_'+serv
                            value = float(df_3.VALUE)
                            xml_txt_box = tree.findall('''.//*[@id='{0}']'''.format(cell_id))[0]
                            list(xml_txt_box)[0].text = '%.1f' % value

    # Export svg to png    
    tempout_path = output.replace('.pdf', '_temporary.svg')
    tree.write(tempout_path)    
    cairosvg.svg2pdf(url=tempout_path, write_to=output)    
    os.remove(tempout_path) 

    return
