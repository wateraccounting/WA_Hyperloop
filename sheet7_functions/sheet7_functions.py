# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:45:59 2017

@author: cmi001
"""

import os
import numpy as np
import netCDF4 as nc
import csv

import wa.General.raster_conversions as RC
import wa.General.data_conversions as DC

from WA_Hyperloop.sheet3_functions import sheet3_functions as sh3

import WA_Hyperloop.becgis as becgis
#%%
## PROVISIONING SERVICES
def mm_to_km3(lu_fh,var_fhs):
    area =  becgis.MapPixelAreakm(lu_fh)
    geo_out, proj, size_X, size_Y = RC.Open_array_info(lu_fh)
    var_new_fhs = list()
    for var_fh in var_fhs:
        var = RC.Open_tiff_array(var_fh)
        var[np.where(var==-9999)]=np.nan
        var_area = (var*area)/1000000
        var_new_fh = var_fh.replace('.tif', '_km3.tif')
        DC.Save_as_tiff(var_new_fh, var_area, geo_out)
        var_new_fhs.append(var_new_fh)
    return var_new_fhs

def livestock_feed(output_folder, lu_fh,ndm_fhs,feed_dict,live_feed,cattle_fh,fraction_fhs):
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
#    area_ha =  Area_converter.Degrees_to_m2(lu_fh)/10000
    area_ha = becgis.MapPixelAreakm(lu_fh) * 100
    LULC = RC.Open_tiff_array(lu_fh)
  #  cattle = RC.Open_tiff_array(cattle_fh)
    geo_out, proj, size_X, size_Y = RC.Open_array_info(lu_fh)
    
    f_pct = np.zeros(LULC.shape)
    for lu_type in feed_dict.keys():
        classes = feed_dict[lu_type]
        mask = np.logical_or.reduce([LULC == value for value in classes])
        f_pct[mask] = live_feed[lu_type]
    feed_fhs_landscape = []
    feed_fhs_incremental = []
    for d in range(len(ndm_fhs)):
        ndm_fh = ndm_fhs[d]
        fraction_fh = fraction_fhs[d]
        year = ndm_fh[-12:-8]
        month = ndm_fh[-6:-4]        
        yield_fract =  RC.Open_tiff_array(fraction_fh)

        out_fh_l = out_folder+'\\feed_prod_landscape_%s_%s.tif' %(year, month)
        out_fh_i = out_folder+'\\feed_prod_incremental_%s_%s.tif' %(year, month)
#        out_fh2 = out_folder+'\\Feed_prod_pH_%s_%s.tif' %(year, month)
        NDM = RC.Open_tiff_array(ndm_fh)
        NDM[np.where(NDM==-9999)]=np.nan
        NDM_feed = NDM * f_pct
        NDM_feed_incremental = NDM_feed  * yield_fract * area_ha/1000000
        NDM_feed_landscape =( NDM_feed *(1-yield_fract )) *area_ha/1000000
        DC.Save_as_tiff(out_fh_l, NDM_feed_landscape, geo_out)
        DC.Save_as_tiff(out_fh_i, NDM_feed_incremental, geo_out)
#        NDM_feed_perHead = NDM_feed / cattle
#        DC.Save_as_tiff(out_fh2, NDM_feed, geo_out)
        feed_fhs_landscape.append(out_fh_l)
        feed_fhs_incremental.append(out_fh_i)
    return feed_fhs_landscape, feed_fhs_incremental

def fuel_wood(output_folder,lu_fh, ndm_fhs,fraction_fhs):
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
        
    area_ha = becgis.MapPixelAreakm(lu_fh) * 100
    LULC = RC.Open_tiff_array(lu_fh)
    geo_out, proj, size_X, size_Y = RC.Open_array_info(lu_fh)
    
    fuel_classes = [1,8,9,10,11,12,13]
    fuel_mask = np.zeros(LULC.shape)
    for fc in fuel_classes:
        fuel_mask[np.where(LULC ==fc)] = 1
    
    fuel_fhs_landscape = []
    fuel_fhs_incremental = []    

    for d in range(len(ndm_fhs)):
        ndm_fh = ndm_fhs[d]
        fraction_fh = fraction_fhs[d]
        yield_fract =  RC.Open_tiff_array(fraction_fh)
        year = ndm_fh[-12:-8]
        month = ndm_fh[-6:-4]
        out_fh_l = out_folder+'\\fuel_prod_landscape_%s_%s.tif' %(year, month)
        out_fh_i = out_folder+'\\fuel_prod_incremental_%s_%s.tif' %(year, month)          
        NDM = RC.Open_tiff_array(ndm_fh)
        NDM[np.where(NDM==-9999)]=np.nan

        NDM_fuel_incremental = NDM * .05 * fuel_mask * yield_fract * area_ha/1000/1000
        NDM_fuel_landscape = NDM  * .05 * fuel_mask *(1-yield_fract) * area_ha/1000/1000
        DC.Save_as_tiff(out_fh_i, NDM_fuel_incremental, geo_out)
        DC.Save_as_tiff(out_fh_l, NDM_fuel_landscape, geo_out)
        fuel_fhs_landscape.append(out_fh_l)
        fuel_fhs_incremental.append(out_fh_i)
    
    return fuel_fhs_landscape, fuel_fhs_incremental

## REGULATING SERVICES

def dry_season_bf(output_folder,WPixOutFile,dry_months):
    data_path = "dry_bf"
    out_folder = os.path.join(output_folder, data_path)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
   
    WPOut = nc.Dataset(WPixOutFile)
    lat = WPOut.variables['latitude']
    lon = WPOut.variables['longitude']
    px_size_lat = np.mean([lat[i+1]-lat[i] for i in range(len(lat)-1)])
    px_size_lon = np.mean([lon[i+1]-lon[i] for i in range(len(lon)-1)])
    geo_out = (np.min(lon)-px_size_lon/2, px_size_lon,0,np.max(lat)-px_size_lat/2,0,px_size_lat)

    times = WPOut.variables['time_yyyymm'][:]
    years = WPOut.variables['time_yyyy'][:]
    months = np.array(times) % 100 #might need this later if not a complete year etc.
    dry_months2 = np.where([months[i] in dry_months for i in range(len(months))])[0]
    bf = WPOut.variables['Baseflow_M'][:].data
    dry_bf = bf[:,:,dry_months2]
    dry_bf = np.nanmean(dry_bf,axis=2)    
    dry_bf_fh = out_folder + '\\dry_bf_%d.tif' %years
    DC.Save_as_tiff(dry_bf_fh, dry_bf, geo_out)
    return  dry_bf_fh

def gw_rchg(output_folder,WPixOutFile): #,moving_avg_len = 3):
    data_path = "gw_rchg"
    out_folder = os.path.join(output_folder, data_path)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)   
        
    WPOut = nc.Dataset(WPixOutFile)
    lat = WPOut.variables['latitude']
    lon = WPOut.variables['longitude']
    px_size_lat = np.mean([lat[i+1]-lat[i] for i in range(len(lat)-1)])
    px_size_lon = np.mean([lon[i+1]-lon[i] for i in range(len(lon)-1)])
    geo_out = (np.min(lon)-px_size_lon/2, px_size_lon,0,np.max(lat)-px_size_lat/2,0,px_size_lat)    
    
    times = WPOut.variables['time_yyyymm'][:]

    perc = WPOut.variables['Percolation_M'][:].data
    padded_perc = np.append(perc[:,:,10:13],perc,axis=2) # add the last 2 months of the 1st year to the start for the trailing average
    gw_rchg_fhs = []
    for t in range(len(times)):
        year = int(times[t]/100)
        month = times[t] % 100
        gw_rch_fh = out_folder+'\\gw_rch_mm_%4d_%02d.tif' %(year, month) 
        gw_rch = np.mean(padded_perc[:,:,t:t+3],axis=2)    
        DC.Save_as_tiff(gw_rch_fh, gw_rch, geo_out)
        gw_rchg_fhs.append(gw_rch_fh)
    return gw_rchg_fhs

def root_zone_storage_Wpx(output_folder,WPixOutFile,WPixInFile):
    Data_Path_RZ = "RZstor"
    out_folder = os.path.join(output_folder, Data_Path_RZ)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    WPOut = nc.Dataset(WPixOutFile)
    WPIn = nc.Dataset(WPixInFile)
    
    time = WPOut.variables['time_yyyymm']
    
    root_depth_sm = WPOut.variables['RootDepthSoilMoisture_M'][:].data
    root_depth = WPIn.variables['RootDepth'][:]
   
    lat = WPOut.variables['latitude']
    lon = WPOut.variables['longitude']
    
    px_size_lat = np.mean([lat[i+1]-lat[i] for i in range(len(lat)-1)])
    px_size_lon = np.mean([lon[i+1]-lon[i] for i in range(len(lon)-1)])
    geo_out = (np.min(lon)-px_size_lon/2, px_size_lon,0,np.max(lat)-px_size_lat/2,0,px_size_lat)
    
    root_storage_fhs = []
    for t in range(len(time)):
        y = str(time[t])[:4]
        m = str(time[t])[-2:]        
        out_fh = out_folder+'\\RZ_storage_mm_%s_%s.tif' %(y, m) 
        root_storage =  root_depth * root_depth_sm[:,:,t]
        DC.Save_as_tiff(out_fh, root_storage.data, geo_out)
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


def recycle(output_folder,et_bg_fhs,recy_ratio,lu_fh,et_type):
    Data_Path_rec = "temp_et_recycle"
    out_folder = os.path.join(output_folder, Data_Path_rec)
    geo_out, proj, size_X, size_Y = RC.Open_array_info(lu_fh)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)    
    recycle_fhs = []
    for et_fh in et_bg_fhs:
        out_fh = out_folder + "\\recycled_et_"+et_type+et_fh[-11:-4]+".tif"
        et = RC.Open_tiff_array(et_fh)
        et[np.where(et==-9999)]=np.nan
        et_recy = et*recy_ratio
        DC.Save_as_tiff(out_fh, et_recy, geo_out)
        recycle_fhs.append(out_fh)
    return recycle_fhs

### Other functions
def lu_type_average(data_fh,lu_fh,lu_dict):
    LULC = RC.Open_tiff_array(lu_fh)
    in_data = RC.Open_tiff_array(data_fh)
    out_data = {}
    for lu_class in lu_dict.keys():
        mask = [LULC == value for value in lu_dict[lu_class]]
        mask = (np.sum(mask,axis=0)).astype(bool)
        out_data[lu_class] = np.nanmean(in_data[mask])
    return out_data
    
def lu_type_sum(data_fh,lu_fh,lu_dict):
    LULC = RC.Open_tiff_array(lu_fh)
    in_data = RC.Open_tiff_array(data_fh)
    out_data = {}
    for lu_class in lu_dict.keys():
        mask = [LULC == value for value in lu_dict[lu_class]]
        mask = (np.sum(mask,axis=0)).astype(bool)
        out_data[lu_class] = np.nansum(in_data[mask])
    return out_data

def split_yield(output_folder,p_fhs, et_blue_fhs,et_green_fhs, ab=(1.0,1.0)):
    Data_Path_split = "split_y"
    out_folder = os.path.join(output_folder, Data_Path_split)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    sp_yield_fhs= []
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
    feed_dict = {'Pasture':[2,3,12,13,14,15,16,17,20,29,34],
                 'Crop':[35,36,37,38,39,40,41,42,43,44,45,54,55,56,57,58,59,60,61,62]}
    
    # above ground biomass for tree and bush classes
    abv_grnd_biomass_ratio = {'1':.26, # protected forest
                              '8':.24, # closed deciduous forest
                              '9':.43, # open deciduous forest
                              '10':.23, # closed evergreen forest
                              '11':.46, # open evergreen forest
                              '12':.48, # closed savanna
                              '13':.48} # open savanna
    # C content as fraction dry matter for Carbon sequestration calculations
    c_fraction = {'default':.47
            }
    
    # 5% of above ground biomass for now
    fuel_dict ={'all':.05}
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
    'UTILIZED':     {'Forest':[8,9,10,11],
                    'Shrubland':[14],
                    'Natural grasslands':[12,13,15,16,20],
                    'Natural water bodies':[23,24,],
                    'Wetlands':[17,19,25,30,31],
                    'Others':[18,21,22,26,27,28,29,32]
                    },
    'MODIFIED':     {'Rainfed crops': [34,35,36,37,38,39,40,41,42,43,44],
                    'Forest plantations':[33],
                    'Settlements':[47,48,49,50,51],
                    'Others':[45,46]
                    },
    'MANAGED': {'Irrigated crops':[52,53,54,55,56,57,58,59,60,61,62],
                'Managed water bodies':[63,65,74],
                'Residential':[68,69,71,72],
                'Industry':[67,70,76],
                'Others':[75,78,77],
                'Indoor domestic':[66],
                'Indoor industry':[0],
                'Greenhouses':[64],
                'Livestock and husbandry':[73],
                'Power and energy':[79,80],
                }

    }

    sheet7_lulc_classes =dict()
    for k in sheet7_lulc.keys():
        l = []
        for k2 in sheet7_lulc[k].keys():
            l.append(sheet7_lulc[k][k2])
        sheet7_lulc_classes[k]=[item for sublist in l for item in sublist]
    return live_feed, feed_dict, abv_grnd_biomass_ratio,fuel_dict,sheet7_lulc_classes,c_fraction


def create_csv(results,output_fh):
    """
    Create the csv-file needed to plot sheet 7.
    
    Parameters
    ----------
    results : dict
        Dictionary generated by calc_sheet7.
    output_fh : str
        Filehandle to store the csv-file.
    """
    first_row = ['LAND_USE','VARIABLE','SERVICE','VALUE','UNITS']    
    if not os.path.exists(os.path.split(output_fh)[0]):
        os.makedirs(os.path.split(output_fh)[0])
    
    csv_file = open(output_fh, 'wb')
    writer = csv.writer(csv_file, delimiter=';')
    writer.writerow(first_row)
    lu_classes = ['PROTECTED', 'UTILIZED','MODIFIED', 'MANAGED']
    for lu_class in lu_classes:
        writer.writerow([lu_class,'Total Runoff','Non-consumptive', '{0:.3f}'.format(results['tot_runoff'][lu_class]),'km3'])
        writer.writerow([lu_class,'Groundwater Recharge','Non-consumptive', '{0:.3f}'.format(results['gw_rech'][lu_class]),'km3'])
    #    writer.writerow(['PROTECTED','Natural water storage in lakes', 'Non-consumptive',  '{0:.2f}'.format(results['nat_stor']),'km3'])
    #    writer.writerow(['PROTECTED','Inland Capture Fishery', 'Non-consumptive', '{0:.2f}'.format(results['fish']),'t'])
        writer.writerow([lu_class,'Natural Feed Production', 'Incremental ET natural', '{0:.3f}'.format(results['feed_incremental'][lu_class]),'t'])
        writer.writerow([lu_class,'Natural Feed Production', 'Landscape ET', '{0:.3f}'.format(results['feed_landscape'][lu_class]),'t'])
        writer.writerow([lu_class,'Natural Fuel Wood Production', 'Incremental ET natural', '{0:.3f}'.format(results['fuel_incremental'][lu_class]),'t'])
        writer.writerow([lu_class,'Natural Fuel Wood Production', 'Landscape ET', '{0:.3f}'.format(results['fuel_landscape'][lu_class]),'t'])
        writer.writerow([lu_class,'Dry Season Baseflow','Non-consumptive', '{0:.3f}'.format(results['baseflow'][lu_class]),'km3'])
    #    writer.writerow(['PROTECTED','Groundwater Recharge', 'SURFACE WATER', 'Flood', 0.])
        writer.writerow([lu_class,'Root Zone Water Storage', 'Non-consumptive', '{0:.3f}'.format(results['root_storage'][lu_class])])
        writer.writerow([lu_class,'Atmospheric Water Recycling', 'Incremental ET natural','{0:.3f}'.format(results['atm_recycl_incremental'][lu_class])])
        writer.writerow([lu_class,'Atmospheric Water Recycling', 'Landscape ET','{0:.3f}'.format(results['atm_recycl_landscape'][lu_class])])
    
    csv_file.close()
    
