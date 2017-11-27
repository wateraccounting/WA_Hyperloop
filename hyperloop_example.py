# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:52:31 2017

@author: bec
"""
import WA_Hyperloop.sheet1_functions as sh1
import WA_Hyperloop.sheet2_functions as sh2
import WA_Hyperloop.sheet3_functions as sh3
import WA_Hyperloop.sheet4_functions as sh4
import WA_Hyperloop.sheet5_functions as sh5
import WA_Hyperloop.hyperloop as hl

###
# Define basin specific parameters
###
basins = dict()

ID = 2
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Hong',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'SWfile':                   [r"D:\WA_HOME\Loop_SW\Simulations\Simulation_201\Sheet_5\Discharge_CR1_Simulation201_monthly_m3_012003_122003.nc",
                                         r"D:\WA_HOME\Loop_SW\Simulations\Simulation_202\Sheet_5\Discharge_CR1_Simulation202_monthly_m3_012004_122004.nc",
                                         r"D:\WA_HOME\Loop_SW\Simulations\Simulation_203\Sheet_5\Discharge_CR1_Simulation203_monthly_m3_012005_122005.nc",
                                         r"D:\WA_HOME\Loop_SW\Simulations\Simulation_204\Sheet_5\Discharge_CR1_Simulation204_monthly_m3_012006_122006.nc",
                                         r"D:\WA_HOME\Loop_SW\Simulations\Simulation_205\Sheet_5\Discharge_CR1_Simulation205_monthly_m3_012007_122007.nc",
                                         r"D:\WA_HOME\Loop_SW\Simulations\Simulation_206\Sheet_5\Discharge_CR1_Simulation206_monthly_m3_012008_122008.nc",
                                         r"D:\WA_HOME\Loop_SW\Simulations\Simulation_207\Sheet_5\Discharge_CR1_Simulation207_monthly_m3_012009_122009.nc",
                                         r"D:\WA_HOME\Loop_SW\Simulations\Simulation_208\Sheet_5\Discharge_CR1_Simulation208_monthly_m3_012010_122010.nc",
                                         r"D:\WA_HOME\Loop_SW\Simulations\Simulation_209\Sheet_5\Discharge_CR1_Simulation209_monthly_m3_012011_122011.nc",
                                         r"D:\WA_HOME\Loop_SW\Simulations\Simulation_210\Sheet_5\Discharge_CR1_Simulation210_monthly_m3_012012_122012.nc",
                                         r"D:\WA_HOME\Loop_SW\Simulations\Simulation_211\Sheet_5\Discharge_CR1_Simulation211_monthly_m3_012013_122013.nc",
                                         r"D:\WA_HOME\Loop_SW\Simulations\Simulation_212\Sheet_5\Discharge_CR1_Simulation212_monthly_m3_012014_122014.nc"
                                        ],
            'outflow_nodes':            r"D:\project_ADB\subproject_Catchment_Map\outlets\Basins_outlets_basin_{0}.shp".format(ID),
            'inflow_nodes':             r"D:\project_ADB\subproject_Catchment_Map\inlets\Basins_inlets_basin_2.shp",
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'in_text':                  None,
            
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\seasons_basin02_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\seasons_basin02_crop36.csv', 'Potato', 'Non-cereals', 'Root/tuber crops', 36.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\seasons_perennial.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 38.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\seasons_perennial.csv', 'Tea', 'Beverage crops', '-', 42.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\seasons_basin02_crop43.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\seasons_basin02_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\seasons_basin02_crop57.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\seasons_perennial.csv', 'Peanut', 'Fruit & vegetables', 'Fruits & nuts', 58.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
            
            # Increase or decrease et, p, qin and qout with a scale        
            'et_scale':                 1.0,
            'p_scale':                  1.0,
            'qin_scale':                1.0,
            'qout_scale':               1.0,
            
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'fraction_altitude_xs':     [50, 600, 50, 600],
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[0],2:[0],3:[0],4:[1,2,3]},
            'dico_out':                 {1:[4],2:[4],3:[4],4:[0]},
            }
            
###
# Define output folder and WaterPix file
###
output_dir = r"D:\project_ADB\Catchments\Vietnam"
waterpix = r"D:\Products\Waterpix\SEAsia\output_20k_v2_20171102.nc"

###
# Define some paths of static data
###
global_data = dict()
global_data["equiped_sw_irrigation"] = r"D:\Products\Landuse\Global\GMIA_FAO\gmia_v5_aeisw_pct_aei_asc\gmia_v5_aeisw_pct_aei.asc"
global_data["wpl_tif"] = r"D:\Products\GreyWaterFootprint\WPL.tif"
global_data["environ_water_req"] = r"D:\Products\EF\EWR.tif"
global_data["population_tif"] = r"D:\Products\WorldPop\VNM-POP\VNM_pph_v2b_2009.tif"
global_data["dem"] = r"K:\Products\HydroSHED\DEM\HydroSHED\DEM\DEM_HydroShed_m_15s.tif"
global_data["dir"] = r"K:\Products\HydroSHED\DEM\HydroSHED\DIR\DIR_HydroShed_-_15s.tif"

###
# Define paths of folders with temporal tif files (file should be named "*_yyyymm.tif") covering the entire domain (i.e. spanning across all basins)
###
data = dict()
data["p_folder"] = r"K:\Products\WaterPix_SEAsia\P"
data["et_folder"] = r"K:\Products\WaterPix_SEAsia\ET"
data["n_folder"] = r"K:\Products\WaterPix_SEAsia\N"
data["ndm_folder"] = r"K:\Products\MODIS_17_NDM"
data["lai_folder"] = r"K:\Products\WaterPix_SEAsia\LAI"
data["etref_folder"] = r"K:\Products\WaterPix_SEAsia\ETREF"
data["bf_folder"] = r"K:\Products\WATERPIX\Baseflow_M" #hl.SortWaterPix(waterpix, 'Baseflow_M', r"K:\Products\WATERPIX") # This functions export tifs from the waterpix-nc file for the entire domain
data["sr_folder"] = r"K:\Products\WATERPIX\SurfaceRunoff_M" #hl.SortWaterPix(waterpix, 'SurfaceRunoff_M', r"K:\Products\WATERPIX")
data["tr_folder"] = r"K:\Products\WATERPIX\TotalRunoff_M" #hl.SortWaterPix(waterpix, 'TotalRunoff_M', r"K:\Products\WATERPIX")
data["r_folder"] = r"K:\Products\WATERPIX\Percolation_M" # hl.SortWaterPix(waterpix, 'Percolation_M', r"K:\Products\WATERPIX")

#%%
###
# Copy the data into the wa-surfwat framework, to prevent downloading per basin. (only DEM and DIR will be donwloaded per basin)
# This step can be disabled if Surfwat is run beforehand and the variable metadata['SWfile'] is already set.
###
#hl.prepareSurfWatLoop(data, global_data)
#for ID, metadata in basins.items()[0:1]:
#        
#    metadata['SWfile'] = hl.LoopSurfWat(waterpix, metadata, global_data, big_basins = [2])
#    print 'SurfWat Finished'
#  
  
###
# Start hyperloop
###
import os
import matplotlib

for ID, metadata in basins.items()[12:13]:
    
    print 'Start basin {0}'.format(ID)
    
    matplotlib.pyplot.close("all")
    
    if not os.path.exists(r"D:\project_ADB\Catchments\Vietnam\{0}\etb".format(metadata['name'])):
        complete_data = hl.sort_data(data, metadata, global_data, output_dir)
    else:
        complete_data = hl.sort_data_short(output_dir, metadata)
        print 'Sort Data Finished'
    
    if not os.path.exists(r"D:\project_ADB\Catchments\Vietnam\{0}\yearly_sheet2".format(metadata['name'])):
        print 'Running Sheet 2'
        complete_data = sh2.create_sheet2(complete_data, metadata, output_dir)
        print 'Sheet 2 Finished'
    
    if not os.path.exists(r"D:\project_ADB\Catchments\Vietnam\{0}\sheet3".format(metadata['name'])):
        print 'Running Sheet 3'
        complete_data = sh3.create_sheet3(complete_data, metadata, output_dir)
        print 'Sheet 3 Finished'
    
    if not os.path.exists(r"D:\project_ADB\Catchments\Vietnam\{0}\sheet4_yearly".format(metadata['name'])):
        print 'Running Sheet 4 and 6'
        complete_data = sh4.create_sheet4_6(complete_data, metadata, output_dir, global_data)
        print 'Sheet 4 and 6 Finished'
    
    if not os.path.exists(r"D:\project_ADB\Catchments\Vietnam\{0}\sheet5_yearly".format(metadata['name'])):
        print 'Running Sheet 5'
        complete_data = sh5.create_sheet5(complete_data, metadata, output_dir, global_data)
        print 'Sheet 5 Finished'

    if not os.path.exists(r"D:\project_ADB\Catchments\Vietnam\{0}\csvs_yearly".format(metadata['name'])):
        print 'Running Sheet 1'
        complete_data, all_sh1_results = sh1.create_sheet1(complete_data, metadata, output_dir, global_data)
        print 'Sheet 1 Finished'
    
    print '{0} Finished'.format(ID)

    hl.diagnosis(metadata, complete_data, output_dir, all_sh1_results, waterpix)    

