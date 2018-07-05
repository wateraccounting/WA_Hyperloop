# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:52:31 2017

@author: bec
"""
from WA_Hyperloop.sheet1_functions import sheet1_functions as sh1
from WA_Hyperloop.sheet2_functions import sheet2_functions as sh2
from WA_Hyperloop.sheet3_functions import sheet3_functions as sh3
from WA_Hyperloop.sheet4_functions import sheet4_functions as sh4
from WA_Hyperloop.sheet5_functions import sheet5_functions as sh5
from WA_Hyperloop import hyperloop as hl
import matplotlib.pyplot as plt

###
# Define basin specific parameters
###
basins = dict()
     
ID = 3
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Ma',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'up_basin_masks':           r"D:\project_ADB\subproject_Catchment_Map\upstream_ID3",
            'alpha_min':                None,
    
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin03_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Tea', 'Beverage crops', '-', 42.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin03_crop43.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin03_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin03_crop57.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ],

            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
                                  
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[0], 2:[1], 3:[0], 4:[2,3]},
            'dico_out':                 {1:[2], 2:[4], 3:[4], 4:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
###
# Define output folder and WaterPix file
###
output_dir      = r"D:\project_ADB\Catchments\Vietnam"
waterpix        = r"K:\Products\WATERPIX\out_SEAsia_0point075.nc"
waterpix_in     = r"K:/Products/WATERPIX/in_SEAsia_0point075.nc"

###
# Define some paths of static data
###
global_data = dict()
global_data["equiped_sw_irrigation"]    = r"K:\Products\Landuse\Global\GMIA_FAO\gmia_v5_aeisw_pct_aei_asc\gmia_v5_aeisw_pct_aei.asc"
global_data["wpl_tif"]                  = r"D:\Products\GreyWaterFootprint\WPL.tif"
global_data["environ_water_req"]        = r"D:\Products\EF\EWR.tif"
global_data["population_tif"]           = r"D:\Products\WorldPop\VNM-POP\VNM_pph_v2b_2009.tif"
global_data["dem"]                      = r"K:\Products\HydroSHED\DEM\HydroSHED\DEM\DEM_HydroShed_m_15s.tif"
global_data["dir"]                      = r"K:\Products\HydroSHED\DEM\HydroSHED\DIR\DIR_HydroShed_-_15s.tif"
global_data["waterpix"]                 = r"D:\Products\Waterpix\SEAsia\output_20k_v2_20171102.nc"

###
# Define paths of folders with temporal tif files (file should be named "*_yyyymm.tif") covering the entire domain (i.e. spanning across all basins)
###
data = dict()
data["ndm_folder"]          = r"K:\Products\MODIS_17_NDM"
data["p_folder"]            = r"K:\Products\WATERPIX\Output\Precipitation_M" 
data["et_folder"]           = r"K:\Products\WATERPIX\Output\Evapotranspiration_M" 
data["n_folder"]            = r"K:\Products\WATERPIX\Output\RainyDays_M" 
data["lai_folder"]          = r"K:\Products\WATERPIX\Output\LeafAreaIndex_M" 
data["etref_folder"]        = r"K:\Products\WATERPIX\Output\ReferenceET_M" 
data["bf_folder"]           = r"K:\Products\WATERPIX\Output\Baseflow_M" 
data["sr_folder"]           = r"K:\Products\WATERPIX\Output\SurfaceRunoff_M" 
data["tr_folder"]           = r"K:\Products\WATERPIX\Output\TotalRunoff_M"
data["perc_folder"]         = r"K:\Products\WATERPIX\Output\Percolation_M"
data["dperc_folder"]        = r"K:\Products\WATERPIX\Output\IncrementalPercolation_M" 
data["supply_total_folder"] = r"K:\Products\WATERPIX\Output\Supply_M"
data["dro"]                 = r"K:\Products\WATERPIX\Output\IncrementalRunoff_M"
data["etb_folder"]          = r"K:\Products\WATERPIX\Output\ETblue_M"
data["etg_folder"]          = r"K:\Products\WATERPIX\Output\ETgreen_M"

#data = dict()
#data["ndm_folder"]          = r"K:\Products\MODIS_17_NDM"
#data["p_folder"]            = hl.WP_NetCDF_to_Rasters(waterpix_in, 'Precipitation_M', r"K:\Products\WATERPIX\Output")
#data["et_folder"]           = hl.WP_NetCDF_to_Rasters(waterpix_in, 'Evapotranspiration_M', r"K:\Products\WATERPIX\Output")
#data["n_folder"]            = hl.WP_NetCDF_to_Rasters(waterpix_in, 'RainyDays_M', r"K:\Products\WATERPIX\Output")
#data["lai_folder"]          = hl.WP_NetCDF_to_Rasters(waterpix_in, 'LeafAreaIndex_M', r"K:\Products\WATERPIX\Output")
#data["etref_folder"]        = hl.WP_NetCDF_to_Rasters(waterpix_in, 'ReferenceET_M', r"K:\Products\WATERPIX\Output")
#data["bf_folder"]           = hl.WP_NetCDF_to_Rasters(waterpix, 'Baseflow_M', r"K:\Products\WATERPIX\Output")
#data["sr_folder"]           = hl.WP_NetCDF_to_Rasters(waterpix, 'SurfaceRunoff_M', r"K:\Products\WATERPIX\Output")
#data["tr_folder"]           = hl.WP_NetCDF_to_Rasters(waterpix, 'TotalRunoff_M', r"K:\Products\WATERPIX\Output")
#data["perc_folder"]         = hl.WP_NetCDF_to_Rasters(waterpix, 'Percolation_M', r"K:\Products\WATERPIX\Output")
#data["dperc_folder"]        = hl.WP_NetCDF_to_Rasters(waterpix, 'IncrementalPercolation_M', r"K:\Products\WATERPIX\Output")
#data["supply_total_folder"] = hl.WP_NetCDF_to_Rasters(waterpix, 'Supply_M', r"K:\Products\WATERPIX\Output")
#data["dro"]                 = hl.WP_NetCDF_to_Rasters(waterpix, 'IncrementalRunoff_M', r"K:\Products\WATERPIX\Output")
#data["etb_folder"]          = hl.WP_NetCDF_to_Rasters(waterpix, 'ETblue_M', r"K:\Products\WATERPIX\Output")
#data["etg_folder"]          = hl.WP_NetCDF_to_Rasters(waterpix, 'ETgreen_M', r"K:\Products\WATERPIX\Output")

steps = dict()
steps['Reproject data']                  = True
steps['Create Sheet 4 and 6']            = True
steps['Create Sheet 2']                  = False
steps['Create Sheet 3']                  = False
steps['Create Sheet 5']                  = True
steps['Create Sheet 1']                  = True

#%%
###
# Start hyperloop
###
for ID, metadata in basins.items()[1:2]:
    
    print 'Start basin {0}: {1}'.format(ID, metadata['name'])
    plt.close("all")
    
    if steps['Reproject data']:
        complete_data = hl.sort_data(data, metadata, global_data, output_dir)
    else:
        complete_data = hl.sort_data_short(output_dir, metadata)

    if steps['Create Sheet 4 and 6']:
        complete_data = sh4.create_sheet4_6(complete_data, metadata, output_dir, global_data)

    if steps['Create Sheet 2']:
        complete_data = sh2.create_sheet2(complete_data, metadata, output_dir)

    if steps['Create Sheet 3']:
        complete_data = sh3.create_sheet3(complete_data, metadata, output_dir)

    if steps['Create Sheet 5']:
        complete_data = sh5.create_sheet5(complete_data, metadata, output_dir, global_data, data)

    if steps['Create Sheet 1']:
        complete_data, all_sh1_results = sh1.create_sheet1(complete_data, metadata, output_dir, global_data)