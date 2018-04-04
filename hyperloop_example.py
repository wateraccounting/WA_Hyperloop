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
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'up_basin_masks':           r"D:\project_ADB\subproject_Catchment_Map\upstream_ID{0}".format(ID),
            'alpha_min':                None,
            
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin02_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin02_crop36.csv', 'Potato', 'Non-cereals', 'Root/tuber crops', 36.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 38.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Tea', 'Beverage crops', '-', 42.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin02_crop43.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin02_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin02_crop57.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Peanut', 'Fruit & vegetables', 'Fruits & nuts', 58.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
            
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[0],2:[0],3:[0],4:[1,2,3]},
            'dico_out':                 {1:[4],2:[4],3:[4],4:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
     
ID = 3
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Ma',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
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
            
ID = 4
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Ca',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'up_basin_masks':           r"D:\project_ADB\subproject_Catchment_Map\upstream_ID4",
            'alpha_min':                None,            
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin04_crop43.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin04_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
           
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[0], 2:[], 3:[1,2], 4:[]},
            'dico_out':                 {1:[3], 2:[3], 3:[0], 4:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
ID = 5
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Ba',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'alpha_min':                None,            
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Rubber', 'Other crops', '-', 33.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin05_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin05_crop39.csv', 'Almond', 'Fruit & vegetables', 'Fruits & nuts', 39.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Mango', 'Fruit & vegetables', 'Vegetables & melons', 40.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Coffee', 'Beverage crops', '-', 42.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin05_crop43.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin05_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin05_crop57.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Pepper', 'Beverage crops', '-', 61.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
           
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[], 2:[], 3:[1,2], 4:[]},
            'dico_out':                 {1:[3], 2:[3], 3:[0], 4:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }

ID = 6
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Serepok',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID6.tif",
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'alpha_min':                None,            
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Rubber', 'Other crops', '-', 33.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin06_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Potato', 'Non-cereals', 'Root/tuber crops', 36.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin06_crop39.csv', 'Peanut', 'Fruit & vegetables', 'Fruits & nuts', 39.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Mango', 'Fruit & vegetables', 'Vegetables & melons', 40.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Coffee', 'Beverage crops', '-', 42.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin06_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Pepper', 'Beverage crops', '-', 61.0) ,
                                        ],
                    
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
                       
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[], 2:[], 3:[1,2], 4:[]},
            'dico_out':                 {1:[3], 2:[3], 3:[0], 4:[0],},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
ID = 7
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Se_San',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'alpha_min':                None,            
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin07_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin07_crop36.csv', 'Potato', 'Non-cereals', 'Root/tuber crops', 36.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Peanut', 'Fruit & vegetables', 'Fruits & nuts', 39.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Coffee', 'Beverage crops', '-', 42.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin07_crop43.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
        
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[], 2:[], 3:[1,2], 4:[]},
            'dico_out':                 {1:[3], 2:[3], 3:[0], 4:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
ID = 8
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Dong_Nai',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'alpha_min':                None,            
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin08_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin08_crop37.csv', 'Potato', 'Non-cereals', 'Leguminous crops', 37.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin08_crop38.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 38.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Coffee', 'Beverage crops', '-', 42.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin08_crop43.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin08_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin08_crop57.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},

            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
           'recycling_ratio':          0.02,
            'dico_in':                  {1:[], 2:[], 3:[], 4:[], 5:[1,2,3]},
            'dico_out':                 {1:[5], 2:[5], 3:[5], 4:[0],5:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
ID = 9
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Bang_Giang',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'alpha_min':                None,           
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin09_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Potato', 'Non-cereals', 'Root/tuber crops', 36.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 38.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin09_crop43.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},

            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[], 2:[]},
            'dico_out':                 {1:[0],2:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
#ID = 10
#basins[ID] = {
#            # Give name and ID of basin, set ID equal to key.
#            'name':                     'Cuu_Long',
#            'id':                       ID,
#            
#            # Give LU-map, SW-file, 
#            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
#            # folder with subbasin masks (name: subbasinname_ID.tif)
#            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
#            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
#            'SWfile':                   r"D:\WA_HOME\Loop_SW\Simulations\Simulation_{0}\Sheet_5\Discharge_CR1_Simulation{0}_monthly_m3_012003_122014.nc".format(ID),
#            'outflow_nodes':            r"D:\project_ADB\subproject_Catchment_Map\outlets\Basins_outlets_basin_{0}.shp".format(ID),
#            'inflow_nodes':             None,
#            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
#            'in_text':                  {1: "txtfilewithinflowsfrommekong.txt"},
#            
#            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
#            'crops':                    [
#                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin10_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
#                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Fodder', 'Other crops', '-', 43.0) ,
#                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin10_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
#                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin10_crop57.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
#                                        ],
#            
#            # Provide non-crop data, set to None if not available.
#            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
#                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
#                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
#                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
#                   
#            # Increase or decrease et, p, qin and qout with a scale        
#            'et_scale':                 1.0,
#            'p_scale':                  1.0,
#            'qin_scale':                1.0,
#            'qout_scale':               1.0,
#            
#            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
#            'fraction_altitude_xs':     [50, 600, 50, 600],
#            'recycling_ratio':          0.02,
#            'dico_in':                  {1:[0]},
#            'dico_out':                 {1:[0]},
#            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
#            }
            
ID = 11
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Gianh',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'alpha_min':                None,           
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin11_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
     
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[]},
            'dico_out':                 {1:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
ID = 12
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Thach_Han',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'alpha_min':                None,          
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin12_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin12_crop43.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin12_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
         
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[],2:[],3:[]},
            'dico_out':                 {1:[0],2:[0],3:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
ID = 13
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Huong',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'alpha_min':                0.1,
            
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin13_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin13_crop36.csv', 'Potato', 'Non-cereals', 'Root/tuber crops', 36.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin13_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
            
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[],2:[],3:[],4:[2,3],5:[]},
            'dico_out':                 {1:[0], 2:[4], 3:[4], 4:[0], 5:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
ID = 14
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'VGTB',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'alpha_min':                None,            
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin14_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
         
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[], 2:[], 3:[], 4:[1,2]},
            'dico_out':                 {1:[4],2:[4],3:[0],4:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
ID = 15
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Tra_Khuc',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'alpha_min':                None,           
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin15_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin15_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
 
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[], 2:[], 3:[], 4:[]},
            'dico_out':                 {1:[0], 2:[0], 3:[0], 4:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
ID = 16
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'Kone',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'alpha_min':                None,
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin16_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin16_crop54.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin16_crop57.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
                        
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[], 2:[], 3:[]},
            'dico_out':                 {1:[0], 2:[0], 3:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
ID = 17
basins[ID] = {
            # Give name and ID of basin, set ID equal to key.
            'name':                     'SERC',
            'id':                       ID,
            
            # Give LU-map, SW-file, 
            # shapefile with outlets of (sub-)basins (the second column should give a subbasin ID), 
            # folder with subbasin masks (name: subbasinname_ID.tif)
            'lu':                       r"D:\project_ADB\subproject_WALU\Clipped_final\Basins_Vietnam_ID_{0}.tif".format(ID),
            'full_basin_mask':          r"D:\project_ADB\subproject_Catchment_Map\Basins_exploded\Raster\ID{0}.tif".format(ID),
            'masks':                    r"D:\project_ADB\subproject_Catchment_Map\Basins_large\Subbasins_masks\ID{0}".format(ID),
            'alpha_min':                None,
            
            # Give start and enddates growingseasons, classifications to select Harvest Index and Water Content, LU-classification number
            'crops':                    [
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Rubber', 'Other crops', '-', 33.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin17_crop35.csv', 'Rice - Rainfed', 'Cereals', '-', 35.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin17_crop38.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 38.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin17_crop39.csv', 'Peanut', 'Fruit & vegetables', 'Fruits & nuts', 39.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin17_crop43.csv', 'Fodder', 'Other crops', '-', 43.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_perennial.csv', 'Rice - Irrigated', 'Cereals', '-', 54.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin17_crop57.csv', 'Sugar cane', 'Non-cereals', 'Sugar crops', 57.0) ,
                                        ('D:\\project_ADB\\subproject_Crop_Calendars\\consolidation\\seasons_basin17_crop58.csv', 'Mellon', 'Fruit & vegetables', 'Fruits & nuts', 58.0) ,
                                        ],
            
            # Provide non-crop data, set to None if not available.
            'non_crop':                 {'meat':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_meat.csv".format(ID),
                                         'milk':        r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_milk.csv".format(ID),
                                         'timber':      r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_timber.csv".format(ID),
                                         'aquaculture': r"D:\project_ADB\validation_data\STATS_GSOV\Main\yield_per_catchment\Yearly_Yields_WPs_{0}.0_aquaculture.csv".format(ID)},
          
            # set variables needed for sheet 5 and 1. keys in dico_out and dico_in refer to subbasin-IDs, list to subbasin-IDs to the respective subbasin in or outflow point. Give most upstream subbasins the lowest value, downstream basins high values.
            'recycling_ratio':          0.02,
            'dico_in':                  {1:[], 2:[], 3:[],4:[], 5:[], 6:[],7:[], 8:[], 9:[]},
            'dico_out':                 {1:[0], 2:[0], 3:[0],4:[0], 5:[0], 6:[0],7:[0], 8:[0], 9:[0]},
            'GRACE':                    r"D:\project_ADB\validation_data\GRACE\basin_{0}_GSFC_mmwe.csv".format(str(ID).zfill(2)),
            }
            
###
# Define output folder and WaterPix file
###
output_dir      = r"D:\project_ADB\Catchments\Vietnam"
waterpix        = r"D:\Products\Waterpix\NEW\output_vClaire_10k_v2.nc"
waterpix_in     = r"D:\Products\Waterpix\NEW\input_10k_v2.nc"

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
data["p_folder"]            = r"K:\Products\WATERPIX\Precipitation_M" #hl.WP_NetCDF_to_Rasters(waterpix_in, 'Precipitation_M', r"K:\Products\WATERPIX")
data["et_folder"]           = r"K:\Products\WATERPIX\Evapotranspiration_M" #hl.WP_NetCDF_to_Rasters(waterpix_in, 'Evapotranspiration_M', r"K:\Products\WATERPIX")
data["n_folder"]            = r"K:\Products\WATERPIX\RainyDays_M" #hl.WP_NetCDF_to_Rasters(waterpix_in, 'RainyDays_M', r"K:\Products\WATERPIX")
data["lai_folder"]          = r"K:\Products\WATERPIX\LeafAreaIndex_M" #hl.WP_NetCDF_to_Rasters(waterpix_in, 'LeafAreaIndex_M', r"K:\Products\WATERPIX")
data["etref_folder"]        = r"K:\Products\WATERPIX\ReferenceET_M" #hl.WP_NetCDF_to_Rasters(waterpix_in, 'ReferenceET_M', r"K:\Products\WATERPIX")
data["bf_folder"]           = r"K:\Products\WATERPIX\Baseflow_M" #hl.WP_NetCDF_to_Rasters(waterpix, 'Baseflow_M', r"K:\Products\WATERPIX")
data["sr_folder"]           = r"K:\Products\WATERPIX\SurfaceRunoff_M" #hl.WP_NetCDF_to_Rasters(waterpix, 'SurfaceRunoff_M', r"K:\Products\WATERPIX")
data["tr_folder"]           = r"K:\Products\WATERPIX\TotalRunoff_M" #hl.WP_NetCDF_to_Rasters(waterpix, 'TotalRunoff_M', r"K:\Products\WATERPIX")
data["perc_folder"]         = r"K:\Products\WATERPIX\Percolation_M" #hl.WP_NetCDF_to_Rasters(waterpix, 'Percolation_M', r"K:\Products\WATERPIX")
data["dperc_folder"]        = r"K:\Products\WATERPIX\IncrementalPercolation_M" #hl.WP_NetCDF_to_Rasters(waterpix, 'IncrementalPercolation_M', r"K:\Products\WATERPIX")
data["supply_total_folder"] = r"K:\Products\WATERPIX\Supply_M" #hl.WP_NetCDF_to_Rasters(waterpix, 'Supply_M', r"K:\Products\WATERPIX")
data["dro"]                 = r"K:\Products\WATERPIX\IncrementalRunoff_M" #hl.WP_NetCDF_to_Rasters(waterpix, 'IncrementalRunoff_M', r"K:\Products\WATERPIX")
data["etb_folder"]          = r"K:\Products\WATERPIX\ETblue_M" #hl.WP_NetCDF_to_Rasters(waterpix, 'ETblue_M', r"K:\Products\WATERPIX")
data["etg_folder"]          = r"K:\Products\WATERPIX\ETgreen_M" #hl.WP_NetCDF_to_Rasters(waterpix, 'ETgreen_M', r"K:\Products\WATERPIX")

#data = dict()
#data["ndm_folder"]          = r"K:\Products\MODIS_17_NDM"
#data["p_folder"]            = hl.WP_NetCDF_to_Rasters(waterpix_in, 'Precipitation_M', r"K:\Products\WATERPIX")
#data["et_folder"]           = hl.WP_NetCDF_to_Rasters(waterpix_in, 'Evapotranspiration_M', r"K:\Products\WATERPIX")
#data["n_folder"]            = hl.WP_NetCDF_to_Rasters(waterpix_in, 'RainyDays_M', r"K:\Products\WATERPIX")
#data["lai_folder"]          = hl.WP_NetCDF_to_Rasters(waterpix_in, 'LeafAreaIndex_M', r"K:\Products\WATERPIX")
#data["etref_folder"]        = hl.WP_NetCDF_to_Rasters(waterpix_in, 'ReferenceET_M', r"K:\Products\WATERPIX")
#data["bf_folder"]           = hl.WP_NetCDF_to_Rasters(waterpix, 'Baseflow_M', r"K:\Products\WATERPIX")
#data["sr_folder"]           = hl.WP_NetCDF_to_Rasters(waterpix, 'SurfaceRunoff_M', r"K:\Products\WATERPIX")
#data["tr_folder"]           = hl.WP_NetCDF_to_Rasters(waterpix, 'TotalRunoff_M', r"K:\Products\WATERPIX")
#data["perc_folder"]         = hl.WP_NetCDF_to_Rasters(waterpix, 'Percolation_M', r"K:\Products\WATERPIX")
#data["dperc_folder"]        = hl.WP_NetCDF_to_Rasters(waterpix, 'IncrementalPercolation_M', r"K:\Products\WATERPIX")
#data["supply_total_folder"] = hl.WP_NetCDF_to_Rasters(waterpix, 'Supply_M', r"K:\Products\WATERPIX")
#data["dro"]                 = hl.WP_NetCDF_to_Rasters(waterpix, 'IncrementalRunoff_M', r"K:\Products\WATERPIX")
#data["etb_folder"]          = hl.WP_NetCDF_to_Rasters(waterpix, 'ETblue_M', r"K:\Products\WATERPIX")
#data["etg_folder"]          = hl.WP_NetCDF_to_Rasters(waterpix, 'ETgreen_M', r"K:\Products\WATERPIX")

steps = dict()
steps['Reproject data']                  = True
steps['Create Sheet 4 and 6']            = True
steps['Create Sheet 2']                  = True
steps['Create Sheet 3']                  = True
steps['Create Sheet 5']                  = True
steps['Create Sheet 1']                  = True

#%%
###
# Start hyperloop
###
for ID, metadata in basins.items()[10:11]:
    
    
    
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