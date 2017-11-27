# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:28:17 2017

@author: bec
"""
import os
import WA_Hyperloop.becgis as becgis
from WA_Hyperloop.sheet5_functions import sheet5_functions as sh5
from WA_Hyperloop.sheet2_functions import sheet2_functions as sh2
from WA_Hyperloop.sheet3_functions import sheet3_functions as sh3
import WA_Hyperloop.get_dictionaries as gd
import glob
import davgis
import netCDF4
import wa.Generator.Sheet5.main as Sheet5
from shutil import copyfile
from WA_Hyperloop.sheet1_functions import sheet1_functions as sh1
import matplotlib.pyplot as plt
import numpy as np

def diagnosis(metadata, complete_data, output_dir, all_results, waterpix):

    output_dir = os.path.join(output_dir, metadata['name'], "diagnosis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    runoff = SortWaterPix(waterpix, 'TotalRunoff_M', output_dir)
    becgis.MatchProjResNDV(metadata['lu'], becgis.ListFilesInFolder(runoff), os.path.join(output_dir, "runoff_matched"))
    complete_data['ro'] = becgis.SortFiles(os.path.join(output_dir, "runoff_matched"), [-10,-6], month_position = [-6,-4])[0:2]
    
    LU = becgis.OpenAsArray(metadata['lu'], nan_values = True)
    
    common_dates = becgis.CommonDates([complete_data['p'][1],complete_data['et'][1],complete_data['ro'][1], complete_data['etb'][1]])
    
    becgis.AssertProjResNDV([complete_data['p'][0],complete_data['et'][0],complete_data['ro'][0]])
    
    balance = np.array([])
    p_y = np.array([])
    et_y = np.array([])
    ro_y = np.array([])
    
    area = becgis.MapPixelAreakm(metadata['lu'])
    
    for date in common_dates:
        
        P = complete_data['p'][0][complete_data['p'][1] == date][0]
        ET = complete_data['et'][0][complete_data['et'][1] == date][0]
        RO = complete_data['ro'][0][complete_data['ro'][1] == date][0]
        
        p = becgis.OpenAsArray(P, nan_values = True) * 0.001 * 0.001 * area
        et = becgis.OpenAsArray(ET, nan_values = True) * 0.001 * 0.001 * area
        ro = becgis.OpenAsArray(RO, nan_values = True) * 0.001 * 0.001 * area
    
        p[np.isnan(LU)] = et[np.isnan(LU)] = ro[np.isnan(LU)] = np.nan
    
        balance = np.append(balance, np.nansum(p) - np.nansum(et) - np.nansum(ro))
        p_y = np.append(p_y, np.nansum(p))
        et_y = np.append(et_y, np.nansum(et))
        ro_y = np.append(ro_y, np.nansum(ro))
      
    ##
    # BASIC WATERBALANCE (PRE-SHEETS)
    ##
    plt.figure(1)
    plt.clf()
    plt.plot(common_dates, np.cumsum(balance), label = 'dS [CHIRPS, ETens, WPro]')
    plt.plot(common_dates, np.cumsum(p_y), label = 'CHIRPS')
    plt.plot(common_dates, np.cumsum(et_y) + np.cumsum(ro_y), label = 'ETens + WPro')
    plt.legend()
    plt.title('CHIRPS, ETens, WPrunoff')
    plt.xlabel('months [km3/month]')
    plt.ylabel('flux [km3/month]')
    plt.savefig(os.path.join(output_dir, "WB_presheet1.jpg"))
    
    ###
    # CHECK ET
    ###
    plt.figure(2)
    plt.clf()
    et = sh1.get_ts(all_results, 'et_advection') + sh1.get_ts(all_results, 'p_recycled')
    plt.scatter(et_y, et)
    plt.xlabel('ETens [km3/month]')
    plt.ylabel('Sheet1 [km3/month]')
    plt.title('EVAPO, SCORE = {0}'.format(np.sum(et_y) - np.sum(et)))
    plt.savefig(os.path.join(output_dir, "CHECK_ET.jpg"))
    
    ##
    #CHECK P
    ##
    plt.figure(3)
    plt.clf()
    p = sh1.get_ts(all_results, 'p_advection') + sh1.get_ts(all_results, 'p_recycled')
    plt.scatter(p_y, p)
    plt.xlabel('CHIRPS [km3/month]')
    plt.ylabel('Sheet1 [km3/month]')
    plt.title('PRECIPITATION, SCORE = {0}'.format(np.sum(p_y) - np.sum(p)))
    plt.savefig(os.path.join(output_dir, "CHECK_P.jpg"))
    
    ##
    # CHECK Q
    ##
    plt.figure(4)
    plt.clf()
    q = sh1.get_ts(all_results, 'q_out_sw') -  sh1.get_ts(all_results, 'q_in_sw')  +  sh1.get_ts(all_results, 'q_out_gw')  -  sh1.get_ts(all_results, 'q_in_gw') +  sh1.get_ts(all_results, 'q_outflow') - sh1.get_ts(all_results, 'q_in_desal')
    plt.scatter(ro_y, q)
    plt.xlabel('Waterpix_runoff [km3/month]')
    plt.ylabel('Sheet1 [km3/month]')
    plt.title('RUNOFF, SCORE = {0}'.format(np.sum(ro_y) - np.sum(q)))
    plt.savefig(os.path.join(output_dir, "CHECK_Q.jpg"))
    
    ###
    # CHECK dS
    ###
    plt.figure(5)
    plt.clf()
    ds = sh1.get_ts(all_results, 'dS')
    plt.scatter(np.cumsum(balance), np.cumsum(ds) * -1)
    plt.xlabel('CHIRPS, ETens, WPrunoff [km3/month]')
    plt.ylabel('Sheet1')
    plt.title('dS, SCORE = {0}'.format(np.sum(np.cumsum(balance)) - np.sum(np.cumsum(ds) * -1)))
    plt.savefig(os.path.join(output_dir, "CHECK_dS.jpg"))
    
    ###
    # CHECK WATERBALANCE (POST-SHEET1)
    ###
    balance_sheet1 = p - et - q
    plt.figure(6)
    plt.clf()
    
    plt.plot(common_dates, np.cumsum(balance), label = 'dS [CHIRPS, ETens, WPro]')
    plt.plot(common_dates, np.cumsum(balance_sheet1), label = 'dS [P(sh1), ET(sh1) + Q(sh1)]')
    plt.plot(common_dates, np.cumsum(ds) * -1, label = 'dS [sh1]')
    
    plt.plot(common_dates, np.cumsum(p_y), label = 'CHIRPS')
    plt.plot(common_dates, np.cumsum(p), label = 'P [sh1]')
    
    plt.plot(common_dates, np.cumsum(et_y) + np.cumsum(ro_y), label = 'ETens + WPro')
    plt.plot(common_dates, np.cumsum(et) + np.cumsum(q), label = 'ET [sh1] + Q [sh1]')
    
    plt.legend()
    plt.xlabel('Months')
    plt.ylabel('Flux [km3/month]')
    plt.savefig(os.path.join(output_dir, "CHECK_WB.jpg"))
    
def prepareSurfWatLoop(data, global_data):
    """
    
    
    """

    data_needed = ["etref_folder", "et_folder", "p_folder"]

    data_dst = {"etref_folder": r"ETref\Monthly",
                "et_folder":    r"Evaporation\ETensV1_0",
                "p_folder":     r"Precipitation\CHIRPS\Monthly"}

    for data_name in data_needed:
        print data_name
        files, dates = becgis.SortFiles(data[data_name], [-10, -6],
                                        month_position=[-6, -4])[0:2]

        for f, d in zip(files, dates):

            fp = os.path.split(f)[1]

            dst = os.path.join(os.environ["WA_HOME"],
                               'Loop_SW', data_dst[data_name],
                               fp[:-4] +
                               "_monthly_{0}.{1}.01.tif".format(d.year,
                                                                str(d.month).zfill(2)))

            folder = os.path.split(dst)[0]

            if not os.path.exists(folder):
                os.makedirs(folder)

            copyfile(f, dst)

    pt = os.path.join(os.environ["WA_HOME"], 'Loop_SW', 'HydroSHED', 'DIR')

    if not os.path.exists(pt):
        os.makedirs(pt)

    copyfile(global_data['dir'], os.path.join(pt, "DIR_HydroShed_-_15s.tif"))

def LoopSurfWat(waterpix, metadata, global_data, big_basins = None):
    
    dst = os.path.join(os.environ["WA_HOME"], "LU", "Loop_SW.tif")
    if os.path.exists(dst):
        os.remove(dst)
        
    copyfile(metadata['full_basin_mask'], dst)
    
    Basin = 'Loop_SW'
    
    P_Product = 'CHIRPS'
    ET_Product = 'ETensV1_0'
    Inflow_Text_Files = []
    Reservoirs_GEE_on_off = 0
    Supply_method = "Fraction"
    ID = metadata['id']
    
    pt = os.path.join(os.environ["WA_HOME"], 'Loop_SW', 'HydroSHED', 'DEM')
    pt2 = os.path.join(os.environ["WA_HOME"], 'Loop_SW', 'HydroSHED', 'DIR')

    if os.path.exists(os.path.join(pt, "DEM_HydroShed_m_15s.tif")):
        os.remove(os.path.join(pt, "DEM_HydroShed_m_15s.tif"))
    if os.path.exists(os.path.join(pt2, "DIR_HydroShed_-_15s.tif")):
        os.remove(os.path.join(pt2, "DIR_HydroShed_-_15s.tif"))

    start = str(netCDF4.Dataset(waterpix).variables["time_yyyymm"][:][0])
    end = str(netCDF4.Dataset(waterpix).variables["time_yyyymm"][:][-1])
    
    Startdate = "{0}-{1}-01".format(start[0:4], start[4:6])
    Enddate = "{0}-{1}-31".format(end[0:4], end[4:6])
    
    print Startdate, Enddate
    
    if ID in big_basins:
        
        surfwater_path = list()
        
        years = range(int(Startdate[0:4]), int(Enddate[0:4])+1)
        
        ID = ID * 100
        
        for year in years:
            
            ID += 1
            Startdate = "{0}-01-01".format(year)
            Enddate = "{0}-12-31".format(year)          
            
            filename = os.path.join(os.environ["WA_HOME"], "Loop_SW", "Simulations", "Simulation_{0}".format(ID), "Sheet_5", "Discharge_CR1_Simulation{0}_monthly_m3_01{1}_12{1}.nc".format(ID, int(year)))
            
            if not os.path.exists(filename):
                Sheet5.Calculate(Basin, 
                                 P_Product, 
                                 ET_Product, 
                                 Inflow_Text_Files, 
                                 waterpix, 
                                 Reservoirs_GEE_on_off, 
                                 Supply_method, 
                                 Startdate, 
                                 Enddate, 
                                 ID)
                os.remove(os.path.join(pt, "DEM_HydroShed_m_15s.tif"))
                os.remove(os.path.join(pt2, "DIR_HydroShed_-_15s.tif"))
       
            else:
                pass
            
            surfwater_path.append(filename)

    else:

        surfwater_path = os.path.join(os.environ["WA_HOME"], "Loop_SW", "Simulations", "Simulation_{0}".format(ID), "Sheet_5", "Discharge_CR1_Simulation{0}_monthly_m3_012003_122014.nc".format(ID))
 
        if not os.path.exists(surfwater_path):   
            Sheet5.Calculate(Basin, 
                             P_Product, 
                             ET_Product, 
                             Inflow_Text_Files, 
                             waterpix, 
                             Reservoirs_GEE_on_off, 
                             Supply_method, 
                             Startdate, 
                             Enddate, 
                             ID)
            os.remove(os.path.join(pt, "DEM_HydroShed_m_15s.tif"))
            os.remove(os.path.join(pt2, "DIR_HydroShed_-_15s.tif"))
        else:
            pass
        

    return surfwater_path
    
def sort_data_short(output_dir, metadata):
    data = ['p', 'et', 'n', 'ndm', 'lai', 'etref', 'etb', 'etg', 'i', 't', 'r', 'bf', 'sr', 'tr']
    complete_data = dict()
    for datatype in data:
        try:
            folder = os.path.join(output_dir, metadata['name'], datatype)
            for fn in glob.glob(folder + "\\*_km3.tif"):
                os.remove(fn)
            files, dates = becgis.SortFiles(folder, [-10,-6], month_position = [-6,-4])[0:2]
            complete_data[datatype] = (files, dates)
        except: 
            print datatype
            continue
    data_2 = ['SUPPLYsw','RETURNFLOW_gwsw','RETURNFLOW_swsw',r'fractions\fractions']
    
    data_2dict = {'SUPPLYsw': 'supply_sw',
                  'RETURNFLOW_gwsw': 'return_flow_gw_sw',
                  'RETURNFLOW_swsw': 'return_flow_sw_sw',
                  r'fractions\fractions': 'fractions'}

    for datatype in data_2:
        try:
            folder = os.path.join(output_dir, metadata['name'], datatype)
            for fn in glob.glob(folder + "\\*_km3.tif"):
                os.remove(fn) 
            files, dates = becgis.SortFiles(folder, [-11,-7], month_position = [-6,-4])[0:2]
            complete_data[data_2dict[datatype]] = (files, dates)
        except: 
            print datatype
            continue
    
    return complete_data

def sort_data(data, metadata, global_data, output_dir):
    output_dir = os.path.join(output_dir, metadata['name'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    complete_data = dict()
    for key in data.keys():
        files, dates = becgis.SortFiles(data[key], [-10,-6], month_position = [-6,-4])[0:2]
        scales = {'p': metadata['p_scale'], 'et': metadata['et_scale'], 'n': 1.0, 'ndm': 1.0, 'lai': 1.0, 'etref': 1.0, 'r': 1.0, 'bf': 1.0, 'tr': 1.0, 'sr': 1.0}
        var_name = key.split('_folder')[0]
        files = becgis.MatchProjResNDV(metadata['lu'], files, os.path.join(output_dir, var_name), resample = 'near', dtype = 'float32', scale = scales[var_name])
        complete_data[var_name] = (files, dates)

    complete_data['fractions'] = sh5.calc_fractions(complete_data['p'][0], complete_data['p'][1], os.path.join(output_dir, 'fractions'), global_data['dem'], metadata['lu'])

    i_files, i_dates, t_files, t_dates = sh2.splitET_ITE(complete_data['et'][0], complete_data['et'][1], complete_data['lai'][0], complete_data['lai'][1], complete_data['p'][0], complete_data['p'][1], complete_data['n'][0], complete_data['n'][1], complete_data['ndm'][0], complete_data['ndm'][1], output_dir, ndm_max_original = False, plot_graph = True, save_e = False)

    complete_data['i'] = (i_files, i_dates)
    complete_data['t'] = (t_files, t_dates)
    
    gb_cats, mvg_avg_len = gd.get_bluegreen_classes(version = '1.0')
    etblue_files, etblue_dates, etgreen_files, etgreen_dates = sh3.splitET_BlueGreen(complete_data['et'][0], complete_data['et'][1], complete_data['etref'][0], complete_data['etref'][1], complete_data['p'][0], complete_data['p'][1], metadata['lu'], output_dir, 
                  moving_avg_length = mvg_avg_len, green_blue_categories = gb_cats, plot_graph = False, 
                  method = 'tail', scale = 1.1, basin = metadata['name'])
                      
    complete_data['etb'] = (etblue_files, etblue_dates)
    complete_data['etg'] = (etgreen_files, etgreen_dates)
    
    return complete_data

def SortWaterPix(nc, variable, output_folder):
    spa_ref = davgis.Spatial_Reference(4326)
    nc1 = netCDF4.Dataset(nc)
    time = nc1.variables['time_yyyymm'][:]
    for time_value in time:
        output_dir = os.path.join(output_folder, variable)
        output_tif = os.path.join(output_dir, "{0}_{1}.tif".format(variable, time_value))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        davgis.NetCDF_to_Raster(input_nc=nc,
                                output_tiff=output_tif,
                                ras_variable=variable,
                                x_variable='longitude', y_variable='latitude',
                                crs=spa_ref,
                                time={'variable': 'time_yyyymm', 'value': time_value})
    return output_dir
