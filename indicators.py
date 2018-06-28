# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:54:17 2017

@author: cmi001
"""

import os
import csv
import glob
import numpy as np
import pandas as pd
import datetime

class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def sheet1_indicators(dir1):

    exploitable_water_fractions = np.array([])
    storage_change_fractions = np.array([])
    available_water_fractions = np.array([])
    basin_closure_fractions = np.array([])
    rainfall_dependency_fractions = np.array([]) 
    utilizable_outflow_fractions = np.array([])
    reserved_outflows_fractions = np.array([])
    dates = np.array([])

    file_list = glob.glob(dir1+'\sheet*.csv')

    
    for f in file_list:
        
        explod = ''.join(os.path.split(f)[1].split('.')[:-1]).split('_')
        try:
            date = datetime.date(int(explod[-2]), int(explod[-1]), 1)
        except:
            date = datetime.date(int(explod[-1]), 1, 1)        
            
        with open(f) as f1:
        
            data = csv.reader(f1)
            DATA = Vividict()
            next(data)
            for row in data:
                splitr= row[0].split(';')
                DATA[splitr[0]][splitr[1]][splitr[2]]=float(splitr[3])
            
            gross_inflow = 0
            for k1 in DATA['INFLOW'].keys():
                for k2 in DATA['INFLOW'][k1]:
                    gross_inflow += DATA['INFLOW'][k1][k2]
            total_storage_change = 0
            for k1 in DATA['STORAGE'].keys():
                for k2 in DATA['STORAGE'][k1]:
                    total_storage_change += DATA['STORAGE'][k1][k2]    

            net_inflow = gross_inflow + total_storage_change
            
            landscape_et = sum([DATA['OUTFLOW']['ET LANDSCAPE'][k1] for k1 in DATA['OUTFLOW']['ET LANDSCAPE'].keys()])
            
            exploitable_water = net_inflow - landscape_et
            DeltaS_fresh_water = total_storage_change

            
            utilised_flow = sum([DATA['OUTFLOW']['ET UTILIZED FLOW'][k1] for k1 in DATA['OUTFLOW']['ET UTILIZED FLOW'].keys()])                                     
            reserved_outflows = np.max([DATA['OUTFLOW']['RESERVED'][k1] for k1 in DATA['OUTFLOW']['RESERVED'].keys()])                   
            
            non_utlisiable_outflow = DATA['OUTFLOW']['OTHER']['Non-utilizable']                                 
            available_water = exploitable_water - reserved_outflows - non_utlisiable_outflow
            
            QSWout = sum([DATA['OUTFLOW']['SURFACE WATER'][k1] for k1 in DATA['OUTFLOW']['SURFACE WATER'].keys()])
            QGWout = sum([DATA['OUTFLOW']['GROUNDWATER'][k1] for k1 in DATA['OUTFLOW']['GROUNDWATER'].keys()])
            
            # Calculate Indicators for sheet1 from Karimi et al
            exploitable_water_fraction = exploitable_water/net_inflow
            storage_change_fraction = DeltaS_fresh_water/exploitable_water
            available_water_fraction = available_water/exploitable_water
            basin_closure_fraction = utilised_flow/available_water
            # Added indicators from Wim
            total_precip = sum([DATA['INFLOW']['PRECIPITATION'][k1] for k1 in DATA['INFLOW']['PRECIPITATION'].keys()])
            rainfall_dependency_fraction = total_precip / net_inflow
            utilizable_outflow_fraction = 1-basin_closure_fraction

            
            if (QSWout + QGWout)!=0:
                reserved_outflows_fraction = reserved_outflows/(QSWout + QGWout)
            else:
                reserved_outflows_fraction = -9999
                
        exploitable_water_fractions = np.append(exploitable_water_fractions, exploitable_water_fraction)
        storage_change_fractions = np.append(storage_change_fractions, storage_change_fraction)
        available_water_fractions = np.append(available_water_fractions, available_water_fraction)
        basin_closure_fractions = np.append(basin_closure_fractions, basin_closure_fraction)
        rainfall_dependency_fractions = np.append(rainfall_dependency_fractions, rainfall_dependency_fraction)
        utilizable_outflow_fractions = np.append(utilizable_outflow_fractions, utilizable_outflow_fraction)
        reserved_outflows_fractions = np.append(reserved_outflows_fractions, reserved_outflows_fraction)
        dates = np.append(dates, date)

    sheet1_indicators = {
            'expl._wat.': exploitable_water_fractions,
            'strg_chng.': storage_change_fractions,
            'avlb._wat.': available_water_fractions,
            'bsn._clsr.': basin_closure_fractions,
            #'rainfall_dependency': rainfall_dependency_fractions,
            #'utilizable_outflow': utilizable_outflow_fractions,
            'rsrvd._of.': reserved_outflows_fractions,
            'dates': dates,
            }
    
    return sheet1_indicators

def sheet2_indicators(dir1):
        
    file_list = glob.glob(os.path.join(dir1, "*.csv"))

    transpiration_fractions = np.array([])
    beneficial_fractions  = np.array([])
    managed_fractions = np.array([])
    agricultural_ET_fractions = np.array([])
    irrigated_agricultural_ET_fractions = np.array([])
    dates = np.array([])
            
    for f in file_list:
        
        explod = ''.join(os.path.split(f)[1].split('.')[:-1]).split('_')
        try:
            date = datetime.date(int(explod[-2]), int(explod[-1]), 1)
        except:
            date = datetime.date(int(explod[-1]), 1, 1)  
            
        with open(f) as f1:
            data = csv.reader(f1)    
            DATA = Vividict()
            next(data)
            mline = []
            for row in data:
                splitr= row[0].split(';')
                DATA[splitr[0]][splitr[1]]=np.array(splitr[2:]).astype('float')
                line = np.array(splitr[2:]).astype('float')
                mline = np.append(mline, line)
            m = mline.reshape(len(mline)/len(line),len(line))
            mt = np.transpose(m)   

            T = sum(mt[0])
            ET = sum(mt[0])+sum(mt[1])+sum(mt[2])+sum(mt[3])
            ET_benef = ET - sum(mt[9])
            ET_managed_nc = sum(mt[0][13:19])+sum(mt[1][13:19])+sum(mt[2][13:19])+sum(mt[3][13:19])
            ET_managed_c = sum(mt[0][23:28])+sum(mt[1][23:28])+sum(mt[2][23:28])+sum(mt[3][23:28])     
            agricultural_ET = sum(m[22][0:4])+sum(m[25][0:4])
            irrigated_agricultural_ET = sum(m[25][0:4])

            transpiration_fractions = np.append(transpiration_fractions, T/ET)
            beneficial_fractions  = np.append(beneficial_fractions, (ET_benef)/ET)
            managed_fractions = np.append(managed_fractions, (ET_managed_nc + ET_managed_c)/ET)
            agricultural_ET_fractions = np.append(agricultural_ET_fractions, agricultural_ET/ET)
            irrigated_agricultural_ET_fractions = np.append(irrigated_agricultural_ET_fractions, irrigated_agricultural_ET/agricultural_ET)
        
        dates = np.append(dates, date)
        
    sheet2_indicators = {
            't_fraction': transpiration_fractions,
            'benefi_ET': beneficial_fractions,
            'mngd_ET': managed_fractions,
            'agr_ET': agricultural_ET_fractions,
            'irr_agr_ET': irrigated_agricultural_ET_fractions,
            'dates': dates,
            }
    
    return sheet2_indicators
            
def sheet3_indicators(dir1):

    a_files = glob.glob(dir1+'\sheet3a*.csv')
    b_files = glob.glob(dir1+'\sheet3b*.csv')
    
    files = zip(a_files,b_files)

    land_productivity_cropss = np.array([])
    water_productivity_r_cropss = np.array([])
    water_productivity_i_cropss = np.array([])
    food_irrigation_dependencys = np.array([])
    years = np.array([])
        
    for (fa, fb) in files:
        years = np.append(years, int(fa[-8:-4]))
        
        dfa= pd.read_csv(fa,delimiter=';')
        dfa2 = dfa.loc[~np.isnan(dfa.WATER_CONSUMPTION)]
        
        dfb = pd.read_csv(fb,delimiter=";")
        dfb2 = dfb.loc[~np.isnan(dfb.LAND_PRODUCTIVITY)]
        
        crop_TYPE = ['Cereals','Beverage crops','Feed crops','Fruit & vegetables','Non-cereals','Oilseeds','Other crops']
        production_1 = []
        area_1 =[]
        
        wat_cons = {'RAINFED':[] ,
                    'IRRIGATED':[] }         
        wat_prod = {'RAINFED':[] ,
                    'IRRIGATED':[] }  
        production_totals = {'RAINFED':[] ,
                             'IRRIGATED':[] }                      
        for crop in crop_TYPE:
            dfa_crop = dfa2.loc[dfa2.TYPE == crop]
            df_crop = dfb2.loc[dfb2.TYPE == crop]
            for cl in ['IRRIGATED','RAINFED']:
                dfa_cropclass = dfa_crop.loc[(dfa_crop.CLASS == cl)]
                if cl == 'IRRIGATED':
                    df_cropclass = df_crop.loc[(df_crop.CLASS == cl) & (df_crop.SUBCLASS == 'Total yield')]
                else:
                    df_cropclass = df_crop.loc[(df_crop.CLASS == cl) & (df_crop.SUBCLASS == 'Yield')]
                if not df_cropclass.empty:
                    lp = np.array(df_cropclass.LAND_PRODUCTIVITY)
                    area = np.array(df_cropclass.Crop_Area)
                    ratio = np.array(df_cropclass.Area_ratio_for_DBLCROP)
                    subtype = df_cropclass.SUBTYPE.tolist()
                    for ti in range(len(subtype)):
                        t = subtype[ti]
                        wat_cons[cl].append(np.sum((dfa_cropclass.loc[dfa_cropclass.SUBTYPE == t]).WATER_CONSUMPTION))
                        wat_prod[cl].append(float((df_cropclass.loc[df_cropclass.SUBTYPE == t]).Crop_Area) * float((df_cropclass.loc[df_cropclass.SUBTYPE == t]).LAND_PRODUCTIVITY))
                    
                    production_1.append(np.sum(lp*area)) 
                    area_1.append(area[0]/ratio[0])
                    production_totals[cl].append(np.sum(lp*area))
                    
        land_productivity_crops = np.sum(production_1)/np.sum(area_1)
        water_productivity_r_crops = np.sum(wat_prod['RAINFED'])/np.sum(wat_cons['RAINFED'])/10000000
        water_productivity_i_crops =  np.sum(wat_prod['IRRIGATED'])/np.sum(wat_cons['IRRIGATED'])/10000000
        food_irrigation_dependency = np.sum(production_totals['IRRIGATED'])/(np.sum(production_totals['IRRIGATED'])+np.sum(production_totals['RAINFED']))*100
        
        land_productivity_cropss = np.append(land_productivity_cropss, land_productivity_crops)
        water_productivity_r_cropss = np.append(water_productivity_r_cropss, water_productivity_r_crops)
        water_productivity_i_cropss = np.append(water_productivity_i_cropss, water_productivity_i_crops)
        food_irrigation_dependencys = np.append(food_irrigation_dependencys, food_irrigation_dependency)
        
    return land_productivity_cropss, water_productivity_r_cropss, water_productivity_i_cropss, food_irrigation_dependencys, years
    
def sheet4_indicators(dir1):
             
    files = glob.glob(dir1+'\sheet*.csv')
    
    groundwater_withdrawl_fractions = np.array([])
    irrigation_efficiencys = np.array([])
    recoverable_fractions = np.array([])
    dates = np.array([])
    
    for f in files:
        year = int(f[-11:-7])
        month = int(f[-6:-4])
        dfa  = pd.read_csv(f,delimiter=';')
        
        recoverable = (np.nansum(dfa.RECOVERABLE_SURFACEWATER)+np.nansum(dfa.RECOVERABLE_GROUNDWATER))
        tot_withdrawl = (np.nansum(dfa.SUPPLY_GROUNDWATER)+np.nansum(dfa.SUPPLY_SURFACEWATER))
        df_irrcrop = dfa.loc[(dfa.LANDUSE_TYPE == "Irrigated crops")]
        et_consumption = float(df_irrcrop.CONSUMED_ET)
        
        groundwater_withdrawl_fraction = np.nansum(dfa.SUPPLY_GROUNDWATER)/tot_withdrawl
        
        if et_consumption != 0.0:
            irrigation_efficiency = float(df_irrcrop.CONSUMED_ET) / (float(df_irrcrop.SUPPLY_GROUNDWATER)+float(df_irrcrop.SUPPLY_SURFACEWATER))
        else:
            irrigation_efficiency = np.nan
        recoverable_fraction = recoverable / tot_withdrawl
        
        groundwater_withdrawl_fractions = np.append(groundwater_withdrawl_fractions, groundwater_withdrawl_fraction)
        irrigation_efficiencys = np.append(irrigation_efficiencys, irrigation_efficiency)
        recoverable_fractions = np.append(recoverable_fractions, recoverable_fraction)
        dates = np.append(dates, datetime.date(year, month, 1))
        
    sheet4_indicators = {
            'gw_wthdrwl': groundwater_withdrawl_fractions,
            'irr._fcncy': irrigation_efficiencys,
            'recovarble': recoverable_fractions,
            'dates': dates,
            }
    
    return sheet4_indicators