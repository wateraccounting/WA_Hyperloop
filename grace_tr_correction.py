# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:47:04 2017

@author: bec
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import scipy.optimize as optimization
from datetime import datetime
from scipy import interpolate
from calendar import monthrange as mr
from WA_Hyperloop import becgis

import WA_Hyperloop.get_dictionaries as gd


def get_ts_from_complete_data(complete_data, mask, keys, dates = None):
    
    if keys == None:
        keys = complete_data.keys()
    
    common_dates = becgis.CommonDates([complete_data[key][1] for key in keys])
    becgis.AssertProjResNDV([complete_data[key][0] for key in keys])
    
    MASK = becgis.OpenAsArray(mask, nan_values = True)
    
    tss = dict()
    
    for key in keys:
        
        var_mm = np.array([])
        
        for date in common_dates:
            
            tif = complete_data[key][0][complete_data[key][1] == date][0]
            
            DATA = becgis.OpenAsArray(tif, nan_values = True)
            DATA[np.isnan(DATA)] = 0.0
            
            DATA[np.isnan(MASK)] = np.nan
            
            var_mm = np.append(var_mm, np.nanmean(DATA))
        
        tss[key] = (common_dates, var_mm)

    return tss


def get_ts_from_complete_data_spec(complete_data, lu_fh, keys, a, dates = None):
    
    if keys == None:
        keys = complete_data.keys()
    
    common_dates = becgis.CommonDates([complete_data[key][1] for key in keys])
    becgis.AssertProjResNDV([complete_data[key][0] for key in keys])
    
    MASK = becgis.OpenAsArray(lu_fh, nan_values = True)
    
    lucs = lucs = gd.get_sheet4_6_classes()
    gw_classes = list()
    for subclass in ['Forests','Rainfed Crops','Shrubland','Forest Plantations']:
        gw_classes += lucs[subclass]
    mask_gw = np.logical_or.reduce([MASK == value for value in gw_classes])
    
    tss = dict()
    
    for key in keys:
        
        var_mm = np.array([])
        
        for date in common_dates:
            
            tif = complete_data[key][0][complete_data[key][1] == date][0]
            
            DATA = becgis.OpenAsArray(tif, nan_values = True)
            DATA[np.isnan(DATA)] = 0.0
            
            DATA[np.isnan(MASK)] = np.nan
            
            alpha = np.ones(np.shape(DATA)) * a
            
            alpha[mask_gw] = 0.0
            
            var_mm = np.append(var_mm, np.nanmean(DATA * alpha))
        
        tss[key] = (common_dates, var_mm)

    return tss


def endofmonth(dates):
    dts = np.array([datetime(dt.year, dt.month, 
                             mr(dt.year, dt.month)[1]) for dt in dates])
    return dts
    

def calc_var_correction(metadata, complete_data, output_dir,
                        formula = 'p-et-tr+supply_total', plot = True,
                        slope = True, return_slope = False):
    
    if plot:
        output_dir = os.path.join(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    keys, operts = split_form(formula)
    
    tss = get_ts_from_complete_data(complete_data, metadata['lu'], keys)
    
    grace = read_grace_csv(metadata['GRACE'])
    grace = interp_ts(grace, (endofmonth(tss[keys[0]][0]), -9999))
    
    new_dates = np.array([datetime(dt.year, dt.month, 1) for dt in grace[0]])
    
    grace = (new_dates, grace[1])
    
    if slope:
        grace = (grace[0], grace[1] - np.mean(grace[1]))
    
    msk = ~np.isnan(grace[1])

    def func(x, alpha, beta, theta, slope = slope):
        
        ds = np.zeros(msk.sum())
        for key, oper in zip(keys, operts):
            
            scalar_array = alpha * (np.cos((x - theta) * (np.pi / 6)) * 0.5 + 0.5) + (beta * (1 - alpha))
            
            if key == keys[-1]:
                new_data = tss[key][1][msk][x] * scalar_array
            else:
                new_data = tss[key][1][msk][x]
            
            if oper == '+':
                ds += new_data
            elif oper == '-':
                ds -= new_data
            elif oper == '/':
                ds /= new_data
            elif oper == '*':
                ds *= new_data
            else:
                raise ValueError('Unknown operator in formula')
        
        if slope:
            return np.cumsum(ds) - np.mean(np.cumsum(ds))
        else:
            return np.cumsum(ds)
        
    x = np.where(msk == True)[0].astype(int).tolist()
    
    x0 = (x[0], grace[0][msk][0])
    
    print "Starting alpha optimization"
    
    p0 = [0.0, 0.5, 6.0]
    
    a, b = optimization.curve_fit(func, x, grace[1][msk], p0 = p0 , bounds=(0, [1.0, 1., 12.]))                   
    
    if plot:
        plt.figure(1)
        plt.clf()
        plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 100)
        ax = plt.gca()
        plt.plot(*tss['supply_total'], label = 'Total Supply', color = 'k')
        
        ax.fill_between(*tss['supply_total'], color = '#6bb8cc', label = 'SW Supply')
        ax.fill_between(*calc_gwsupply(tss['supply_total'], a), color = '#c48211', label = 'GW Supply')
                        
        plt.scatter(*tss['supply_total'], color = 'k')
        plt.ylim([0, np.nanmax(tss['supply_total'][1]) * 1.2])
        plt.xlim([tss['supply_total'][0][0],tss['supply_total'][0][-1]])
        plt.title('Surface and Groundwater Supplies')
        plt.legend()
        plt.savefig(os.path.join(output_dir, metadata['name'], 'SWGW_Supply'))
        
        plt.figure(2)
        plt.clf()
        plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
        ax = plt.gca()
        scalar_array = a[0] * (np.cos((x - a[2]) * (np.pi / 6)) * 0.5 + 0.5) + (a[1] * (1 - a[0]))
        plt.plot(tss['supply_total'][0], scalar_array, color = 'k')
        plt.ylim([0, 1])
        plt.xlim([tss['supply_total'][0][0],tss['supply_total'][0][-1]])
        #plt.title('SW:GW Ratio')
    #    plt.suptitle(r'$SW = (\alpha \cdot [\cos(t \cdot \frac{2\pi}{12} - \
    #                 \theta) \cdot \frac{1}{2} + \frac{1}{2}] + \beta \cdot \
    #                 (1 - \alpha)) \cdot TS$' + '\n' + r'$\alpha = $' + 
    #                 '{0:.2f}'.format(a[0]) + r', $\beta = $' + 
    #                 '{0:.2f}'.format(a[1]) + r', $\theta = $' + 
    #                 '{0:.2f}'.format(a[2]))
        plt.savefig(os.path.join(output_dir, metadata['name'], 'SWGW_Ratio'))

    grace = read_grace_csv(metadata['GRACE'])
    grace = interp_ts(grace, (endofmonth(tss[keys[0]][0]), -9999))
    new_dates = np.array([datetime(dt.year, dt.month, 1) for dt in grace[0]])
    grace = (new_dates, grace[1])
    
    if plot:
        
        plt.figure(3)
        plt.clf()
        plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
        ax = plt.gca()
        plt.plot(*grace, label = 'GRACE', color = 'r')
        plt.plot(*calc_polyfit(grace, order = 1), color = 'r', linestyle = ':')
        plt.plot(grace[0], func(x, a[0], a[1], a[2], slope = False), color = 'k', label = 'WAplus')
        plt.plot(*calc_polyfit((grace[0], func(x, a[0], a[1], a[2], slope = False)), order = 1), color = 'k', linestyle = ':')
        plt.legend()
        plt.xlim([grace[0][0], grace[0][-1]])
        plt.ylabel('dS/dt [mm/month]')    
        plt.savefig(os.path.join(output_dir, metadata['name'], 'dSdt_Grace'))
        
    if return_slope:
        ts = (grace[0], func(x, a[0], a[1], a[2], slope = False))
        dts_ordinal = toord(ts)
        p_WA = np.polyfit(dts_ordinal[~np.isnan(ts[1])],
                                   ts[1][~np.isnan(ts[1])], 1)
        
        dt = dts_ordinal[-1] - dts_ordinal[0]
        
        print "dS/dt = {0} mm / day".format(p_WA[0])
        print "dS = dS/dt * {0} = {1}".format(dt, p_WA[0]*dt)
        
        ts = grace
        dts_ordinal = toord(ts)
        p_GRACE = np.polyfit(dts_ordinal[~np.isnan(ts[1])],
                                   ts[1][~np.isnan(ts[1])], 1)
        
        print "dS/dt_GRACE = {0} mm / day".format(p_GRACE[0]) 
        print "dS = dS/dt * {0} = {1}".format(dt, p_GRACE[0]*dt)
        
        startend = (grace[0][0], grace[0][-1])
        
        return p_WA, p_GRACE, dt, startend
    
    else:
        return a, x0


def calc_gwsupply(total_supply, params):
    x = range(len(total_supply[0]))
    scalar_array = params[0] * (np.cos((x - params[2]) * (np.pi / 6)) * 0.5 + 0.5) + (params[1] * (1 - params[0]))
    gw_supply = (total_supply[0], total_supply[1] - (total_supply[1] * scalar_array))
    return gw_supply


def correct_var_test(metadata, complete_data, output_dir, formula,
                new_var = None, slope = False):
    
    var = split_form(formula)[0][-1]
    
    a, x0 = calc_var_correction(metadata, complete_data, output_dir,
                            formula = formula, slope = slope)
    
    LULC = becgis.OpenAsArray(metadata['lu'])
    geo_info = becgis.GetGeoInfo(metadata['lu'])
    lucs = gd.get_sheet4_6_classes()
    
    gw_classes = list()
    for subclass in ['Forests','Rainfed Crops','Shrubland','Forest Plantations']:
        gw_classes += lucs[subclass]
        
    mask_gw = np.logical_or.reduce([LULC == value for value in gw_classes])
            
    for fn, date in zip(complete_data[var][0], complete_data[var][1]):
        
        time_scale = 'monthly'
        if time_scale == 'monthly':
            x = 12 * (date.year - x0[1].year) + (date.month - 1) + x0[0]
        
        scalar = a[0] * (np.cos((x - a[2]) * (np.pi / 6)) * 0.5 + 0.5) + (a[1] * (1 - a[0]))
        
        geo_info = becgis.GetGeoInfo(fn)
        
        data = becgis.OpenAsArray(fn, nan_values = True)
        
        scalar_array = np.ones(np.shape(LULC)) * scalar
        #scalar_array[mask_gw] = 0.0
        
        data *= scalar_array
        
        if new_var != None:
            flder = os.path.join(output_dir, metadata['name'], 'data', new_var)
            if not os.path.exists(flder):
                os.makedirs(flder)
            bla = os.path.split(fn)[1].split('_')[-1]
            filen = 'supply_sw_' + bla[0:4] + '_' + bla[4:6] + '.tif'
            fn = os.path.join(flder, filen)
        becgis.CreateGeoTiff(fn, data, *geo_info)
        
    if new_var != None:
        meta = becgis.SortFiles(flder, [-11,-7], month_position = [-6,-4])[0:2]
        return a, meta
    else:
        return a
    
def correct_var(metadata, complete_data, output_dir, formula,
                new_var = None, slope = False):
    var = split_form(formula)[0][-1]
    a = calc_var_correction(metadata, complete_data, output_dir,
                            formula = formula, slope = slope, plot = True)
    for fn in complete_data[var][0]:
        geo_info = becgis.GetGeoInfo(fn)
        data = becgis.OpenAsArray(fn, nan_values = True) * a[0]
        if new_var != None:
            folder = os.path.join(output_dir, metadata['name'],
                                  'data', new_var)
            if not os.path.exists(folder):
                os.makedirs(folder)
            bla = os.path.split(fn)[1].split('_')[-1]
            filen = 'supply_sw_' + bla[0:4] + '_' + bla[4:6] + '.tif'
            fn = os.path.join(folder, filen)
        becgis.CreateGeoTiff(fn, data, *geo_info)
    if new_var != None:
        meta = becgis.SortFiles(folder, [-11,-7], month_position = [-6,-4])[0:2]
        return a, meta
    else:
        return a


def toord(dates):
    return np.array([dt.toordinal() for dt in dates[0]])


def interp_ts(source_ts, destin_ts):
    f = interpolate.interp1d(toord(source_ts), source_ts[1], 
                             bounds_error = False, fill_value = np.nan)
    values = f(toord(destin_ts))
    return (destin_ts[0], values)


def read_grace_csv(csv_file):
    
    df = pd.read_csv(csv_file)
    dt = [datetime.strptime(dt, '%Y-%m-%d').date() for dt in df['date'].values]
    grace_mm = np.array(df['dS [mm]'].values)
    
    return np.array(dt), grace_mm


def calc_polyfit(ts, order = 1):
    dts_ordinal = toord(ts)
    p = np.polyfit(dts_ordinal[~np.isnan(ts[1])],
                               ts[1][~np.isnan(ts[1])], order)
    vals = np.polyval(p, dts_ordinal)
    dts = [datetime.fromordinal(dt).date() for dt in dts_ordinal]
    return (dts, vals)


def calc_form(tss, formula):
    
    varias, operts = split_form(formula)
        
    assert len(varias) == len(operts)
    
    ds = np.zeros(tss[varias[0]][0].shape)
    
    for vari, oper in zip(varias, operts):
        
        if oper == '+':
            ds += tss[vari][1]
        elif oper == '-':
            ds -= tss[vari][1]
        elif oper == '/':
            ds /= tss[vari][1]
        elif oper == '*':
            ds *= tss[vari][1]
        else:
            raise ValueError('Unknown operator in formula')
    
    return (tss[varias[0]][0], ds)


def split_form(formula):
    varias = re.split("\W", formula)
    operts = re.split("[a-z]+", formula, flags=re.IGNORECASE)[1:-1]
    
    while '_' in operts:
        operts.remove('_')
        
    if len(varias) > len(operts):
        operts.insert(0, '+')
    
    return varias, operts






