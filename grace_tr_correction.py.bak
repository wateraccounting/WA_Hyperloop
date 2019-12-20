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
from dateutil.relativedelta import relativedelta

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
                        formula = 'p-et-tr+supply_swa', plot = True,
                        slope = False, bounds = ([0.0, 0.0, 1.], [1., 1., 12.])):
    
    bounds = np.array(bounds)
    
    if plot:
        output_dir = os.path.join(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    keys, operts = split_form(formula)
    
    tss = get_ts_from_complete_data(complete_data, metadata['lu'], keys)
    
    # Load the Grace timeseries.
    grace = read_grace_csv(metadata['GRACE'])
    
    # The monthly datasets are cumulatives, i.e. precipitation for "2003-03-01" shows the precitpiation between 03-01 and 03-31.
    # Grace dates are interpolated and corrected here to match with those cumulatives. The new grace array will have the same length
    # as the other timeseries (in tss), i.e. the array is padded with NaNs.
    grace = interp_ts(grace, (endofmonth(tss[keys[0]][0]), -9999))
    new_dates = np.array([datetime(dt.year, dt.month, 1) for dt in grace[0]])
    grace = (new_dates, grace[1])
    
    # Fit to storage values (slope = False) or to the trend in storage values (slope = True).
    if slope:
        grace = (grace[0], grace[1] - np.nanmean(grace[1]))
    
    # Function to determine the balance
    def func(x, alpha, beta, theta, slope = slope):
        
        ds = np.zeros(x.size)
        
        for key, oper in zip(keys, operts):
            
            scalar_array = alpha * (np.cos((x - theta) * (np.pi / 6)) * 0.5 + 0.5) + (beta * (1 - alpha))
            
            if key == keys[-1]:
                new_data = tss[key][1][x] * scalar_array
            else:
                new_data = tss[key][1][x]
            
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
    
    # Check which Grace datapoints are missing and make a list of indice-numbers for which
    # data is available.
    msk = ~np.isnan(grace[1])
    x = np.where(msk == True)[0].astype(int).tolist()
    
    print "Starting alpha optimization"
    
    # Initial values for alpha, beta and theta
    p0 = (bounds[0] + bounds[1])/2.
    
    # Find values for alpha, beta and theta so that the waterbalance matches as closely
    # as possible with Grace.
    a, b = optimization.curve_fit(func, x, grace[1][msk], p0 = p0 , bounds= bounds)
    
    if plot:
        
        var_name = keys[-1]
        
        plt.figure(1, figsize = (10,6))
        plt.clf()
        plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 100)
        ax = plt.gca()
        plt.plot(*tss[var_name], label = "Total Supply", color = 'k')
        plt.xlabel("Time [date]")
        plt.ylabel("Total Supply [mm]")
        ax.fill_between(*tss[var_name], color = '#6bb8cc', label = 'SW Supply')
        ax.fill_between(*calc_gwsupply(tss[var_name], a), color = '#c48211', label = 'GW Supply')
        ax.set_facecolor('lightgray')          
        plt.scatter(*tss[var_name], color = 'k')
        plt.ylim([0, np.nanmax(tss[var_name][1]) * 1.2])
        plt.xlim([tss[var_name][0][0],tss[var_name][0][-1]])
        plt.title('Surface and Groundwater Supplies')
        plt.legend()
        plt.savefig(os.path.join(output_dir, metadata['name'], 'SWGW_Supply'))
        
        plt.figure(2, figsize = (10,6))
        plt.clf()
        plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
        ax = plt.gca()
        ax.set_facecolor('lightgray')
        X = np.arange(tss[var_name][0].size)
        scalar_array = a[0] * (np.cos((X - a[2]) * (np.pi / 6)) * 0.5 + 0.5) + (a[1] * (1 - a[0]))
        plt.plot(tss[var_name][0], scalar_array, color = 'darkblue')
        plt.scatter(tss[var_name][0], scalar_array, color = 'darkblue')
        plt.ylim([0, 1])
        plt.xlim([tss[var_name][0][0],tss[var_name][0][-1]])
        plt.xlabel("Time [date]")
        plt.ylabel(r"$Supply_{SW}\ /\ Supply_{tot}\ [-]$")
        plt.suptitle(r"$Supply_{SW} = (\alpha \cdot [\cos(t \cdot \frac{2\pi}{12} - \theta) \cdot \frac{1}{2} + \frac{1}{2}] + \beta \cdot (1 - \alpha)) \cdot Supply_{tot}$" + '\n' + r"$\alpha = $" + 
                     '{0:.2f}'.format(a[0]) + r', $\beta = $' + 
                     '{0:.2f}'.format(a[1]) + r', $\theta = $' + 
                     '{0:.2f}'.format(a[2]))
        plt.savefig(os.path.join(output_dir, metadata['name'], 'SWGW_Ratio'))
        
        plt.figure(3, figsize = (10,6))
        plt.clf()
        plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
        ax = plt.gca()
        ax.set_facecolor('lightgray')  
        plt.plot(*grace, label = 'GRACE', color = 'r')
        plt.plot(*calc_polyfit(grace, order = 1), color = 'r', linestyle = ':')
        
        if metadata['lu_based_supply_split']:
            plt.plot(grace[0], func(X, 0., 1., 1., slope = False), color = 'darkblue', label = 'WAplus (lu)')
            plt.plot(*calc_polyfit((grace[0], func(X, 0., 1., 1., slope = False)), order = 1), color = 'darkblue', linestyle = ':')
            plt.plot(grace[0], func(X, a[0], a[1], a[2], slope = False), color = 'k', label = 'WAplus (lu + grace)')
            plt.plot(*calc_polyfit((grace[0], func(X, a[0], a[1], a[2], slope = False)), order = 1), color = 'k', linestyle = ':')
        else:
            plt.plot(grace[0], func(X, a[0], a[1], a[2], slope = False), color = 'k', label = 'WAplus (grace)')
            plt.plot(*calc_polyfit((grace[0], func(X, a[0], a[1], a[2], slope = False)), order = 1), color = 'k', linestyle = ':')            
            
        plt.legend()
        plt.xlim([grace[0][0], grace[0][-1]])
        plt.ylabel('dS/dt [mm/month]')
        plt.xlabel("Time [date]")
        plt.savefig(os.path.join(output_dir, metadata['name'], 'dSdt_Grace'))

    x0 = tss[keys[0]][0][0]
    
    return a, x0


def calc_gwsupply(total_supply, params):
    x = range(len(total_supply[0]))
    scalar_array = params[0] * (np.cos((x - params[2]) * (np.pi / 6)) * 0.5 + 0.5) + (params[1] * (1 - params[0]))
    gw_supply = (total_supply[0], total_supply[1] - (total_supply[1] * scalar_array))
    return gw_supply
    
def correct_var(metadata, complete_data, output_dir, formula,
                new_var, slope = False, bounds = (0, [1.0, 1., 12.])):
    
    var = split_form(formula)[0][-1]
    
    a, x0 = calc_var_correction(metadata, complete_data, output_dir,
                            formula = formula, slope = slope, plot = True, bounds = bounds)
    
    for date, fn in zip(complete_data[var][1], complete_data[var][0]):
        
        geo_info = becgis.GetGeoInfo(fn)
        
        data = becgis.OpenAsArray(fn, nan_values = True)
        
        x = calc_delta_months(x0, date)
        
        fraction = a[0] * (np.cos((x - a[2]) * (np.pi / 6)) * 0.5 + 0.5) + (a[1] * (1 - a[0]))
        
        data *= fraction
        
        folder = os.path.join(output_dir, metadata['name'],
                              'data', new_var)
        
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        bla = os.path.split(fn)[1].split('_')[-1]
        filen = 'supply_sw_' + bla[0:4] + '_' + bla[4:6] + '.tif'
        fn = os.path.join(folder, filen)
            
        becgis.CreateGeoTiff(fn, data, *geo_info)
        
    meta = becgis.SortFiles(folder, [-11,-7], month_position = [-6,-4])[0:2]
    return a, meta

def calc_delta_months(x0, date):
    
    if isinstance(x0, datetime):
        x0 = x0.date()
    if isinstance(date, datetime):
        date = date.date()
        
    x = 0
    checker = x0
    direction = {False: -1, True: 1}[date > x0]
    while checker != date:
        checker += relativedelta(months = 1) * direction
        x += 1 * direction
    return x

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
