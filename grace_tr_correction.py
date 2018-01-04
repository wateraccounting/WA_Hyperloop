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
from WA_Hyperloop import becgis


def get_ts_from_complete_data(complete_data, mask, keys):
    
    common_dates = becgis.CommonDates([complete_data[key][1] for key in keys])
    becgis.AssertProjResNDV([complete_data[key][0] for key in keys])
    
    MASK = becgis.OpenAsArray(mask, nan_values = True)
    
    tss = dict()
    
    for key in keys:
        
        var_mm = np.array([])
        
        for date in common_dates:
            
            tif = complete_data[key][0][complete_data[key][1] == date][0]
            
            DATA = becgis.OpenAsArray(tif, nan_values = True)
            
            DATA[np.isnan(MASK)] = np.nan
            
            var_mm = np.append(var_mm, np.nanmean(DATA))
        
        tss[key] = (common_dates, var_mm)

    return tss


def calc_tr_correction(metadata, complete_data, output_dir, plot = True):

    a_guess = np.array([1.0, 0.0])
    
    output_dir = os.path.join(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    formula_dsdt = 'p-et-tr'
    keys, operts = split_form(formula_dsdt)
    
    tss = get_ts_from_complete_data(complete_data, metadata['lu'], keys)
    
    grace = read_grace_csv(metadata['GRACE'])
    dgracedt = (grace[0][1:], np.diff(grace[1]))
    dgracedt = interp_ts(dgracedt, tss[keys[0]])

    msk = ~np.isnan(dgracedt[1])


    def func(x, a, b):
        
        ds = np.zeros(msk.sum())
        for key, oper in zip(keys, operts):
            if key == 'tr':
                new_data = tss[key][1][msk][x] * a + b
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

        return ds


    x = range(msk.sum())
    
    a,b = optimization.curve_fit(func, x, dgracedt[1][msk], a_guess)
    
    tss['trnew'] = (tss['tr'][0], a[0]*tss['tr'][1] + a[1])
    formula_dsdt_new = formula_dsdt.replace('tr', 'trnew')
    
    path = os.path.join(output_dir, metadata['name'], metadata['name'])
    
    if plot:
        plot_storage(tss, formula_dsdt, dgracedt, a_guess)
        plt.savefig(path + '_orig_dsdt.png')
        plt.close()
        
        plot_cums(tss, dgracedt, formula_dsdt, formula_dsdt_new)
        plt.savefig(path + '_cumulatives_wb.png')
        plt.close()
        
        plot_storage(tss, formula_dsdt_new, dgracedt, a)
        plt.savefig(path + '_corr_dsdt.png')
        plt.close()

    return a


def correct_tr(metadata, complete_data, output_dir):
    a = calc_tr_correction(metadata, complete_data, output_dir, plot = True)
    for fn in complete_data['tr'][0]:
        geo_info = becgis.GetGeoInfo(fn)
        data = np.maximum(0.0, becgis.OpenAsArray(fn, nan_values = True) * a[0] + a[1])
        becgis.CreateGeoTiff(fn, data, *geo_info)


def toord(variable):
    return [dt.toordinal() for dt in variable[0]]


def interp_ts(source_ts, destin_ts):
    dts_ordinal = [dt.toordinal() for dt in source_ts[0]]
    f = interpolate.interp1d(dts_ordinal, source_ts[1], 
                             bounds_error = False, fill_value = np.nan)
    dts_destin_ordinal = [dt.toordinal() for dt in destin_ts[0]]
    return (destin_ts[0], f(dts_destin_ordinal))


def read_grace_csv(csv_file):
    
    from datetime import datetime as dtim
    
    df = pd.read_csv(csv_file)
    dt = [dtim.strptime(dt, '%Y-%m-%d').date() for dt in df['date'].values]
    grace_mm = np.array(df['dS [mm]'].values)
    
    return np.array(dt), grace_mm
    

def plot_storage(tss, formula, dgracedt, a):

    varias = re.split("\W", formula)
    
    plt.figure(figsize = (10,8))
    plt.clf()
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    linestyles = ['-', '--', '-.', ':']
    
    for i, vari in enumerate(varias):
        ax1.plot(*tss[vari], linestyle=linestyles[i], color='k', label=vari)
    
    dstoragedt = calc_form(tss, formula)
    storage = (dstoragedt[0], np.cumsum(dstoragedt[1]))
    
    ax2.plot(*storage, linestyle='-', color='r', label='Storage (WB)')
    ax2.plot(*calc_polyfit(storage), linestyle=':', color='r')
    
    if dgracedt:
        grace = (dgracedt[0][~np.isnan(dgracedt[1])], 
                 np.cumsum(dgracedt[1][~np.isnan(dgracedt[1])]))
        ax2.plot(*grace, linestyle='--', color='r', label='Storage (GRACE)')
        ax2.plot(*calc_polyfit(grace), linestyle=':', color='r')
    
    ax1.set_ylabel('Flux [mm/month]')
    ax2.set_ylabel('S [mm]')
    
    plt.xlabel('Time')
    plt.title('dSdt = {0}, {1} = {2} * tr + {3}'.format(formula,
              varias[-1], a[0], a[1]))
    ax1.legend(loc = 'upper left')
    ax2.legend(loc = 'upper right')


def plot_cums(tss, dsdt, formula_dsdt, formula_dsdt_new):
    plt.figure(figsize = (10,8))
    plt.clf()
    plt.grid(b=True, which='Major', color='0.65',linestyle='--', zorder = 0)
    
    plt.plot(tss['p'][0], np.cumsum(tss['p'][1]), label = '$\sum P$')
    plt.plot(tss['et'][0], np.cumsum(tss['et'][1]), label = '$\sum ET$')
    plt.plot(tss['tr'][0], np.cumsum(tss['tr'][1]), label = '$\sum TR$')
    
    plt.plot(tss['trnew'][0], np.cumsum(tss['trnew'][1]),
             label = '$\sum TR_{corr}$')
    
    msk = ~np.isnan(dsdt[1])
    plt.plot(dsdt[0][msk], np.cumsum(dsdt[1][msk]), label = 'S_grace')
    
    dsdt_orig = calc_form(tss, formula_dsdt)
    plt.plot(dsdt_orig[0], np.cumsum(dsdt_orig[1]), label = 'S_orig')
    
    dsdt_new = calc_form(tss, formula_dsdt_new)
    plt.plot(dsdt_new[0], np.cumsum(dsdt_new[1]), label = 'S_corr')
    
    plt.xlabel('Time')
    plt.ylabel('Stock [mm]')
    plt.legend()  


def calc_polyfit(ts, order = 1):
    dts_ordinal = np.array([dt.toordinal() for dt in ts[0]])
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
    
    if len(varias) > len(operts):
        operts.insert(0, '+')
    
    return varias, operts






