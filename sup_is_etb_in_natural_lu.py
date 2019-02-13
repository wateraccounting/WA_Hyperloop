# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:03:36 2018

@author: cmi001
"""
import os
import numpy as np
import becgis
import WA_Hyperloop.get_dictionaries as gd

def supply_return_natural_lu(metadata, complete_data):
    
    lu_tif = metadata['lu']
    LULC = becgis.OpenAsArray(lu_tif, nan_values = True)
    lucs = gd.get_sheet4_6_classes()
    
    #new directories:
    directory_sup = os.path.split(complete_data['supply_total'][0][0])[0]+'_corr'
    directory_dro = os.path.split(complete_data['dro'][0][0])[0]+'_corr'
    directory_dperc = os.path.split(complete_data['dperc'][0][0])[0]+'_corr'
    directory_sro = os.path.split(complete_data['sr'][0][0])[0]+'_corr'
    directory_tr = os.path.split(complete_data['tr'][0][0])[0]+'_corr'
    
    if not os.path.exists(directory_sup):
        os.makedirs(directory_sup)
    if not os.path.exists(directory_dro):
        os.makedirs(directory_dro)
    if not os.path.exists(directory_dperc):
        os.makedirs(directory_dperc)
    if not os.path.exists(directory_sro):
        os.makedirs(directory_sro)
    if not os.path.exists(directory_tr):
        os.makedirs(directory_tr)
# 
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_tif)
    
    common_dates = becgis.CommonDates([complete_data['supply_total'][1], #complete_data['etb'][1], 
                                      complete_data['dro'][1], 
                                      complete_data['dperc'][1], #,
                                      complete_data['tr'][1]])
    for date in common_dates:        
        total_supply_tif = complete_data['supply_total'][0][complete_data['supply_total'][1] == date][0]
        SUP = becgis.OpenAsArray(total_supply_tif, nan_values = True)
        
        dperc_tif = complete_data['dperc'][0][complete_data['dperc'][1] == date][0]
        DPERC = becgis.OpenAsArray(dperc_tif, nan_values = True)
        DPERC[np.isnan(DPERC)] = 0        
        
        dro_tif = complete_data['dro'][0][complete_data['dro'][1] == date][0]
        DRO = becgis.OpenAsArray(dro_tif, nan_values = True)
        DRO[np.isnan(DRO)] = 0
        
        sro_tif = complete_data['sr'][0][complete_data['sr'][1] == date][0]
        SRO = becgis.OpenAsArray(sro_tif, nan_values = True)
        SRO[np.isnan(SRO)] = 0
        
#        et_blue_tif = complete_data['etb'][0][complete_data['etb'][1] == date][0]
#        ETB = becgis.OpenAsArray(et_blue_tif, nan_values = True)

        tr_tif = complete_data['tr'][0][complete_data['tr'][1] == date][0]
        TR = becgis.OpenAsArray(tr_tif, nan_values = True)
        
#        perc_tif = complete_data['perc'][0][complete_data['perc'][1] == date][0]
#        PERC = becgis.OpenAsArray(perc_tif, nan_values = True)
        
        natural_lus = ['Forests',
                       'Shrubland',
                       'Rainfed Crops',
                       'Forest Plantations',
                       'Natural Water Bodies',
                       'Wetlands',
                       'Natural Grasslands',
                       'Other (Non-Manmade)']
        natural_lu_codes = []
        for lu_c in natural_lus:
            natural_lu_codes.extend(lucs[lu_c])
        for code in natural_lu_codes:
#            PERC[LULC == code] = PERC[LULC == code] - DPERC[LULC == code]
            TR[LULC == code] = TR[LULC == code] - DRO[LULC == code]
            SRO[LULC == code] = SRO[LULC == code] - DRO[LULC == code]
            SUP[LULC == code] = SUP[LULC == code] - DPERC[LULC == code] - DRO[LULC == code]
            DRO[LULC == code] = 0
            DPERC[LULC == code] = 0
        
        
        outfile_sup = os.path.join(directory_sup, os.path.basename(total_supply_tif))
        becgis.CreateGeoTiff(outfile_sup, SUP, driver, NDV, xsize, ysize, GeoT, Projection)
        
        outfile_dro = os.path.join(directory_dro, os.path.basename(dro_tif))
        becgis.CreateGeoTiff(outfile_dro, DRO, driver, NDV, xsize, ysize, GeoT, Projection)
        
        outfile_sro = os.path.join(directory_sro, os.path.basename(sro_tif))
        becgis.CreateGeoTiff(outfile_sro, SRO, driver, NDV, xsize, ysize, GeoT, Projection)

        outfile_dperc = os.path.join(directory_dperc, os.path.basename(dperc_tif))
        becgis.CreateGeoTiff(outfile_dperc, DPERC, driver, NDV, xsize, ysize, GeoT, Projection)
        
#        outfile_perc = os.path.join(directory_perc, os.path.basename(perc_tif))
#        becgis.CreateGeoTiff(outfile_perc, PERC, driver, NDV, xsize, ysize, GeoT, Projection)

        outfile_tr = os.path.join(directory_tr, os.path.basename(tr_tif))
        becgis.CreateGeoTiff(outfile_tr, TR, driver, NDV, xsize, ysize, GeoT, Projection)


    complete_data['supply_total'] = becgis.SortFiles(directory_sup, [-10,-6], month_position = [-6,-4])[0:2]
    complete_data['dro'] = becgis.SortFiles(directory_dro, [-10,-6], month_position = [-6,-4])[0:2]
    complete_data['sr'] = becgis.SortFiles(directory_sro, [-10,-6], month_position = [-6,-4])[0:2]
    complete_data['dperc'] = becgis.SortFiles(directory_dperc, [-10,-6], month_position = [-6,-4])[0:2]
#    complete_data['perc'] = becgis.SortFiles(directory_perc, [-10,-6], month_position = [-6,-4])[0:2]
    complete_data['tr'] = becgis.SortFiles(directory_tr, [-10,-6], month_position = [-6,-4])[0:2]

    return complete_data

def bf_reduction_with_gwsup(metadata, complete_data):
    lu_tif = metadata['lu']
    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_tif)
    #new directories:
    directory_bf = os.path.split(complete_data['bf'][0][0])[0]+'_corrbf'
    if not os.path.exists(directory_bf):
        os.makedirs(directory_bf)
    directory_ro = os.path.split(complete_data['tr'][0][0])[0]+'_corrbf'
    if not os.path.exists(directory_ro):
        os.makedirs(directory_ro)
        
    common_dates = becgis.CommonDates([complete_data['supply_gw'][1],
                                      complete_data['bf'][1]])

    for date in common_dates:        
        gw_supply_tif = complete_data['supply_gw'][0][complete_data['supply_gw'][1] == date][0]
        SUP_GW = becgis.OpenAsArray(gw_supply_tif, nan_values = True)
        
        bf_tif = complete_data['bf'][0][complete_data['bf'][1] == date][0]
        BF = becgis.OpenAsArray(bf_tif, nan_values = True)

        ro_tif = complete_data['tr'][0][complete_data['tr'][1] == date][0]

        sro_tif = complete_data['sr'][0][complete_data['sr'][1] == date][0]
        SRO = becgis.OpenAsArray(sro_tif, nan_values = True)
        
        BF_new = BF - SUP_GW
        BF_new[BF_new < 0] = 0.
        
        RO_new = BF_new + SRO
        
        outfile_bf = os.path.join(directory_bf, os.path.basename(bf_tif))
        becgis.CreateGeoTiff(outfile_bf, BF_new, driver, NDV, xsize, ysize, GeoT, Projection)
        outfile_tr = os.path.join(directory_ro, os.path.basename(ro_tif))
        becgis.CreateGeoTiff(outfile_tr, RO_new, driver, NDV, xsize, ysize, GeoT, Projection)
    complete_data['bf'] = becgis.SortFiles(directory_bf, [-10,-6], month_position = [-6,-4])[0:2]
    complete_data['tr'] = becgis.SortFiles(directory_ro, [-10,-6], month_position = [-6,-4])[0:2]
    return complete_data
