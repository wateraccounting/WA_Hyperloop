# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:33:21 2017

@author: cmi001
"""
import os
import ogr
import numpy as np
import netCDF4 as nc
import csv
import datetime
import pandas as pd
import wa.General.raster_conversions as RC
from WA_Hyperloop.sheet1_functions import sheet1_functions as sh1
import WA_Hyperloop.becgis as becgis
import xml.etree.ElementTree as ET
from WA_Hyperloop.sheet4_functions import sheet4_functions as sh4
from WA_Hyperloop.sheet7_functions import sheet7_functions as sh7
import subprocess

def create_sheet5(complete_data, metadata, output_dir, global_data, template = r"C:\Users\bec\Dropbox\UNESCO\Scripts\claire\sheet5\sheet5_blank_template_new.svg"):
    
    if type(metadata['SWfile']) == list:
        SW_dates = np.array([datetime.date.fromordinal(t) for t in nc.MFDataset(metadata['SWfile']).variables['time'][:]])
    else:
        SW_dates = np.array([datetime.date.fromordinal(t) for t in nc.Dataset(metadata['SWfile']).variables['time'][:]])
    
    date_list = becgis.CommonDates([complete_data['tr'][1],complete_data['supply_sw'][1], complete_data['return_flow_gw_sw'][1], complete_data['return_flow_sw_sw'][1],
                                    complete_data['sr'][1],complete_data['bf'][1], complete_data['fractions'][1], SW_dates])
    
    date_list = becgis.ConvertDatetimeDate(date_list, out = 'datetime')
    man_dict = dictionary()
    
    #landuse
    lu_fh  = metadata['lu']
    output_folder = os.path.join(output_dir, metadata['name'])
    #Subbasin mask - Name structure: subbasincode_subbasinname.shp
    
    sb_temp =  os.path.join(output_folder, 'temp_sb')
    sb_fhs = becgis.ListFilesInFolder(metadata['masks'])
    sb_fhs = becgis.MatchProjResNDV(lu_fh, sb_fhs, sb_temp)
    
    fnames = [sb_fh.split('\\')[-1] for sb_fh in sb_fhs]
    sb_codes = [fname.split('_')[0] for fname in fnames]
    sb_names = [fname.split('_')[1] for fname in fnames]
    sb_names = [sb_name.split('.')[0] for sb_name in sb_names]
    sb_fhs_code_names = zip(sb_fhs,sb_codes,sb_names)
    #Outlets .shp
    outlets = metadata['outflow_nodes'] #Shapefile's 2nd field (after id) should contain the name of the subbasin as written in sb_names
    
    # subbasin connectivity dictionaries
    dico_in = metadata['dico_in']
    dico_out = metadata['dico_out']
    
    fractions_fhs = complete_data['fractions'][0]
    
    live_feed, feed_dict, abv_grnd_biomass_ratio,fuel_dict,sheet7_lulc_classes,c_fractions = sh7.get_sheet7_classes()
    lu_dict = sheet7_lulc_classes
    
    ### Get inflow from SurfWat
    in_text = dict()
    if metadata['inflow_nodes']:
        discharge_in, subbasins_in = discharge_at_points(metadata['inflow_nodes'],metadata['SWfile'], date_list)
#        if type(metadata['SWfile']) == list:
#            time = nc.MFDataset(metadata['SWfile']).variables['time'][:]
#        else:
#            time = nc.Dataset(metadata['SWfile']).variables['time'][:]
        time = np.array([datetime.datetime.toordinal(dt) for dt in date_list])
        idee = {n:s for n,s in zip(sb_names, sb_codes)}
        
        for sb in np.unique(subbasins_in):
            
            first = True
            for ix in [idx for idx, word in enumerate(subbasins_in) if word == sb]:
                if first:
                    q_in = discharge_in[ix]
                    first = False
                else:
                    q_in = np.sum([q_in, discharge_in[ix]], axis = 0)
            
            with open(os.path.join(output_folder, '{0}.txt'.format(sb)), 'wb') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ')
                spamwriter.writerow(["{0}".format('[this is text]')])
                for t, v in zip(time, q_in):
                    spamwriter.writerow([int(t),int(v)])
            
            in_text[int(idee[sb])] = os.path.join(output_folder, '{0}.txt'.format(sb))
            
    elif metadata['in_text']:
        in_text = metadata['in_text']
        
    else:
        in_text = None
    
    added_inflow = {}
    if in_text:
        for k in in_text.keys():
            inflowtext = os.path.join(in_text[k])
            added_inflow[k] = read_inflow_file(inflowtext,date_list)
    
    #%% Runoff_Surface
    surf_ro_fhs = complete_data['sr'][0].tolist()
    surf_ro_fhs = sh7.mm_to_km3(lu_fh,surf_ro_fhs) #convert runoff from mm to km3
    
    #%% Runoff_baseflow        
    base_ro_fhs = complete_data['bf'][0].tolist()
    base_ro_fhs = sh7.mm_to_km3(lu_fh,base_ro_fhs) #convert runoff from mm to km3
    
    #%% Runoff_total       
    ro_fhs = complete_data['tr'][0].tolist()
    ro_fhs = sh7.mm_to_km3(lu_fh,ro_fhs) #convert runoff from mm to km3
    
    withdr_fhs = complete_data['supply_sw'][0].tolist()
    withdr_fhs = sh7.mm_to_km3(lu_fh,withdr_fhs) 

    return_gw_sw_fhs = complete_data['return_flow_gw_sw'][0].tolist()
    return_gw_sw_fhs = sh7.mm_to_km3(lu_fh,return_gw_sw_fhs)
    
    return_sw_sw_fhs = complete_data['return_flow_sw_sw'][0].tolist()
    return_sw_sw_fhs = sh7.mm_to_km3(lu_fh,return_sw_sw_fhs)
    
    #%% Outflow
#    discharge, subbasins = discharge_at_points(outlets,metadata['SWfile'], date_list)
#    
#    discharge_sum = {}
#    
#    for i in range(len(sb_names)):
#        sb  = sb_names[i]
#        sb_code = sb_codes[i]
#        discharge_sum[sb_code]=np.sum(np.array([discharge[j]/1000000000. for j in range(len(discharge)) if subbasins[j]==sb]),axis = 0)
#    
#    import matplotlib.pyplot as plt
#    plt.plot(discharge_sum['1'], label='1')
    
    discharge_out_from_wp = True
    
    if discharge_out_from_wp:
        
        discharge_sum = dict()
        subbasins2 = list()
        
        for temp_sb, sb_code in zip(sb_fhs, sb_codes):
            
            subbasins2.append(sb_name)
            arr = np.array([])
            
            mask = becgis.OpenAsArray(temp_sb, nan_values = True)
            
            for fh, dt in zip(ro_fhs, becgis.ConvertDatetimeDate(complete_data['tr'][1], out = 'datetime')):
                
                if dt in date_list:
                    RO = becgis.OpenAsArray(fh, nan_values = True)
                    RO[mask != 1] = np.nan
                
                    arr = np.append(arr, np.nansum(RO))

            discharge_sum[str(sb_code)] = arr

#    plt.plot(discharge_sum['1'], label='2')
#    plt.legend()

    #%% Splitting up the outflow
    split_discharge = discharge_split(global_data["wpl_tif"],global_data["environ_water_req"],discharge_sum,ro_fhs,fractions_fhs,sb_fhs_code_names,date_list)

    #%% Add arrows to template when possible (dependent on subbasin structure)
    svg_template = sheet_5_dynamic_arrows(dico_in,dico_out,test_svg = template,outpath = template.replace('.svg', '_temp.svg'))
    
    #%% Write CSV file
    results = Vividict()
    dt = 0
    print "starting sheet 5 loop"
    for d in date_list:
        print "sheet 5 {0} done".format(d)
        
        datestr1 = "%04d_%02d" %(d.year,d.month)
        datestr2 = "%04d%02d" %(d.year,d.month)
        #datestr2 = "%04d_%d" %(d.year,d.month)
        for sb in sb_codes:
            results["%04d" %(d.year)]["%02d" %(d.month)]['total_outflow'][sb] = split_discharge["%04d" %(d.year)]["%02d" %(d.month)]['total_outflow'][sb]
            results["%04d" %(d.year)]["%02d" %(d.month)]['committed_outflow'][sb] = split_discharge["%04d" %(d.year)]["%02d" %(d.month)]['committed_outflow'][sb]
            results["%04d" %(d.year)]["%02d" %(d.month)]['non_utilizable_outflow'][sb] = split_discharge["%04d" %(d.year)]["%02d" %(d.month)]['non_utilizable_outflow'][sb] 
            results["%04d" %(d.year)]["%02d" %(d.month)]['utilizable_outflow'][sb] = split_discharge["%04d" %(d.year)]["%02d" %(d.month)]['utilizable_outflow'][sb]
            results["%04d" %(d.year)]["%02d" %(d.month)]['non_recoverable_outflow'][sb] = split_discharge["%04d" %(d.year)]["%02d" %(d.month)]['non_recoverable_outflow'][sb]
        for s in range(1,len(sb_codes)+1):
            outflow = {}
            outflow = split_discharge["%04d" %(d.year)]["%02d" %(d.month)]['total_outflow']
            if not 0 in dico_in[s]:
                results["%04d" %(d.year)]["%02d" %(d.month)]['inflows'][sb_codes[s-1]] = np.sum([outflow[sb_codes[j-1]] for j in dico_in[s]])
            else:
                results["%04d" %(d.year)]["%02d" %(d.month)]['inflows'][sb_codes[s-1]] = np.sum([outflow[sb_codes[j-1]] for j in dico_in[s] if j!=0]) + added_inflow[s][dt]/1000000000.
        #define filehandles for the correct time
        
        surf_ro_fh = surf_ro_fhs[np.where([datestr2 in surf_ro_fhs[i] for i in range(len(surf_ro_fhs))])[0][0]]
        base_ro_fh = base_ro_fhs[np.where([datestr2 in base_ro_fhs[i] for i in range(len(base_ro_fhs))])[0][0]]
        ro_fh = ro_fhs[np.where([datestr2 in ro_fhs[i] for i in range(len(ro_fhs))])[0][0]]
        
        withdr_fh = withdr_fhs[np.where([datestr1 in withdr_fhs[i] for i in range(len(withdr_fhs))])[0][0]]
        
        return_gw_sw_fh = return_gw_sw_fhs[np.where([datestr1 in return_gw_sw_fhs[i] for i in range(len(return_gw_sw_fhs))])[0][0]]
        return_sw_sw_fh = return_gw_sw_fhs[np.where([datestr1 in return_gw_sw_fhs[i] for i in range(len(return_gw_sw_fhs))])[0][0]]
        
        results["%04d" %(d.year)]["%02d" %(d.month)]['surf_runoff'] = lu_type_sum_subbasins(surf_ro_fh,lu_fh,lu_dict,sb_fhs_code_names)
        results["%04d" %(d.year)]["%02d" %(d.month)]['base_runoff'] = lu_type_sum_subbasins(base_ro_fh,lu_fh,lu_dict,sb_fhs_code_names)
        
        results["%04d" %(d.year)]["%02d" %(d.month)]['total_runoff'] = sum_subbasins(ro_fh,sb_fhs_code_names)
        
        results["%04d" %(d.year)]["%02d" %(d.month)]['withdrawls'] = lu_type_sum_subbasins(withdr_fh,lu_fh,man_dict,sb_fhs_code_names)
        
        results["%04d" %(d.year)]["%02d" %(d.month)]['return_gw_sw'] = sum_subbasins(return_gw_sw_fh,sb_fhs_code_names)
        results["%04d" %(d.year)]["%02d" %(d.month)]['return_sw_sw'] = sum_subbasins(return_sw_sw_fh,sb_fhs_code_names)
        
        for j in results["%04d" %(d.year)]["%02d" %(d.month)]['surf_runoff'][sb_codes[0]].keys():
            results["%04d" %(d.year)]["%02d" %(d.month)]['surf_runoff']['basin'][j] = np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['surf_runoff'][k][j] for k in sb_codes])
            results["%04d" %(d.year)]["%02d" %(d.month)]['base_runoff']['basin'][j] = np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['base_runoff'][k][j] for k in sb_codes])            
             
        results["%04d" %(d.year)]["%02d" %(d.month)]['total_runoff']['basin'] = np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['total_runoff'][k] for k in sb_codes])
        
        results["%04d" %(d.year)]["%02d" %(d.month)]['withdrawls']['basin']['man'] = np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['withdrawls'][k]['man'] for  k in sb_codes])
        results["%04d" %(d.year)]["%02d" %(d.month)]['withdrawls']['basin']['natural'] = np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['withdrawls'][k]['natural'] for k in sb_codes])            
             
        results["%04d" %(d.year)]["%02d" %(d.month)]['return_sw_sw']['basin'] = np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['return_sw_sw'][k] for k in sb_codes])
        results["%04d" %(d.year)]["%02d" %(d.month)]['return_gw_sw']['basin'] = np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['return_sw_sw'][k] for k in sb_codes])            
        
        outs = [j for j in dico_out.keys() if 0 in dico_out[j]]
        results["%04d" %(d.year)]["%02d" %(d.month)]['total_outflow']['basin'] = np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['total_outflow'][sb_codes[k-1]] for k in outs])
        results["%04d" %(d.year)]["%02d" %(d.month)]['committed_outflow']['basin'] =  np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['committed_outflow'][sb_codes[k-1]] for k in outs])
        results["%04d" %(d.year)]["%02d" %(d.month)]['non_utilizable_outflow']['basin'] = np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['non_utilizable_outflow'][sb_codes[k-1]] for k in outs])
        results["%04d" %(d.year)]["%02d" %(d.month)]['utilizable_outflow']['basin'] = np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['utilizable_outflow'][sb_codes[k-1]] for k in outs])
        results["%04d" %(d.year)]["%02d" %(d.month)]['non_recoverable_outflow']['basin'] =  np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['non_recoverable_outflow'][sb_codes[k-1]] for k in outs])
        
        ins = [j for j in dico_in.keys() if 0 in dico_in[j]]
        results["%04d" %(d.year)]["%02d" %(d.month)]['inflows']['basin'] = np.nansum([added_inflow[k][dt]/1000000000. for k in ins])
        
        output_fh = output_folder +"\\sheet5_monthly\\sheet5_"+datestr1+".csv"
        create_csv(results["%04d" %(d.year)]["%02d" %(d.month)],output_fh)
        output = output_folder + '\\sheet5_monthly\\sheet5_'+datestr1+'.png'
        create_sheet5_inception(metadata['name'], sb_codes, datestr1, 'km3', output_fh, output, template=svg_template)
        dt += 1
    
    fhs, dates, years, months, days = becgis.SortFiles(output_folder +"\\sheet5_monthly", [-11,-7], month_position = [-6,-4], extension = 'csv')
    
    years, counts = np.unique(years, return_counts = True)
    
    fhs = sh4.create_csv_yearly(os.path.join(output_folder, "sheet5_monthly"), os.path.join(output_folder, "sheet5_yearly"), year_position =[-11,-7], month_position = [-6,-4], header_rows = 1, header_columns = 2, minus_header_colums = -1)
    
    for fh in fhs:
        output = fh.replace('csv', 'png')
        create_sheet5_inception(metadata['name'], sb_codes, datestr1, 'km3', fh, output, template=svg_template)
        
    return complete_data

def sheet_5_dynamic_arrows(dico_in,dico_out,test_svg = 'D:\Code\sheeet5_dyn.svg',outpath = 'sheet5_with_arrows_temp.svg'):
    normal_arrow = {}
    straight_out = {}
    straight_in = {}
    join_one_below = {}
    join_one_top = {}
    
    lower_arrow = {}
    lower_join_one_below = {}
    
    element1 = '<path d="M 63.071871,264.267 V 271.30428 H 78.457001 V 154.10108 L 85.774098,154.10077 V 156.582 L 87.774096,157.531 89.773756,156.582 89.774174,150.12138 74.457001,150.12171 V 267.32467 L 67.07187,267.32475 67.072128,264.267 65.08986,265.203 Z" id="normal_arrow" style="fill:#73b7d1;fill-opacity:1;stroke:#fbfbfb;stroke-width:0.55299997px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccccccccccc" ns1:connector-curvature="0" ns1:label="#path8251" />'
    element1 = 'path d="M 63.071871,264.267 V 271.30428 H 78.457001 V 154.10108 L 85.774098,154.10077 V 156.582 L 87.774096,157.531 89.773756,156.582 89.774174,150.12138 74.457001,150.12171 V 267.32467 L 67.07187,267.32475 67.072128,264.267 65.08986,265.203 Z" id="normal_arrow" style="fill:#73b7d1;fill-opacity:1;stroke:#fbfbfb;stroke-width:0.55299997px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccccccccccc" ns1:connector-curvature="0" ns1:label="#path8251" '
    element2 = 'path d="M 63.154,271.035 H 86 V 267.455 L 66.6,267.45471 V 264.7052 L 64.872,265.528 63.154476,264.71131" id="low_bar" style="opacity:1;fill:#73b7d1;fill-opacity:1;stroke:none;stroke-width:0.26458332px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccc" ns1:connector-curvature="0" ns1:label="#path8255" '
    element3 = 'path d="M 62.851212,148.62795 62.853138,156.5571 64.853137,157.71202 66.852797,156.5571 66.853215,148.69463 64.85134,149.76798 Z" id="top_in" style="opacity:1;fill:#73b7d1;fill-opacity:1;stroke:#ffffff;stroke-width:0.55299997;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccc" ns1:connector-curvature="0" ns1:label="#path8255" ' 
    element4 = 'path d="M 66.872,264.29 66.873585,272.19258 64.873925,273.3475 62.873926,272.19258 62.872,264.29 64.872128,265.203 Z" id="bottom_out" style="fill:#73b7d1;fill-opacity:1;stroke:#ffffff;stroke-width:0.55299997;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccc" ns1:connector-curvature="0" ns1:label="#path8274" '
    element5 = 'path d="M 67.468749,150.121 74.457001,150.12171 89.774172,150.12138 89.773754,156.582 87.774094,157.531 85.774098,156.582 V 154.10077 L 67.468748,154.101 Z" id="top_connect_1" style="fill:#73b7d1;fill-opacity:1;stroke:#fcfcfc;stroke-width:0.55299997;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccccc" ns1:connector-curvature="0" ns1:label="#path8250" />\n<ns0:path d="M 69.297345,150.399 H 65.921953 L 65.911237,153.823 H 69.279 Z" id="top_connect_2" style="fill:#73b7d1;fill-opacity:1;stroke:none;stroke-width:0.26458332px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccc" ns1:connector-curvature="0" ns1:label="#path8262" '
    element6 = '<rect height="3.4260001" id="rect_join_mask" style="stroke-width:0.25686634;fill:#73b7d1;fill-opacity:1" transform="translate(%.2f)" width="1.9203489" x="62.342754" y="150.399" ns1:label="#rect3787" /> '
    element7 = '<rect height="3.4300554" id="rect_low_mask" style="fill:#73b7d1;fill-opacity:1;stroke-width:0.26458332"  transform="translate(%.2f)" width="1.4658357" x="66.373032" y="267.59995" ns1:label="#rect8273" />'
  
    expected_nb_arrows = np.sum([len(dico_out[k]) for k in dico_out.keys()]) + np.sum([0 in dico_in[k] for k in dico_in.keys()])
    
    dico_in_b = {}
    for k in dico_in.keys():
        dico_in_b[k] = [dico_in[k][j] for j in range(len(dico_in[k])) if dico_in[k][j] != 0]
    dico_out_b = {}
    for k in dico_in.keys():
        dico_out_b[k] = [dico_out[k][j] for j in range(len(dico_out[k])) if dico_out[k][j] != 0]
  
    for sb in dico_out.keys():
        if len(dico_out_b[sb]) ==1:
#            if dico_out_b[sb][0] == 0:
#                straight_out[sb] = 1
            if (len(dico_in_b[dico_out_b[sb][0]]) == 1) & (dico_out_b[sb] == sb+1):
                normal_arrow[sb] = 1
            else:     
                prev_list = [i for i in range(dico_out_b[sb][0]-len(dico_in_b[dico_out_b[sb][0]]),dico_out_b[sb][0])]
                contrib = dico_in_b[dico_out_b[sb][0]]
                if (prev_list == contrib) & (contrib !=[]):
                    normal_arrow[np.max(contrib)] = 1
                    contrib_2 = [i for i in contrib if i!=(np.max(contrib))]
                    for i in range(len(contrib_2)):
                        join_one_below[contrib_2[i]]=1
               # else to add in later with extra levels of complexity
        elif len(dico_out_b[sb])>0:
            going_to = dico_out_b[sb]
            next_list = [i for i in range(sb+1,sb+1+len(dico_out_b[sb]))]
            if going_to == next_list:
                normal_arrow[sb]=1
                going_to_2 = [i for i in going_to if i!=(np.max(going_to))]
                for j in range(len(going_to_2)):
                    join_one_top[going_to_2[j]]=1
    
    for sb in dico_out.keys():
        if 0 in dico_out[sb]:
            straight_out[sb]=1
    
    for sb in dico_in.keys():
        if 0 in dico_in[sb]:
            straight_in[sb]=1
      
    actual_nb_arrows = len(normal_arrow) + len(straight_out) + len(straight_in) \
                        + len(join_one_below) + len(join_one_top)\
                        + len(lower_arrow) + len(lower_join_one_below)
       
    if expected_nb_arrows == actual_nb_arrows:
        tree = ET.parse(test_svg)   
        off1 = 22.66 
        for n in normal_arrow.keys():
            offset = (n - 1) * off1
            child = ET.Element(element1 %offset)
            layer = tree.findall('''*[@id='layer1']''')[0]
            layer.append(child)
           
        for j in straight_in.keys():    
            offset = (j - 1) * off1 + .22
            child = ET.Element(element3 %offset)
            layer = tree.findall('''*[@id='layer1']''')[0]
            layer.append(child)
        
        for j in straight_out.keys():    
            offset = (j - 1) * off1 + .22
            child = ET.Element(element4 %offset)
            layer = tree.findall('''*[@id='layer1']''')[0]
            layer.append(child)
        
        for j in join_one_top.keys():    
            offset = (j - 1) * off1
            child = ET.Element(element5 %(offset,offset))
            layer = tree.findall('''*[@id='layer1']''')[0]
            layer.append(child)
        #    child = ET.Element(element5b %offset)
        #    layer = tree.findall('''*[@id='layer1']''')[0]
        #    layer.append(child)
                
        for j in join_one_below.keys():    
            offset = (j - 1) * off1 +.22
            child = ET.Element(element2 %offset)
            layer = tree.findall('''*[@id='layer1']''')[0]
            layer.append(child)
        
        for j in straight_in.keys():
            if len(dico_in[j]) > 1 :
                offset = (j - 1) * off1
                child = ET.Element(element6 %offset)
                layer = tree.findall('''*[@id='layer1']''')[0]
                layer.append(child) 
                
        for j in straight_out.keys():
            if len(dico_out[j]) > 1 :
                offset = (j - 1) * off1
                child = ET.Element(element7 %offset)
                layer = tree.findall('''*[@id='layer1']''')[0]
                layer.append(child)  
        
        tree.write(outpath)
    else:
            outpath = test_svg
            print ('ERROR: unexpected number of arrows.\n      Basin structure too complex to generate Sheet7 arrows automatically.\
                         \n      Standard template returned as output.')
    return outpath

def calc_fractions(p_fhs, p_dates, output_dir, dem_fh, lu_fh):
    
    dem_reproj_fhs = becgis.MatchProjResNDV(lu_fh, np.array([dem_fh]), output_dir)
    fraction_altitude_xs = [11, 423, 11, 423]
    # Determine the per pixel fractions needed to calculate the non-utilizable outflow.
    upstream_fh = sh4.upstream_of_lu_class(dem_fh, lu_fh, output_dir, clss = None)
    fractions_altitude_fh = os.path.join(output_dir, 'fractions_altitude.tif')
    sh4.linear_fractions(lu_fh, upstream_fh, dem_reproj_fhs[0], fractions_altitude_fh, fraction_altitude_xs, unit = 'm', quantity = 'Altitude')
    
    p_months = np.array([date.month for date in p_dates])
    
    fractions_fhs = np.array([])
    
    for date in p_dates:
        # Create some filehandles to store results.
        std_fh = os.path.join(output_dir, 'std_means', 'std_{0}.tif'.format(str(date.month).zfill(2)))
        mean_fh = os.path.join(output_dir, 'std_means', 'mean_{0}.tif'.format(str(date.month).zfill(2)))
        fractions_dryness_fh = os.path.join(output_dir, 'fractions_dryness', 'fractions_dryness_{0}_{1}.tif'.format(date.year, str(date.month).zfill(2)))
        fractions_fh = os.path.join(output_dir, 'fractions', 'fractions_{0}_{1}.tif'.format(date.year, str(date.month).zfill(2)))
        
        # If not done yet, calculate the mean and std of the precipitation for the current month of the year.
        if not np.any([os.path.isfile(std_fh), os.path.isfile(mean_fh)]):        
            becgis.CalcMeanStd(p_fhs[p_months == date.month], std_fh, mean_fh)
        
        # Determine fractions regarding dryness to determine non-utilizable outflow.
        sh1.dryness_fractions(p_fhs[p_dates == date][0], std_fh, mean_fh, fractions_dryness_fh, base = -0.5, top = 0.0)
        
        # Multiply the altitude and dryness fractions.
        becgis.Multiply(fractions_altitude_fh, fractions_dryness_fh, fractions_fh)
        
        fractions_fhs = np.append(fractions_fhs, fractions_fh)
        
    return (fractions_fhs, p_dates)

def dictionary():
    lucs = {
    'Forests':              [1, 8, 9, 10, 11, 17],
    'Shrubland':            [2, 12, 14, 15],
    'Rainfed Crops':        [34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    'Forest Plantations':   [33, 44],
    'Natural Water Bodies': [4, 19, 23, 24],
    'Wetlands':             [5, 25, 30, 31],
    'Natural Grasslands':   [3, 13, 16, 20],
    'Other (Non-Manmade)':  [6, 7, 18, 21, 22, 26, 27, 28, 29, 32, 45, 46, 48, 49, 50, 51],
    'Irrigated crops':      [52,53,54,55,56,57,58,59,60,61,62],
    'Managed water bodies': [63,74,75,77],
    'Aquaculture':          [65],
    'Residential':          [47, 66, 68, 72],
    'Greenhouses':          [64],
    'Other':                [68,69,70,71,76,78]}
    
    manmade_categories = ['Irrigated crops','Managed water bodies','Aquaculture','Residential','Greenhouses','Other']
    man_lu = [lucs[m] for m in manmade_categories]
    flat_man = [item for sublist in man_lu for item in sublist]
    
    natural_categories = ['Forests','Shrubland','Rainfed Crops','Forest Plantations','Natural Water Bodies','Wetlands','Natural Grasslands','Other (Non-Manmade)']
    nat_lu = [lucs[m] for m in natural_categories]
    flat_natural = [item for sublist in nat_lu for item in sublist]
    
    man_dict = {'man':flat_man,'natural':flat_natural}
    
    return man_dict

def create_sheet5_inception(basin,sb_codes, period, units, data, output, template=False):
    
    df = pd.read_csv(data, sep=';')
    if not template:
#        path = os.path.dirname(os.path.abspath(__file__))
#        svg_template_path = os.path.join(path, 'svg', 'sheet_7.svg')
        svg_template_path = 'C:\Anaconda2\Lib\site-packages\wa\Sheets\svg\sheet_5_2.svg'
    else:
        svg_template_path = os.path.abspath(template)
        
    tree = ET.parse(svg_template_path)

    xml_txt_box = tree.findall('''.//*[@id='basin']''')[0]
    xml_txt_box.getchildren()[0].text = 'Basin: ' + basin

    xml_txt_box = tree.findall('''.//*[@id='period']''')[0]
    xml_txt_box.getchildren()[0].text = 'Period: ' + period
    
    line_id0 = [31633,30561,30569,30577,30585,30905,30913,30921,30929,
                31873,31993,32001,32026,32189,32197,32318,32465,32609,
               31273,31281,31289,31297,32817]
#    line_id0 = [31633,30557,30565,30571,30579,30877,30913,30921,30929,
 #               31869,31993,32001,32025,32189,32197,32317,32465,32609,
#                31273,31281,31289,31297,32817]
    
    line_lengths = [1,4,4,4,4,4,4,4,4,1,2,2,1,2,2,1,1,1,4,4,4,4,1]
    line_names = ['Inflow',
                  'Fast Runoff: PROTECTED','Fast Runoff: UTILIZED','Fast Runoff: MODIFIED','Fast Runoff: MANAGED',
                  'Slow Runoff: PROTECTED','Slow Runoff: UTILIZED','Slow Runoff: MODIFIED','Slow Runoff: MANAGED',
                  'Total Runoff',
                  'SW withdr. manmade','SW withdr. natural',
                  'SW withdr. total',
                  'Return Flow SW','Return Flow GW',
                  'Total Return Flow',
                  'Interbasin Transfer','SW storage change',
                  'Outflow: Committed','Outflow: Non Recoverable','Outflow: Non Utilizable','Outflow: Utilizable','Outflow: Total']
    current_variables = ['Inflow',
                  'Fast Runoff: PROTECTED','Fast Runoff: UTILIZED','Fast Runoff: MODIFIED','Fast Runoff: MANAGED',
                  'Slow Runoff: PROTECTED','Slow Runoff: UTILIZED','Slow Runoff: MODIFIED','Slow Runoff: MANAGED',
                  'SW withdr. manmade','SW withdr. natural',
                  'SW withdr. total',                  
                  'Return Flow SW','Return Flow GW','Total Return Flow',
                  'Total Runoff','Outflow: Committed','Outflow: Non Recoverable','Outflow: Non Utilizable','Outflow: Utilizable','Outflow: Total'
                  ] #,
                  #'Outflow: Total']
#    current_variables = line_names
    for var1 in current_variables:
        line_nb = [i for i in range(len(line_names)) if line_names[i] == var1][0]
        line_0 = line_id0[line_nb]
#        print line_0
        line_len = line_lengths[line_nb]
        df_var = df.loc[df.VARIABLE == var1]
        sb_order = sb_codes
        value_sum = 0
        for sb in range(len(sb_order)):
            if (var1 == 'Inflow') & (sb ==0):
                df_sb = df_var.loc[df_var.SUBBASIN == sb_order[sb]]
                cell_id = 'g' + str(32609 + 8*sb*line_len)
           #     cell_id = 'tspan' + str(line_0 + 8*sb*line_len)
                xml_txt_box = tree.findall('''.//*[@id='{0}']'''.format(cell_id))[0]
     #           xml_txt_box[0].text = '-'
                xml_txt_box[0].text = '%.1f' %(df_sb.VALUE)               
            else:
       # for sb in range(10):
                df_sb = df_var.loc[df_var.SUBBASIN == sb_order[sb]]  
                cell_id = 'g' + str(line_0 + 8*sb*line_len)
           #     cell_id = 'tspan' + str(line_0 + 8*sb*line_len)
                xml_txt_box = tree.findall('''.//*[@id='{0}']'''.format(cell_id))[0] 
     #           xml_txt_box[0].text = '-'
                xml_txt_box[0].text = '%.1f' %(df_sb.VALUE)
                value_sum += float(df_sb.VALUE)

        cell_id = 'g' + str(line_0 + 8*9*line_len)
        df_sb = df_var.loc[df_var.SUBBASIN == 'basin']
        xml_txt_box = tree.findall('''.//*[@id='{0}']'''.format(cell_id))[0] 
        xml_txt_box[0].text = '%.1f' %(df_sb.VALUE)           

#            try:
#                xml_txt_box[0][0].text = ''
#                xml_txt_box[0][0][0].text = ''
#            except:
#                print 'fail at %s' %cell_id
#                pass

#            cell_id = 'g' + str(line_0 + 8*sb*line_len)


    tempout_path = output.split('.')[0]+'.svg'        
    tree.write(tempout_path)
    out_png = output
    #out_png = 'sheet7_%d_%02d.png' %(year,month)
    subprocess.call(['C:\Program Files\Inkscape\inkscape.exe',tempout_path,'--export-dpi=300','--export-png='+out_png])
    os.remove(tempout_path)
    return

### Other functions
class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def lu_type_sum_subbasins(data_fh,lu_fh,lu_dict,sb_fhs_code_names):
    """
    Returns totals in a dict split by subbasin and land use type (PLU, ULU etc)
    Parameters
    ----------
    data_fh : str
        location of the map of the data to split
    lu_fh : str
        location of landuse map
    lu_dict : dict
        lu_class : list of landuses in class
    sb_fhs_code_names : list of tuples
        (sb_fhs,sb_codes,sb_names)
    """ 
    LULC = RC.Open_tiff_array(lu_fh)
    in_data = RC.Open_tiff_array(data_fh)
    out_data = Vividict()
    sb_fhs = zip(*sb_fhs_code_names)[0]
    sb_codes = zip(*sb_fhs_code_names)[1]
    for j in range(len(sb_fhs)):
        sb_fh = sb_fhs[j]
        sb_code = sb_codes[j]
        sb_mask = RC.Open_tiff_array(sb_fh)
        sb_mask[sb_mask==-9999]=0
        sb_mask = sb_mask.astype('bool')      
        for lu_class in lu_dict.keys():
            mask = [LULC == value for value in lu_dict[lu_class]]
            mask = (np.sum(mask,axis=0)).astype(bool)
            mask = mask * sb_mask
            out_data[sb_code][lu_class] = np.nansum(in_data[mask])
    return out_data

def sum_subbasins(data_fh,sb_fhs_code_names):
    """
    Returns totals in a dict split by subbasin
    Parameters
    ----------
    data_fh : str
        location of the map of the data to split
    sb_fhs_code_names : list of tuples
        (sb_fhs,sb_codes,sb_names)
    """ 
    in_data = RC.Open_tiff_array(data_fh)
    out_data = Vividict()
    sb_fhs = zip(*sb_fhs_code_names)[0]
    sb_codes = zip(*sb_fhs_code_names)[1]
    for j in range(len(sb_fhs)):
        sb_fh = sb_fhs[j]
        sb_code = sb_codes[j]
        sb_mask = RC.Open_tiff_array(sb_fh)
        sb_mask[sb_mask==-9999]=0
        sb_mask = sb_mask.astype('bool')      
        out_data[sb_code] = np.nansum(in_data[sb_mask])
    return out_data

def read_inflow_file(inflowtext,date_list):
    df = pd.read_csv(inflowtext,delimiter = ' ',skiprows=1,header =None,names=['date','inflow'])
    date_py = np.array([datetime.datetime.fromordinal(dt) for dt in df.date])
#    for j in range(len(df.date)):
#        delta = datetime.timedelta(int(df.date[j])-1)
#        date_py.append( datetime.datetime(1,1,1) + delta) 
#    date_py = np.array(date_py) 
    date_index = [np.where(date_py == k)[0][0] for k in date_list]
    inflows = np.array(df.inflow)[date_index]
    return inflows

def discharge_at_points(PointShapefile, SWpath, date_list):
    if type(SWpath) == list:
        SW = nc.MFDataset(SWpath)
    else:
        SW = nc.Dataset(SWpath)
    
    dates = [datetime.datetime.fromordinal(x) for x in SW.variables['time'][:]]
    
    mask = np.array([(dt in date_list) for dt in dates])
    
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(PointShapefile, 0)
    layer = dataSource.GetLayer()
    featureCount = layer.GetFeatureCount()
    discharge = []
    subbasins = []
    for pt in range(featureCount):
        feature = layer.GetFeature(pt)
        subbasins.append(feature.GetField(1))
        geometry = feature.GetGeometryRef()
        x = geometry.GetX()
        y = geometry.GetY()
        pos_x = (np.abs(SW.variables['longitude'][:]-x)).argmin()
        pos_y = (np.abs(SW.variables['latitude'][:]-y)).argmin()
        if 'Discharge_End_CR' in SW.variables.keys():
            discharge.append(SW.variables['Discharge_End_CR'][:,pos_y,pos_x][mask])
        else:
            discharge.append(SW.variables['Discharge_CR1'][:,pos_y,pos_x][mask])
            
    assert len(discharge[0]) == len(date_list)
    
    return discharge, subbasins

def discharge_split(wpl_fh,ewr_fh,discharge_sum,ro_fhs,fractions_fhs,sb_fhs_code_names,date_list):
    results = Vividict()  
    gray_water_fraction = {}
    ewr_percentage = {}

    sb_fhs = zip(*sb_fhs_code_names)[0]
    sb_codes = zip(*sb_fhs_code_names)[1]
    
    long_disch_mean = np.mean([discharge_sum[k] for k in sb_codes],axis=1)
    
    for i in range(len(sb_fhs)):
        sb_fh = sb_fhs[i]
        sb_code = sb_codes[i]
        gray_water_fraction[sb_code] = sh1.calc_basinmean(wpl_fh, sb_fh)
        ewr_percentage[sb_code] = sh1.calc_basinmean(ewr_fh, sb_fh)        
 #   runoff = QGIS.OpenAsArray(ro_fh, nan_values = True)
    t = 0
    for d in date_list:
        datestr1 = "%04d_%02d" %(d.year,d.month)
        datestr2 = "%04d%02d" %(d.year,d.month)
        ro_fh = ro_fhs[np.where([datestr2 in ro_fhs[i] for i in range(len(ro_fhs))])[0][0]]
        runoff = becgis.OpenAsArray(ro_fh, nan_values = True)
        fractions_fh = fractions_fhs[np.where([datestr1 in fractions_fhs[i] for i in range(len(fractions_fhs))])[0][0]]
        fractions =  becgis.OpenAsArray(fractions_fh, nan_values = True)

        non_utilizable_runoff = runoff * fractions       
        non_utilizable_sum = {}
        for i in range(len(sb_fhs)):
            sb_fh = sb_fhs[i]
            sb_code = sb_codes[i]
            sb_mask = RC.Open_tiff_array(sb_fh)
            sb_mask[sb_mask==-9999]=0
            sb_mask = sb_mask.astype('bool')      
            non_utilizable_sum[sb_code] = np.nansum(non_utilizable_runoff[sb_mask])        
            
        
            results["%04d" %(d.year)]["%02d" %(d.month)]['non_recoverable_outflow'][sb_code] = gray_water_fraction[sb_code] * discharge_sum[sb_code][t]
            reserved_outflow_demand = long_disch_mean[i] * ewr_percentage[sb_code]
        
            non_consumed_water = discharge_sum[sb_code][t] - results["%04d" %(d.year)]["%02d" %(d.month)]['non_recoverable_outflow'][sb_code]
        
            results["%04d" %(d.year)]["%02d" %(d.month)]['non_utilizable_outflow'][sb_code] = min(non_consumed_water, max(0.0, non_utilizable_sum[sb_code]))
            # note: committed = reserved_outflow_actual
            results["%04d" %(d.year)]["%02d" %(d.month)]['committed_outflow'][sb_code] = min(non_consumed_water - results["%04d" %(d.year)]["%02d" %(d.month)]['non_utilizable_outflow'][sb_code], reserved_outflow_demand)
            results["%04d" %(d.year)]["%02d" %(d.month)]['utilizable_outflow'][sb_code] = max(0.0, non_consumed_water - results["%04d" %(d.year)]["%02d" %(d.month)]['non_utilizable_outflow'][sb_code] - results["%04d" %(d.year)]["%02d" %(d.month)]['committed_outflow'][sb_code])
            results["%04d" %(d.year)]["%02d" %(d.month)]['total_outflow'][sb_code] =  discharge_sum[sb_code][t]
        t+=1
        
    return results


def create_csv(results,output_fh):
    
    """
    Create the csv-file for sheet 5.
    
    Parameters
    ----------
    results : dict
        Dictionary of results generated in sheet5_run.py
    output_fh : str
        Filehandle to store the csv-file.
    """
    first_row = ['SUBBASIN','VARIABLE','VALUE','UNITS']    
    if not os.path.exists(os.path.split(output_fh)[0]):
        os.makedirs(os.path.split(output_fh)[0])
    
    csv_file = open(output_fh, 'wb')
    writer = csv.writer(csv_file, delimiter=';')
    writer.writerow(first_row)
    lu_classes = ['PROTECTED', 'UTILIZED','MODIFIED', 'MANAGED']
    for sb in results['surf_runoff'].keys():
        writer.writerow([sb,'Inflow','{0:.3f}'.format(results['inflows'][sb]),'km3'])
        for lu_class in lu_classes:
            writer.writerow([sb,'Fast Runoff: '+lu_class,'{0:.3f}'.format(results['surf_runoff'][sb][lu_class]),'km3'])
            writer.writerow([sb,'Slow Runoff: ' +lu_class,'{0:.3f}'.format(results['base_runoff'][sb][lu_class]),'km3'])
        writer.writerow([sb,'Total Runoff','{0:.3f}'.format(results['total_runoff'][sb]),'km3'])
        writer.writerow([sb,'SW withdr. manmade','{0:.3f}'.format(results['withdrawls'][sb]['man']),'km3'])
        writer.writerow([sb,'SW withdr. natural','{0:.3f}'.format(results['withdrawls'][sb]['natural']),'km3'])
        writer.writerow([sb,'SW withdr. total','{0:.3f}'.format(results['withdrawls'][sb]['man']+results['withdrawls'][sb]['natural']),'km3'])
        writer.writerow([sb,'Return Flow SW','{0:.3f}'.format(results['return_sw_sw'][sb]),'km3'])
        writer.writerow([sb,'Return Flow GW','{0:.3f}'.format(results['return_gw_sw'][sb]),'km3'])
        writer.writerow([sb,'Total Return Flow','{0:.3f}'.format(results['return_sw_sw'][sb]+results['return_gw_sw'][sb]),'km3'])
#        writer.writerow([sb,'Interbasin Transfer','{0:.3f}'.format(results['interbasin_transfer'][sb]),'km3'])
#        writer.writerow([sb,'SW storage change','{0:.3f}'.format(results['delta_S_SW'][sb]),'km3'])
        writer.writerow([sb,'Outflow: Total','{0:.3f}'.format(results['total_outflow'][sb]),'km3'])
        writer.writerow([sb,'Outflow: Committed','{0:.3f}'.format(results['committed_outflow'][sb]),'km3'])
        writer.writerow([sb,'Outflow: Non Recoverable','{0:.3f}'.format(results['non_recoverable_outflow'][sb]),'km3'])
        writer.writerow([sb,'Outflow: Non Utilizable','{0:.3f}'.format(results['non_utilizable_outflow'][sb]),'km3'])
        writer.writerow([sb,'Outflow: Utilizable','{0:.3f}'.format(results['utilizable_outflow'][sb]),'km3'])
        
    csv_file.close()
    return