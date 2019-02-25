# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:33:21 2017

@author: cmi001
"""
import os
import csv
import datetime
from datetime import date
import subprocess
import shutil
import tempfile as tf
import xml.etree.ElementTree as ET
from itertools import groupby
from operator import itemgetter
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import ogr
import types

import WA_Hyperloop.becgis as becgis
from WA_Hyperloop import hyperloop as hl
import WA_Hyperloop.get_dictionaries as gd
from WA_Hyperloop.paths import get_path

def create_sheet5(complete_data, metadata, output_dir, global_data):
    output_folder = os.path.join(output_dir, metadata['name'], 'sheet5')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    template = get_path('sheet5_svg')
    complete_data['fractions'] = calc_fractions(complete_data['p'],
                                                os.path.join(output_dir, metadata['name'], 'data', 'fractions'),
                                                global_data['dem'], metadata['lu'],
                                                metadata['fraction_xs'])
    if 'SWfile' in metadata.keys():
        if isinstance(metadata['SWfile'], types.ListType):
            SW_dates = np.array([datetime.date.fromordinal(t)
                                 for t in nc.MFDataset(metadata['SWfile']).variables['time'][:]])
        else:
            SW_dates = np.array([datetime.date.fromordinal(t)
                                 for t in nc.Dataset(metadata['SWfile']).variables['time'][:]])

        date_list = becgis.CommonDates([complete_data['tr'][1], complete_data['supply_sw'][1],
                                        complete_data['fractions'][1],
                                        SW_dates])
    else:
        date_list = becgis.CommonDates([complete_data['tr'][1], complete_data['supply_sw'][1],
                                        complete_data['fractions'][1]])

    date_list = becgis.ConvertDatetimeDate(date_list, out='datetime')
    man_dict = dictionary()
    lu_fh = metadata['lu']

    sb_codes = sorted(metadata['masks'].keys())
    sb_fhs = [metadata['masks'][sb][1] for sb in sb_codes]
    #reproject subbasin masks
    sb_temp = os.path.join(output_folder, 'temp_sb')
    sb_fhs = becgis.MatchProjResNDV(lu_fh, sb_fhs, sb_temp)
    sb_names = [metadata['masks'][sb][0] for sb in sb_codes]
    sb_fhs_code_names = zip(sb_fhs, sb_codes, sb_names)
    
    # subbasin connectivity dictionaries
    dico_in = metadata['dico_in']
    dico_out = metadata['dico_out']

    fractions_fhs = complete_data['fractions'][0]

    _, _, _, _, lu_dict, _ = gd.get_sheet7_classes()

    discharge_out_from_wp = metadata['discharge_out_from_wp']

    surf_ro_fhs = complete_data['sr'][0].tolist()
    base_ro_fhs = complete_data['bf'][0].tolist()
    ro_fhs = complete_data['tr'][0].tolist()
    withdr_fhs = complete_data['supply_sw'][0].tolist()
    return_gw_sw_fhs = complete_data['return_flow_gw_sw'][0].tolist()
    return_sw_sw_fhs = complete_data['return_flow_sw_sw'][0].tolist()

    if discharge_out_from_wp:
        AREA = becgis.MapPixelAreakm(lu_fh)
        added_inflow = dict()
        discharge_sum = dict()
        interbasin_transfers = dict()
        deltaSW = dict()

        for temp_sb, sb_code in zip(sb_fhs, sb_codes):
            in_list = np.array(metadata['dico_in'][sb_code])
            out_list = np.array(metadata['dico_out'][sb_code])
            AVAIL_sb = np.array([])
            ro = np.array([])
            wth = np.array([])
            interbasin_transfers[sb_code] = np.zeros(len(date_list))
            mask = becgis.OpenAsArray(temp_sb, nan_values=True)

            for dt in date_list:
                rofh = np.array(ro_fhs)[complete_data['tr'][1] == dt.date()][0]
                wfh = np.array(withdr_fhs)[complete_data['supply_sw'][1] == dt.date()][0]

                RO = becgis.OpenAsArray(rofh, nan_values=True) * AREA / 1e6
                RO[mask != 1] = np.nan

                W = becgis.OpenAsArray(wfh, nan_values=True) * AREA / 1e6
                W[mask != 1] = np.nan

                AVAIL = np.nansum(RO)-np.nansum(W)
                AVAIL_sb = np.append(AVAIL_sb, AVAIL)
                ro = np.append(ro, np.nansum(RO))
                wth = np.append(wth, np.nansum(W))

            # Add inflow from outside sources to available runoff
            # Add or remove interbasin transfers as well
            if 0 in in_list:
                if len(metadata['masks'][sb_code][2]) == 0:
                    print 'Warning, missing inflow textfile, proceeding without added inflow'
                else:
                    added_inflow[sb_code] = 0
                    for inflow_file in metadata['masks'][sb_code][2]:
                        added_inflow[sb_code] += read_inflow_file(inflow_file, date_list)
                AVAIL_sb += added_inflow[sb_code]
                
            if len(metadata['masks'][sb_code][3]) > 0: # check if any interbasin transfers are listed
                for transfer_file in metadata['masks'][sb_code][3]:
                    interbasin_transfers[sb_code] += read_inflow_file(transfer_file, date_list)
                AVAIL_sb += interbasin_transfers[sb_code]         

            inflow = np.zeros(len(AVAIL_sb))
            for inflow_sb in in_list[in_list != 0]:
                AVAIL_sb += discharge_sum[sb_codes[inflow_sb-1]]
                inflow += discharge_sum[sb_codes[inflow_sb-1]]

            deltaS = np.zeros(len(AVAIL_sb))
            for i in np.where(AVAIL_sb < 0):
                deltaS[i] = AVAIL_sb[i] # DeltaS = RO(includes_returns) + otherInflows - W  where negative

            AVAIL_sb[AVAIL_sb < 0] = 0
            discharge_sum[sb_code] = AVAIL_sb # Discharge = RO(includes_returns) + otherInflows - W where positive

            deltaSW[sb_code] = deltaS
            # Spread out DeltaS to previous timesteps if possible
            if len(out_list) == 0:
                deltaSW[sb_code] += discharge_sum[sb_code]
                discharge_sum[sb_code] = discharge_sum[sb_code] * 0
                
            else:
                ds = deltaS
                ro = np.array(ro)
                ds = np.array(ds)
                wth = np.array(wth)
                data = list(np.where(ds < 0)[0])
                prev_end = 0
                for k, g in groupby(enumerate(data), lambda (i, x): i-x):
                    decr_st = map(itemgetter(1), g)
                    start = decr_st[0]
                    end = decr_st[-1]
                    deltaS_season = np.sum(ds[start:end+1])
                    avail_per_month = ro[prev_end:start] + inflow[prev_end:start] - wth[prev_end:start]
                    avail_prev_season = np.sum(ro[prev_end:start] + inflow[prev_end:start] - wth[prev_end:start])
                    weight_per_month = avail_per_month/avail_prev_season
                    if start > 0:
                        if avail_prev_season < abs(deltaS_season):
                            print 'Warning, insufficient runoff and inflows in previous months to meet DeltaS'
                            ds[prev_end:start] = np.min((avail_per_month, -deltaS_season * weight_per_month), axis=0)
                            discharge_sum[sb_code][prev_end:start] = discharge_sum[sb_code][prev_end:start] - ds[prev_end:start]
                        else:
                            ds[prev_end:start] = -deltaS_season * weight_per_month
                            discharge_sum[sb_code][prev_end:start] = discharge_sum[sb_code][prev_end:start] - ds[prev_end:start]
                    prev_end = end + 1
                deltaSW[sb_code] = ds

    if discharge_out_from_wp == False:
        PointShapefile = metadata['OutletPoints']
        SWpath = metadata['surfwat']
        sw_time, discharge_natural, discharge_end, stat_name = discharge_at_points(PointShapefile, SWpath)
#        discharge_sum_end = {}
#        discharge_sum_nat = {}
        discharge_sum = {}
        # fill in outlets which are only on "natural rivers" with discharge_natural values
        for i in range(len(discharge_end)):
            if np.isnan(discharge_end[i].all()):
                discharge_end[i] = discharge_natural[i]
        for sb_code in sb_codes:
            discharge_sum[sb_code] = np.sum(np.array(discharge_end)[np.where(np.array(stat_name) == sb_code)[0]], axis=0)/1e9
            in_list = np.array(metadata['dico_in'][sb_code])
            if 0 in in_list:
                if len(metadata['masks'][2]) == 0:
                    print 'Warning, missing inflow textfile, proceeding without added inflow'
                else:
                    added_inflow[sb_code] = 0
                    for inflow_file in metadata['masks'][2]:
                        added_inflow[sb_code] += read_inflow_file(inflow_file, date_list)

    #Splitting up the outflow into committed/ non_utilizable/ utilizable/ non_recoverable
    split_discharge = discharge_split(global_data["wpl_tif"], global_data["environ_water_req"],
                                      discharge_sum, ro_fhs, fractions_fhs,
                                      sb_fhs_code_names, date_list)
    #Add arrows to template when possible (dependent on subbasin structure)
    svg_template = sheet_5_dynamic_arrows(dico_in, dico_out, template,
                                          os.path.join(output_folder, 'temp_sheet5.svg'))
    results = Vividict()
    dt = 0
    print "starting sheet 5 loop"
    for d in date_list:
        print 'sheet 5 {0} started'.format(d)
        datestr1 = "%04d_%02d" %(d.year, d.month)
        datestr2 = "%04d%02d" %(d.year, d.month)
        ystr = "%04d" %(d.year)
        mstr = "%02d" %(d.month)
        for sb in sb_codes:
            results[ystr][mstr]['total_outflow'][sb] = split_discharge[ystr][mstr]['total_outflow'][sb]
            results[ystr][mstr]['committed_outflow'][sb] = split_discharge[ystr][mstr]['committed_outflow'][sb]
            results[ystr][mstr]['non_utilizable_outflow'][sb] = split_discharge[ystr][mstr]['non_utilizable_outflow'][sb]
            results[ystr][mstr]['utilizable_outflow'][sb] = split_discharge[ystr][mstr]['utilizable_outflow'][sb]
            results[ystr][mstr]['non_recoverable_outflow'][sb] = split_discharge[ystr][mstr]['non_recoverable_outflow'][sb]
            results[ystr][mstr]['deltaS'][sb] = deltaSW[sb][dt]
        for s in range(1, len(sb_codes)+1):
            outflow = {}
            outflow = split_discharge[ystr][mstr]['total_outflow']
            if not 0 in dico_in[s]:
                results[ystr][mstr]['inflows'][sb_codes[s-1]] = np.sum([outflow[sb_codes[j-1]] for j in dico_in[s]])
            else:
                results[ystr][mstr]['inflows'][sb_codes[s-1]] = np.sum([outflow[sb_codes[j-1]] for j in dico_in[s] if j != 0]) + added_inflow[s][dt]
        #define filehandles for the correct time
        surf_ro_fh = surf_ro_fhs[np.where([datestr2 in surf_ro_fhs[i] for i in range(len(surf_ro_fhs))])[0][0]]
        base_ro_fh = base_ro_fhs[np.where([datestr2 in base_ro_fhs[i] for i in range(len(base_ro_fhs))])[0][0]]
        ro_fh = ro_fhs[np.where([datestr2 in ro_fhs[i] for i in range(len(ro_fhs))])[0][0]]

        withdr_fh = withdr_fhs[np.where([datestr2 in withdr_fhs[i] for i in range(len(withdr_fhs))])[0][0]]

        return_gw_sw_fh = return_gw_sw_fhs[np.where([datestr2 in return_gw_sw_fhs[i] for i in range(len(return_gw_sw_fhs))])[0][0]]
        return_sw_sw_fh = return_sw_sw_fhs[np.where([datestr2 in return_sw_sw_fhs[i] for i in range(len(return_sw_sw_fhs))])[0][0]]

        results[ystr][mstr]['surf_runoff'] = lu_type_sum_subbasins(surf_ro_fh, lu_fh, lu_dict, sb_fhs_code_names)
        results[ystr][mstr]['base_runoff'] = lu_type_sum_subbasins(base_ro_fh, lu_fh, lu_dict, sb_fhs_code_names)

        results[ystr][mstr]['total_runoff'] = sum_subbasins(ro_fh, sb_fhs_code_names)

        results[ystr][mstr]['withdrawls'] = lu_type_sum_subbasins(withdr_fh, lu_fh, man_dict, sb_fhs_code_names)

        results["%04d" %(d.year)]["%02d" %(d.month)]['interbasin_transfers'][sb] = interbasin_transfers[sb][dt]

        results[ystr][mstr]['return_gw_sw'] = sum_subbasins(return_gw_sw_fh, sb_fhs_code_names)
        results[ystr][mstr]['return_sw_sw'] = sum_subbasins(return_sw_sw_fh, sb_fhs_code_names)

        for j in results[ystr][mstr]['surf_runoff'][sb_codes[0]].keys():
            results[ystr][mstr]['surf_runoff']['basin'][j] = np.nansum([results[ystr][mstr]['surf_runoff'][k][j] for k in sb_codes])
            results[ystr][mstr]['base_runoff']['basin'][j] = np.nansum([results[ystr][mstr]['base_runoff'][k][j] for k in sb_codes])

        results[ystr][mstr]['total_runoff']['basin'] = np.nansum([results[ystr][mstr]['total_runoff'][k] for k in sb_codes])

        results[ystr][mstr]['withdrawls']['basin']['man'] = np.nansum([results[ystr][mstr]['withdrawls'][k]['man'] for  k in sb_codes])
        results[ystr][mstr]['withdrawls']['basin']['natural'] = np.nansum([results[ystr][mstr]['withdrawls'][k]['natural'] for k in sb_codes])

        results["%04d" %(d.year)]["%02d" %(d.month)]['interbasin_transfers']['basin'] = np.nansum([results["%04d" %(d.year)]["%02d" %(d.month)]['interbasin_transfers'][k] for k in sb_codes])
        results[ystr][mstr]['deltaS']['basin'] = np.nansum([results[ystr][mstr]['deltaS'][k] for k in sb_codes])

        results[ystr][mstr]['return_sw_sw']['basin'] = np.nansum([results[ystr][mstr]['return_sw_sw'][k] for k in sb_codes])
        results[ystr][mstr]['return_gw_sw']['basin'] = np.nansum([results[ystr][mstr]['return_sw_sw'][k] for k in sb_codes])

        outs = [j for j in dico_out.keys() if 0 in dico_out[j]]
        results[ystr][mstr]['total_outflow']['basin'] = np.nansum([results[ystr][mstr]['total_outflow'][sb_codes[k-1]] for k in outs])
        results[ystr][mstr]['committed_outflow']['basin'] = np.nansum([results[ystr][mstr]['committed_outflow'][sb_codes[k-1]] for k in outs])
        results[ystr][mstr]['non_utilizable_outflow']['basin'] = np.nansum([results[ystr][mstr]['non_utilizable_outflow'][sb_codes[k-1]] for k in outs])
        results[ystr][mstr]['utilizable_outflow']['basin'] = np.nansum([results[ystr][mstr]['utilizable_outflow'][sb_codes[k-1]] for k in outs])
        results[ystr][mstr]['non_recoverable_outflow']['basin'] = np.nansum([results[ystr][mstr]['non_recoverable_outflow'][sb_codes[k-1]] for k in outs])

        ins = [j for j in dico_in.keys() if 0 in dico_in[j]]
        results[ystr][mstr]['inflows']['basin'] = np.nansum([added_inflow[k][dt] for k in ins]) #np.nansum([added_inflow[k][dt]/1000000000. for k in ins])

        output_fh = output_folder +"\\sheet5_monthly\\sheet5_"+datestr1+".csv"
        create_csv(results[ystr][mstr], output_fh)
        output = output_folder + '\\sheet5_monthly\\sheet5_'+datestr1+'.png'
        create_sheet5_svg(metadata['name'], sb_codes, datestr1, 'km3',
                          output_fh, output, svg_template, smart_unit=True)
        dt += 1
    fhs, dates, years, months, days = becgis.SortFiles(output_folder +"\\sheet5_monthly", [-11, -7], month_position=[-6, -4], extension='csv')
    years, counts = np.unique(years, return_counts=True)

    fhs = hl.create_csv_yearly(os.path.join(output_folder, "sheet5_monthly"),
                               os.path.join(output_folder, "sheet5_yearly"), 5,
                               metadata['water_year_start_month'],
                               year_position=[-11, -7], month_position=[-6, -4],
                               header_rows=1, header_columns=2,
                               minus_header_colums=-1)

    for fh in fhs:
        ystr = os.path.basename(fh).split('_')[-1][:4]
        output = fh.replace('csv', 'png')
        create_sheet5_svg(metadata['name'], sb_codes, ystr, 'km3',
                          fh, output, svg_template, smart_unit=False)
    shutil.rmtree(os.path.split(sb_fhs[0])[0])
    os.remove(svg_template)
    print 'Done'
    return complete_data

def sheet_5_dynamic_arrows(dico_in, dico_out, test_svg, outpath):
    normal_arrow = {}
    straight_out = {}
    straight_in = {}
    join_one_below = {}
    join_one_top = {}
    lower_arrow = {}
    lower_join_one_below = {}

#    element1 = '<ns0:path d="M 63.071871,264.267 V 271.30428 H 78.457001 V 154.10108 L 85.774098,154.10077 V 156.582 L 87.774096,157.531 89.773756,156.582 89.774174,150.12138 74.457001,150.12171 V 267.32467 L 67.07187,267.32475 67.072128,264.267 65.08986,265.203 Z" id="normal_arrow" style="fill:#73b7d1;fill-opacity:1;stroke:#fbfbfb;stroke-width:0.55299997px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccccccccccc" ns1:connector-curvature="0" ns1:label="#path8251" />'
#    element1 = 'ns0:path d="M 63.071871,264.267 V 271.30428 H 78.457001 V 154.10108 L 85.774098,154.10077 V 156.582 L 87.774096,157.531 89.773756,156.582 89.774174,150.12138 74.457001,150.12171 V 267.32467 L 67.07187,267.32475 67.072128,264.267 65.08986,265.203 Z" id="normal_arrow" style="fill:#73b7d1;fill-opacity:1;stroke:#fbfbfb;stroke-width:0.55299997px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccccccccccc" ns1:connector-curvature="0" ns1:label="#path8251" '
#    element2 = 'ns0:path d="M 63.154,271.035 H 86 V 267.455 L 66.6,267.45471 V 264.7052 L 64.872,265.528 63.154476,264.71131" id="low_bar" style="opacity:1;fill:#73b7d1;fill-opacity:1;stroke:none;stroke-width:0.26458332px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccc" ns1:connector-curvature="0" ns1:label="#path8255" '
#    element3 = 'ns0:path d="M 62.851212,148.62795 62.853138,156.5571 64.853137,157.71202 66.852797,156.5571 66.853215,148.69463 64.85134,149.76798 Z" id="top_in" style="opacity:1;fill:#73b7d1;fill-opacity:1;stroke:#ffffff;stroke-width:0.55299997;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccc" ns1:connector-curvature="0" ns1:label="#path8255" '
#    element4 = 'ns0:path d="M 66.872,264.29 66.873585,272.19258 64.873925,273.3475 62.873926,272.19258 62.872,264.29 64.872128,265.203 Z" id="bottom_out" style="fill:#73b7d1;fill-opacity:1;stroke:#ffffff;stroke-width:0.55299997;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccc" ns1:connector-curvature="0" ns1:label="#path8274" '
#    element5 = 'ns0:path d="M 67.468749,150.121 74.457001,150.12171 89.774172,150.12138 89.773754,156.582 87.774094,157.531 85.774098,156.582 V 154.10077 L 67.468748,154.101 Z" id="top_connect_1" style="fill:#73b7d1;fill-opacity:1;stroke:#fcfcfc;stroke-width:0.55299997;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccccc" ns1:connector-curvature="0" ns1:label="#path8250" />\n<ns0:path d="M 69.297345,150.399 H 65.921953 L 65.911237,153.823 H 69.279 Z" id="top_connect_2" style="fill:#73b7d1;fill-opacity:1;stroke:none;stroke-width:0.26458332px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccc" ns1:connector-curvature="0" ns1:label="#path8262" '
#    element6 = '<ns0:rect height="3.4260001" id="rect_join_mask" style="stroke-width:0.25686634;fill:#73b7d1;fill-opacity:1" transform="translate(%.2f)" width="1.9203489" x="62.342754" y="150.399" ns1:label="#rect3787" /> '
#    element7 = '<ns0:rect height="3.4300554" id="rect_low_mask" style="fill:#73b7d1;fill-opacity:1;stroke-width:0.26458332"  transform="translate(%.2f)" width="1.4658357" x="66.373032" y="267.59995" ns1:label="#rect8273" />'

    element1 = '<path d="M 63.071871,264.267 V 271.30428 H 78.457001 V 154.10108 L 85.774098,154.10077 V 156.582 L 87.774096,157.531 89.773756,156.582 89.774174,150.12138 74.457001,150.12171 V 267.32467 L 67.07187,267.32475 67.072128,264.267 65.08986,265.203 Z" id="normal_arrow" style="fill:#73b7d1;fill-opacity:1;stroke:#fbfbfb;stroke-width:0.55299997px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccccccccccc" ns1:connector-curvature="0" ns1:label="#path8251" />'
    element1 = 'path d="M 63.071871,264.267 V 271.30428 H 78.457001 V 154.10108 L 85.774098,154.10077 V 156.582 L 87.774096,157.531 89.773756,156.582 89.774174,150.12138 74.457001,150.12171 V 267.32467 L 67.07187,267.32475 67.072128,264.267 65.08986,265.203 Z" id="normal_arrow" style="fill:#73b7d1;fill-opacity:1;stroke:#fbfbfb;stroke-width:0.55299997px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccccccccccc" ns1:connector-curvature="0" ns1:label="#path8251" '
    element2 = 'path d="M 63.154,271.035 H 86 V 267.455 L 66.6,267.45471 V 264.7052 L 64.872,265.528 63.154476,264.71131" id="low_bar" style="opacity:1;fill:#73b7d1;fill-opacity:1;stroke:none;stroke-width:0.26458332px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccc" ns1:connector-curvature="0" ns1:label="#path8255" '
    element3 = 'path d="M 62.851212,148.62795 62.853138,156.5571 64.853137,157.71202 66.852797,156.5571 66.853215,148.69463 64.85134,149.76798 Z" id="top_in" style="opacity:1;fill:#73b7d1;fill-opacity:1;stroke:#ffffff;stroke-width:0.55299997;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccc" ns1:connector-curvature="0" ns1:label="#path8255" '
    element4 = 'path d="M 66.872,264.29 66.873585,272.19258 64.873925,273.3475 62.873926,272.19258 62.872,264.29 64.872128,265.203 Z" id="bottom_out" style="fill:#73b7d1;fill-opacity:1;stroke:#ffffff;stroke-width:0.55299997;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccc" ns1:connector-curvature="0" ns1:label="#path8274" '
    element5 = 'path d="M 67.468749,150.121 74.457001,150.12171 89.774172,150.12138 89.773754,156.582 87.774094,157.531 85.774098,156.582 V 154.10077 L 67.468748,154.101 Z" id="top_connect_1" style="fill:#73b7d1;fill-opacity:1;stroke:#fcfcfc;stroke-width:0.55299997;stroke-linecap:butt;stroke-linejoin:miter;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccccccc" ns1:connector-curvature="0" ns1:label="#path8250" />\n<path d="M 69.297345,150.399 H 65.921953 L 65.911237,153.823 H 69.279 Z" id="top_connect_2" style="fill:#73b7d1;fill-opacity:1;stroke:none;stroke-width:0.26458332px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1" transform="translate(%.2f)" ns2:nodetypes="ccccc" ns1:connector-curvature="0" ns1:label="#path8262" '
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
        if len(dico_out_b[sb]) == 1:
            if (len(dico_in_b[dico_out_b[sb][0]]) == 1) & (dico_out_b[sb] == sb+1):
                normal_arrow[sb] = 1
            else:
                prev_list = [i for i in range(dico_out_b[sb][0]-len(dico_in_b[dico_out_b[sb][0]]), dico_out_b[sb][0])]
                contrib = dico_in_b[dico_out_b[sb][0]]
                if (prev_list == contrib) & (contrib != []):
                    normal_arrow[np.max(contrib)] = 1
                    contrib_2 = [i for i in contrib if i != (np.max(contrib))]
                    for i in range(len(contrib_2)):
                        join_one_below[contrib_2[i]] = 1
               # else to add in later with extra levels of complexity
        elif len(dico_out_b[sb]) > 0:
            going_to = dico_out_b[sb]
            next_list = [i for i in range(sb+1, sb+1+len(dico_out_b[sb]))]
            if going_to == next_list:
                normal_arrow[sb] = 1
                going_to_2 = [i for i in going_to if i != (np.max(going_to))]
                for j in range(len(going_to_2)):
                    join_one_top[going_to_2[j]] = 1

    for sb in dico_out.keys():
        if 0 in dico_out[sb]:
            straight_out[sb] = 1
    for sb in dico_in.keys():
        if 0 in dico_in[sb]:
            straight_in[sb] = 1

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
            child = ET.Element(element5 %(offset, offset))
            layer = tree.findall('''*[@id='layer1']''')[0]
            layer.append(child)

        for j in join_one_below.keys():
            offset = (j - 1) * off1 +.22
            child = ET.Element(element2 %offset)
            layer = tree.findall('''*[@id='layer1']''')[0]
            layer.append(child)

        for j in straight_in.keys():
            if len(dico_in[j]) > 1:
                offset = (j - 1) * off1
                child = ET.Element(element6 %offset)
                layer = tree.findall('''*[@id='layer1']''')[0]
                layer.append(child)

        for j in straight_out.keys():
            if len(dico_out[j]) > 1:
                offset = (j - 1) * off1
                child = ET.Element(element7 %offset)
                layer = tree.findall('''*[@id='layer1']''')[0]
                layer.append(child)
        tree.write(outpath)
    else:
        shutil.copyfile(test_svg, outpath)
        print 'ERROR: unexpected number of arrows.\n      Basin structure too complex to generate Sheet7 arrows automatically.\
               \n      Standard template returned as output.'
    return outpath

def upstream_of_lu_class(dem_fh, lu_fh, output_folder, clss=63):
    """
    Calculate which pixels are upstream of a certain landuseclass.

    Parameters
    ----------
    dem_fh : str
        Filehandle pointing to a Digital Elevation Model.
    lu_fh : str
        Filehandle pointing to a landuse classification map.
    clss : int, optional
        Landuse identifier for which the upstream pixels will be determined.
        Default is 63 (Managed Water Bodies)
    output_folder : str
        Folder to store the map 'upstream.tif', contains value 1 for
        pixels upstream of waterbodies, 0 for pixels downstream.

    Returns
    -------
    upstream_fh : str
        Filehandle pointing to 'upstream.tif' map.
    """
    upstream_fh = os.path.join(output_folder, 'upstream.tif')

    if clss is not None:
        import pcraster as pcr

        temp_folder = tf.mkdtemp()
        extra_temp_folder = os.path.join(temp_folder, "out")
        os.makedirs(extra_temp_folder)

        temp_dem_fh = os.path.join(temp_folder, "dem.map")
        output1 = os.path.join(temp_folder, "catchments.map")
        output2 = os.path.join(temp_folder, "catchments.tif")
        temp2_lu_fh = os.path.join(temp_folder, "lu.map")

        srs_lu, ts_lu, te_lu, ndv_lu = becgis.GetGdalWarpInfo(lu_fh)
        te_lu_new = ' '.join([te_lu.split(' ')[0], te_lu.split(' ')[3], te_lu.split(' ')[2], te_lu.split(' ')[1]])

        GeoT = becgis.GetGeoInfo(dem_fh)[4]

        assert abs(GeoT[1]) == abs(GeoT[5]), "Please provide a DEM with square pixels. Unfortunately, PCRaster does not support rectangular pixels."

        temp1_lu_fh = becgis.MatchProjResNDV(dem_fh, np.array([lu_fh]), temp_folder)

        os.system("gdal_translate -projwin {0} -of PCRaster {1} {2}".format(te_lu_new, dem_fh, temp_dem_fh))
        os.system("gdal_translate -projwin {0} -of PCRaster {1} {2}".format(te_lu_new, temp1_lu_fh[0], temp2_lu_fh))

        dem = pcr.readmap(temp_dem_fh)
        ldd = pcr.lddcreate(dem, 9999999, 9999999, 9999999, 9999999)
        lulc = pcr.nominal(pcr.readmap(temp2_lu_fh))
        waterbodies = (lulc == clss)
        catch = pcr.catchment(ldd, waterbodies)
        pcr.report(catch, output1)

        os.system("gdal_translate -of GTiff {0} {1}".format(output1, output2))

        output3 = becgis.MatchProjResNDV(lu_fh, np.array([output2]), extra_temp_folder)

        upstream = becgis.OpenAsArray(output3[0], nan_values=True)
        upstream[np.isnan(upstream)] = 0.
        upstream = upstream.astype(np.bool)

        shutil.rmtree(temp_folder)

        if upstream_fh is not None:
            driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_fh)
            becgis.CreateGeoTiff(upstream_fh, upstream.astype(np.int8), driver, NDV, xsize, ysize, GeoT, Projection)
    else:
        dummy = becgis.OpenAsArray(lu_fh, nan_values=True) * 0.
        dummy = dummy.astype(np.bool)
        driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_fh)
        becgis.CreateGeoTiff(upstream_fh, dummy.astype(np.int8), driver, NDV, xsize, ysize, GeoT, Projection)

    print "Finished calculating up and downstream areas."
    return upstream_fh

def linear_fractions(lu_fh, upstream_fh, proxy_fh, output_fh, xs, unit='km',
                     quantity='Distance to water', gw_only_classes=None,
                     plot_graph=True):
    """
    Determine the fractions to split water supply or recoverable water into
    ground and surface water per pixel.

    Parameters
    ----------
    lu_fh : str
        Filehandle pointing to a landuse classification map.
    upstream_fh : str
        Filehandle pointing to map indicating areas upstream and downstream
        of managed water bodies.
    proxy_fh : str
        Filehandle pointing to a map with values used to determine a fraction based on
        a linear function.
    output_fh : str
        Filehandle to store results.
    xs : list
        List with 4 floats or integers, like [x1, x2, x3, x4]. The first two
        numbers refer to the linear relation used to determine alpha or beta upstream
        of waterbodies. The last two numbers refer to the linear relation used
        to determine alpha or beta downstream. x1 and x3 are the distances to water
        in kilometers up to which pixels will depend fully on surfacewater.
        x2 and x4 are the distances from which pixels will depend fully on groundwater.
        Pixels with a distance to surfacewater between x1 x2 and x3 x4 will depend on a mixture
        of surface and groundwater. Choose x1=x3 and x2=x4 to make no distinction
        between upstream or downstream.
    unit : str, optional
        Unit of the proxy, default is 'km'.
    quantity :str, optional
        Quantity of the proxy, default is 'Distance to water'.
    gw_only_classes : dict or None, optional
        Dictionary with the landuseclasses per category for sheet4b, i.e. lu_categories
        from the total_supply function. When this dictionary is provided, the pixel values for
        either beta or alpha for the landuseclasses 'Forests', Shrubland',
        'Rainfed Crops' and 'Forest Plantations' are set to zero. Use this to
        set beta to zero for these classes, since they are likely to only use
        groundwater. Default is None.
    plot_graph : boolean, optional
        True, plot a graph. False, dont plot a graph. Default is True.

    Returns
    -------
    alpha_fh : str
        Filehandle pointing to the rasterfile containing the values for alpha
        or beta.
    """

    upstream = becgis.OpenAsArray(upstream_fh).astype(np.bool)
    distances = becgis.OpenAsArray(proxy_fh, nan_values=True)

    f1 = interpolate.interp1d([xs[0], xs[1]], [1, 0], kind='linear',
                              bounds_error=False, fill_value=(1, 0))
    f2 = interpolate.interp1d([xs[2], xs[3]], [1, 0], kind='linear',
                              bounds_error=False, fill_value=(1, 0))

    LULC = becgis.OpenAsArray(lu_fh, nan_values=True)
    distances[np.isnan(LULC)] = np.nan

    alpha = np.zeros(np.shape(distances))
    alpha[upstream] = f1(distances[upstream])
    alpha[~upstream] = f2(distances[~upstream])

    if plot_graph:
        graph_fh = output_fh.replace('.tif', '.png')
        x = np.arange(np.nanmin(distances), np.nanmax(distances), 1)
        fig = plt.figure(figsize=(10, 10))
        plt.clf()
        ax = fig.add_subplot(111)
        ax.plot(x, f1(x), '--k', label='Fraction Upstream')
        ax.plot(x, f2(x), '-k', label='Fraction Downstream')
        ax.set_ylabel('Fraction [-]')
        ax.set_xlabel('{1} [{0}]'.format(unit, quantity))
        ax.set_zorder(2)
        ax.patch.set_visible(False)
        ax.set_ylim([0, 1])
        plt.legend(loc='lower right')
        plt.title('Fractions from {0}'.format(quantity))
        plt.suptitle('x1 = {0:.0f}{4}, x2 = {1:.0f}{4}, x3 = {2:.0f}{4}, x4 = {3:.0f}{4}'.format(xs[0], xs[1], xs[2], xs[3], unit))

        bins = np.arange(np.nanmin(distances), np.nanmax(distances), (np.nanmax(distances) - np.nanmin(distances)) / 35)
        hist_up, bins_up = np.histogram(distances[upstream & ~np.isnan(distances)], bins=bins)
        hist_down, bins = np.histogram(distances[~upstream & ~np.isnan(distances)], bins=bins_up)
        width = bins[1] - bins[0]
        ax2 = ax.twinx()
        ax2.bar(bins[:-1], hist_up, width, color='#6bb8cc', label='Upstream')
        ax2.bar(bins[:-1], hist_down, width, color='#a3db76',
                bottom=hist_up, label='Downstream')
        ax2.set_ylabel('Frequency [-]')
        ax2.set_zorder(1)
        ax2.patch.set_visible(True)
        ax2.set_xlim([x[0], x[-1]])
        plt.legend(loc='upper right')

        ax.legend(loc='upper right', fancybox=True, shadow=True)
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11), fancybox=True, shadow=True, ncol=5)
        plt.savefig(graph_fh)

    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(lu_fh)
    becgis.CreateGeoTiff(output_fh, alpha, driver, NDV, xsize, ysize, GeoT, Projection)

    if gw_only_classes is not None:
        try:
            ls = [gw_only_classes['Forests'],
                  gw_only_classes['Shrubland'],
                  gw_only_classes['Rainfed Crops'],
                  gw_only_classes['Forest Plantations']]
        except KeyError:
            print 'Please provide a dictionary with at least the following keys: \
                Forests, Shrubland, Rainfed Crops and Forest plantations'
        classes = list(becgis.Flatten(ls))
        becgis.set_classes_to_value(output_fh, lu_fh, classes, value=0)


def dryness_fractions(p_fh, std_fh, mean_fh, fractions_dryness_fh, base=-0.5, top=0.0):
    """
    Calculate fractions per pixel based on the distribution P compared to its long-term
    mean and std.

    Parameters
    ----------
    p_fh : str
        Filehandle pointing to precipitation map.
    std_fh : str
        Filehandle pointing to standard deviation map.
    mean_fh : str
        Filehandle pointing to map with mean values.
    fractions_dryness_fh : str
        Filehandle indicating where to store result.
    base : float, optional
        Use base to shift to lower boundary. Fractions are zero for pixels where
        the precipitation is smaller than (mean + base * std). Default is -0.5.
    top : float, optional
        Use top to shift the upper boundary. Fractions are one for pixels where the precipitation
        is larger than (mean + top *std). Default is 0.0.
    """
    P = becgis.OpenAsArray(p_fh, nan_values=True)
    STD = becgis.OpenAsArray(std_fh, nan_values=True)
    MEAN = becgis.OpenAsArray(mean_fh, nan_values=True)

    BASE = MEAN + base * STD
    TOP = MEAN + top * STD

    fractions = np.where(P > BASE, np.min([(1.0 / (TOP - BASE)) * (P - BASE), np.ones(np.shape(P))], axis=0), np.zeros(np.shape(P)))

    driver, NDV, xsize, ysize, GeoT, Projection = becgis.GetGeoInfo(p_fh)

    if not os.path.exists(os.path.split(fractions_dryness_fh)[0]):
        os.makedirs(os.path.split(fractions_dryness_fh)[0])

    becgis.CreateGeoTiff(fractions_dryness_fh, fractions, driver, NDV, xsize, ysize, GeoT, Projection)


def calc_fractions(p_data, output_dir, dem_fh, lu_fh, fraction_altitude_xs):
    p_fhs, p_dates = p_data
    dem_reproj_fhs = becgis.MatchProjResNDV(lu_fh, np.array([dem_fh]), output_dir)
    upstream_fh = upstream_of_lu_class(dem_fh, lu_fh, output_dir, clss=None)
    fractions_altitude_fh = os.path.join(output_dir, 'fractions_altitude.tif')
    linear_fractions(lu_fh, upstream_fh, dem_reproj_fhs[0], fractions_altitude_fh,
                     fraction_altitude_xs, unit='m', quantity='Altitude',
                     plot_graph=False)

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
        dryness_fractions(p_fhs[p_dates == date][0], std_fh, mean_fh,
                          fractions_dryness_fh, base=-0.5, top=0.0)
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
        'Irrigated crops':      [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
        'Managed water bodies': [63, 74, 75, 77],
        'Aquaculture':          [65],
        'Residential':          [47, 66, 68, 72],
        'Greenhouses':          [64],
        'Other':                [68, 69, 70, 71, 76, 78]}

    manmade_categories = ['Irrigated crops', 'Managed water bodies',
                          'Aquaculture', 'Residential', 'Greenhouses', 'Other']
    man_lu = [lucs[m] for m in manmade_categories]
    flat_man = [item for sublist in man_lu for item in sublist]
    natural_categories = ['Forests', 'Shrubland', 'Rainfed Crops',
                          'Forest Plantations', 'Natural Water Bodies',
                          'Wetlands', 'Natural Grasslands', 'Other (Non-Manmade)']
    nat_lu = [lucs[m] for m in natural_categories]
    flat_natural = [item for sublist in nat_lu for item in sublist]

    man_dict = {'man':flat_man, 'natural':flat_natural}

    return man_dict

def create_sheet5_svg(basin, sb_codes, period, units, data, output, template=False, smart_unit=False):

    df = pd.read_csv(data, sep=';')
    scale = 0
    if smart_unit:
        scale_test = np.nanmax(df['VALUE'].values)
        scale = hl.scale_factor(scale_test)
        df['VALUE'] *= 10**scale

    svg_template_path = os.path.abspath(template)

    tree = ET.parse(svg_template_path)

    xml_txt_box = tree.findall('''.//*[@id='unit']''')[0]
    if np.all([smart_unit, scale > 0]):
        list(xml_txt_box)[0].text = 'Sheet 5b: Surface Water ({0} km3)'.format(10**-scale)
    else:
        list(xml_txt_box)[0].text = 'Sheet 5b: Surface Water (km3)'

    xml_txt_box = tree.findall('''.//*[@id='basin']''')[0]
    list(xml_txt_box)[0].text = 'Basin: ' + basin.replace('_', ' ')

    xml_txt_box = tree.findall('''.//*[@id='period']''')[0]
    list(xml_txt_box)[0].text = 'Period: ' + period.replace('_', '-')

    line_id0 = [31633, 30561, 30569, 30577, 30585, 30905, 30913, 30921, 30929,
                31873, 31993, 32001, 32026, 32189, 32197, 32318, 32465, 32609,
                31273, 31281, 31289, 31297, 32817]
    line_lengths = [1, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 2, 1, 2, 2, 1, 1, 1, 4, 4, 4, 4, 1]
    line_names = ['Inflow',
                  'Fast Runoff: PROTECTED', 'Fast Runoff: UTILIZED',
                  'Fast Runoff: MODIFIED', 'Fast Runoff: MANAGED',
                  'Slow Runoff: PROTECTED', 'Slow Runoff: UTILIZED',
                  'Slow Runoff: MODIFIED', 'Slow Runoff: MANAGED',
                  'Total Runoff',
                  'SW withdr. manmade', 'SW withdr. natural',
                  'SW withdr. total',
                  'Return Flow SW', 'Return Flow GW',
                  'Total Return Flow',
                  'Interbasin Transfer', 'SW storage change',
                  'Outflow: Committed', 'Outflow: Non Recoverable',
                  'Outflow: Non Utilizable', 'Outflow: Utilizable',
                  'Outflow: Total']
    current_variables = ['Inflow',
                         'Fast Runoff: PROTECTED', 'Fast Runoff: UTILIZED',
                         'Fast Runoff: MODIFIED', 'Fast Runoff: MANAGED',
                         'Slow Runoff: PROTECTED', 'Slow Runoff: UTILIZED',
                         'Slow Runoff: MODIFIED', 'Slow Runoff: MANAGED',
                         'SW withdr. manmade', 'SW withdr. natural',
                         'SW withdr. total',
                         'Return Flow SW', 'Return Flow GW', 'Total Return Flow',
                         'Total Runoff', 'Outflow: Committed', 'Outflow: Non Recoverable',
                         'Outflow: Non Utilizable', 'Outflow: Utilizable',
                         'Outflow: Total',
                         'Interbasin Transfer',
                         'SW storage change'
                        ]
#    current_variables = line_names
    for var1 in current_variables:
        line_nb = [i for i in range(len(line_names)) if line_names[i] == var1][0]
        line_0 = line_id0[line_nb]
        line_len = line_lengths[line_nb]
        df_var = df.loc[df.VARIABLE == var1]
        sb_order = sb_codes
        value_sum = 0
        for sb in sb_order:
            df_sb = df_var.loc[df_var.SUBBASIN == str(sb)]
            cell_id = 'g' + str(line_0 + 8*(sb-1)*line_len)
            xml_txt_box = tree.findall('''.//*[@id='{0}']'''.format(cell_id))[0]
            xml_txt_box[0].text = '%.1f' %(df_sb.VALUE)
            value_sum += float(df_sb.VALUE)

        cell_id = 'g' + str(line_0 + 8*9*line_len)
        df_sb = df_var.loc[df_var.SUBBASIN == 'basin']
        xml_txt_box = tree.findall('''.//*[@id='{0}']'''.format(cell_id))[0]
        xml_txt_box[0].text = '%.1f' %(df_sb.VALUE)

    tempout_path = output.split('.')[0]+'.svg'
    tree.write(tempout_path)
    out_png = output
    subprocess.call([get_path('inkscape'), tempout_path,
                     '--export-dpi=300', '--export-png='+out_png])
    os.remove(tempout_path)
    return

class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def lu_type_sum_subbasins(data_fh, lu_fh, lu_dict, sb_fhs_code_names):
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
    LULC = becgis.OpenAsArray(lu_fh)
    AREA = becgis.MapPixelAreakm(data_fh)
    in_data = becgis.OpenAsArray(data_fh, nan_values=True) * AREA / 1e6
    out_data = Vividict()
    sb_fhs = zip(*sb_fhs_code_names)[0]
    sb_codes = zip(*sb_fhs_code_names)[1]
    for j in range(len(sb_fhs)):
        sb_fh = sb_fhs[j]
        sb_code = sb_codes[j]
        sb_mask = becgis.OpenAsArray(sb_fh)
        sb_mask[sb_mask != 1] = 0
        sb_mask = sb_mask.astype('bool')
        for lu_class in lu_dict.keys():
            mask = [LULC == value for value in lu_dict[lu_class]]
            mask = (np.sum(mask, axis=0)).astype(bool)
            mask = mask * sb_mask
            out_data[sb_code][lu_class] = np.nansum(in_data[mask])
    return out_data

def sum_subbasins(data_fh, sb_fhs_code_names):
    """
    Returns totals in a dict split by subbasin
    Parameters
    ----------
    data_fh : str
        location of the map of the data to split
    sb_fhs_code_names : list of tuples
        (sb_fhs,sb_codes,sb_names)
    """
    AREA = becgis.MapPixelAreakm(data_fh)
    in_data = becgis.OpenAsArray(data_fh, nan_values=True) * AREA / 1e6
    out_data = Vividict()
    sb_fhs = zip(*sb_fhs_code_names)[0]
    sb_codes = zip(*sb_fhs_code_names)[1]
    for j in range(len(sb_fhs)):
        sb_fh = sb_fhs[j]
        sb_code = sb_codes[j]
        sb_mask = becgis.OpenAsArray(sb_fh)
        sb_mask[sb_mask != 1] = 0
        sb_mask = sb_mask.astype('bool')
        out_data[sb_code] = np.nansum(in_data[sb_mask])
    return out_data

def read_inflow_file(inflowtext, date_list):
    df = pd.read_csv(inflowtext, delimiter=' ', skiprows=1, header=None,
                     names=['date', 'inflow'])
    try:
        date_py = np.array([datetime.datetime.fromordinal(dt) for dt in df.date])
    except:
        date_py = np.array([datetime.datetime.strptime(dt,'%d-%m-%Y') for dt in df.date])
    date_index = [np.where(date_py == k)[0][0] for k in date_list]
    inflows = np.array(df.inflow)[date_index]
    return inflows

def discharge_at_points(PointShapefile, SWpath):
    SW = nc.Dataset(SWpath)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(PointShapefile, 0)
    layer = dataSource.GetLayer()
    featureCount = layer.GetFeatureCount()
    discharge_natural = []
    discharge_end = []
    stat_name = []
    for pt in range(featureCount):
        feature = layer.GetFeature(pt)
        stat_name.append(feature.GetField(1))
        geometry = feature.GetGeometryRef()
        x = geometry.GetX()
        y = geometry.GetY()
        pos_x = (np.abs(SW.variables['longitude'][:]-x)).argmin()
        pos_y = (np.abs(SW.variables['latitude'][:]-y)).argmin()
        discharge_natural.append(SW.variables['discharge_natural'][:, pos_y, pos_x])
        discharge_end.append(SW.variables['discharge_end'][:, pos_y, pos_x])
        sw_time = [date.fromordinal(d) for d in SW.variables['time'][:]]
    return sw_time, discharge_natural, discharge_end, stat_name

def discharge_split(wpl_fh, ewr_fh, discharge_sum, ro_fhs, fractions_fhs,
                    sb_fhs_code_names, date_list):
    results = Vividict()
    gray_water_fraction = {}
    ewr_percentage = {}

    sb_fhs = zip(*sb_fhs_code_names)[0]
    sb_codes = zip(*sb_fhs_code_names)[1]

    long_disch_mean = np.mean([discharge_sum[k] for k in sb_codes], axis=1)

    for i in range(len(sb_fhs)):
        sb_fh = sb_fhs[i]
        sb_code = sb_codes[i]
        gray_water_fraction[sb_code] = becgis.calc_basinmean(wpl_fh, sb_fh)
        ewr_percentage[sb_code] = becgis.calc_basinmean(ewr_fh, sb_fh)
    t = 0
    for d in date_list:
        datestr1 = "%04d_%02d" %(d.year, d.month)
        datestr2 = "%04d%02d" %(d.year, d.month)
        ystr = "%04d" %(d.year)
        mstr = "%02d" %(d.month)
        ro_fh = ro_fhs[np.where([datestr2 in ro_fhs[i] for i in range(len(ro_fhs))])[0][0]]
        AREA = becgis.MapPixelAreakm(ro_fh)
        runoff = becgis.OpenAsArray(ro_fh, nan_values=True) * AREA / 1e6
        fractions_fh = fractions_fhs[np.where([datestr1 in fractions_fhs[i] for i in range(len(fractions_fhs))])[0][0]]
        fractions = becgis.OpenAsArray(fractions_fh, nan_values=True)

        non_utilizable_runoff = runoff * fractions
        non_utilizable_sum = {}
        for i in range(len(sb_fhs)):
            sb_fh = sb_fhs[i]
            sb_code = sb_codes[i]
            sb_mask = becgis.OpenAsArray(sb_fh)
            sb_mask[sb_mask != 1] = 0
            sb_mask = sb_mask.astype('bool')
            non_utilizable_sum[sb_code] = np.nansum(non_utilizable_runoff[sb_mask])

            results[ystr][mstr]['non_recoverable_outflow'][sb_code] = gray_water_fraction[sb_code] * discharge_sum[sb_code][t]
            reserved_outflow_demand = long_disch_mean[i] * ewr_percentage[sb_code]

            non_consumed_water = discharge_sum[sb_code][t] - results[ystr][mstr]['non_recoverable_outflow'][sb_code]

            results[ystr][mstr]['non_utilizable_outflow'][sb_code] = min(non_consumed_water, max(0.0, non_utilizable_sum[sb_code]))
            # note: committed = reserved_outflow_actual
            results[ystr][mstr]['committed_outflow'][sb_code] = min(non_consumed_water - results[ystr][mstr]['non_utilizable_outflow'][sb_code], reserved_outflow_demand)
            results[ystr][mstr]['utilizable_outflow'][sb_code] = max(0.0, non_consumed_water - results[ystr][mstr]['non_utilizable_outflow'][sb_code] - results["%04d" %(d.year)]["%02d" %(d.month)]['committed_outflow'][sb_code])
            results[ystr][mstr]['total_outflow'][sb_code] = discharge_sum[sb_code][t]
        t += 1
    return results


def create_csv(results, output_fh):
    """
    Create the csv-file for sheet 5.

    Parameters
    ----------
    results : dict
        Dictionary of results generated in sheet5_run.py
    output_fh : str
        Filehandle to store the csv-file.
    """
    first_row = ['SUBBASIN', 'VARIABLE', 'VALUE', 'UNITS']
    if not os.path.exists(os.path.split(output_fh)[0]):
        os.makedirs(os.path.split(output_fh)[0])
    csv_file = open(output_fh, 'wb')
    writer = csv.writer(csv_file, delimiter=';')
    writer.writerow(first_row)
    lu_classes = ['PROTECTED', 'UTILIZED', 'MODIFIED', 'MANAGED']
    for sb in results['surf_runoff'].keys():
        writer.writerow([sb, 'Inflow', '{0}'.format(results['inflows'][sb]), 'km3'])
        for lu_class in lu_classes:
            writer.writerow([sb, 'Fast Runoff: '+lu_class, '{0}'.format(results['surf_runoff'][sb][lu_class]), 'km3'])
            writer.writerow([sb, 'Slow Runoff: ' +lu_class, '{0}'.format(results['base_runoff'][sb][lu_class]), 'km3'])
        writer.writerow([sb, 'Total Runoff', '{0}'.format(results['total_runoff'][sb]), 'km3'])
        writer.writerow([sb, 'SW withdr. manmade', '{0}'.format(results['withdrawls'][sb]['man']), 'km3'])
        writer.writerow([sb, 'SW withdr. natural', '{0}'.format(results['withdrawls'][sb]['natural']), 'km3'])
        writer.writerow([sb, 'SW withdr. total', '{0}'.format(results['withdrawls'][sb]['man']+results['withdrawls'][sb]['natural']), 'km3'])
        writer.writerow([sb, 'Return Flow SW', '{0}'.format(results['return_sw_sw'][sb]), 'km3'])
        writer.writerow([sb, 'Return Flow GW', '{0}'.format(results['return_gw_sw'][sb]), 'km3'])
        writer.writerow([sb, 'Total Return Flow', '{0}'.format(results['return_sw_sw'][sb]+results['return_gw_sw'][sb]), 'km3'])
        writer.writerow([sb, 'Outflow: Total', '{0}'.format(results['total_outflow'][sb]), 'km3'])
        writer.writerow([sb, 'Outflow: Committed', '{0}'.format(results['committed_outflow'][sb]), 'km3'])
        writer.writerow([sb, 'Outflow: Non Recoverable', '{0}'.format(results['non_recoverable_outflow'][sb]), 'km3'])
        writer.writerow([sb, 'Outflow: Non Utilizable', '{0}'.format(results['non_utilizable_outflow'][sb]), 'km3'])
        writer.writerow([sb, 'Outflow: Utilizable', '{0}'.format(results['utilizable_outflow'][sb]), 'km3'])
        writer.writerow([sb,'Interbasin Transfer','{0}'.format(results['interbasin_transfers'][sb]),'km3'])
        writer.writerow([sb, 'SW storage change', '{0}'.format(results['deltaS'][sb]), 'km3'])
    csv_file.close()
    return
