# WA_Hyperloop

## Inputs

### Metadata

#### name
**string**, Basin name.

#### lu
**string**, Path to landuse geotiff-file.

#### full_basin_mask
**string**, Path to geotiff-file, 1 is part of basin, 0 is not part of basin.

#### masks
**string**, Path to folder containing geotiff-files for each subbasin.

#### crops
**list**, 

#### non_crop
**list**, 

#### recycling_ratio
**float**, Parameter indicating how much of ET is recycled within the basin.

#### dico_in
**dictionary**, 

#### dico_out
**dictionary**,

#### GRACE
**string**, Path to csv file with timeseries of GRACE data

#### fraction_xs
**list**, 

#### discharge_out_from_wp
**boolean**, Value is true if WaterPix is used directly in sheet5 rather than Surfwat

#### lu_based_supply_split
**boolean**, Value is True if an initial split in SW/GW supply is done based on landuse class and values in get_dictionnaries

#### grace_supply_split
**boolean**, Value is True if GW/SW split is adjusted. Can be true weather or not initial split based on landuse is done. If both of these are False, all supply will be SWsupply

#### grace_split_alpha_bounds
**tuple**, lower and upper bounds of trigonometric function parameters for splitting suply into sw and gw as ([alpha_l, beta_l, theta_l],[alpha_u, beta_u, theta_u]). ([0., 0., 0.], [1.0, 1.0, 12.]) are the widest bounds allowed. alpha controls the mean, beta the amplitude and theta the phase.
![alt text](/docs/alpha_beta_theta.png "alpha_beta_theta")

#### water_year_start_month
**integer**, Start month of water year. Used to compute the yearly sheets.

#### ndm_max_original
**boolean**, True will use original method to determine NDM_max (based on entire domain), false will use a different method dependent on nearby pixels of the same lu-category.
