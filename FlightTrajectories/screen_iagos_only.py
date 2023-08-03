import xarray as xr
import pandas as pd
import numpy as np
import glob
import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import great_circle_calculator.great_circle_calculator as gcc
from misc_geo import haversine

#--path IAGOS files
path_iagos='/bdd/IAGOS/'
#
#--path out
pathout='FLIGHTS/'
#
#--airlines
airlines=['Air France','Cathay Pacific','China Airlines','Hawaiian Airlines','Lufthansa']
#
#--flight IDs
flightids=['D-AIGT','D-AIKO','D-AIHE','N384HA','B-18316','B-18317','B-18806','F-GLZU','F-GZCO','B-HLR']
#
#--year min and max
yr_min=2018
yr_max=2021
#yr_min=2018
#yr_max=2018
#
#--minimum distance in m to be considered for a meaningful IAGOS flight to optimise (4000 km)
#distmin = 4000000.
distmin = 2500000.
print('Minimal cruising distance to screen IAGOS flights=',distmin)
#
#--regions
regions=['europe','southamerica','northamerica','centralamerica','asia','centralasia',\
         'southeastasia','australasia','middleeast','africa','pacific']
lons1={} ; lons2={} ; lats1={} ; lats2={}
#
lons1['europe']=-15.0
lons2['europe']=+30.0
lats1['europe']=+35.0
lats2['europe']=+75.0
#
lons1['northamerica']=-170.0
lons2['northamerica']=-50.0
lats1['northamerica']=+30.0
lats2['northamerica']=+80.0
#
lons1['centralamerica']=-120.0
lons2['centralamerica']=-60.0
lats1['centralamerica']=+10.0
lats2['centralamerica']=+30.0
#
lons1['southamerica']=-100.0
lons2['southamerica']=-30.0
lats1['southamerica']=-60.0
lats2['southamerica']=+10.0
#
lons1['asia']=+90.0
lons2['asia']=+150.0
lats1['asia']=+0.0
lats2['asia']=+60.0
#
lons1['centralasia']=+30.0
lons2['centralasia']=+90.0
lats1['centralasia']=+40.0
lats2['centralasia']=+70.0
#
lons1['southeastasia']=+60.0
lons2['southeastasia']=+90.0
lats1['southeastasia']=+0.0
lats2['southeastasia']=+40.0
#
lons1['australasia']=+90.0
lons2['australasia']=+180.0
lats1['australasia']=-50.0
lats2['australasia']=+0.0
#
lons1['middleeast']=+30.0
lons2['middleeast']=+60.0
lats1['middleeast']=+15.0
lats2['middleeast']=+40.0
#
lons1['africa']={}
lons2['africa']={}
lats1['africa']={}
lats2['africa']={}
lons1['africa'][1]=-20.0
lons2['africa'][1]=+30.0
lats1['africa'][1]=-40.0
lats2['africa'][1]=+35.0
lons1['africa'][2]=+30.0
lons2['africa'][2]=+60.0
lats1['africa'][2]=-40.0
lats2['africa'][2]=+15.0
#
lons1['pacific']=-180.0
lons2['pacific']=-130.0
lats1['pacific']=-30.0
lats2['pacific']=+30.0
#
#--plot map of regions
plate = ccrs.PlateCarree()
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(111, projection=plate)
ax.set_xlim(-180.0,180.0)
ax.set_ylim(-90.0,90.0)
for region in regions:
  if region=='africa':
    ax.plot([lons1[region][1],lons2[region][2]],[lats1[region][1],lats1[region][1]], c='red', lw=2)
    ax.plot([lons1[region][1],lons2[region][1]],[lats2[region][1],lats2[region][1]], c='red', lw=2)
    ax.plot([lons1[region][1],lons1[region][1]],[lats1[region][1],lats2[region][1]], c='red', lw=2)
    ax.plot([lons2[region][2],lons2[region][2]],[lats1[region][2],lats2[region][2]], c='red', lw=2)
  else:
    ax.plot([lons1[region],lons2[region]],[lats1[region],lats1[region]], c='red', lw=2)
    ax.plot([lons1[region],lons2[region]],[lats2[region],lats2[region]], c='red', lw=2)
    ax.plot([lons1[region],lons1[region]],[lats1[region],lats2[region]], c='red', lw=2)
    ax.plot([lons2[region],lons2[region]],[lats1[region],lats2[region]], c='red', lw=2)
fig.gca().coastlines()
fig.gca().add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"),alpha=1)
fig.gca().add_feature(cartopy.feature.BORDERS,color="red",linewidth=0.3)
plt.savefig('GRAPHS/world_map_with_regions.png')
#plt.show()
#
#--is in region
def isinregion(region,lon,lat):
  if region=='africa':
    if lon >= lons1[region][1] and lon <= lons2[region][1] and lat >= lats1[region][1] and lat <= lats2[region][1] or \
       lon >= lons1[region][2] and lon <= lons2[region][2] and lat >= lats1[region][2] and lat <= lats2[region][2] :
       return True
    else:
       return False
  else:
    if lon >= lons1[region] and lon <= lons2[region] and lat >= lats1[region] and lat <= lats2[region]:
       return True
    else:
       return False
#
#--initialise dataframes
df_airline=pd.DataFrame({"Airline": [airline for airline in airlines]})
df_airline.set_index('Airline',inplace=True)
#
df_flightid=pd.DataFrame({"Flight ID": [flightid for flightid in flightids]})
df_flightid.set_index('Flight ID',inplace=True)
#
#--loop on years
for yr in range(yr_min,yr_max+1):
  #--add yr to dataframe
  df_airline[str(yr)+' all']=[0 for airline in airlines]
  df_flightid[str(yr)+' all']=[0 for flightid in flightids]
  df_airline[str(yr)+' screened']=[0 for airline in airlines]
  df_flightid[str(yr)+' screened']=[0 for flightid in flightids]
  #--select all IAGOS files
  files_iagos_all=sorted(glob.glob(path_iagos+str(yr)+'??/*.nc4'))
  #--clean datasets by removing duplicates with different versions
  files_iagos_seeds=sorted(list(set([file[0:55] for file in files_iagos_all])))
  #--create new list
  files_iagos=[]
  for seed in files_iagos_seeds:
      #--we only keep the latest version of each file if duplicates
      file_iagos=[file for file in files_iagos_all if file[0:55]==seed][-1]
      #--append to list
      files_iagos.append(file_iagos)
  print('We found ',len(files_iagos),' independent files for ',yr)
  #
  #--dictionary of output files 
  files={}
  #
  #--open IAGOS files
  for file_iagos in files_iagos:
    #--open file
    print('\n'+file_iagos)
    iagos=xr.open_dataset(file_iagos)
    #extract data from open file
    #get keys in dictionary and value
    #--departure
    airport_departure=iagos.departure_airport
    airport_code_departure=airport_departure.split(',')[0]
    country_departure=airport_departure.split(',')[2]
    coord_departure=iagos.departure_coord.split()
    lon_departure, lat_departure = float(coord_departure[0]), float(coord_departure[1])
    print('departure=',airport_code_departure,country_departure,lon_departure,lat_departure)
    #--arrival
    airport_arrival=iagos.arrival_airport
    airport_code_arrival=airport_arrival.split(',')[0]
    country_arrival=airport_arrival.split(',')[2]
    coord_arrival=iagos.arrival_coord.split()
    lon_arrival, lat_arrival = float(coord_arrival[0]), float(coord_arrival[1])
    print('arrival=',airport_code_arrival,country_arrival,lon_arrival,lat_arrival)
    #--airline
    airline=iagos.platform.split(',')[-1].strip(' ')
    print('airline is ',airline)
    #--flightid
    flightid=iagos.platform.split(',')[-2].strip(' ')
    print('flightid is ',flightid)
    #--increment counter
    df_flightid[str(yr)+' all'][flightid] += 1
    df_airline[str(yr)+' all'][airline]   += 1
    #--extract data
    lat_iagos=iagos['lat']
    lon_iagos=iagos['lon']
    time_iagos=iagos['UTC_time']
    pressure_iagos=iagos['air_press_AC']
    #--define arrays of lon, lat, pressure, temp and rhi values
    lon_iagos_values=lon_iagos.values
    lat_iagos_values=lat_iagos.values
    pressure_iagos_values=pressure_iagos.values
    #--exclude unknown airports    
    if lon_departure < -999. or lat_departure < -999. or lon_arrival < -999. or lat_arrival < -999.: continue
    #--find regions
    found_two_regions=False
    for region1 in regions: 
      if isinregion(region1,lon_departure,lat_departure):
        for region2 in regions: 
           if isinregion(region2,lon_arrival,lat_arrival): 
               print(region1+' => '+region2)
               found_two_regions=True
               break
        break
    #
    if not found_two_regions: 
        print('ATTENTION: one or both of the regions not found') 
        continue
    #
    #--find lon, lat, alt of cruising similar to what FR24 database does
    #--this is empirical and needs to be checked
    #--the threshold depend on the time resolution of the IAGOS data
    ind=np.where((np.abs(np.diff(gaussian_filter1d(pressure_iagos,40)))<50.) & (pressure_iagos[:-1]<35000.))[0]
    #
    #--eliminate low-level flights
    if len(ind) == 0:
        print('This flight is too low to be optimized so we stop here')
        continue
    #
    #--find longitude and latitude of beginning and end of cruising phase
    lon_p1=lon_iagos_values[ind[0]]
    lon_p2=lon_iagos_values[ind[-1]]
    lat_p1=lat_iagos_values[ind[0]]
    lat_p2=lat_iagos_values[ind[-1]]
    #
    print('Departure and arrival points')
    print('lon lat p1=',lon_p1,lat_p1)
    print('lon lat p2=',lon_p2,lat_p2)
    #
    #--eliminate unknown points
    if np.isnan(lat_p1) or np.isnan(lon_p1) or np.isnan(lon_p1) or np.isnan(lon_p2): 
        print('Some of the lat / lon are NaN. We stop here for this flight')
        continue
    #
    #--compute great circle distance
    dist = haversine(lat_p1, lon_p1, lat_p2, lon_p2)
    dist_gcc = gcc.distance_between_points((lon_p1,lat_p1),(lon_p2,lat_p2),unit='meters')
    print('Distance between airports (haversine & gcc) = ',"{:6.3f}".format(dist/1000.),"{:6.3f}".format(dist_gcc/1000.),'km')
    #
    #--eliminate flights with too short cruising distance
    if dist <= distmin: 
        print('This flight is too short to be optimized so we stop here')
        continue
    #
    #--increment counter
    df_flightid[str(yr)+' screened'][flightid] += 1
    df_airline[str(yr)+' screened'][airline]   += 1
    #
    #--list paper files
    if (region1, region2) not in files.keys():
         files[(region1,region2)] = open(pathout+region1+'_to_'+region2+'_'+str(yr)+'.csv','w')
    files[(region1,region2)].write(file_iagos+'\n')
    #
#
#--make sums by year
df_airline.loc["Total"] = df_airline.sum()
df_flightid.loc["Total"] = df_flightid.sum()
#--df output
print('Sampled airlines=',df_airline)
print('Sampled flight ids=',df_flightid)
#--latex output
print(df_airline.style.to_latex())
print(df_flightid.style.to_latex())
