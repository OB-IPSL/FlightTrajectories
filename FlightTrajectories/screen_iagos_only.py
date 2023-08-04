import xarray as xr
import pandas as pd
import numpy as np
import glob
import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import great_circle_calculator.great_circle_calculator as gcc
from FlightTrajectories.misc_geo import haversine

#--path IAGOS files
path_iagos='/bdd/IAGOS/netcdf/'  #--path where IAGOS files are located
lenstr=62                        #--length of string before flight ID
#
#--paths out
pathout='../FLIGHTS/'
pathgraph='../GRAPHS/'
#
#--airlines
airlines=['Air France','Cathay Pacific','China Airlines','Hawaiian Airlines','Lufthansa']
#
#--aircraft IDs and type
aircraft_ids=['D-AIGT','D-AIKO','D-AIHE','N384HA','B-18316','B-18317','B-18806','F-GLZU','F-GZCO','B-HLR']
aircraft_types=['A340-313','A330-343','A340-642','A330-243','A330-302','A330-302','A340-313','A340-313','A330-203','A330-343']
#
#--year min and max
yr_min=2018
yr_max=2021
#
#--minimum distance in m to be considered for a meaningful IAGOS flight to optimise (4000 km)
distmin = 2500000.
print('Minimal cruising distance to screen IAGOS flights=',distmin)
#
#--physical constants
gamma=1.4               #--specific heat ratio
kb=1.38e-23             #--Boltzmann constant
air_density=29.e-3      #--air density (kg/m3)
avogadro=6.02e23        #--Avogadro number
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
plt.savefig(pathgraph+'world_map_with_regions.png')
plt.close()
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
df_aircraft=pd.DataFrame({"Aircraft ID": [aircraft_id for aircraft_id in aircraft_ids],\
                             "Aircraft type": [aircraft_type for aircraft_type in aircraft_types]})
df_aircraft.set_index('Aircraft ID',inplace=True)
#
groundspeed_ave_list=[]
groundspeed_std_list=[]
airspeed_ave_list=[]
airspeed_std_list=[]
machnumber_ave_list=[]
machnumber_std_list=[]
#
#--loop on years
for yr in range(yr_min,yr_max+1):
  #--add yr to dataframe
  df_airline[str(yr)+' all']=[0 for airline in airlines]
  df_aircraft[str(yr)+' all']=[0 for aircraft_id in aircraft_ids]
  df_airline[str(yr)+' screened']=[0 for airline in airlines]
  df_aircraft[str(yr)+' screened']=[0 for aircraft_id in aircraft_ids]
  #
  #--select all IAGOS files
  #--files are located in directories YYYYMM
  files_iagos_all=sorted(glob.glob(path_iagos+str(yr)+'??/*.nc4'))
  #
  #--clean datasets by removing duplicates with different versions
  files_iagos_seeds=sorted(list(set([file[0:lenstr] for file in files_iagos_all])))   #--/bdd/IAGOS/netcdf
  #--create new list
  files_iagos=[]
  for seed in files_iagos_seeds:
      #--we only keep the latest version of each file if duplicates
      file_iagos=[file for file in files_iagos_all if file[0:lenstr]==seed][-1]  #--/bdd/IAGOS/netcdf
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
    #--aircraft_id
    aircraft_id=iagos.platform.split(',')[-2].strip(' ')
    print('aircraft_id is ',aircraft_id)
    #--increment counter
    df_aircraft.loc[aircraft_id,str(yr)+' all'] += 1
    df_airline.loc[airline,str(yr)+' all']      += 1
    #--extract data
    lat_iagos=iagos['lat']
    lon_iagos=iagos['lon']
    time_iagos=iagos['UTC_time']
    pressure_iagos=iagos['air_press_AC']
    temperature_iagos=iagos['air_temp_AC']
    airspeed_iagos=iagos['air_speed_AC']
    windspeed_iagos=iagos['wind_speed_AC']
    groundspeed_iagos=iagos['ground_speed_AC']
    #--define arrays of lon, lat, pressure, and airspeed values
    lon_iagos_values=lon_iagos.values
    lat_iagos_values=lat_iagos.values
    pressure_iagos_values=pressure_iagos.values
    temperature_iagos_values=temperature_iagos.values
    airspeed_iagos_values=airspeed_iagos.values
    windspeed_iagos_values=windspeed_iagos.values
    groundspeed_iagos_values=groundspeed_iagos.values
    #--exclude unknown airports    
    if lon_departure < -999. or lat_departure < -999. or lon_arrival < -999. or lat_arrival < -999.: 
        print('Error: Issue with longitude or latitude of airport')
        continue
    #
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
        print('Error: one or both of the regions not found') 
        continue
    #
    #--find lon, lat, alt of cruising similar to what FR24 database does
    #--this is empirical and needs to be checked
    #--the threshold depend on the time resolution of the IAGOS data
    ind=np.where((np.abs(np.diff(gaussian_filter1d(pressure_iagos,40)))<50.) & (pressure_iagos[:-1]<35000.))[0]
    #
    #--eliminate low-level flights
    if len(ind) == 0:
        print('Error: this flight is too low to be optimized so we stop here')
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
        print('Eror: some of the lat / lon are NaN. We stop here for this flight')
        continue
    #
    #--compute great circle distance
    dist = haversine(lat_p1, lon_p1, lat_p2, lon_p2)
    dist_gcc = gcc.distance_between_points((lon_p1,lat_p1),(lon_p2,lat_p2),unit='meters')
    print('Distance between airports (haversine & gcc) = ',"{:6.3f}".format(dist/1000.),"{:6.3f}".format(dist_gcc/1000.),'km')
    #
    #--eliminate flights with too short cruising distance
    if dist <= distmin: 
        print('Error: this flight is too short to be optimized so we stop here')
        continue
    #
    #--compute groundspeed statistics
    groundspeeds=groundspeed_iagos_values[ind]
    groundspeeds=groundspeeds[groundspeeds>=0]
    groundspeed_ave=np.average(groundspeeds)
    groundspeed_std=np.std(groundspeeds)
    print('Groundspeed = ',"{:6.1f}".format(groundspeed_ave),'+/-',"{:6.2f}".format(groundspeed_std))
    groundspeed_ave_list.append(groundspeed_ave)
    groundspeed_std_list.append(groundspeed_std)
    #
    #--compute airspeed statistics
    airspeeds=airspeed_iagos_values[ind]
    indices=airspeeds>=0
    airspeeds=airspeeds[indices]
    airspeed_ave=np.average(airspeeds)
    airspeed_std=np.std(airspeeds)
    print('Airspeed = ',"{:6.1f}".format(airspeed_ave),'+/-',"{:6.2f}".format(airspeed_std))
    if not np.isnan(airspeed_ave):
       airspeed_ave_list.append(airspeed_ave)
       airspeed_std_list.append(airspeed_std)
    #
    #--compute windspeed statistics
    windspeeds=windspeed_iagos_values[ind]
    windspeeds=windspeeds[indices]
    #
    #--compute Mach number statistics
    temperatures=temperature_iagos_values[ind]
    temperatures=temperatures[indices]
    #
    #--sound of speed for ideal gas according to Wikipedia: sqrt(gamma*kB*T/m)
    soundspeeds=np.sqrt(gamma*kb*temperatures/(air_density/avogadro))
    machnumbers=airspeeds/soundspeeds
    machnumber_ave=np.average(machnumbers)
    machnumber_std=np.std(machnumbers)
    print('Mach number = ',"{:5.4f}".format(machnumber_ave),'+/-',"{:5.4f}".format(machnumber_std))
    if not np.isnan(machnumber_ave):
       machnumber_ave_list.append(machnumber_ave)
       machnumber_std_list.append(machnumber_std)
    #print('Correlation coefficient airspeed - T=',np.corrcoef(airspeeds,temperatures)[0,1])
    #print('Correlation coefficient airspeed - windspeed =',np.corrcoef(airspeeds,windspeeds)[0,1])
    #
    #--increment counter
    df_aircraft.loc[aircraft_id,str(yr)+' screened'] += 1
    df_airline.loc[airline,str(yr)+' screened']      += 1
    #
    #--list paper files
    if (region1, region2) not in files.keys():
         files[(region1,region2)] = open(pathout+region1+'_to_'+region2+'_'+str(yr)+'.csv','w')
    files[(region1,region2)].write(file_iagos+'\n')
    #
#
#--make sums by year
df_airline.loc["Total"] = df_airline.sum()
df_aircraft.loc["Total"] = df_aircraft.sum()
#
#--df output
print('\n')
print('Sampled airlines=',df_airline)
print('Sampled flight ids=',df_aircraft)
#
#--latex output
print(df_airline.style.to_latex())
print(df_aircraft.style.to_latex())
#
#--Mean and st.dev. in speeds in relative terms
print('Averaged ground speeds   =',np.average(groundspeed_ave_list),'+/-',np.std(groundspeed_ave_list))
print('St. dev. of ground speeds=',np.average(groundspeed_std_list))
print('St. dev. of ground speeds=',np.average(groundspeed_std_list)/np.average(groundspeed_ave_list)*100,'%')
print('\n')
print('Averaged air speeds      =',np.average(airspeed_ave_list),'+/-',np.std(airspeed_ave_list))
print('St. dev. of air speeds   =',np.average(airspeed_std_list))
print('St. dev. of air speeds   =',np.average(airspeed_std_list)/np.average(airspeed_ave_list)*100,'%')
print('\n')
print('Averaged Mach number     =',np.average(machnumber_ave_list),'+/-',np.std(machnumber_ave_list))
print('St. dev. of Mach number  =',np.average(machnumber_std_list))
print('St. dev. of Mach number  =',np.average(machnumber_std_list)/np.average(machnumber_ave_list)*100,'%')
#
#--make plots
plt.hist(airspeed_ave_list)
plt.title('Histogram of average cruising air speed (m/s)')
plt.xlabel('Air speed (m/s)')
plt.savefig(pathgraph+'histogram_airspeed_ave.png')
plt.close()
plt.hist(airspeed_std_list)
plt.title('Histogram of standard deviation of cruising air speed (m/s)')
plt.xlabel('St. Dev. of air speed (m/s)')
plt.savefig(pathgraph+'histogram_airspeed_std.png')
plt.close()
plt.hist(groundspeed_std_list)
plt.title('Histogram of standard deviation of cruising ground speed (m/s)')
plt.xlabel('St. Dev. of ground speed (m/s)')
plt.savefig(pathgraph+'histogram_groundspeed_std.png')
plt.close()
plt.hist(machnumber_ave_list)
plt.title('Histogram of average cruising Mach number')
plt.xlabel('Mach number')
plt.savefig(pathgraph+'histogram_machnumber_ave.png')
plt.close()
