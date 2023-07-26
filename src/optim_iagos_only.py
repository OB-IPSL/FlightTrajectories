#==================================================================================
#--Optimal flight route calculation compared against IAGOS actual route
#--From Ed Gryspeerdt and Olivier Boucher
#--October 2022
#==================================================================================
#
#--import numpy packages
import ipdb
from numba import jit
#
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime,timedelta
from scipy.optimize import minimize
from scipy import interpolate
from sklearn.metrics import mean_squared_error
import cartopy.crs as ccrs
import cartopy
import pandas as pd
import great_circle_calculator.great_circle_calculator as gcc
import argparse
import glob, os, sys
import warnings
import time
from math import exp,log
import pickle
from scipy.ndimage import gaussian_filter1d


from misc_geo import haversine, bearing, nearest, closest_argmin, sph2car, car2sph, G, el_foeew, ei_foeew
from optimalrouting import ZermeloLonLat



def process_grid(xr_u200, xr_v200, xr_t200, xr_r200, nbts, 
       lon_p1, lat_p1, lon_p2, lat_p2, lons_wind, lats_wind, 
       lon_iagos_values, lat_iagos_values, lon_key_values, lat_key_values):
    #--prepare plate grid
    plate = ccrs.PlateCarree()
    
    #--prepare meshgrid
    xx,yy=np.meshgrid(lons_wind,lats_wind)
    
    #--Deal with rotated grid
    #--convert lat-lon to radians
    lat_p1_rad=np.deg2rad(lat_p1)
    lon_p1_rad=np.deg2rad(lon_p1)
    lat_p2_rad=np.deg2rad(lat_p2)
    lon_p2_rad=np.deg2rad(lon_p2)
    
    #--convert to cartesian coordinates
    P1=sph2car(lat_p1_rad,lon_p1_rad)
    P2=sph2car(lat_p2_rad,lon_p2_rad)
    
    #--cross product P1^P2 - perpendicular to P1 and P2
    ON=np.cross(P1,P2)
    
    #--coordinates of new Pole in old system
    THETA,PHI=car2sph(ON)
    lat_pole=np.rad2deg(THETA)
    lon_pole=np.rad2deg(PHI)
    print('Coord lat lon new pole in original grid=',"{:5.2f}".format(lat_pole),"{:5.2f}".format(lon_pole))
    
    #--prepare rotated grid
    rotated = ccrs.RotatedPole(pole_latitude=lat_pole,pole_longitude=lon_pole)
    
    #--new coordinates of old pole (diagnostics)
    xyz=rotated.transform_points(plate,np.array([0.]),np.array([90.]))
    lon_pole_t=xyz[0,0]
    lat_pole_t=xyz[0,1]
    print('lon lat pole t=',"{:5.2f}".format(lon_pole_t),"{:5.2f}".format(lat_pole_t))
    
    #--new coordinates of p1,p2 points
    xyz=rotated.transform_points(plate,np.array([lon_p1,lon_p2]),np.array([lat_p1,lat_p2]))
    lon_p1=xyz[0,0] ; lat_p1=xyz[0,1]
    lon_p2=xyz[1,0] ; lat_p2=xyz[1,1]
    print('lon lat p1 t=',"{:5.2f}".format(lon_p1),"{:5.2f}".format(lat_p1))
    print('lon lat p2 t=',"{:5.2f}".format(lon_p2),"{:5.2f}".format(lat_p2))
    if lon_p2 < lon_p1: 
       print('lon_p2 < lon_p1 may pose problems later on - reconsider new pole')
       print('we stop here as this needs to be fixed if ever it happens')
       sys.exit()
    
    #--rotate lon_iagos and lat_iagos
    xyz=rotated.transform_points(plate,lon_iagos_values,lat_iagos_values)
    lon_iagos_values=xyz[:,0] 
    lat_iagos_values=xyz[:,1]
    
    #--rotate lon_key and lat_key values
    xyz=rotated.transform_points(plate,lon_key_values,lat_key_values)
    lon_key_values=xyz[:,0] 
    lat_key_values=xyz[:,1]
    
    #--rotate wind field on original gridpoints
    xr_u200_values_t=np.zeros((xr_u200.values.shape))
    xr_v200_values_t=np.zeros((xr_v200.values.shape))
    for t in range(nbts):
        xr_u200_values_t[t,:,:],xr_v200_values_t[t,:,:]=rotated.transform_vectors(plate,xx,yy,xr_u200.values[t,:,:],xr_v200.values[t,:,:])
    
    #--rotate meshgrid
    xyz=rotated.transform_points(plate,xx,yy)
    xx_t=xyz[:,:,0]
    yy_t=xyz[:,:,1]
    
    #--interpolate u_t and v_t on a regular grid on rotated grid using gridddata
    xx_t_yy_t=np.array([[ixt,iyt] for ixt,iyt in zip(xx_t.flatten(),yy_t.flatten())])
    for t in range(nbts):
       xr_u200.values[t,:,:]=interpolate.griddata(xx_t_yy_t,xr_u200_values_t[t,:,:].flatten(),(xx,yy),method='nearest')
       xr_v200.values[t,:,:]=interpolate.griddata(xx_t_yy_t,xr_v200_values_t[t,:,:].flatten(),(xx,yy),method='nearest')

    return plate, xr_nofly, xyz, lon_pole_t, lat_pole_t, lon_p1, lat_p1, lon_p2, lat_p2

##
##@jit(parallel=True)
#def cost_time(lons, lats, dtprint=False):
#   #--Array of positions (lons, lats)
#   positions=np.array([ [x,y] for x,y in zip(lons,lats) ])
#   #--Ground track info
#   #--Compute bearing(lat1, lon1, lat2, lon2)
#   bearings = bearing(positions[:-1, 1],positions[:-1, 0],positions[1:, 1],positions[1:, 0])
#   #--Compute distances
#   dists =  haversine(positions[:-1, 1],positions[:-1, 0],positions[1:, 1],positions[1:, 0])
#   #--interpolated wind speed on lon,lat array
#   wind_u, wind_v = wind(lons, lats)
#   #--Compute wind in the middles of segments
#   winds_u = (wind_u[:-1]+wind_u[1:])/2.0
#   winds_v = (wind_v[:-1]+wind_v[1:])/2.0
#   #--compute wind bearings
#   windbearings = np.rad2deg(np.arctan2(winds_u, winds_v))
#   #--Compute wind speed
#   windspeed = np.sqrt(winds_u**2+winds_v**2)
#   #--Follow law of cosines - http://hyperphysics.phy-astr.gsu.edu/hbase/lcos.html#c3
#   theta = bearings - windbearings
#   beta = np.rad2deg(np.arcsin(windspeed*np.sin(np.deg2rad(theta))/airspeed))
#   alpha = 180. - theta - beta
#   #--Resultant is the ground track speed
#   resultant = np.sqrt(airspeed**2 + windspeed**2 - 2*airspeed*windspeed*np.cos(np.deg2rad(alpha)))
#   #--Segment time is haversine distance divided by ground speed
#   segment_time = dists / resultant / 3600. # m / (m/s) / (s/hr)= hours
#   #--Compute total time over trajectory
#   total_time = segment_time.sum()  #--hours
#   #--Print if asked for
#   if dtprint: 
#      print('cost pos lat1=',positions[:-1, 1])
#      print('cost pos lon1=',positions[:-1, 0])
#      print('cost pos lat2=',positions[1:, 1])
#      print('cost pos lon2=',positions[1:, 0])
#      print('cost bearing=', bearings)
#      print('dists=',dists)
#      print('wind_u=',winds_u)
#      print('wind_v=',winds_v)
#      print('resultant=',resultant)
#      print('cost time dt=',segment_time)
#      print('total dist=',dists.sum())
#      print('total time=',total_time)
#   #--Return total flight time
#   return total_time
##
##--Cost function: return flight duration squared (this is the function to be minimized)
#def cost_squared(y, x0, lon1, lat1, lon2, lat2, dtprint=False):
#   #--y is the vector to be optimized (excl. departure and arrival)
#   #--x0 is the cordinate (vector of longitudes corresponding to the y)
#   #--lons vector including dep and arr
#   lons = np.array([lon1] + list(x0) + [lon2])
#   #--lats vector including dep and arr
#   lats = np.array([lat1] + list(y)  + [lat2])
#   return cost_time(lons, lats, dtprint=dtprint)**2.0
##
##--Extraction of wind speed components in m/s
##@jit( parallel=True)
#def wind(lons,lats):
#   #--extraction of wind in m/s
#   #--we extract wind on lat using the xarray sel method
#   lons_indices=closest_argmin(lons,lons_wind)  #--we need the indices because lon,time are lumped
#   lats_indices=closest_argmin(lats,lats_wind)
#   lats_closest=lats_wind[lats_indices]         #--we need the closest values
#   #--create new zz axis
#   #ipdb.set_trace()
#   lons_z  = xr.DataArray(lons_indices, dims="zz")
#   lats_z  = xr.DataArray(lats_closest, dims="zz")
#   #--Extract the data along the new axis
#   windu=xr_u200_reduced.isel(z=lons_z).sel(latitude=lats_z)['u'].values
#
#   #xr_u200_reduced.sel(z=lons_z.values).rename_dims({"z": "zz",}).sel(latitude=lats_z)['u'].values
#   windv=xr_v200_reduced.isel(z=lons_z).sel(latitude=lats_z)['v'].values
#   #xr_v200_reduced.sel(z=lons_z.values).rename_dims({"z": "zz",}).sel(latitude=lats_z)['v'].values
#
#   #windu=xr_u200_reduced.interp(latitude=lats_z,z=lons_z,method='linear')['u'].values
#   #windv=xr_v200_reduced.interp(latitude=lats_z,z=lons_z,method='linear')['v'].values
#   #--Return u,v components of the wind
#   return windu,windv
##
##--Compute shortest route
#def shortest_route(dep_loc, arr_loc, npoints, even_spaced=True):
#    #--Returns npoints points along the shortest (great circle) route
#    #--dep_loc and arr_loc are in the order (lon, lat)
#    latlon_crs = ccrs.PlateCarree()
#    #--projection for transforming trajectory
#    crs = ccrs.Gnomonic(central_longitude=dep_loc[0], central_latitude=dep_loc[1])
#    gno_aloc = crs.transform_point(arr_loc[0], arr_loc[1], latlon_crs)
#    #--two options here
#    #--points are evenly spaced in distance (this is what we want)
#    if even_spaced:
#        # Gnomic projection so distances from centre scale with tan
#        # Map distance = R * tan (true dist/R)
#        # Get distances in gnomic space that are evenly spaced in true distance
#        point_gdists = np.tan(np.linspace(0, np.arctan(np.sqrt(np.sum(np.array(gno_aloc)**2))/R_earth), npoints+2))
#        # Moving on straight line in gnomic projection, so can scale x,y coordinates separately
#        scale_gdists = point_gdists/point_gdists[-1]
#        newpoints = latlon_crs.transform_points(
#            crs,
#            scale_gdists*gno_aloc[0],
#            scale_gdists*gno_aloc[1])
#    #--points are not evenly spaced in distance
#    else:
#        newpoints = latlon_crs.transform_points(
#            crs,
#            np.linspace(0, gno_aloc[0], npoints+2),
#            np.linspace(0, gno_aloc[1], npoints+2))
#    #--Define trajectory points
#    x0 = newpoints[:, 0] #--longitudes
#    y0 = newpoints[:, 1] #--latitudes
#    #--Return the array of points
#    return x0, y0
##
##--Compute quickest route
#def quickest_route(dep_loc, arr_loc, npoints, lat_iagos):
#    #
#    #--bounds
#    bnds = tuple((-89.9,89.9) for i in range(npoints))
#    #
#    #--First compute shortest route
#    x0, y0 = shortest_route(dep_loc, arr_loc, npoints, even_spaced=True)
#    #
#    #--Minimization with y0 from shortest route as initial conditions
#    res = minimize(cost_squared,y0[1:-1],args=(x0[1:-1],*dep_loc,*arr_loc),method=method,bounds=bnds,options={'maxiter':maxiter,'disp':disp} )
#    y = np.append(np.insert(res['x'],0,dep_loc[1]),arr_loc[1])
#    quickest_time=cost_time(x0, y, dtprint=False)
#    #
#    #--Perform several other minimizations with y0 shifted southwards and northwards
#    n=len(x0)
#    for dymax in [-21.,-18.,-15.,-12.,-9.,-6.,-3.,3.,6.,9.,12.,15.,18.,21.]:
#    ##for dymax in [-27.,-21.,-15.,-9.,-3.,3.,9.,15.,21.]:
#     for imid in [n//2, n//3, 2*n//3]:
#      dy=[dymax*float(i)/float(imid) for i in range(imid)]+[dymax*float(n-i)/float(n-imid) for i in range(imid,n)]
#      y0p=y0+dy
#      #--minimization with y0p as initial conditions
#      res = minimize(cost_squared,y0p[1:-1],args=(x0[1:-1],*dep_loc,*arr_loc),method=method,bounds=bnds,options={'maxiter':maxiter,'disp':disp})
#      y_2 = np.append(np.insert(res['x'],0,dep_loc[1]),arr_loc[1])
#      quickest_time_2=cost_time(x0, y_2)
#      if quickest_time_2 < quickest_time:
#         quickest_time = quickest_time_2
#         y = y_2   #--new best minimum
#    #
#    ##--y between -180 and 180
#    ##y = (y+90.0)%180.0 - 90.0
#    ##
#    #--Solution to optimal route
#    return (x0, y)

#-----------------------------------------------------------------------------------------
# MAIN CODE 
#-----------------------------------------------------------------------------------------
#
#--Read list of IAGOS files
#--Default: use inputfile

def read_data(iagos_file):
    #--print out file
    print('\n')
    print(iagos_file)
    
    #--open IAGOS file
    iagos=xr.open_dataset(iagos_file)
    
    #--get IAGOS id from file name
    iagos_id=iagos_file.split('/')[-1].split('_')[2]
    
    #--extract metadata from flight
    dep_airport_iagos=iagos.departure_airport.split(',')[0]
    arr_airport_iagos=iagos.arrival_airport.split(',')[0]
    dep_time_iagos=datetime.strptime(iagos.departure_UTC_time,"%Y-%m-%dT%H:%M:%SZ")
    arr_time_iagos=datetime.strptime(iagos.arrival_UTC_time,"%Y-%m-%dT%H:%M:%SZ")
    ave_time_iagos=dep_time_iagos+(arr_time_iagos-dep_time_iagos)/2.
    flightid_iagos=iagos.platform.split(',')[3].lstrip().rstrip()
    print('Flightid=',flightid_iagos,dep_airport_iagos,arr_airport_iagos)
    print('Flight departure time=',dep_time_iagos)
    print('Flight arrival time=',arr_time_iagos)
    print('Flight average time=',ave_time_iagos)
    
    #--extract data
    lat_iagos=iagos['lat']
    lon_iagos=iagos['lon']
    time_iagos=iagos['UTC_time']
    pressure_iagos=iagos['air_press_AC']
    
    #--define arrays of lon, lat, and pressure values
    lon_iagos_values=lon_iagos.values
    lat_iagos_values=lat_iagos.values
    pressure_iagos_values=pressure_iagos.values
    
    #--extract and convert IAGOS departure date
    yr_iagos=dep_time_iagos.year
    mth_iagos=dep_time_iagos.month
    day_iagos=dep_time_iagos.day
    hr_iagos=dep_time_iagos.hour
    hr_iagos_closest,hr_iagos_ind=nearest([i*Dt_ERA for i in range(24//Dt_ERA)],hr_iagos)
    stryr=dep_time_iagos.strftime("%Y")
    strmth=dep_time_iagos.strftime("%m")
    strday=dep_time_iagos.strftime("%d")
    
    #--find lon, lat, alt of cruising similar to what FR24 database does
    #--this is empirical and needs to be checked
    #--the threshold depend on the time resolution of the IAGOS data
    ind=np.where((np.abs(np.diff(gaussian_filter1d(pressure_iagos,40)))<50.) & (pressure_iagos[:-1]<35000.))[0]
    
    #--eliminate low-level flights
    if len(ind) == 0:
        print('This flight is too low to be optimized so we stop here')
        continue
    
    #--find longitude and latitude of beginning and end of cruising phase
    lon_p1=lon_iagos_values[ind[0]]
    lon_p2=lon_iagos_values[ind[-1]]
    lat_p1=lat_iagos_values[ind[0]]
    lat_p2=lat_iagos_values[ind[-1]]
    
    lon_key_values=np.array([lon_iagos_values[0],lon_p1,lon_p2,lon_iagos_values[-1]])
    lat_key_values=np.array([lat_iagos_values[0],lat_p1,lat_p2,lat_iagos_values[-1]])
    alt_key_values=np.array([pressure_iagos_values[0],pressure_iagos_values[ind[0]],pressure_iagos_values[ind[-1]],pressure_iagos_values[-1]])
    
    print('Departure and arrival points')
    print('lon lat p1=',lon_p1,lat_p1)
    print('lon lat p2=',lon_p2,lat_p2)

    return iagos_id, dep_airport_iagos, arr_airport_iagos, dep_time_iagos, 
           arr_time_iagos, ave_time_iagos, flightid_iagos, lat_iagos,
           lon_iagos, time_iagos, pressure_iagos, yr_iagos, mth_iagos, day_iagos,
           hr_iagos, hr_iagos_closest, hr_iagos_ind, stryr, strmth, strday,
           ind, lon_p1, lon_p2, lat_p1, lat_p2, lon_key_values, lat_key_values, alt_key_values

def compute_IAGOS_route(lon_shortest, lon_iagos_value, lat_iagos_values, 
        pressure_iagos_values, lon_p1, lon_p2, lat_p1, lat_p2):

    #--interpolated latitude of IAGOS flight with similar sampling
    nlon=len(lon_shortest)
    imid=nlon//2
    idxmid=(np.abs(lon_iagos_values-lon_shortest[imid])).argmin()
    lon_iagos_cruising=[lon_iagos_values[idxmid]]
    lat_iagos_cruising=[lat_iagos_values[idxmid]]
    pressure_iagos_cruising=[pressure_iagos_values[idxmid]]
    #--eliminate dodgy cases
    if idx1 >= idxmid or idx2 <= idxmid: 
        print('This is a dodgy case with idxmid not in between idx1 and idx2 - would need some investigation')
        continue
    #--flight is eastbound in new coordinates (lon_p1 < lon_p2)
    for i in range(imid+1,len(lon_shortest)):
       ilon=np.max(np.where(lon_iagos_values[idxmid:idx2+1]<=lon_shortest[i]))
       lon_iagos_cruising=lon_iagos_cruising+[lon_iagos_values[idxmid+ilon]]
       lat_iagos_cruising=lat_iagos_cruising+[lat_iagos_values[idxmid+ilon]]
       pressure_iagos_cruising=pressure_iagos_cruising+[pressure_iagos_values[idxmid+ilon]]
    for i in range(imid-1,-1,-1):
       ilon=np.min(np.where(lon_iagos_values[idx1:idxmid]>=lon_shortest[i]))
       lon_iagos_cruising=[lon_iagos_values[idx1+ilon]]+lon_iagos_cruising
       lat_iagos_cruising=[lat_iagos_values[idx1+ilon]]+lat_iagos_cruising
       pressure_iagos_cruising=[pressure_iagos_values[idx1+ilon]]+pressure_iagos_cruising
    
    #--put the correct departure and arrival coordinates
    lon_iagos_cruising[0]=lon_p1
    lon_iagos_cruising[-1]=lon_p2
    lat_iagos_cruising[0]=lat_p1
    lat_iagos_cruising[-1]=lat_p2
    
    #--conversion to np array
    lon_iagos_cruising=np.array(lon_iagos_cruising)
    lat_iagos_cruising=np.array(lat_iagos_cruising)
    pressure_iagos_cruising=np.array(pressure_iagos_cruising)
    
    #--IAGOS route
    p1_iagos=(lon_iagos_cruising[0],lat_iagos_cruising[0])
    p2_iagos=(lon_iagos_cruising[-1],lat_iagos_cruising[-1])
    dist_iagos = haversine(p1_iagos[1], p1_iagos[0], p2_iagos[1], p2_iagos[0])
    dist_gcc_iagos = gcc.distance_between_points(p1_iagos,p2_iagos,unit='kilometers')
    dt_iagos_2=cost_time(lon_iagos_cruising, lat_iagos_cruising, dtprint=False)
    print('IAGOS cruising flight time actual and sampled estimated =',"{:6.4f}".format(dt_iagos_1),"{:6.4f}".format(dt_iagos_2),'hours')
    
    return lon_iagos_cruising, lat_iagos_cruising, pressure_iagos_cruising, 
           p1_iagos, p2_iagos, dist_iagos, dist_gcc_iagos, dt_iagos_2

def make_plot(rotated, lon_iagos_values, lat_iagos_values, lon_key_values, lat_key_values, alt_key_values,
              lon_shortest, lat_shortest, lon_quickest, lat_quickest, lon_ed, lat_ed, 
              lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced,
              iagos_id, flightid_iagos, dep_airport_iagos, arr_airport_iagos, stryr, strmth, strday, # TODO hide inside dict..
              optim_level, dt_shortest, dt_quickest, dt_ed_LD, dt_iagos_2, pathout, yr, 
              pressure_iagos):
        ):
    fig=plt.figure(figsize=(10,5))
    ax=fig.add_subplot(111, projection=rotated)
    #--lon-lat
    ax.plot(lon_iagos_values, lat_iagos_values, c='black', lw=2, label='IAGOS')
    ax.scatter(lon_key_values, lat_key_values, c='red', marker='X', lw=1)
    ax.plot(lon_shortest, lat_shortest, c='green', lw=2, label='Shortest',zorder=10)
    ax.plot(lon_quickest, lat_quickest, c='blue',  lw=2, label='Gradient descent')
    if solution: ax.plot(lon_ed, lat_ed, c='cyan', lw=2, label='Zermelo method')
    #--wind field
    ax.quiver(lons_wind[::10],lats_wind[::10],np.transpose(xr_u200_reduced['u'].values[::10,::10]),np.transpose(xr_v200_reduced['v'].values[::10,::10]),scale=1200)
    #--define domain with a margin
    ax.set_xlim(np.min(lon_shortest)-20.,np.max(lon_shortest)+20.)
    ax.set_ylim(np.max([-90.,np.min(lat_shortest)-20.]),np.min([90.,np.max(lat_shortest)+20.]))
    #--make plot nice
    ax.legend(loc='lower left',fontsize=10)
    fig.gca().coastlines()
    fig.gca().add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"),alpha=1)
    fig.gca().add_feature(cartopy.feature.BORDERS,color="red",linewidth=0.3)
    #--plot title
    plt.title(iagos_id+' '+str(flightid_iagos)+' '+dep_airport_iagos+'=>'+arr_airport_iagos+' '+' '+stryr+strmth+strday+\
              ' level='+str(optim_level)+' hPa'+'\n'+ 'Shortest='+"{:.3f}".format(dt_shortest)+\
              ' Gradient descent='+"{:.3f}".format(dt_quickest)+' Zermelo method='+"{:.3f}".format(dt_ed_LD)+' IAGOS='+"{:.3f}".format(dt_iagos_2),fontsize=11)
    #--save, show and close
    basefile=pathout+'traj_'+str(iagos_id)+'_lev'+str(optim_level)+'_'+str(yr)
    plt.savefig(basefile+'.png',dpi=150,bbox_inches='tight')
    if pltshow: plt.show()
    plt.close()
    
    #--plot 2: height
    #----------------
    fig,ax = plt.subplots()
    ax.set_ylim(1030,-300)
    ax.set_ylabel("Pressure (hPa)",color="black",fontsize=14)
    ax.set_xlabel("Longitude",color="black",fontsize=14)
    ax.plot(lon_iagos_values, pressure_iagos/100., c='black', lw=2, label='IAGOS flight pressure')
    ax.plot(lon_iagos_values[ind], pressure_iagos[ind]/100., c='red', lw=2, label='Flight pressure (cruising)',zorder=10)
    ax.plot(lon_iagos_values[:-1], np.diff(gaussian_filter1d(pressure_iagos,40)), c='blue', lw=1, label='Finite difference')
    ax.plot(lon_iagos_values[[0,-1]], [350.,350.], c='black', linestyle='solid', lw=1)
    ax.plot(lon_iagos_values[[0,-1]], [50.,50.], c='black', linestyle='dashed', lw=1)
    ax.plot(lon_iagos_values[[0,-1]], [-50.,-50.], c='black', linestyle='dashed', lw=1)
    ax.scatter(lon_key_values, alt_key_values/100., c='red', marker='X', lw=1, label='Origin - Cruising - Destination')
    #--make plot nice
    plt.title(iagos_id+' '+str(flightid_iagos)+' '+dep_airport_iagos+'=>'+arr_airport_iagos+' '+' '+stryr+strmth+strday+' level='+str(optim_level)+' hPa')
    plt.legend()
    #--save, show and close
    basefile=pathout+'alt_'+str(iagos_id)+'_lev'+str(optim_level)+'_'+str(yr)
    plt.savefig(basefile+'.png',dpi=150,bbox_inches='tight')
    if pltshow: plt.show()
    plt.close()
    
    #--save a pickle to redo the plots later if needed
    dico={'iagos_id':iagos_id,'flightid_iagos':flightid_iagos,'airspeed':airspeed,'dist_gcc':dist_gcc,\
          'lat_pole': lat_pole,'lon_pole':lon_pole,'lon_iagos_values':lon_iagos_values,'lon_iagos_cruising':lon_iagos_cruising,'lat_iagos_values':lat_iagos_values,\
          'lon_key_values':lon_key_values,'optim_level':optim_level,'solution':solution,\
          'lat_key_values':lat_key_values,'lon_shortest':lon_shortest,'lat_shortest':lat_shortest,'pressure_iagos':pressure_iagos,'alt_key_values':alt_key_values,\
          'lon_quickest':lon_quickest,'lat_quickest':lat_quickest,'lon_ed':lon_ed,'lat_ed':lat_ed,'lons_wind':lons_wind,'lats_wind':lats_wind,\
          'xr_u200':xr_u200_reduced,'xr_v200':xr_v200_reduced,'dep_airport_iagos':dep_airport_iagos,'arr_airport_iagos':arr_airport_iagos,\
          'stryr':stryr,'strmth':strmth,'strday':strday,'dt_shortest':dt_shortest,'dt_quickest':dt_quickest,'dt_ed':dt_ed_LD,'dt_iagos_2':dt_iagos_2}
    basefile=pathout+'data_'+str(iagos_id)+'_lev'+str(optim_level)+'_'+str(yr)

def ED_quickest_route(p1, p2, airspeed, lon_p1, lon_p2, lat_p1, lat_p2, 
                      lat_shortest, lat_quickest, lat_iagos_cruising):
    start_time = time.time()
    # Create the zermelo solver. Note the default values
    #--max_dest_dist is in metres
    #--sub_factor: number of splits for next round if solution is bounded by pairs of trajectories
    #--psi_range: +/- angle for the initial bearing
    #--psi_res: resolution within psi_range bounds, could try 0.2 instead
    zermelolonlat = ZermeloLonLat(cost_func=lambda x, y, z: np.ones(np.atleast_1d(x).shape),
                                  wind_func=wind, timestep=60, psi_range=60, psi_res=0.5,
                                  length_factor=1.4, max_dest_distance=75000., sub_factor=80)
    
    initial_psi = zermelolonlat.bearing_func(*p1, *p2)
    psi_vals = np.linspace(initial_psi-60, initial_psi+60, 30)
    #--This prodcues a series of Zermelo trajectories for the given initial directions
    zloc, zpsi, zcost = zermelolonlat.zermelo_path(np.repeat(np.array(p1)[:, None], len(psi_vals), axis=-1),
                        # This 90 is due to an internal conversion between bearings and angles
                        # - which is obviously a bad idea... noramlly it is hidden internally
                        ##   90-psi_vals, nsteps=800, airspeed=250, dtime=dep_time_iagos)
                        90-psi_vals, nsteps=800, airspeed=airspeed, dtime=0) #--modif OB
    
    # This identifies the optimal route
    solution, fpst, ftime, flocs, fcost = zermelolonlat.route_optimise(np.array(p1), np.array(p2), airspeed=airspeed, dtime=0)
    #--if solution was found
    if solution: 
      lon_ed=flocs[:,0]
      lat_ed=flocs[:,1]
      #
      #--compute Ed's time by stretching slightly the trajectory to the same endpoints
      npoints_ed=len(lon_ed)
      print('npoints_ed=',npoints_ed)
      lon_ed=lon_ed+(lon_p2-lon_ed[-1])*np.arange(npoints_ed)/float(npoints_ed-1)
      lat_ed=lat_ed+(lat_p2-lat_ed[-1])*np.arange(npoints_ed)/float(npoints_ed-1)
      #--compute corresponding time 
      dt_ed_HD=cost_time(lon_ed, lat_ed, dtprint=False)
      print('Cruising flight time ED (high res) =',"{:6.4f}".format(dt_ed_HD),'hours')
      lon_ed_LD=np.append(lon_ed[::npoints_ed//npoints],[lon_ed[-1]])
      lat_ed_LD=np.append(lat_ed[::npoints_ed//npoints],[lat_ed[-1]])
      dt_ed_LD=cost_time(lon_ed_LD, lat_ed_LD, dtprint=False)
      print('Cruising flight time ED (low res) =',"{:6.4f}".format(dt_ed_LD),'hours')
    else: 
      print('No solution found by Zermelo')  
      lon_ed=float('inf')
      lat_ed=float('inf')
      dt_ed_HD=float('inf')
      dt_ed_LD=float('inf')
    end_time = time.time()
    time_elapsed_EG=end_time-start_time
    print('Time elapsed for Zermelo method=',"{:3.1f}".format(time_elapsed_EG),'s')
    
    #--computing indices of quality of fit
    rmse_shortest=mean_squared_error(lat_shortest,lat_iagos_cruising)**0.5
    rmse_quickest=mean_squared_error(lat_quickest,lat_iagos_cruising)**0.5
    lat_max_shortest=np.max(np.abs(lat_shortest-lat_iagos_cruising))
    lat_max_quickest=np.max(np.abs(lat_quickest-lat_iagos_cruising))
    print('rmse and lat max=',rmse_shortest,rmse_quickest,lat_max_shortest,lat_max_quickest)
    
    ##--fill DataFrame - not very efficient but ok as dataframe is short
    #final_df.loc[len(final_df)]=[iagos_file,iagos_id,flightid_iagos,optim_level,yr,dep_airport_iagos,arr_airport_iagos,\
    #                             dep_time_iagos,arr_time_iagos,time_iagos.values[idx1],time_iagos.values[idx2],lon_p1,lat_p1,\
    #                             lon_p2,lat_p2,dt_shortest,dt_quickest,dt_ed_LD,dt_iagos_2,rmse_shortest,rmse_quickest,\
    #                             time_elapsed_OB,time_elapsed_EG,airspeed,dist_gcc]
    #
    ##--save quickest route for Ed in a Dataframe
    #route_df=pd.DataFrame(columns=['Total_time_IAGOS','Total_time_quickest','longitudes IAGOS',\
    #                               'latitudes IAGOS','longitudes quickest','latitudes quickest'])
    #for i in range(len(lat_quickest)):
    #   new_df = pd.DataFrame({'Total_time_IAGOS':dt_iagos_2*3600.,'Total_time_quickest':dt_quickest*3600.,\
    #                          'longitudes IAGOS':lon_iagos_cruising[i],'latitudes IAGOS':lat_iagos_cruising[i],\
    #                          'longitudes quickest':lon_quickest[i],'latitudes quickest':lat_quickest[i]},index=[i])
    #   route_df = pd.concat([route_df,new_df]) 
    #route_df.to_csv(pathout+str(iagos_id)+'_lev'+str(optim_level)+'_'+str(yr)+'.csv')



def opti(mth, inputfile, route, level,  maxiter,
         method, path_iagos, path_ERA5, pathout,
         nbmeters, airspeed, R_earth, disp, pltshow):

    if inputfile != '': 
        #--single file
        iagos_files=[inputfile]
        #--overwrite year from filename in this case
        yr=int(inputfile.split('/')[-1].split('_')[2][0:4])
    elif route != '': 
        #--Otherwise use the selected route
        #--read route flights 
        csvfile = 'FLIGHTS/'+route+'_'+str(yr)+'.csv'
        #--open file
        print('csvfile=',csvfile)
        iagos_files = pd.read_csv(csvfile,header=None,names=['file'])
        iagos_files = list(iagos_files['file'].values)
    else:
        #--Otherwise use year+month
        #--all IAGOS files from selected year and month
        iagos_files=sorted(glob.glob(path_iagos+str(yr)+str(mth).zfill(2)+'/*.nc4'))


    print('We have found '+str(len(iagos_files))+' IAGOS files.')
 
    #--Initialise dataframe for saving output
    final_df=pd.DataFrame(columns=['file_iagos','flightname_iagos','flightid_iagos','level','year','dep_airport','arr_airport',\
                                   'dep_time','arr_time','time_start_cruising','time_end_cruising','lon_start_cruising','lat_start_cruising',\
                                   'lon_end_cruising','lat_end_cruising','time shortest','time OB','time EG','time_iagos',\
                                   'rmse lat shortest','rmse lat quickest','time elapsed OB','time elapsed EG','airspeed','dist_gcc'])
    
    #--Open ERA5 wind files as xarray objects
    stryr=str(yr)
    file_u=pathERA5+'u.'+stryr+'.GLOBAL.nc'
    file_v=pathERA5+'v.'+stryr+'.GLOBAL.nc'
    file_t=pathERA5+'ta.'+stryr+'.GLOBAL.nc'
    file_r=pathERA5+'r.'+stryr+'.GLOBAL.nc'

    print(file_u)
    print(file_v)
    print(file_t)
    print(file_r)

    xr_u=xr.open_dataset(file_u)  
    xr_v=xr.open_dataset(file_v)  
    xr_t=xr.open_dataset(file_t)  
    xr_r=xr.open_dataset(file_r)  

    #--Extract coordinates
    levels_wind=list(xr_u['level'].values)
    lons_wind=xr_u['longitude'].values
    lats_wind=xr_u['latitude'].values
 
    #--Loop on flights
    for iagos_file in iagos_files:
        (iagos_id, dep_airport_iagos, arr_airport_iagos, dep_time_iagos, 
         arr_time_iagos, ave_time_iagos, flightid_iagos, lat_iagos,
         lon_iagos, time_iagos, pressure_iagos, yr_iagos, mth_iagos, day_iagos,
         hr_iagos, hr_iagos_closest, hr_iagos_ind, stryr, strmth, strday,
         ind, lon_p1, lon_p2, lat_p1, lat_p2, lon_key_values, lat_key_values, alt_key_values) = read_data(iagos_file)

        #--compute great circle distance
        dist = haversine(lat_p1, lon_p1, lat_p2, lon_p2)
        dist_gcc = gcc.distance_between_points((lon_p1,lat_p1),(lon_p2,lat_p2),unit='meters')
        print('Distance between airports (haversine & gcc) = ',"{:6.3f}".format(dist/1000.),"{:6.3f}".format(dist_gcc/1000.),'km')
        
        # --compute number of legs
        npoints = int(dist // nbmeters)
        
        #--select IAGOS datapoints as close as possible to FR24 datapoints
        idx1 = ((lon_iagos_values-lon_p1)**2.0+(lat_iagos_values-lat_p1)**2.0).argmin()
        idx2 = ((lon_iagos_values-lon_p2)**2.0+(lat_iagos_values-lat_p2)**2.0).argmin()
        
        #--compute actual IAGOS time flight during cruising (in hours)
        dt_iagos_1=float(time_iagos.values[idx2]-time_iagos.values[idx1])/3600./1.e9 #--convert nanoseconds => hr
        
        #--compute average IAGOS pressure (in hPa)
        ave_pressure_iagos=np.average(pressure_iagos[idx1:idx2])/100.
        print('Pressure levels in ERA5 file=',levels_wind)
        #--find closest pressure level in ERA5 data
        pressure_iagos_closest,pressure_ind_closest=nearest(levels_wind,ave_pressure_iagos)
        print('Average pressure=',"{:5.2f}".format(ave_pressure_iagos),'hPa closest to',pressure_iagos_closest,'hPa')
        
        #--select pressure level for optimisation
        if level == -1:
           optim_level=pressure_iagos_closest
        else:
           optim_level=level
        
        #--time ERA5 preparation
        start_time = time.time()
        
        #--pre-sample times 
        nbts=int(dt_iagos_1/Dt_ERA)+2
        
        #--times to extract (3-hourly) from start to end of flight
        times_to_extract=[datetime(yr_iagos,mth_iagos,day_iagos,hr_iagos_closest,0)+timedelta(hours=i*Dt_ERA) for i in range(nbts)]
        
        #--find closest time and corresponding index to flight average time
        ave_time_iagos_closest,ave_time_ind_closest=nearest(times_to_extract,ave_time_iagos)
        print('Flight average time closest to ERA=',ave_time_iagos_closest)
        
        #--preload the data for a range of nbts times
        xr_u200=xr_u.sel(level=optim_level,time=times_to_extract).load()
        xr_v200=xr_v.sel(level=optim_level,time=times_to_extract).load()
        
        #--select array (m/s)
        xr_u200_values=xr_u200['u'].values
        xr_v200_values=xr_v200['v'].values
        
        #--prepare plate grid
        process_grid(xr_u200, xr_v200, xr_t200, xr_r200, nbts, 
             lon_p1, lat_p1, lon_p2, lat_p2, lons_wind, lats_wind, 
             lon_iagos_values, lat_iagos_values, lon_key_values, lat_key_values)

        #--substitute data back in their original xarray objects
        xr_u200['u'].values=xr_u200_values
        xr_v200['v'].values=xr_v200_values
        
        #--dt per degree of longitude
        dtime_per_degree=dt_iagos_1/abs(lon_p2-lon_p1) #--this assumes lon_p2 > lon_p1
        
        #--compute times_era assuming uniform sampling of longitudes
        times_wind=[]
        for lon in list(lons_wind):
          if lon<lon_p1:
            time_to_append=datetime(yr_iagos,mth_iagos,day_iagos,hr_iagos_closest,0)
          elif lon<lon_p2:
            time_to_append,time_ind_closest=nearest(times_to_extract,dep_time_iagos+timedelta(hours=dtime_per_degree*(lon-lon_p1)))
          else:
            time_to_append,time_ind_closest=nearest(times_to_extract,arr_time_iagos+timedelta(minutes=30))
          #--append list of position times
          times_wind.append(time_to_append)
        #--convert times_wind to np array
        times_wind=np.array(times_wind)
        #--define new joint lon-time axis
        lons_z = xr.DataArray(lons_wind, dims="z")
        times_z = xr.DataArray(times_wind, dims="z")
        
        #--preload the data for a range of nbts times
        xr_u200_reduced=xr_u200.sel(longitude=lons_z,time=times_z,latitude=lats_wind).load()
        xr_v200_reduced=xr_v200.sel(longitude=lons_z,time=times_z,latitude=lats_wind).load()
        xr_u200_reduced.chunk(chunks={"z":"auto"})
        xr_v200_reduced.chunk(chunks={"z":"auto"})
        xr_u200_reduced.persist()
        xr_v200_reduced.persist()
        
        
        end_time = time.time()
        time_elapsed_ERA=end_time-start_time
        print('Time elapsed for ERA5=',"{:3.1f}".format(time_elapsed_ERA),'s')
        
        #--define p1 and p2 as tuples
        p1 = (lon_p1, lat_p1)
        p2 = (lon_p2, lat_p2)
        
        #--test longitude range
        if not (lons_wind[0] <= lon_p1 and lon_p1 <= lons_wind[-1] and lons_wind[0] <= lon_p2 and lon_p2 <= lons_wind[-1]):
            print('lons_wind range is ',lons_wind[0],lons_wind[-1])
            print('There is a problem with the longitude range')
            sys.exit()
        
        #--flatten arrays and create coordinate vector
        xx_yy=np.array([[ixx,iyy] for ixx,iyy in zip(xx.flatten(),yy.flatten())])
        
        #--check longitudes are mostly monotonic in IAGOS flight and stop otherwise
        print('monotonicity of IAGOS longitudes =',np.sum(np.diff(lon_iagos_values) >= 0)/len(lon_iagos_values), \
                                                   np.sum(np.diff(lon_iagos_values) <= 0)/len(lon_iagos_values))
        if (np.sum(np.diff(lon_iagos_values) >= 0)/len(lon_iagos_values) < 0.90 and \
            np.sum(np.diff(lon_iagos_values) <= 0)/len(lon_iagos_values) < 0.90): 
            print('Flight longitudes are not monotonic enough so we stop here for this flight')
            continue
        
        #--compute shortest route
        lon_shortest, lat_shortest = shortest_route(p1, p2, npoints, even_spaced=True)
        
        #--compute time of shortest route 
        dt_shortest=cost_time(lon_shortest, lat_shortest, dtprint=False)
        print('Cruising flight time shortest =',"{:6.4f}".format(dt_shortest),'hours')
        
        #---------------------
        #--compute IAGOS route
        #---------------------
        (lon_iagos_cruising, lat_iagos_cruising, 
         pressure_iagos_cruising, p1_iagos, p2_iagos, 
         dist_iagos, dist_gcc_iagos, dt_iagos_2) = compute_IAGOS_route(lon_shortest, lon_iagos_value, lat_iagos_values, 
                                                                        pressure_iagos_values, lon_p1, lon_p2,
                                                                        lat_p1, lat_p2)

        #---------------------------
        #--compute OB quickest route
        #---------------------------
        start_time = time.time()
        lon_quickest, lat_quickest = quickest_route(p1, p2, npoints, lat_iagos_cruising)
        dt_quickest=cost_time(lon_quickest, lat_quickest, dtprint=False)
        end_time = time.time()
        time_elapsed_OB=end_time-start_time
        print('Time elapsed for gradient descent method =',"{:3.1f}".format(time_elapsed_OB),'s')
        print('Cruising flight time quickest gradient descent=',"{:6.4f}".format(dt_quickest),'hours')
        
        #---------------------------
        #--compute ED quickest route
        #---------------------------
        ED_quickest_route(p1, p2, airspeed, lon_p1, lon_p2, lat_p1, lat_p2, 
                      lat_shortest, lat_quickest, lat_iagos_cruising):
                
        #--fill DataFrame - not very efficient but ok as dataframe is short
        final_df.loc[len(final_df)]=[iagos_file,iagos_id,flightid_iagos,optim_level,yr,dep_airport_iagos,arr_airport_iagos,\
                                     dep_time_iagos,arr_time_iagos,time_iagos.values[idx1],time_iagos.values[idx2],lon_p1,lat_p1,\
                                     lon_p2,lat_p2,dt_shortest,dt_quickest,dt_ed_LD,dt_iagos_2,rmse_shortest,rmse_quickest,\
                                     time_elapsed_OB,time_elapsed_EG,airspeed,dist_gcc]
        
        #--save quickest route for Ed in a Dataframe
        route_df=pd.DataFrame(columns=['Total_time_IAGOS','Total_time_quickest','longitudes IAGOS',\
                                       'latitudes IAGOS','longitudes quickest','latitudes quickest'])
        for i in range(len(lat_quickest)):
           new_df = pd.DataFrame({'Total_time_IAGOS':dt_iagos_2*3600.,'Total_time_quickest':dt_quickest*3600.,\
                                  'longitudes IAGOS':lon_iagos_cruising[i],'latitudes IAGOS':lat_iagos_cruising[i],\
                                  'longitudes quickest':lon_quickest[i],'latitudes quickest':lat_quickest[i]},index=[i])
           route_df = pd.concat([route_df,new_df]) 
        route_df.to_csv(pathout+str(iagos_id)+'_lev'+str(optim_level)+'_'+str(yr)+'.csv')
        
        #-plot 1: prepare map traj plot
        #------------------------------
        make_plot(rotated, lon_iagos_values, lat_iagos_values, lon_key_values, lat_key_values, alt_key_values,
              lon_shortest, lat_shortest, lon_quickest, lat_quickest, lon_ed, lat_ed, 
              lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced,
              iagos_id, flightid_iagos, dep_airport_iagos, arr_airport_iagos, stryr, strmth, strday, # TODO hide inside dict..
              optim_level, dt_shortest, dt_quickest, dt_ed_LD, dt_iagos_2, pathout, yr, 
              pressure_iagos):

        
        #--save a pickle to redo the plots later if needed
        dico={'iagos_id':iagos_id,'flightid_iagos':flightid_iagos,'airspeed':airspeed,'dist_gcc':dist_gcc,\
              'lat_pole': lat_pole,'lon_pole':lon_pole,'lon_iagos_values':lon_iagos_values,'lon_iagos_cruising':lon_iagos_cruising,'lat_iagos_values':lat_iagos_values,\
              'lon_key_values':lon_key_values,'optim_level':optim_level,'solution':solution,\
              'lat_key_values':lat_key_values,'lon_shortest':lon_shortest,'lat_shortest':lat_shortest,'pressure_iagos':pressure_iagos,'alt_key_values':alt_key_values,\
              'lon_quickest':lon_quickest,'lat_quickest':lat_quickest,'lon_ed':lon_ed,'lat_ed':lat_ed,'lons_wind':lons_wind,'lats_wind':lats_wind,\
              'xr_u200':xr_u200_reduced,'xr_v200':xr_v200_reduced,'dep_airport_iagos':dep_airport_iagos,'arr_airport_iagos':arr_airport_iagos,\
              'stryr':stryr,'strmth':strmth,'strday':strday,'dt_shortest':dt_shortest,'dt_quickest':dt_quickest,'dt_ed':dt_ed_LD,'dt_iagos_2':dt_iagos_2}
        basefile=pathout+'data_'+str(iagos_id)+'_lev'+str(optim_level)+'_'+str(yr)
        with open(basefile+'.pickle', 'wb') as f: pickle.dump(dico, f)
        #
        #--save dataframe (and overwrite)
        if route != '':
          final_df.to_csv(pathout+str(route)+'_'+str(yr)+'_lev'+str(level)+'.csv')
        else:
          final_df.to_csv(pathout+str(yr)+str(mth).zfill(2)+'_lev'+str(level)+'.csv')
    
    #--final stats and plots
    print('\n')
    print('rmse lat shortest '+str(yr)+'=',final_df['rmse lat shortest'].mean())
    print('rmse lat quickest '+str(yr)+'=',final_df['rmse lat quickest'].mean())

def main():
    #  NOTE 
    #  OB: gradient descent
    #  EG: Zermelo method
    
    #  INPUT PARAMETERS FROM COMMAND LINE
    #  example on how to call the python script
    #  python optim_iagos_only.py --yr=2019 --level=200
    #  level=-1 => the script uses the pressure level that is closest to the average IAGOS flight pressure
    #  one can prescribe inputfile, otherwise route, otherwise mth
    
    parser = argparse.ArgumentParser(
            prog="FlightTrajectories",
            description="Optime flight trajectories",
            epilog="""example on how to call the python script:\n"""
                """python optim_iagos_only.py --yr=2019 --level=200\n"""
                """level=-1 => the script uses the pressure level that is closest to the average IAGOS flight pressure \n"""
                """one can prescribe inputfile, otherwise route, otherwise mth."""

            )
    parser.add_argument('--yr', type=int, choices=[2018,2019,2020,2021,2022], default=2018, help='year')
    parser.add_argument('--mth', type=int, choices=[-1,1,2,3,4,5,6,7,8,9,10,11,12], default=1, help='month')
    parser.add_argument('--inputfile', type=str, default='IAGOS_timeseries_2019010510370802_L2_3.1.0.nc4', help='input file')
    parser.add_argument('--route', type=str, default='', help='route')
    parser.add_argument('--level', type=int, choices=[-1,150,175,200,225,250], default=-1, help='level (hPa)')
    parser.add_argument('--maxiter', type=int, default=100, help='max number of iterations')
    parser.add_argument('--method', type=str, default='SLSQP', choices=['SLSQP','BFGS','L-BFGS-B'], help='minimization method')
    parser.add_argument('--iagos', type=str, default='/bdd/IAGOS', help='path to the IAGOS files')
    parser.add_argument('--era5', type=str, default='/projsu/cmip-work/oboucher/ERA5/', help='path to the ERA5 files')
    parser.add_argument('--output', type=str, default='./output', help='path to the output folder')


    #--time sampling of ERA5 data
    Dt_ERA=1 #--hourly
    #--path to store output plots
    #pathout='/projsu/cmip-work/oboucher/FR24/ROUTE/'+str(maxiter)+'/'+method+'/'

    #--get arguments from command line
    args = parser.parse_args()
    
    #--copy arguments into variables
    yr=args.yr
    mth=args.mth
    inputfile=args.inputfile
    route=args.route
    level=args.level
    maxiter=args.maxiteruu
    method=args.method
    path_iagos=args.iagos
    path_ERA5=args.era5
    if args.output.endswith('/'):
        output = args.output[:-1]
    else:
        output = args.output
    pathout=os.path.join(output+str(maxiter),method)
  
    #--print input parameters to output
    print('yr=',yr)
    print('mth=',mth)
    print('inputfile=',inputfile)
    print('route=',route)
    print('level=',level)
    print('maxiter=',maxiter)
    print('method=',method)
    
    #--stop unwanted warnings from xarray
    warnings.filterwarnings("ignore")
    
    #--a little more verbose
    print('We are dealing with IAGOS flights for year '+str(yr))
    if level == -1:
      print('Variable level in optimisation')
    else:
      print('Fixed level in optimisation='+str(level)+' hPa')
    
    
    if not os.path.exists(pathout): os.makedirs(pathout)
    # --number of m for one leg
    nbmeters = 50000.
    #--typical aircraft airspeed in m/s 
    airspeed = 241.
    # --Earth's radius in m (same value as in misc_geo)
    R_earth = 6372800.
    #--print out details of the minimization
    disp=False
    #--show plots
    pltshow=False
    #pltshow=True
    #--save plots
    #pltsave=True
    #
    #--Definition of the cost function for the flight time under wind conditions
    #--y is an array of latitude (does not include departure and arrival points) that is being optimized
    #--x0 is a set of fixed longitude
    #--return flight duration (in hours) accounting for winds
    opti(mth,
    inputfile,
    route,
    level,
    maxiter,
    method,
    path_iagos,
    path_ERA5,
    pathout,
    nbmeters,
    airspeed,
    R_earth,
    disp,
    pltshow)

if __name__ == "__main__":
    main()
