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


def cost_time(lons, lats, lons_wind, lats_wind, dtprint=False):
   #--Array of positions (lons, lats)
   positions=np.array([ [x,y] for x,y in zip(lons,lats) ])
   #--Ground track info
   #--Compute bearing(lat1, lon1, lat2, lon2)
   bearings = bearing(positions[:-1, 1],positions[:-1, 0],positions[1:, 1],positions[1:, 0])
   #--Compute distances
   dists =  haversine(positions[:-1, 1],positions[:-1, 0],positions[1:, 1],positions[1:, 0])
   #--interpolated wind speed on lon,lat array
   wind_u, wind_v = wind(lons, lats, lons_wind, lats_wind)
   #--Compute wind in the middles of segments
   winds_u = (wind_u[:-1]+wind_u[1:])/2.0
   winds_v = (wind_v[:-1]+wind_v[1:])/2.0
   #--compute wind bearings
   windbearings = np.rad2deg(np.arctan2(winds_u, winds_v))
   #--Compute wind speed
   windspeed = np.sqrt(winds_u**2+winds_v**2)
   #--Follow law of cosines - http://hyperphysics.phy-astr.gsu.edu/hbase/lcos.html#c3
   theta = bearings - windbearings
   beta = np.rad2deg(np.arcsin(windspeed*np.sin(np.deg2rad(theta))/airspeed))
   alpha = 180. - theta - beta
   #--Resultant is the ground track speed
   resultant = np.sqrt(airspeed**2 + windspeed**2 - 2*airspeed*windspeed*np.cos(np.deg2rad(alpha)))
   #--Segment time is haversine distance divided by ground speed
   segment_time = dists / resultant / 3600. # m / (m/s) / (s/hr)= hours
   #--Compute total time over trajectory
   total_time = segment_time.sum()  #--hours
   #--Print if asked for
   if dtprint: 
      print('cost pos lat1=',positions[:-1, 1])
      print('cost pos lon1=',positions[:-1, 0])
      print('cost pos lat2=',positions[1:, 1])
      print('cost pos lon2=',positions[1:, 0])
      print('cost bearing=', bearings)
      print('dists=',dists)
      print('wind_u=',winds_u)
      print('wind_v=',winds_v)
      print('resultant=',resultant)
      print('cost time dt=',segment_time)
      print('total dist=',dists.sum())
      print('total time=',total_time)
   #--Return total flight time
   return total_time

#--Cost function: return flight duration squared (this is the function to be minimized)
def cost_squared(y, x0, lon1, lat1, lon2, lat2,lons_wind, lats_wind, dtprint=False):
   #--y is the vector to be optimized (excl. departure and arrival)
   #--x0 is the cordinate (vector of longitudes corresponding to the y)
   #--lons vector including dep and arr
   lons = np.array([lon1] + list(x0) + [lon2])
   #--lats vector including dep and arr
   lats = np.array([lat1] + list(y)  + [lat2])
   return cost_time(lons, lats, lons_wind, lats_wind, dtprint=dtprint)**2.0

#--Extraction of wind speed components in m/s
def wind(lons,lats, lons_wind, lats_wind):
   #--extraction of wind in m/s
   #--we extract wind on lat using the xarray sel method
   lons_indices=closest_argmin(lons,lons_wind)  #--we need the indices because lon,time are lumped
   lats_indices=closest_argmin(lats,lats_wind)
   lats_closest=lats_wind[lats_indices]         #--we need the closest values
   #--create new zz axis
   #ipdb.set_trace()
   lons_z  = xr.DataArray(lons_indices, dims="zz")
   lats_z  = xr.DataArray(lats_closest, dims="zz")
   #--Extract the data along the new axis
   windu=xr_u200_reduced.isel(z=lons_z).sel(latitude=lats_z)['u'].values

   #xr_u200_reduced.sel(z=lons_z.values).rename_dims({"z": "zz",}).sel(latitude=lats_z)['u'].values
   windv=xr_v200_reduced.isel(z=lons_z).sel(latitude=lats_z)['v'].values
   #xr_v200_reduced.sel(z=lons_z.values).rename_dims({"z": "zz",}).sel(latitude=lats_z)['v'].values

   #windu=xr_u200_reduced.interp(latitude=lats_z,z=lons_z,method='linear')['u'].values
   #windv=xr_v200_reduced.interp(latitude=lats_z,z=lons_z,method='linear')['v'].values
   #--Return u,v components of the wind
   return windu,windv
#
#--Compute shortest route
def shortest_route(dep_loc, arr_loc, npoints, R_earth, even_spaced=True):
    #--Returns npoints points along the shortest (great circle) route
    #--dep_loc and arr_loc are in the order (lon, lat)
    latlon_crs = ccrs.PlateCarree()
    #--projection for transforming trajectory
    crs = ccrs.Gnomonic(central_longitude=dep_loc[0], central_latitude=dep_loc[1])
    gno_aloc = crs.transform_point(arr_loc[0], arr_loc[1], latlon_crs)
    #--two options here
    #--points are evenly spaced in distance (this is what we want)
    if even_spaced:
        # Gnomic projection so distances from centre scale with tan
        # Map distance = R * tan (true dist/R)
        # Get distances in gnomic space that are evenly spaced in true distance
        point_gdists = np.tan(np.linspace(0, np.arctan(np.sqrt(np.sum(np.array(gno_aloc)**2))/R_earth), npoints+2))
        # Moving on straight line in gnomic projection, so can scale x,y coordinates separately
        scale_gdists = point_gdists/point_gdists[-1]
        newpoints = latlon_crs.transform_points(
            crs,
            scale_gdists*gno_aloc[0],
            scale_gdists*gno_aloc[1])
    #--points are not evenly spaced in distance
    else:
        newpoints = latlon_crs.transform_points(
            crs,
            np.linspace(0, gno_aloc[0], npoints+2),
            np.linspace(0, gno_aloc[1], npoints+2))
    #--Define trajectory points
    x0 = newpoints[:, 0] #--longitudes
    y0 = newpoints[:, 1] #--latitudes
    #--Return the array of points
    return x0, y0
#
#--Compute quickest route
def quickest_route(dep_loc, arr_loc, npoints, lat_iagos):
    #
    #--bounds
    bnds = tuple((-89.9,89.9) for i in range(npoints))
    #
    #--First compute shortest route
    x0, y0 = shortest_route(dep_loc, arr_loc, npoints, even_spaced=True)
    #
    #--Minimization with y0 from shortest route as initial conditions
    res = minimize(cost_squared,y0[1:-1],args=(x0[1:-1],*dep_loc,*arr_loc),method=method,bounds=bnds,options={'maxiter':maxiter,'disp':disp} )
    y = np.append(np.insert(res['x'],0,dep_loc[1]),arr_loc[1])
    quickest_time=cost_time(x0, y, dtprint=False)
    #
    #--Perform several other minimizations with y0 shifted southwards and northwards
    n=len(x0)
    for dymax in [-21.,-18.,-15.,-12.,-9.,-6.,-3.,3.,6.,9.,12.,15.,18.,21.]:
    ##for dymax in [-27.,-21.,-15.,-9.,-3.,3.,9.,15.,21.]:
     for imid in [n//2, n//3, 2*n//3]:
      dy=[dymax*float(i)/float(imid) for i in range(imid)]+[dymax*float(n-i)/float(n-imid) for i in range(imid,n)]
      y0p=y0+dy
      #--minimization with y0p as initial conditions
      res = minimize(cost_squared,y0p[1:-1],args=(x0[1:-1],*dep_loc,*arr_loc),method=method,bounds=bnds,options={'maxiter':maxiter,'disp':disp})
      y_2 = np.append(np.insert(res['x'],0,dep_loc[1]),arr_loc[1])
      quickest_time_2=cost_time(x0, y_2)
      if quickest_time_2 < quickest_time:
         quickest_time = quickest_time_2
         y = y_2   #--new best minimum
    #
    ##--y between -180 and 180
    ##y = (y+90.0)%180.0 - 90.0
    ##
    #--Solution to optimal route
    return (x0, y)


