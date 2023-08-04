import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import numpy as np
import cartopy
import warnings
import argparse
import glob
from math import exp,log

pltshow=True

parser = argparse.ArgumentParser()
parser.add_argument('--maxiter', type=int, default=100, help='max number of iterations')
parser.add_argument('--method', type=str, default='SLSQP', choices=['SLSQP','BFGS','L-BFGS-B'], help='minimization method')
parser.add_argument('--flightid', type=str, default=2019010510370802, help='flight number')
#
#--get arguments from command line
args = parser.parse_args()
#
#--copy arguments into variables
maxiter=args.maxiter
method=args.method
flightid0=args.flightid

#--pathout with output data
pathout='/projsu/cmip-work/oboucher/FR24/ROUTE/'+str(maxiter)+'/'+method+'/'
#
#--stop warnings
warnings.filterwarnings("ignore")
#
for flightid in [flightid0]:
  #
  #--picklefile
  picklefile=glob.glob(pathout+'data_'+str(flightid)+'_lev*_????.pickle')
  print(picklefile)
  # Load dictionary from disk and display
  dico = pickle.load(open(picklefile[0],'rb'))
  #--extract dictionary
  iagos_id=dico['iagos_id']
  flightid_iagos=dico['flightid_iagos']
  lat_pole=dico['lat_pole']
  lon_pole=dico['lon_pole']
  lon_iagos_values=dico['lon_iagos_values']
  lat_iagos_values=dico['lat_iagos_values']
  lon_iagos_cruising=dico['lon_iagos_cruising']
  pressure_iagos=dico['pressure_iagos']
  lon_key_values=dico['lon_key_values']
  lat_key_values=dico['lat_key_values']
  alt_key_values=dico['alt_key_values']
  lon_shortest=dico['lon_shortest']
  lat_shortest=dico['lat_shortest']
  lon_quickest=dico['lon_quickest']
  lat_quickest=dico['lat_quickest']
  solution=dico['solution']
  lon_ed=dico['lon_ed']
  lat_ed=dico['lat_ed']
  lons_wind=dico['lons_wind']
  lats_wind=dico['lats_wind']
  xr_u200=dico['xr_u200']
  xr_v200=dico['xr_v200']
  dep_airport_iagos=dico['dep_airport_iagos']
  arr_airport_iagos=dico['arr_airport_iagos']
  stryr=dico['stryr']
  strmth=dico['strmth']
  strday=dico['strday']
  dt_shortest=dico['dt_shortest']
  dt_quickest=dico['dt_quickest']
  dt_ed=dico['dt_ed']
  dt_iagos_2=dico['dt_iagos_2']
  optim_level=dico['optim_level']
  #
  yr=int(stryr)
  #
  #----------------------------------------------------------------------------------------------------------------------
  #--PLOT 1
  #--prepare rotated grid
  rotated = ccrs.RotatedPole(pole_latitude=lat_pole,pole_longitude=lon_pole)
  #-prepare map traj plot
  #------------------------------
  fig=plt.figure(figsize=(10,5))
  ax=fig.add_subplot(111, projection=rotated)
  #--lon-lat
  ax.plot(lon_iagos_values, lat_iagos_values, c='black', lw=2, label='IAGOS')
  ax.scatter(lon_key_values, lat_key_values, c='red', marker='X', lw=1)
  ax.plot(lon_shortest, lat_shortest, c='green', lw=2, label='Shortest',zorder=10)
  ax.plot(lon_quickest, lat_quickest, c='blue',  lw=2, label='Gradient descent')
  if solution: ax.plot(lon_ed, lat_ed, c='cyan', lw=2, label='Zermelo method')
  #--wind field
  ax.quiver(lons_wind[::10],lats_wind[::10],np.transpose(xr_u200['u'].values[::10,::10]),np.transpose(xr_v200['v'].values[::10,::10]),scale=1200)
  #--define domain with a margin
  ax.set_xlim(np.min(lon_shortest)-20.,np.max(lon_shortest)+20.)
  ax.set_ylim(np.max([-90.,np.min(lat_shortest)-20.]),np.min([90.,np.max(lat_shortest)+20.]))
  #--make plot nice
  ax.legend(loc='lower left',fontsize=10)
  fig.gca().coastlines()
  fig.gca().add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"),alpha=1)
  fig.gca().add_feature(cartopy.feature.BORDERS,color="red",linewidth=0.3)
  #--plot title
  plt.title(iagos_id+' '+str(iagos_id)+' '+dep_airport_iagos+'=>'+arr_airport_iagos+' '+stryr+strmth+strday+\
            ' level='+str(optim_level)+' hPa'+'\n'+ 'Shortest='+"{:.3f}".format(dt_shortest)+\
            ' Gradient descent='+"{:.3f}".format(dt_quickest)+' Zermelo method='+"{:.3f}".format(dt_ed)+' IAGOS='+"{:.3f}".format(dt_iagos_2),fontsize=11)
  #--save, show and close
  basefile=pathout+'traj_'+str(iagos_id)+'_lev'+str(optim_level)+'_'+str(yr)
  plt.savefig(basefile+'.png',dpi=150,bbox_inches='tight')
  if pltshow: plt.show()
  plt.close()
  #
  #----------------------------------------------------------------------------------------------------------------------
  #--PLOT 2
  #-- height
  #----------------
  fig,ax = plt.subplots()
  ax.plot(lon_iagos_values, pressure_iagos.values/100., c='black', lw=2, label='Pressure IAGOS')
  ax.set_ylim(1030,0)
  ax.set_ylabel("Pressure (hPa)",color="black",fontsize=14)
  ax.scatter(lon_key_values, alt_key_values/100., c='red', marker='X', lw=1, label='FL - Cruising')
  #--make plot nice
  plt.title(str(iagos_id)+' '+dep_airport_iagos+'=>'+arr_airport_iagos+' '+stryr+strmth+strday+' level='+str(optim_level)+' hPa')
  plt.legend()
  #--save, show and close
  basefile=pathout+'alt_'+str(iagos_id)+'_lev'+str(optim_level)+'_'+str(yr)
  plt.savefig(basefile+'.png',dpi=150,bbox_inches='tight')
  if pltshow: plt.show()
  plt.close()
