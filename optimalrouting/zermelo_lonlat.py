from .zermelo import ZermeloBase
import numpy as np
import math
from misc_geo import haversine, lonlat_bearing

R_EARTH = 6372800 #--OB

#####################
# Utility functions #
#####################
def cosec(val):
    return 1/np.sin(val)


def sec(val):
    return 1/np.cos(val)


def cot(val):
    return 1/np.tan(val)


class ZermeloLonLat(ZermeloBase):
    def distance_func(self, x1, y1, x2, y2):
        return haversine(x1, y1, x2, y2)

    def bearing_func(self, x1, y1, x2, y2):
        return lonlat_bearing(x1, y1, x2, y2)

    def dpsi_dt_func(self, plon, plat, psi, airspeed, dtime):
        ###u, v = self.wind_func(plon, plat, dtime)
        u, v = self.wind_func(plon, plat)
        I = self.cost_func(plon, plat, dtime)

        # Used for calculating numerical gradients
        dval = 0.25
        
        ###dudlon, dvdlon = self.wind_func(plon+dval, plat, dtime)
        dudlon, dvdlon = self.wind_func(plon+dval, plat)
        dudlon, dvdlon = (dudlon-u)/np.deg2rad(dval), (dvdlon-v)/np.deg2rad(dval)
        ###dudlat, dvdlat = self.wind_func(plon, plat+dval, dtime)
        dudlat, dvdlat = self.wind_func(plon, plat+dval)
        dudlat, dvdlat = (dudlat-u)/np.deg2rad(dval), (dvdlat-v)/np.deg2rad(dval)

        lat_r = np.deg2rad(plat)
        lon_r = np.deg2rad(plon)
        psi_r = np.deg2rad(90-psi)

        dpsi = 1/R_EARTH * (
            - dvdlon*sec(lat_r)*(np.cos(psi_r))**2
            + np.tan(lat_r)*np.sin(psi_r)*(airspeed+u*np.sin(psi_r)+v*np.cos(psi_r))
            + np.sin(psi_r)*np.cos(psi_r)*(-dudlon*sec(lat_r)+dvdlat)
            + dudlat*(np.sin(psi_r))**2)

        return np.rad2deg(-dpsi)

    def position_update_func(self, dep_loc, psi, airspeed, u, v):
        # dep_loc in degrees, psi in degrees, airspeed, u, v in ms-1
        # Calculate bearing and distance adjusted for wind, assuming flat earth
        dx = (u + airspeed*np.cos(np.deg2rad(psi))) * self.timestep
        dy = (v + airspeed*np.sin(np.deg2rad(psi))) * self.timestep
        # Allows for poles (I think)
        distance = np.sqrt(dx**2 + dy**2)
        Ad = distance/R_EARTH

        angle = np.arctan2(dy, dx)
        bearing = np.pi/2 - angle

        olon, olat = np.deg2rad(dep_loc)
        nlat = np.arcsin(
            np.sin(olat)*np.cos(Ad) +
            np.cos(olat)*np.sin(Ad)*np.cos(bearing))           

        nlon = olon + np.arctan2(
            np.sin(bearing)*np.sin(Ad)*np.cos(olat),
            np.cos(Ad)-np.sin(olat)*np.sin(nlat))

        return np.rad2deg(nlon), np.rad2deg(nlat)
