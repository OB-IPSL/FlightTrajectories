'''Zermelo solution on a 2D plane'''
from .zermelo import ZermeloBase
import numpy as np


class ZermeloXY(ZermeloBase):
    def distance_func(self, x1, y1, x2, y2):
        return np.sqrt((x2-x1)**2+(y2-y1)**2)

    def bearing_func(self, x1, y1, x2, y2):
        # Note that these are backwards from the 'traditional' argument
        # order for arctan2 as we are calculating the bearing, not the
        # angle
        return np.rad2deg(np.arctan2((x2-x1), (y2-y1)))

    def dpsi_dt_func(self, px, py, psi, airspeed, dtime):
        u, v = self.wind_func(px, py, dtime)
        I = self.cost_func(px, py, dtime)

        dval = 0.1

        dudx, dvdx = self.wind_func(px+dval, py, dtime)
        dudx, dvdx = (dudx-u)/dval, (dvdx-v)/dval
        dudy, dvdy = self.wind_func(px, py+dval, dtime)
        dudy, dvdy = (dudy-u)/dval, (dvdy-v)/dval
        dIdx = (self.cost_func(px+dval, py, dtime)-I)/dval
        dIdy = (self.cost_func(px, py+dval, dtime)-I)/dval

        psi_r = np.deg2rad(psi)
        Nv = airspeed + u*np.cos(psi_r) + v*np.sin(psi_r)

        # Careful with the signs here! The cost signs were originally wrong
        dpsi = (
            + dvdx*(np.sin(psi_r))**2
            + np.sin(psi_r)*np.cos(psi_r)*(dudx-dvdy)
            - dudy*(np.cos(psi_r))**2
            - (Nv/I)*np.sin(psi_r)*dIdx
            + (Nv/I)*np.cos(psi_r)*dIdy)

        return np.rad2deg(dpsi)

    def position_update_func(self, dep_loc, psi, airspeed, u, v):
        # dep_loc in degrees, psi in degrees, airspeed, u, v in ms-1
        dx = (u + airspeed*np.cos(np.deg2rad(psi))) * self.timestep
        dy = (v + airspeed*np.sin(np.deg2rad(psi))) * self.timestep
        return (dep_loc[0]+dx,
                dep_loc[1]+dy)

