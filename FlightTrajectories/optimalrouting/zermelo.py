import numpy as np

class ZermeloBase(object):
    def __init__(self,
                 cost_func=lambda x, y: np.ones(np.atleast_1d(x).shape),
                 ###wind_func=lambda x, y, t: (np.zeros(np.atleast_1d(x).shape),
                 ###                           np.zeros(np.atleast_1d(x).shape)),
                 wind_func=lambda x, y: (np.zeros(np.atleast_1d(x).shape),
                                         np.zeros(np.atleast_1d(x).shape)),
                 timestep=1,
                 psi_range=30,
                 psi_res=0.25,
                 length_factor=1.4,
                 max_dest_distance=100,
                 sub_factor=None):
        self.cost_func = cost_func
        self.wind_func = wind_func
        self.timestep = timestep
        self.psi_range = psi_range
        self.psi_res = psi_res
        self.length_factor = length_factor
        self.max_dest_distance = max_dest_distance
        self.sub_factor = sub_factor
        
    def distance_func(self, x1, y1, x2, y2):
        return

    def bearing_func(self, x1, y1, x2, y2):
        return

    def dpsi_dt_func(self, plon, plat, psi, airspeed, dtime):
        return

    def position_update_func(self, dep_loc, psi, airspeed, u, v):
        return

    def zermelo_path(self, dep_loc, lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, initial_psi, nsteps, airspeed, dtime):
        loc = [dep_loc]
        psi = [initial_psi]
        cost = [np.zeros(np.atleast_1d(initial_psi).shape)]
        u, v = self.wind_func(*loc[-1],  lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced)
        step = 0

        while step < nsteps:
            dpsi_dt = self.dpsi_dt_func(loc[-1][0], loc[-1][1], psi[-1], airspeed, dtime,  lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced)
            psi.append(psi[-1]+dpsi_dt*self.timestep)
            u, v = self.wind_func(*loc[-1],  lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced)
            newloc = self.position_update_func(loc[-1], psi[-1], airspeed, u, v)
            loc.append(newloc)
            cost.append(self.cost_func(loc[-1][0], loc[-1][1], dtime))
            step += 1

        return np.array(loc).astype('float'), np.array(psi).astype('float'), np.array(cost).astype('float')

    def trajectory_passingside_func(self, arr_loc, traj_locs):
        '''For each trajectory in locs, label it as left/right (0/1) depending if the goal is left or right
        of the flight track at closest approach'''
        # Calculate pairs either side of the destination
        solution_bearing = self.bearing_func(
            arr_loc[0],
            arr_loc[1],
            traj_locs[:, 0],
            traj_locs[:, 1])

        traj_bearing = self.bearing_func(
            traj_locs[:-1, 0],
            traj_locs[:-1, 1],
            traj_locs[1:, 0],
            traj_locs[1:, 1])

        # Find index of closest approach
        destination_dists = self.distance_func(
            traj_locs[:, 0], traj_locs[:, 1],
            arr_loc[0], arr_loc[1])
        min_loc = np.argmin(destination_dists, axis=0).clip(0, traj_bearing.shape[0]-1)
        tdirection = np.take_along_axis(traj_bearing, min_loc[None, :], 0)[0]
        sdirection = np.take_along_axis(solution_bearing, min_loc[None, :], 0)[0]
        bearing_diff = (sdirection-tdirection)%360
        bearing_diff[bearing_diff>180] -= 360
        passing_side = np.sign(bearing_diff)

        return passing_side

    def route_optimise(self, dep_loc, arr_loc,  lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, airspeed, dtime, debug=False):
        solution=True  #--OB
        initial_psi = 90-self.bearing_func(*dep_loc, *arr_loc) # psi is angle, not bearing (not the best idea...)
        if isinstance(self.psi_range, int) or isinstance(self.psi_range, float):
            initial_psi = np.arange(
                int(initial_psi-self.psi_range),
                int(initial_psi+self.psi_range),
                self.psi_res)
        else:
            initial_psi = np.arange(
                int(self.psi_range[0]),
                int(self.psi_range[1]),
                self.psi_res)
        start_loc = np.zeros((2, len(initial_psi)))
        start_loc[0] = dep_loc[0]
        start_loc[1] = dep_loc[1]

        total_dist = self.distance_func(*dep_loc, *arr_loc) # In metres
        # Steps to reach destination
        print('zermelo=',self.length_factor,total_dist,airspeed,self.timestep)
        nsteps = int(self.length_factor*(total_dist/airspeed)/self.timestep) 

        loc, psi, cost = self.zermelo_path(start_loc,  lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, initial_psi, nsteps, airspeed, dtime)

        destination_dists = self.distance_func(loc[:, 0], loc[:, 1], arr_loc[0], arr_loc[1])
        index = np.zeros(len(initial_psi))
        # if sub_factor is not None, we need to go further and re-calculate a higher-resolution set of trajectories
        if self.sub_factor:
            for sub_iter in range(5):
                passing_side = self.trajectory_passingside_func(arr_loc, loc)

                signchange_inds = np.where((np.diff(passing_side)!=0)&(np.diff(index)==0))

                new_initial_psi = [np.linspace(initial_psi[a], initial_psi[a+1], self.sub_factor+1)
                                   for a in signchange_inds]
                index = np.array([np.zeros((self.sub_factor+1))+a
                                  for a in range(len(signchange_inds[0]))]).ravel()
                initial_psi = np.array(new_initial_psi).ravel()
                initial_psi.sort()
                start_loc = np.zeros((2, len(initial_psi)))
                start_loc[0] = dep_loc[0]
                start_loc[1] = dep_loc[1]

                loc, psi, cost = self.zermelo_path(start_loc,  lons_wind, lats_wind, xr_u200_reduced, xr_v200_reduced, initial_psi, nsteps, airspeed, dtime)

                destination_dists = self.distance_func(loc[:, 0], loc[:, 1], arr_loc[0], arr_loc[1])

                try:
                    min_ind, length_ind = self.min_locate(destination_dists)
                    break # Minimum located!
                except:
                    continue
            else:
                # Max iterations reached with no solution
                #raise ValueError('Solution not found')
                solution=False
                min_ind=0
                length_ind=0

        else:
            min_ind, length_ind = self.min_locate(destination_dists)
        if debug:
            return (
                initial_psi[min_ind],
                self.timestep*length_ind,
                loc[:length_ind, :, min_ind],
                cost[:length_ind, min_ind],
                [initial_psi,
                 loc,
                 cost,
                 min_ind,
                 length_ind,
                 index])
        return solution, initial_psi[min_ind], self.timestep*length_ind, loc[:length_ind, :, min_ind], cost[:length_ind, min_ind]

    def min_locate(self, destination_dists):
        # Potential solution
        p_solns = np.where(np.min(destination_dists, axis=0) < self.max_dest_distance)
        p_min_ind = np.argmin(np.argmin(destination_dists, axis=0)[p_solns])
        min_ind = p_solns[0][p_min_ind]

        length_ind = np.argmin(destination_dists[:, min_ind])
        return min_ind, length_ind
