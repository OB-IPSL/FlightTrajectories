import math
import numpy as np
import sys
import numpy as np
from heapq import heappush, heappop
import scipy.sparse
import math

#--LMDZ formulation
#--liquid water vapour saturation
def el_foeew(Tair,celsius=False):
   rtt = 273.16
   if celsius:
     T=rtt+Tair #--K
   else:
     T=Tair
   restt = 611.14
   rkbol = 1.380658E-23
   rnavo = 6.0221367E+23
   r = rnavo*rkbol
   rmd = 28.9644
   rmv = 18.0153
   rd = 1000.*r/rmd
   rv = 1000.*r/rmv
   #r2es = restt*rd/rv #--this is what is in the model
   r2es = restt        #--needed to predict sat vap pressure in Pa
   r3les = 17.269
   r4les = 35.86
   return r2es*np.exp(r3les*(T-rtt)/(T-r4les))

#--LMDZ formulation
#--ice water vapour saturation
def ei_foeew(Tair,celsius=False):
   rtt = 273.16
   if celsius:
     T=rtt+Tair #--K
   else:
     T=Tair
   restt = 611.14
   rkbol = 1.380658E-23
   rnavo = 6.0221367E+23
   r = rnavo*rkbol
   rmd = 28.9644
   rmv = 18.0153
   rd = 1000.*r/rmd
   rv = 1000.*r/rmv
   #r2es = restt*rd/rv #--this is what is in the model
   r2es = restt        #--needed better to predict sat vap pressure in Pa
   r3ies = 21.875
   r4ies = 7.66
   return r2es*np.exp(r3ies*(T-rtt)/(T-r4ies))

#--compute G slope
#--Scmidt-Appleman criterion
def G(p):
  #--input: p pressure in Pa
  eps_w = 0.622 #--ratio of molecular masses of water and dry air (kg H2O kg air -1)
  RCPD=1004.    #--J/kg air/K
  EiH2O=1.25    #--emission index of water vapour for kerosene (kg kg-1)
  Qheat=43.E6   #--specific combustion heat for kerosene (J kg-1)
  eta=0.3       #--aircraft efficiency
  #--slope of dilution line in exhaust
  #--kg H2O/kg fuel * J kg air-1 K-1 * Pa / (kg H2O / kg air * J kg fuel-1)
  Gcontr = EiH2O * RCPD * p / (eps_w*Qheat*(1.-eta))
  #--Tcontr threshold (in K)
  Tcontr = 226.69+9.43*np.log(Gcontr-0.053)+0.72*(np.log(Gcontr-0.053))**2.0
  #--return arrays
  return Gcontr, Tcontr
#
#--find nearest item in items to pivot
def nearest(items, pivot):
   minitems=min(items, key=lambda x: abs(x - pivot))
   return minitems, items.index(minitems)
#
#--find nearest items in items to pivots - very very slow
def nearests(items, pivots):
   closest_list=[] ; closest_ind=[]
   for pivot in list(pivots):
      closest,ind=nearest(list(items),pivot) 
      closest_list.append(closest) 
      closest_ind.append(ind)
   return np.array(closest_list), np.array(closest_ind)
#
#--find array of indices of nearest for all items of A in B
def closest_argmin(A, B):
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & (np.abs(A-sorted_B[sorted_idx-1]) < np.abs(A-sorted_B[sorted_idx]))
    return sidx_B[sorted_idx-mask]
#
#--spherical to cartesian coordinates
def sph2car(theta,phi):
#--theta=latitude (in rad)
#--phi=longitude (in rad)
   x=np.cos(theta)*np.cos(phi)
   y=np.cos(theta)*np.sin(phi)
   z=np.sin(theta)
   return np.array([x,y,z])
#
#--cartesian to spherical coordinates
def car2sph(xyz):
   #--get coordinates from normalised vector
   x,y,z=xyz/np.linalg.norm(xyz,ord=2)
   #--cartesian to spherical (in rad)
   theta=np.arcsin(z)
   phi=np.sign(y)*np.arccos(x/np.cos(theta))
   return theta,phi

#--Earth radius in m
R_E = 6372800.0 #see http://rosettacode.org/wiki/Haversine_formula

def compdist(latp,lonp,lat1,lon1,lat2,lon2):
    '''Compares the distance from p to 1 and 2. Returns True if 2 is closer'''
    #print (haversine(latp,lonp,lat1,lon1), haversine(latp,lonp,lat2,lon2))
    #if not np.isfinite(haversine(latp,lonp,lat1,lon1)):
    #    return True
    return (haversine(latp,lonp,lat1,lon1) > haversine(latp,lonp,lat2,lon2))

def haversine(lat1,lon1,lat2,lon2):
    '''Computes the Haversine distance between two points in m'''
    dlat = math.pi*(lat2-lat1)/180.
    dlon = math.pi*(lon2-lon1)/180.
    lat1 = math.pi*(lat1)/180.
    lat2 = math.pi*(lat2)/180.
    arc = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2)**2)
    c= 2*np.arcsin(np.sqrt(arc))
    return R_E*c

def bearing(lat1, lon1, lat2, lon2):
    ''' Calculates the bearing between two points
    All arguments in degrees
    Code from: https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/'''
    dlon = np.deg2rad(lon2-lon1)
    X = np.cos(np.deg2rad(lat2)) * np.sin(dlon)
    Y = np.cos(np.deg2rad(lat1))*np.sin(np.deg2rad(lat2))-np.sin(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*np.cos(dlon)
    return np.rad2deg(np.arctan2(X, Y))

def lonlat_bearing(lon1, lat1, lon2, lat2):
    ''' Calculates the bearing between two points
    NOTE: switched lon/lat order to match TASIC code
    All arguments in degrees
    Code from: https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/'''
    dlon = np.deg2rad(lon2-lon1)
    X = np.cos(np.deg2rad(lat2)) * np.sin(dlon)
    Y = np.cos(np.deg2rad(lat1))*np.sin(np.deg2rad(lat2))-np.sin(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*np.cos(dlon)
    return np.rad2deg(np.arctan2(X, Y))

def theta(temperature,pressure):
    '''Calculated potential temperature
    Args:
       temp in K
       presure in hPa'''
    theta = temperature * (100000/pressure) ** 0.288
    return theta

def coordinate_rotate(lon, lat, centre_lon, centre_lat):
    '''Rotates a set of spherical coordinates so that the centre location
    is on the equator'''
    if abs(centre_lon)<10:
        icentre_lon = np.deg2rad(centre_lon-90)
    else:
        icentre_lon = np.deg2rad(centre_lon)
    icentre_lat = np.deg2rad(centre_lat)
    ilon = np.deg2rad(lon)
    ilat = np.deg2rad(lat)
    alpha = ilon - icentre_lon
    x1 = np.cos(ilat) * np.cos(alpha)
    x2 = np.cos(ilat) * np.sin(alpha)
    x3 = np.sin(ilat)

    x1p = np.cos(icentre_lat) * x1 + np.sin(icentre_lat) * x3
    x2p = x2
    x3p = -np.sin(icentre_lat) * x1 + np.cos(icentre_lat) * x3

    if abs(centre_lon)>10:
        return (np.rad2deg(np.arctan2(x2p, x1p)),
                np.rad2deg(np.arcsin(x3p)))
    else:
        return (np.rad2deg(np.arcsin(x3p)),
                -1*np.rad2deg(np.arctan2(x2p, x1p)))
    
class hKDTree(object):
    """
    hkd-tree for quick nearest-neighbor lookup

    """

    def __init__(self, data_lat, data_lon, leafsize=10):
        """Construct a haversine distance kd-tree.

        Parameters
        ----------
        data_lat, data_lon
                 : array_like, shape (n)
            The data points to be indexed. This array is not copied, and
            so modifying this data will result in bogus results.
        leafsize : positive int
            The number of points at which the algorithm switches over to
            brute-force.
        """
        self.data = np.concatenate((np.asarray(data_lat)[:,None],
                                    np.asarray(data_lon)[:,None]),axis=1)
        self.n, self.m = np.shape(self.data)
        self.leafsize = int(leafsize)
        if self.leafsize<1:
            raise ValueError("leafsize must be at least 1")
        self.maxes = np.amax(self.data,axis=0)
        self.mins = np.amin(self.data,axis=0)

        self.tree = self.__build(np.arange(self.n), self.maxes, self.mins)

    class node(object):
        if sys.version_info[0] >= 3:
            def __lt__(self, other): id(self) < id(other)
            def __gt__(self, other): id(self) > id(other)
            def __le__(self, other): id(self) <= id(other)
            def __ge__(self, other): id(self) >= id(other)
            def __eq__(self, other): id(self) == id(other)
    class leafnode(node):
        def __init__(self, idx):
            self.idx = idx
            self.children = len(idx)
    class innernode(node):
        def __init__(self, split_dim, split, less, greater):
            self.split_dim = split_dim
            self.split = split
            self.less = less
            self.greater = greater
            self.children = less.children+greater.children

    def __build(self, idx, maxes, mins):
        if len(idx)<=self.leafsize:
            return hKDTree.leafnode(idx)
        else:
            data = self.data[idx]
            #maxes = np.amax(data,axis=0)
            #mins = np.amin(data,axis=0)
            d = np.argmax(maxes-mins)
            maxval = maxes[d]
            minval = mins[d]
            if maxval==minval:
                # all points are identical; warn user?
                return hKDTree.leafnode(idx)
            data = data[:,d]

            # sliding midpoint rule; see Maneewongvatana and Mount 1999
            # for arguments that this is a good idea.
            split = (maxval+minval)/2
            less_idx = np.nonzero(data<=split)[0]
            greater_idx = np.nonzero(data>split)[0]
            if len(less_idx)==0:
                split = np.amin(data)
                less_idx = np.nonzero(data<=split)[0]
                greater_idx = np.nonzero(data>split)[0]
            if len(greater_idx)==0:
                split = np.amax(data)
                less_idx = np.nonzero(data<split)[0]
                greater_idx = np.nonzero(data>=split)[0]
            if len(less_idx)==0:
                # _still_ zero? all must have the same value
                assert np.all(data==data[0]), "Troublesome data array: %s" % data
                split = data[0]
                less_idx = np.arange(len(data)-1)
                greater_idx = np.array([len(data)-1])

            lessmaxes = np.copy(maxes)
            lessmaxes[d] = split
            greatermins = np.copy(mins)
            greatermins[d] = split
            return hKDTree.innernode(d, split,
                    self.__build(idx[less_idx],lessmaxes,mins),
                    self.__build(idx[greater_idx],maxes,greatermins))

    def __query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf):

        side_distances = np.maximum(0,np.maximum(x-self.maxes,self.mins-x))
        if p!=np.inf:
            side_distances**=p
            min_distance = np.sum(side_distances)
        else:
            min_distance = np.amax(side_distances)

        # priority queue for chasing nodes
        # entries are:
        #  minimum distance between the cell and the target
        #  distances between the nearest side of the cell and the target
        #  the head node of the cell
        q = [(min_distance,
              tuple(side_distances),
              self.tree)]
        # priority queue for the nearest neighbors
        # furthest known neighbor first
        # entries are (-distance**p, i)
        neighbors = []

        if eps==0:
            epsfac=1
        elif p==np.inf:
            epsfac = 1/(1+eps)
        else:
            epsfac = 1/(1+eps)**p

        if p!=np.inf and distance_upper_bound!=np.inf:
            distance_upper_bound = distance_upper_bound**p

        while q:
            min_distance, side_distances, node = heappop(q)
            if isinstance(node, hKDTree.leafnode):
                # brute-force
                data = self.data[node.idx]
                ds = haversine(data[:,0],data[:,1],x[np.newaxis,:][:,0],x[np.newaxis,:][:,1])
                for i in range(len(ds)):
                    if ds[i]<distance_upper_bound:
                        if len(neighbors)==k:
                            heappop(neighbors)
                        heappush(neighbors, (-ds[i], node.idx[i]))
                        if len(neighbors)==k:
                            distance_upper_bound = -neighbors[0][0]
            else:
                # we don't push cells that are too far onto the queue at all,
                # but since the distance_upper_bound decreases, we might get
                # here even if the cell's too far
                if min_distance>distance_upper_bound*epsfac:
                    # since this is the nearest cell, we're done, bail out
                    break
                # compute minimum distances to the children and push them on
                if x[node.split_dim]<node.split:
                    near, far = node.less, node.greater
                else:
                    near, far = node.greater, node.less

                # near child is at the same distance as the current node
                heappush(q,(min_distance, side_distances, near))

                # far child is further by an amount depending only
                # on the split value
                sd = list(side_distances)
                if p == np.inf:
                    min_distance = max(min_distance, abs(node.split-x[node.split_dim]))
                elif p == 1:
                    sd[node.split_dim] = np.abs(node.split-x[node.split_dim])
                    min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]
                else:
                    sd[node.split_dim] = np.abs(node.split-x[node.split_dim])**p
                    min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]

                # far child might be too far, if so, don't bother pushing it
                if min_distance<=distance_upper_bound*epsfac:
                    heappush(q,(min_distance, tuple(sd), far))
        return sorted([(-d,i) for (d,i) in neighbors])
        
    def query(self, x_lats, x_lons, k=1, eps=0, p=2, distance_upper_bound=np.inf):
        """
        query the kd-tree for nearest neighbors

        Parameters
        ----------

        x_lats,x_lons
            : array-like, (n)
            An array of points to query.
        k : integer
            The number of nearest neighbors to return.
        eps : nonnegative float
            Return approximate nearest neighbors; the kth returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real kth nearest neighbor.
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance. This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.

        Returns
        -------

        d : array of floats
            The distances to the nearest neighbors.
            If x has shape tuple+(self.m,), then d has shape tuple if
            k is one, or tuple+(k,) if k is larger than one.  Missing
            neighbors are indicated with infinite distances.  If k is None,
            then d is an object array of shape tuple, containing lists
            of distances. In either case the hits are sorted by distance
            (nearest first).
        i : array of integers
            The locations of the neighbors in self.data. i is the same
            shape as d.

        """
        x = np.concatenate((np.asarray(x_lats)[:,None],np.asarray(x_lons)[:,None]),axis=1)
        if np.shape(x)[-1] != self.m:
            raise ValueError("x must consist of vectors of length %d but has shape %s" % (self.m, np.shape(x)))
        if p<1:
            raise ValueError("Only p-norms with 1<=p<=infinity permitted")
        retshape = np.shape(x)[:-1]
        if retshape!=():
            if k is None:
                dd = np.empty(retshape,dtype=np.object)
                ii = np.empty(retshape,dtype=np.object)
            elif k>1:
                dd = np.empty(retshape+(k,),dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape+(k,),dtype=np.int)
                ii.fill(self.n)
            elif k==1:
                dd = np.empty(retshape,dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(retshape,dtype=np.int)
                ii.fill(self.n)
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")
            for c in np.ndindex(retshape):
                hits = self.__query(x[c], k=k, p=p, distance_upper_bound=distance_upper_bound)
                if k is None:
                    dd[c] = [d for (d,i) in hits]
                    ii[c] = [i for (d,i) in hits]
                elif k>1:
                    for j in range(len(hits)):
                        dd[c+(j,)], ii[c+(j,)] = hits[j]
                elif k==1:
                    if len(hits)>0:
                        dd[c], ii[c] = hits[0]
                    else:
                        dd[c] = np.inf
                        ii[c] = self.n
            return dd, ii
        else:
            hits = self.__query(x, k=k, p=p, distance_upper_bound=distance_upper_bound)
            if k is None:
                return [d for (d,i) in hits], [i for (d,i) in hits]
            elif k==1:
                if len(hits)>0:
                    return hits[0]
                else:
                    return np.inf, self.n
            elif k>1:
                dd = np.empty(k,dtype=np.float)
                dd.fill(np.inf)
                ii = np.empty(k,dtype=np.int)
                ii.fill(self.n)
                for j in range(len(hits)):
                    dd[j], ii[j] = hits[j]
                return dd, ii
            else:
                raise ValueError("Requested %s nearest neighbors; acceptable numbers are integers greater than or equal to one, or None")

    def __query_ball_point(self, x, r, p=2., eps=0):
        R = Rectangle(self.maxes, self.mins)
        
        def traverse_checking(node, rect):
            if rect.min_distance_point(x, p) > r / (1. + eps):
                return []
            elif rect.max_distance_point(x, p) < r * (1. + eps):
                return traverse_no_checking(node)
            elif isinstance(node, hKDTree.leafnode):
                d = self.data[node.idx]
                return node.idx[haversine(d[:,0],d[:,1],x[np.newaxis,:][:,0],
                                          x[np.newaxis,:][:,1]) <= r].tolist()
            else:
                less, greater = rect.split(node.split_dim, node.split)
                return traverse_checking(node.less, less) + \
                    traverse_checking(node.greater, greater)
            
        def traverse_no_checking(node):
            if isinstance(node, hKDTree.leafnode):
                return node.idx.tolist()
            else:
                return traverse_no_checking(node.less) + \
                    traverse_no_checking(node.greater)
            
        return traverse_checking(self.tree, R)

    def query_ball_point(self, x_lats, x_lons, r, p=2., eps=0):
        """Find all points within distance r of point(s) x.
        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r : positive float
            The radius of points to return.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.
        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.
        Notes
        -----
        If you have many points whose neighbors you want to find, you may save
        substantial amounts of time by putting them in a KDTree and using
        query_ball_tree.
        Examples
        --------
        >>> from scipy import spatial
        >>> x, y = np.mgrid[0:4, 0:4]
        >>> points = zip(x.ravel(), y.ravel())
        >>> tree = spatial.KDTree(points)
        >>> tree.query_ball_point([2, 0], 1)
        [4, 8, 9, 12]
        """
        x = np.concatenate((np.asarray(x_lats)[:,None],np.asarray(x_lons)[:,None]),axis=1)
        if x.shape[-1] != self.m:
            raise ValueError("Searching for a %d-dimensional point in a "
                             "%d-dimensional KDTree" % (x.shape[-1], self.m))
        if len(x.shape) == 1:
            return self.__query_ball_point(x, r, p, eps)
        else:
            retshape = x.shape[:-1]
            result = np.empty(retshape, dtype=np.object)
            for c in np.ndindex(retshape):
                result[c] = self.__query_ball_point(x[c], r, p=p, eps=eps)
            return result
