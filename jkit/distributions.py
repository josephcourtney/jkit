import numpy as np

from .math_util import  sample_disjoint_uniform, merge_angular_intervals,\
                        random_point_on_hypersphere, hypersphere_intersection_with_circle,\
                        point_in_hypersphere, SparseKDGrid, plane_unit_vectors
from .util import random_pop


class Disk(object):
    def __init__(self, point, r):
        point = np.asarray(point).reshape(-1)
        self.dim = point.shape[0]
        self.center = point
        self.r = r

    def covers(self, point):
        return point_in_hypersphere(self.center, self.r, point)

    def intersection_with_circle(self, center, r, uv):
        return hypersphere_intersection_with_circle(self.center, self.r, center, r, uv = uv)

class BlueDistribution(object):
    '''
        implements Spoke Darts for Efficient High Dimensional Blue Noise Sampling, Ebeida et al.
        https://arxiv.org/pdf/1408.1118.pdf
        As per the conclusions of the paper, only degenerate plane spoke darts (circles) are
        used as they are more efficient and simpler to implement.
    '''

    def __init__(self, dim, radius, density = lambda pt: 1.0, n_tries = 5):
        self.dim = dim
        self.radius = radius
        self.n_tries = n_tries
        self.density = density
        self.reset()

    def get_radius(self, pt):
        return self.radius * self.density(pt)


    def reset(self):
        self.grid = SparseKDGrid(self.dim, 2*self.radius, buffer = 4*self.radius)
        self.disks = dict()

    def trim_spoke(self, spoke_center, plane_spoke):
        '''
            Given a plane_spoke (circle) sample a position from regions on its perimeter
            that are not covered by existing disks.
        '''
        spoke_center = np.asarray(spoke_center)
        nearby_disks = [self.disks[tuple(p)] for p in self.grid.points_near(spoke_center)]
        u, v = plane_unit_vectors(spoke_center, plane_spoke[:2])

        spoke_radius = np.sqrt(np.sum((plane_spoke[0] - spoke_center)**2))

        # u and v define plane on which plane spoke resides

        raw_angular_intervals = list()
        for d in nearby_disks:
            res = d.intersection_with_circle(spoke_center, spoke_radius, (u,v))
            if res is None:
                continue
            else:
                (pt_0, pt_1), (theta_0, theta_1) = res
            if theta_1 < theta_0:
                theta_1 += 2*np.pi
            raw_angular_intervals.append( (theta_0, theta_1) )

        available_intervals = merge_angular_intervals(raw_angular_intervals, False, True)
        theta = sample_disjoint_uniform(available_intervals)
        theta = theta or 0
        new_point = (np.cos(theta)*u + np.sin(theta)*v)*spoke_radius + spoke_center

        for d in nearby_disks:
            if d.covers(new_point):
                return None
        else:
            return new_point

    def random_plane_spoke(self, c_pt):
        pt = random_point_on_hypersphere(self.dim, (2,)).T * self.get_radius(c_pt) + c_pt
        spoke = [pt[0], pt[1]]
        return spoke

    def sample(self):
        self.reset()
        initial_point = np.random.random(size=self.dim)
        sample_set = [initial_point]
        self.disks[tuple(initial_point)] = Disk(initial_point, self.get_radius(initial_point))
        active_set = [initial_point]

        while(len(active_set)>0):
            core_point = random_pop(active_set)

            n_reject = 0
            while n_reject <= self.n_tries:
                raw_point = self.random_plane_spoke(core_point)
                refined_point = self.trim_spoke(core_point, raw_point)
                if not refined_point is None:
                    # keep track of points beyond the unit square to avoid bunching on the edge
                    spoke_radius = np.sqrt(np.sum((refined_point - core_point)**2))
                    self.grid.append(refined_point)
                    self.disks[tuple(refined_point)] = Disk(refined_point, spoke_radius)
                    if np.all(refined_point >= 0-spoke_radius) and np.all(refined_point <= 1+spoke_radius):
                        sample_set.append(refined_point)
                        active_set.append(refined_point)
                        n_reject = 0
                        print(len(sample_set))
                    else:
                        n_reject += 1

                else:
                    n_reject += 1

        pts = np.array(sample_set)
        pts = pts[np.all(pts <= 1, 1)]
        pts = pts[np.all(pts >= 0, 1)]
        return pts
