import collections

import numpy as np


def logsumexp(ary, axis=-1):
    a = np.max(ary)
    return a + np.log(np.sum(np.exp(ary-a), axis=axis))


def nnls(phi, y, t=1e-6, beta=1):
    m, n = phi.shape
    p = set()
    z = set(range(n))
    w = np.zeros((n, 1))
    y = y.reshape((m, 1))
    w_tmp = np.dot(phi.T, (y - np.dot(phi, w)))
    w_hist = []
    while len(z) != 0 and np.max(w_tmp) > t:
        j = np.where(w_tmp == np.max(w_tmp))[0][0]
        p.add(j)
        z.remove(j)
        phi_p = phi[:, sorted(tuple(p))]
        s = np.zeros((n, 1))
        s[sorted(tuple(p)), :] = np.dot(np.linalg.inv(np.dot(phi_p.T, phi_p)), np.dot(phi_p.T, y))
        while np.any(s[sorted(tuple(p)), :] <= t):
            alpha = np.min(w[sorted(tuple(p)), :]/(w[sorted(tuple(p)), :]-s[sorted(tuple(p)), :]))
            w = w + alpha * (s - w)
            z |= (p & set(np.where(w == 0)[0]))
            p -= set(np.where(w == 0)[0])
            phi_p = phi[:, sorted(tuple(p))]
            s[sorted(tuple(p))] = np.dot(np.linalg.inv(np.dot(phi_p.T, phi_p)), np.dot(phi_p.T, y))
            s[sorted(tuple(z))] = 0
        w = s[:]
        w_tmp = np.dot(phi.T, (y-np.dot(phi, w)))

        if np.any([np.allclose(w, ex) for ex in w_hist]):
            break
        w_hist.append(w)

        sigma = beta*np.linalg.inv(np.dot(phi.T, phi))

    return w, sigma


def sample_categorical(weights):
    w = weights / np.sum(weights)
    return w.cumsum().searchsorted(np.random.random())


def plane_unit_vectors(origin, points):
    u = (points[0] - origin) / np.sqrt(np.sum((points[0] - origin)**2))
    v = (points[1] - origin) - np.dot(np.dot(u, (points[1] - origin).T), u)
    v = v / np.sqrt(np.sum(v**2))
    return u, v


def merge_angular_intervals(
    angle_interval_iterable,
    return_merged_intervals=True,
    return_open_intervals=False,
    return_open_angle=False
):
    ret_val = list()
    merged_angle_intervals = list()
    for (theta_0, theta_1) in angle_interval_iterable:
        for ivl in merged_angle_intervals:
            if ivl[0] <= theta_0 <= ivl[1]:
                merged_angle_intervals.remove(ivl)
                merged_angle_intervals.append((ivl[0], theta_1))
                break
            elif ivl[0] <= theta_1 <= ivl[1]:
                merged_angle_intervals.remove(ivl)
                merged_angle_intervals.append((theta_0, ivl[1]))
                break
        else:
            merged_angle_intervals.append((theta_0, theta_1))
    merged_angle_intervals = sorted(merged_angle_intervals, key=lambda x: x[0])
    if return_merged_intervals:
        ret_val.append(merged_angle_intervals)

    if return_open_intervals or return_open_angle:
        open_intervals = [
            (merged_angle_intervals[i-1][1], merged_angle_intervals[i][0])
            for i in range(len(merged_angle_intervals))
        ]
        for i in range(len(open_intervals)):
            if open_intervals[i][0] > open_intervals[i][1]:
                open_intervals[i] = (open_intervals[i][0], open_intervals[i][1]+2*np.pi)
        if return_open_intervals:
            ret_val.append(open_intervals)
        if return_open_angle:
            if len(open_intervals) > 0:
                open_angle = sum([ivl[1]-ivl[0] for ivl in open_intervals])
            else:
                open_angle = 2*np.pi
            ret_val.append(open_angle)

    if len(ret_val) == 0:
        return None
    elif len(ret_val) == 1:
        return ret_val[0]
    else:
        return tuple(ret_val)


def sample_disjoint_uniform(interval_iterable):
    try:
        iter(interval_iterable)
    except:
        raise ValueError("interval_iterable must be iterable")

    ivl = [sorted(e) for e in interval_iterable]
    d = sum([e[1]-e[0] for e in ivl])
    e = np.random.random()*d

    for i in range(len(ivl)):
        if e <= ivl[i][1]-ivl[i][0]:
            return e + ivl[i][0]
        else:
            e -= (ivl[i][1]-ivl[i][0])
    else:
        return None


def random_point_on_hypersphere(dim, shape):
    x = np.random.normal(size=(dim, np.product(shape)))
    x = x / np.sqrt(np.sum(x**2, 0))[None, :]
    return x.reshape([dim]+list(shape))


def random_point_in_hyperannulus(dim, r_min, r_max, shape):
    n = np.product(shape)
    phi = np.random.normal(size=(dim, n))
    phi = phi / np.sqrt(np.sum(phi**2, 0))[None, :]
    r = (np.random.random(size=n) * (r_max**(dim+1) - r_min**(dim+1)) + r_min**(dim+1))**(1/(dim+1))
    return (phi * r).T.reshape(shape+(dim,))


def random_point_in_hypersphere(dim, shape):
    n = np.product(shape)
    phi = np.random.normal(size=(dim, n))
    phi = phi / np.sqrt(np.sum(phi**2, 0))[None, :]
    r = np.random.random(size=n)**(1/(dim+1))
    return (phi * r).T.reshape(shape+(dim,))


def hypersphere_intersection_with_line(center, radius, line):
    # intersection of a line and a hypersphere https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    o = np.asarray(line[0]).reshape(-1)
    l = np.asarray(line[1]).reshape(-1) - o
    l_2 = np.sum(l**2)
    omc = (o-center)
    a = -2 * np.dot(l, omc) / (2*l_2)

    e = (2*np.dot(l, omc))**2 - 4*l_2*(np.sum(omc**2) - radius**2)
    if e < 0:
        return None
    else:
        b = np.sqrt(e) / (2*l_2)
        if 0 <= a+b <= np.sqrt(l_2):
            if 0 <= a-b <= np.sqrt(l_2):
                d = min(a-b, a+b)
            else:
                d = a+b
        else:
            if 0 <= a-b <= np.sqrt(l_2):
                d = a-b
            else:
                return None
        return o+d*l


def hypersphere_intersection_with_circle(
    hypersphere_center,
    hypersphere_radius,
    circle_center,
    circle_radius,
    uv=None,
    perimeter_points=None
):
    if uv is None and perimeter_points is None:
        raise ValueError("You must provide either unit vectors on circle plane or two points on circle perimeter")
    if uv is None:
        # define unit vectors of hyperplane that circle lies on
        u = (perimeter_points[0] - circle_center) / np.sqrt(np.sum((perimeter_points[0] - circle_center)**2))
        v = (perimeter_points[1] - circle_center) - np.dot(np.dot(u, (perimeter_points[1] - circle_center).T), u)
        v = v / np.sqrt(np.sum(v**2))
    else:
        u, v = uv

    # center of circular intersection of hypersphere and hyperplane
    # in u-v coordinates with origin at circle center
    c_s_c = np.array([np.dot(hypersphere_center - circle_center, u), np.dot(hypersphere_center - circle_center, v)])
    c2c_d = np.sqrt(np.sum(c_s_c**2))
    if c2c_d > hypersphere_radius + circle_radius or\
       c2c_d < np.abs(hypersphere_radius - circle_radius) or\
       c2c_d == 0:
        return None
    a = (hypersphere_radius**2 - circle_radius**2 + c2c_d**2) / (2*c2c_d)
    h = np.sqrt(hypersphere_radius**2 - a**2)
    p_0 = c_s_c - a * c_s_c / c2c_d

    circle_center_uv = np.array([np.dot(circle_center, u), np.dot(circle_center, v)])

    # left point
    p_ix_uv_0 = np.array([
        (p_0[0] + h * c_s_c[1] / c2c_d),
        (p_0[1] - h * c_s_c[0] / c2c_d)
    ])

    p_ix_0 = (p_ix_uv_0[0]*u + p_ix_uv_0[1]*v + circle_center)

    # left point
    p_ix_uv_1 = np.array([
        (p_0[0] - h * c_s_c[1] / c2c_d),
        (p_0[1] + h * c_s_c[0] / c2c_d)
    ])

    p_ix_1 = (p_ix_uv_1[0]*u + p_ix_uv_1[1]*v + circle_center)

    d = np.dot((p_ix_uv_0 - circle_center_uv), (p_ix_uv_1 - circle_center_uv))

    theta_0 = np.arctan2((p_ix_uv_0 - circle_center_uv)[1], (p_ix_uv_0 - circle_center_uv)[0])
    theta_1 = np.arctan2((p_ix_uv_1 - circle_center_uv)[1], (p_ix_uv_1 - circle_center_uv)[0])

    if d < 0:
        p_left, p_right = p_ix_0, p_ix_1
    else:
        p_left, p_right = p_ix_1, p_ix_0

    theta_left = np.arctan2(p_ix_uv_0[1], p_ix_uv_0[0])
    theta_right = np.arctan2(p_ix_uv_1[1], p_ix_uv_1[0])

    return (p_left, p_right), (theta_left, theta_right)


def point_in_hypersphere(center, radius, point):
    point = np.asarray(point).reshape(-1)
    assert point.shape[0] == center.shape[0]
    r = np.sqrt(np.sum((center - point)**2))
    return radius >= r


class SparseKDGrid(object):
    """
        sparse k-dimensional grid facilitating quick lookup of nearby points
    """
    def __init__(self, dim, delta, buffer=0.0):
        self.buffer = buffer
        self.dim = dim
        self.delta = delta
        self.points = collections.defaultdict(list)
        self.n_max = int(np.floor((1+2*self.buffer)/self.delta))

    def append(self, pt):
        pt = np.asarray(pt)
        idx = tuple(np.floor((pt+self.buffer) / self.delta).astype(int))
        self.points[idx].append(pt)

    def points_near(self, pt):
        pt = np.asarray(pt)
        idx = np.floor((pt+self.buffer) / self.delta).astype(int)
        offset_idx = np.indices((3,) * self.dim).reshape(self.dim, -1).T
        offsets = np.r_[-1, 0, 1].take(offset_idx)
        # offsets = offsets[np.any(offsets, 1)]
        neighbors = idx + offsets
        valid = np.all((neighbors < np.array([self.n_max]*self.dim)) & (neighbors >= 0), axis=1)
        neighbors = neighbors[valid]

        pt_list = list()
        for idx in neighbors:
            p = self.points.get(tuple(idx))
            if p is not None:
                pt_list += p
        nr_pts = np.array(pt_list).reshape(-1, self.dim)
        nr_pts = nr_pts[np.any(nr_pts != pt[None, :], 1)]

        return nr_pts
