# !/usr/bin/env python
# encoding:utf-8

import itertools

import matplotlib.pyplot as plt
import numpy as np

from .plot import make_plot_grid


def tsp_brute_force(dmat):
    best_path = None
    best_path_length = np.inf
    for path in itertools.permutations(range(dmat.shape[0]), dmat.shape[0]):
        path_length = np.sum(dmat[(path[:-1], path[1:])])
        if path_length < best_path_length:
            best_path_length = path_length
            best_path = path
    return best_path, best_path_length


def tsp_greedy(dmat, best_path=None, best_path_length=np.inf):
    num_cities = dmat.shape[0]

    for i in range(num_cities):
        path = [i]
        while len(path) < num_cities:
            path.append([i for i in np.argsort(dmat[path[-1]]) if i not in path][0])
        path_length = np.sum(dmat[(path[:-1], path[1:])])
        if path_length < best_path_length:
            best_path_length = path_length
            best_path = path
    return best_path, best_path_length


def tsp_k_opt(dmat, k=3, best_path=None, best_path_length=np.inf, method='boltzmann', temperature=1.0, n_iter=100):
    num_cities = dmat.shape[0]

    if best_path is None:
        best_path = list(range(num_cities))
        best_path_length = np.sum(dmat[(best_path[:-1], best_path[1:])])

    if method == 'brute_force':
        cuts = [
            np.sort(np.concatenate((
                [0],
                np.random.choice(range(1, num_cities - 1), size=k, replace=False),
                [num_cities]
            )))
            for _ in range(n_iter)
        ]
        n_iter = len(cuts)
    elif method == 'stochastic':
        cuts = [
            np.sort(np.concatenate(([0], e, [num_cities])))
            for e in itertools.permutations(range(1, num_cities - 1), k)
        ]

    mu_dmat = np.mean(dmat)

    improvement = True
    for step in range(n_iter):
        if method in ['brute_force', 'stochastic']:
            idx = cuts[step]
        elif method == 'boltzmann':
            d_neighbor = np.diagonal(dmat[best_path, :][:, best_path], 1)
            p = np.exp(-(1 / d_neighbor) / (temperature * mu_dmat))[:-1]
            p /= np.sum(p)
            idx = np.sort(np.concatenate((
                [0],
                np.random.choice(range(1, num_cities - 1), size=k, replace=False, p=p),
                [num_cities]
            )))

        if method == 'brute_force' and not improvement:
            break
        improvement = False
        chunks = [best_path[idx[i]:idx[i + 1]] for i in range(k + 1)]
        dmat_sel = np.zeros((k * 2 + 2, k * 2 + 2))
        for i in range(k + 1):
            for j in range(k + 1):
                if i != j:
                    dmat_sel[i, j] = dmat[chunks[i][0], chunks[j][0]]  # start to start
                    dmat_sel[i, j + k + 1] = dmat[chunks[i][0], chunks[j][-1]]  # start to end
                    dmat_sel[i + k + 1, j] = dmat[chunks[i][-1], chunks[j][0]]  # end to start
                    dmat_sel[i + k + 1, j + k + 1] = dmat[chunks[i][-1], chunks[j][-1]]  # end to end
                else:
                    dmat_sel[i, j] = 1e12
                    dmat_sel[i + k + 1, j + k + 1] = 1e12
        path_sel, path_length_sel = tsp_brute_force(dmat_sel)
        chunks_aug = (chunks + [e[::-1] for e in chunks])
        new_path = sum([chunks_aug[i] for i in path_sel[::2]], [])

        new_path_length = np.sum(dmat[(new_path[:-1], new_path[1:])])
        if new_path_length < best_path_length:
            best_path = new_path[:]
            best_path_length = new_path_length
            improvement = True

    return best_path, best_path_length


def tsp_insertion(dmat, method='nearest'):
    num_cities = dmat.shape[0]
    unvisited = list(range(num_cities))

    path = [0]
    unvisited.remove(0)
    while unvisited:
        if method == 'arbitrary':
            k = np.random.choice(unvisited)
        elif method == 'nearest':
            dmat_sel = dmat[path, :][:, unvisited]
            idx = np.where(dmat_sel == np.min(dmat_sel))[1][0]
            k = unvisited[idx]
        elif method == 'farthest':
            dmat_sel = dmat[path, :][:, unvisited]
            idx = np.where(dmat_sel == np.max(dmat_sel))[1][0]
            k = unvisited[idx]
        elif method == 'cheapest':
            best_k = 0
            best_cost = np.inf
            if len(path) > 1:
                for k in unvisited:
                    cost = np.min(dmat[path[:-1], k] + dmat[k, path[1:]] - dmat[(path[:-1], path[1:])])
                    if cost < best_cost:
                        best_cost = cost
                        best_k = k
                k = best_k
            else:
                dmat_sel = dmat[path, :][:, unvisited]
                idx = np.where(dmat_sel == np.min(dmat_sel))[1][0]
                k = unvisited[idx]

        if len(path) > 1:
            i = np.argmin(dmat[path[:-1], k] + dmat[k, path[1:]] - dmat[(path[:-1], path[1:])])
        else:
            i = 1
        path = path[:i] + [k] + path[i:]
        unvisited.remove(k)

    path_length = np.sum(dmat[(path[:-1], path[1:])])

    return path, path_length


def tsp(dmat, initialization='arbitrary insertion'):
    if initialization == 'greedy':
        best_path, best_path_length = tsp_greedy(dmat)
    elif initialization == 'arbitrary insertion':
        best_path, best_path_length = tsp_insertion(dmat, method='arbitrary')
    elif initialization == 'nearest insertion':
        best_path, best_path_length = tsp_insertion(dmat, method='nearest')
    elif initialization == 'farthest insertion':
        best_path, best_path_length = tsp_insertion(dmat, method='farthest')
    elif initialization == 'cheapest insertion':
        best_path, best_path_length = tsp_insertion(dmat, method='cheapest')
    else:
        best_path, best_path_length = tsp_k_opt(dmat)

    best_path, best_path_length = tsp_k_opt(dmat, k=3, best_path=best_path, best_path_length=best_path_length,
                                            method='boltzmann', n_iter=10)
    return best_path, best_path_length


def tsp_brute_force_plot(data, dmat):
    best_path = None
    best_path_length = np.inf
    paths = list(itertools.permutations(range(dmat.shape[0]), dmat.shape[0]))
    for i, path in enumerate(paths):
        print('brute force: {:6d}/{:6d}'.format(i, len(paths)))
        path_length = np.sum(dmat[(path[:-1], path[1:])])

        path_loc = data[path, :]
        path_loc = np.vstack((path_loc, path_loc[:1, :]))
        fig, ax = make_plot_grid(1, 1)
        ax[0].scatter(data[:, 0], data[:, 1], marker='.', s=1)
        ax[0].plot(path_loc[:, 0], path_loc[:, 1], lw=0.5)
        ax[0].set_title('path length = {:6.5f} ({:6.5f})'.format(path_length, best_path_length))
        plt.savefig('./plot/brute_force_{:06d}.png'.format(i))
        plt.close()

        if path_length < best_path_length:
            best_path_length = path_length
            best_path = path
    return best_path, best_path_length


def tsp_greedy_plot(data, dmat, best_path=None, best_path_length=np.inf):
    num_cities = dmat.shape[0]
    path = [0]
    for i in range(num_cities - 1):
        print('greedy: {:6d}/{:6d}'.format(len(path), num_cities))
        path.append([j for j in np.argsort(dmat[path[-1]]) if j not in path][0])
        path_length = np.sum(dmat[(path[:-1], path[1:])])

        path_loc = data[path, :]
        path_loc = np.vstack((path_loc, path_loc[:1, :]))

        fig, ax = make_plot_grid(1, 1)
        ax[0].scatter(data[:, 0], data[:, 1], marker='.', s=1)
        ax[0].plot(path_loc[:, 0], path_loc[:, 1], lw=0.5)
        ax[0].set_title('path length = {:6.5f}'.format(path_length))
        plt.savefig('./plot/greedy_{:06d}.png'.format(i))
        plt.close()
    return path, path_length


def tsp_insertion_plot(data, dmat, method='nearest'):
    num_cities = dmat.shape[0]
    unvisited = list(range(num_cities))

    path = [0]
    unvisited.remove(0)
    for i in range(num_cities - 1):
        print('{:s} insertion: {:6d}/{:6d}'.format(method, len(path), num_cities))
        if method == 'arbitrary':
            k = np.random.choice(unvisited)
        elif method == 'nearest':
            dmat_sel = dmat[path, :][:, unvisited]
            idx = np.where(dmat_sel == np.min(dmat_sel))[1][0]
            k = unvisited[idx]
        elif method == 'farthest':
            dmat_sel = dmat[path, :][:, unvisited]
            idx = np.where(dmat_sel == np.max(dmat_sel))[1][0]
            k = unvisited[idx]
        elif method == 'cheapest':
            best_k = 0
            best_cost = np.inf
            if len(path) > 1:
                for k in unvisited:
                    cost = np.min(dmat[path[:-1], k] + dmat[k, path[1:]] - dmat[(path[:-1], path[1:])])
                    if cost < best_cost:
                        best_cost = cost
                        best_k = k
                k = best_k
            else:
                dmat_sel = dmat[path, :][:, unvisited]
                idx = np.where(dmat_sel == np.min(dmat_sel))[1][0]
                k = unvisited[idx]

        if len(path) > 1:
            j = np.argmin(dmat[path[:-1], k] + dmat[k, path[1:]] - dmat[(path[:-1], path[1:])])
        else:
            j = 1
        path = path[:j] + [k] + path[j:]
        unvisited.remove(k)

        path_length = np.sum(dmat[(path[:-1], path[1:])])
        path_loc = data[path, :]
        path_loc = np.vstack((path_loc, path_loc[:1, :]))

        fig, ax = make_plot_grid(1, 1)
        ax[0].scatter(data[:, 0], data[:, 1], marker='.', s=1)
        ax[0].plot(path_loc[:, 0], path_loc[:, 1], lw=0.5)
        ax[0].set_title('path length = {:6.5f}'.format(path_length))
        plt.savefig('./plot/{:s}_insertion_{:06d}.png'.format(method, i))
        plt.close()

    return path, path_length


def tsp_k_opt_plot(data, dmat, k=3, best_path=None, best_path_length=np.inf, method='boltzmann', temperature=1.0,
                   n_iter=100):
    num_cities = dmat.shape[0]

    if best_path is None:
        best_path = list(range(num_cities))
        best_path_length = np.sum(dmat[(best_path[:-1], best_path[1:])])

    if method == 'brute_force':
        cuts = [
            np.sort(np.concatenate((
                [0],
                np.random.choice(range(1, num_cities - 1), size=k, replace=False),
                [num_cities]
            )))
            for _ in range(n_iter)
        ]
        n_iter = len(cuts)
    elif method == 'stochastic':
        cuts = [
            np.sort(np.concatenate(([0], e, [num_cities])))
            for e in itertools.permutations(range(1, num_cities - 1), k)
        ]
    mu_dmat = np.mean(dmat)

    improvement = True
    for step in range(n_iter):
        if method in ['brute_force', 'stochastic']:
            idx = cuts[step]
        elif method == 'boltzmann':
            d_neighbor = np.diagonal(dmat[best_path, :][:, best_path], 1)
            p = np.exp(-(1 / d_neighbor) / (temperature * mu_dmat))[:-1]
            p /= np.sum(p)
            idx = np.sort(np.concatenate((
                [0],
                np.random.choice(range(1, num_cities - 1), size=k, replace=False, p=p),
                [num_cities]
            )))
            print(idx)

        print('{:s} {:d} opt: {:6d}/{:6d}'.format(method, k, step, n_iter))

        # if not stochastic and not improvement:
        #     break
        improvement = False
        chunks = [best_path[idx[i]:idx[i + 1]] for i in range(k + 1)]
        dmat_sel = np.zeros((k * 2 + 2, k * 2 + 2))
        for i in range(k + 1):
            for j in range(k + 1):
                if i != j:
                    dmat_sel[i, j] = dmat[chunks[i][0], chunks[j][0]]  # start to start
                    dmat_sel[i, j + k + 1] = dmat[chunks[i][0], chunks[j][-1]]  # start to end
                    dmat_sel[i + k + 1, j] = dmat[chunks[i][-1], chunks[j][0]]  # end to start
                    dmat_sel[i + k + 1, j + k + 1] = dmat[chunks[i][-1], chunks[j][-1]]  # end to end
                else:
                    dmat_sel[i, j] = 1e12
                    dmat_sel[i + k + 1, j + k + 1] = 1e12
        path_sel, path_length_sel = tsp_brute_force(dmat_sel)
        chunks_aug = (chunks + [e[::-1] for e in chunks])
        new_path = sum([chunks_aug[i] for i in path_sel[::2]], [])

        new_path_length = np.sum(dmat[(new_path[:-1], new_path[1:])])
        if new_path_length < best_path_length:
            print('k opt: {:6.5f}'.format(new_path_length))
            best_path = new_path[:]
            best_path_length = new_path_length

            path_loc = data[new_path, :]
            path_loc = np.vstack((path_loc, path_loc[:1, :]))

            fig, ax = make_plot_grid(1, 1)
            ax[0].scatter(data[:, 0], data[:, 1], marker='.', s=1)
            ax[0].plot(path_loc[:, 0], path_loc[:, 1], lw=0.5)
            ax[0].set_title('path length = {:6.5f} ({:6.5f})'.format(new_path_length, best_path_length))
            plt.savefig('./plot/{:s}_{:d}_opt_{:06d}.png'.format(method, k, step))
            plt.close()

    return best_path, best_path_length
