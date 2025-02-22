import numpy as np
import numba as nb
from math import sqrt
from tqdm import tqdm
import networkx as nx
from typing import Tuple, List
from collections import defaultdict


@nb.njit(fastmath=True)
def r2norm(x):
    """compute 2-norm of a vector

    Parameters
    ----------
    x : np.ndarray
        shape (2, ) array

    Returns
    -------
    float
        2-norm of x
    """
    return sqrt(x[0] * x[0] + x[1] * x[1])


def random_point_og(og: np.ndarray, rnd_gen: np.random.Generator = None) -> np.ndarray:
    """Get a random point in free space from the occupancyGrid.

    Parameters
    ----------
    og : np.ndarray
        occupancyGrid. 1 is obstacle, 0 is free space.
    rnd_gen: np.random.Generator
        A randomness generator (to allow a deterministic output).

    Returns
    -------
    np.ndarray
        randomly-sampled point
    """
    free = np.argwhere(og == 0)
    generator = np.random.randint if rnd_gen is None else rnd_gen.integers
    return free[generator(low=0, high=free.shape[0])]


############# RRT BASE CLASS ##########################################################


class RRT(object):
    def __init__(
        self,
        og: np.ndarray,
        n: int,
        costfn: callable = None,
        pbar: bool = True,
        seed: int = 0,
    ):
        # whether to display a progress bar
        self.pbar = pbar
        # array containing vertex points
        self.n = n
        # occupancyGrid free space
        self.free = np.argwhere(og == 0)
        self.og = og

        # define simple r2norm cost function if
        # no cost function is passed in

        if costfn is None:

            def costfn(
                vcosts: np.ndarray,
                points: np.ndarray,
                v: int,
                x: np.ndarray,
            ) -> float:
                return vcosts[v] + r2norm(points[v] - x)

        self.cost = costfn
        self.not_a_point = [np.inf, np.inf]
        self.not_a_dist = np.inf

        # Initialize the random generator with the given seed
        self.rand_gen = np.random.default_rng(seed)

    def route2gv(self, T: nx.DiGraph, gv) -> List[int]:
        """
        Returns a list of vertices on `T` that are the shortest path from the root of
        `T` to the node `gv`. This is a wrapper of `nx.shortest_path`.

        see also, `vertices_as_ndarray` method

        Parameters
        ----------
        T : nx.DiGraph
            The RRT DiGraph object, returned by rrt.make()
        gv : int
            vertex of the goal on `T`.

        Returns
        -------
        List[int]
            ordered list of vertices on `T`. Iterating through this list will give the
            vertices of the tree in order from the root to `gv`.
        """
        return nx.shortest_path(T, source=0, target=gv, weight="dist")

    def vertices_as_ndarray(self, T: nx.DiGraph, path: list) -> np.ndarray:
        """
        Helper method for obtaining an ordered list of vertices on `T` as an (M x 2)
        array.

        Parameters
        ----------
        T : nx.DiGraph
            The RRT DiGraph object, returned by rrt.make()
        path : list
            List of vertices on the RRT DiGraph object

        Returns
        -------
        np.ndarray
            (M x 2) array of points. M is the number of vertices in `T`.
        """
        lines = []
        for i in range(len(path) - 1):
            lines.append([T.nodes[path[i]]["pt"], T.nodes[path[i + 1]]["pt"]])
        return np.array(lines)

    @staticmethod
    def near(points: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Obtain all points in `points` that are near `x`, sorted in ascending order of
        distance.

        Parameters
        ----------
        points : np.ndarray
            (M x 2) array of points
        x : np.ndarray
            (2, ) array, point to find near

        Returns
        -------
        np.ndarray
            (M x 2) sorted array of points)
        """
        # vector from x to all points
        p2x = points - x
        # norm of that vector
        dist = np.linalg.norm(p2x, axis=1)
        # sorted norms
        sorted = np.argsort(dist)
        return sorted

    @staticmethod
    def within(points: np.ndarray, x: np.ndarray, r: float) -> np.ndarray:
        """Obtain un-ordered array of points that are within `r` of the point `x`.

        Parameters
        ----------
        points : np.ndarray
            (M x 2) array of points
        x : np.ndarray
            (2, ) array, point to find near
        r : float
            radius to search within

        Returns
        -------
        np.ndarray
            (? x 2) array of points within `r` of `x`
        """
        # vector from x to all points
        p2x = points - x
        # dot dist with self to get r2
        d2 = p2x[:, 0] * p2x[:, 0] + p2x[:, 1] * p2x[:, 1]
        # get indices of points within r2
        near_idx = np.argwhere(d2 < r * r)
        return np.atleast_1d(np.squeeze(near_idx))

    @staticmethod
    @nb.njit()
    def collisionfree(og, a, b) -> bool:
        """Check occupancyGrid for collisions between points `a` and `b`.

        Parameters
        ----------
        og : np.ndarray
            the occupancyGrid where 0 is free space. Anything other than 0 is treated as an obstacle.
        a : np.ndarray
            (2, ) array of integer coordinates point a
        b : np.ndarray
            (2, ) array of integer coordinates point b

        Returns
        -------
        bool
            whether or not there is a collision between the two points
        """
        x0 = a[0]
        y0 = a[1]
        x1 = b[0]
        y1 = b[1]
        dx = abs(x1 - x0)
        if x0 < x1:
            sx = 1
        else:
            sx = -1
        dy = -abs(y1 - y0)
        if y0 < y1:
            sy = 1
        else:
            sy = -1
        err = dx + dy
        while True:
            if og[x0, y0] != 0:
                return False
            elif x0 == x1 and y0 == y1:
                return True
            else:
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

    def sample_all_free(self):
        """
        Sample uniformly from free space in this object's occupancy grid.

        Returns
        -------
        np.ndarray
            (2, ) array of a point in free space of occupancy grid.
        """
        return self.free[self.rand_gen.choice(self.free.shape[0])]

    def plan(self, xstart: np.ndarray, xgoal: np.ndarray):
        """
        Compute a plan from `xstart` to `xgoal`. Raises an exception if called on the
        base class.

        Parameters
        ----------
        xstart : np.ndarray
            starting point
        xgoal : np.ndarray
            goal point

        Raises
        ------
        NotImplementedError
            if called on the base class
        """
        raise NotImplementedError("This method is not implemented in the base class.")

    def set_og(self, og_new: np.ndarray):
        """
        Set a new occupancy grid. This method also updates free space in the passed
        occupancy grid.

        Parameters
        ----------
        og_new : np.ndarray
            the new occupancy grid
        """
        self.og = og_new
        self.free = np.argwhere(og_new == 0)

    def set_n(self, n: int):
        """Update the number of attempted sample points, `n`.

        Parameters
        ----------
        n : int
            new number of sample points attempted when calling plan() from this object.
        """
        self.n = n

    def go2goal(self, vcosts, points, xgoal, j, children, parents):
        """
        Attempt to find a path from an existing point on the tree to the goal. This will
        update `vcosts`, `points`, `children`, and `parents` if a path is found, and
        return a vertex corresponding to the goal. If no path can be computed from the
        tree to the goal, the nearest vertex to the goal is selected.

        Parameters
        ----------
        vcosts : np.ndarray
            (M x 1) array of costs to vertices)
        points : np.ndarray
            (Mx2) array of points
        xgoal : np.ndarray
            (2, ) array of goal point
        j : int
            counter for the "current" vertex's index
        children : dict
            dict of children of vertices
        parents : dict
            dict of parents of children

        Returns
        -------
        tuple (int, dict, dict, np.ndarray, np.ndarray)
            tuple containing (vgoal, children, parents, points, vcosts)
        """
        # cost for all existing points
        costs = np.empty(vcosts.shape)
        for i in range(points.shape[0]):
            costs[i] = self.cost(vcosts, points, i, xgoal)

        found_goal = False
        for idx in np.argsort(costs):
            if self.collisionfree(self.og, points[idx], xgoal):
                vgoal = j
                points = np.concatenate((points, xgoal[np.newaxis, :]), axis=0)
                vcosts = np.concatenate((vcosts, [costs[idx]]), axis=0)
                points[vgoal] = xgoal
                vcosts[vgoal] = costs[idx]
                children[idx].append(vgoal)
                parents[vgoal] = idx
                found_goal = True
                break
        if not found_goal:
            # shrink points array
            dists = np.linalg.norm(points - xgoal)
            vgoal = np.argmin(dists)
        return vgoal, children, parents, points, vcosts

    def build_graph(self, vgoal, points, parents, vcosts):
        """
        Build the networkx DiGraph object from the points, parents dict, costs dict.

        Parameters
        ----------
        vgoal : int
            index of the goal vertex
        points : np.ndarray
            (Mx2) array of points
        parents : dict
            array of parents. Each vertex has a single parent (hence, the tree), except
            the root node which does not have a parent.
        vcosts : np.ndarray
            (M, 1) array of costs to vertices

        Returns
        -------
        nx.DiGraph
            Graph of the tree. Edges have attributes: `cost` (the cost of the edge's
            leaf -- remember, costs are additive!), `dist` (distance between the
            vertices).
        """
        assert points.max() < self.not_a_point[0] - 1
        # build graph
        T = nx.DiGraph()
        T.add_node(vgoal, pt=points[vgoal])
        for i, p in enumerate(points):
            T.add_node(i, pt=p)

        for child, parent in parents.items():
            if parent is not None:
                p1, p2 = points[parent], points[child]
                dist = r2norm(p2 - p1)
                T.add_edge(parent, child, dist=dist, cost=vcosts[child])
        return T


############# RRT STANDARD ####################################################


class RRTStandard(RRT):
    def __init__(
        self,
        og: np.ndarray,
        n: int,
        costfn: callable = None,
        pbar=True,
        seed: int = 0,
    ):
        super().__init__(og, n, costfn=costfn, pbar=pbar, seed=seed)

    def plan(self, xstart: np.ndarray, xgoal: np.ndarray) -> Tuple[nx.DiGraph, int]:
        """
        Compute a plan from `xstart` to `xgoal`. Using the Standard RRT algorithm.

        Compute a plan from `xstart` to `xgoal`. The plan is a tree, with the root at
        `xstart` and a leaf at `xgoal`. If xgoal could not be connected to the tree, the
        leaf nearest to xgoal is considered the "goal" leaf.

        Parameters
        ----------
        xstart : np.ndarray
            (2, ) start point
        xgoal : np.ndarray
            (2, ) goal point

        Returns
        -------
        Tuple[nx.DiGraph, int]
            DiGraph of the tree, and the vertex of the goal leaf (if goal could be
            reached) or the closest tree node.
        """
        sampled = set()
        points = np.full((self.n, 2), dtype=int, fill_value=self.not_a_point)
        vcosts = np.full((self.n,), fill_value=self.not_a_dist)
        parents, children = {}, defaultdict(list)
        points[0] = xstart
        vcosts[0] = 0
        parents[0] = None
        i, j = 0, 1
        if self.pbar:
            pbar = tqdm(total=self.n)

        while i < self.n:
            if self.pbar:
                pbar.update(1)
            xnew = self.sample_all_free()
            vnearest = self.near(points, xnew)[0]
            xnearest = points[vnearest]
            nocoll = self.collisionfree(self.og, xnearest, xnew)
            if nocoll and tuple(xnew) not in sampled and j != self.n:
                sampled.add(tuple(xnew))
                # check least cost path to xnew
                vbest = vnearest
                # store new point
                vnew = j
                points[vnew] = xnew
                vcosts[vnew] = self.cost(vcosts, points, vbest, xnew)
                # store new edge
                children[vbest].append(vnew)
                parents[vnew] = vbest
                j += 1
            i += 1

        # go to goal if possible
        vgoal, children, parents, points, vcosts = self.go2goal(
            vcosts, points, xgoal, j, children, parents
        )

        # build graph
        T = self.build_graph(vgoal, points, parents, vcosts)

        return T, vgoal


########### RRT STAR ##########################################################


class RRTStar(RRT):
    def __init__(
        self,
        og: np.ndarray,
        n: int,
        r_rewire: float,
        costfn: callable = None,
        pbar=True,
        seed: int = 0,
    ):
        super().__init__(og, n, costfn=costfn, pbar=pbar, seed=seed)
        self.r_rewire = r_rewire

    def plan(self, xstart: np.ndarray, xgoal: np.ndarray):
        """
        Compute a plan from `xstart` to `xgoal`. Using the RRT* algorithm.

        The plan is a tree, with the root at
        `xstart` and a leaf at `xgoal`. If xgoal could not be connected to the tree, the
        leaf nearest to xgoal is considered the "goal" leaf.

        Parameters
        ----------
        xstart : np.ndarray
            (2, ) start point
        xgoal : np.ndarray
            (2, ) goal point

        Returns
        -------
        Tuple[nx.DiGraph, int]
            DiGraph of the tree, and the vertex of the goal leaf (if goal could be
            reached) or the closest tree node.
        """
        sampled = set()
        points = np.full((self.n, 2), dtype=int, fill_value=self.not_a_point)
        vcosts = np.full((self.n,), fill_value=self.not_a_dist)
        children, parents = defaultdict(list), {}
        points[0] = xstart
        vcosts[0] = 0
        parents[0] = None
        i, j = 0, 1
        if self.pbar:
            pbar = tqdm(total=self.n)

        while i < self.n:
            if self.pbar:
                pbar.update(1)

            xnew = self.sample_all_free()
            vnearest = self.near(points, xnew)[0]
            xnearest = points[vnearest]

            nocoll = self.collisionfree(self.og, xnearest, xnew)
            if nocoll and tuple(xnew) not in sampled and j != self.n:
                sampled.add(tuple(xnew))

                # check least cost path to xnew
                vbest = vnearest
                cbest = self.cost(vcosts, points, vbest, xnew)
                vnear = self.within(points, xnew, self.r_rewire)

                for vn in vnear:
                    xn = points[vn]
                    cn = self.cost(vcosts, points, vn, xnew)
                    if cn < cbest:
                        if self.collisionfree(self.og, xn, xnew):
                            vbest = vn
                            cbest = cn

                # store new point
                vnew = j
                points[vnew] = xnew
                vcosts[vnew] = cbest
                # store new edge
                parents[vnew] = vbest
                children[vbest].append(vnew)

                # tree rewire
                for vn in vnear:
                    xn = points[vn]
                    cn = vcosts[vn]
                    cmaybe = self.cost(vcosts, points, vn, xnew)
                    if cmaybe < cn:
                        if self.collisionfree(self.og, xn, xnew):
                            parent = parents[vn]
                            if parent is not None:
                                # reassign parent
                                try:
                                    children[parent].remove(vn)
                                    parents[vn] = vnew
                                    vcosts[vn] = cmaybe
                                except ValueError:
                                    pass
                j += 1
            i += 1
        # go to goal if possible
        vgoal, children, parents, points, vcosts = self.go2goal(
            vcosts, points, xgoal, j, children, parents
        )
        # build graph
        T = self.build_graph(vgoal, points, parents, vcosts)

        return T, vgoal


########### RRT STAR INFORMED #################################################


class RRTStarInformed(RRT):
    def __init__(
        self,
        og: np.ndarray,
        n: int,
        r_rewire: float,
        r_goal: float,
        costfn: callable = None,
        pbar: bool = True,
        seed: int = 0,
    ):
        super().__init__(og, n, costfn=costfn, pbar=pbar, seed=seed)
        self.r_rewire = r_rewire
        self.r_goal = r_goal
        # store the ellipses for plotting later
        self.ellipses = {}

    def unitball(self):
        """draw a point from a uniform distribution bounded by the ball:
        U(x1, x2) ~ 1 > (x1)^2 + (x2)^2"""
        r = self.rand_gen.uniform(0, 1)
        theta = 2 * np.pi * self.rand_gen.uniform(0, 1)
        x = np.sqrt(r) * np.cos(theta)
        y = np.sqrt(r) * np.sin(theta)
        unif = np.array([x, y])
        return unif

    def sample_ellipse(self, xstart, xgoal, c, clamp=True):
        xcent = (xstart + xgoal) / 2
        CL = self.get_ellipse_xform(xstart, xgoal, c)
        xball = self.unitball()
        x, y = tuple(np.dot(CL, xball) + xcent)

        if clamp:
            # clamp to finite og
            x = int(max(0, min(self.og.shape[0] - 1, x)))
            y = int(max(0, min(self.og.shape[1] - 1, y)))
        return np.array((x, y))

    def rotation_to_world_frame(self, xstart, xgoal):
        """calculate the rotation matrix from the world-frame to the frame given
        by the hyperellipsoid with focal points at xf1=xstart and xf2=xgoal. a unit
        ball multiplied by this matrix will produce an oriented ellipsoid with those
        focal points."""
        a1 = np.atleast_2d((xgoal - xstart) / np.linalg.norm(xgoal - xstart))
        M = np.outer(a1, np.atleast_2d([1, 0]))
        try:
            U, _, V = np.linalg.svd(M)
        except np.linalg.LinAlgError:
            # handle SVD not converging
            U, _, V = np.linalg.svd(M, full_matrices=False)
        return U @ np.diag([np.linalg.det(U), np.linalg.det(V)]) @ V.T

    def get_ellipse_xform(self, xstart, xgoal, cmax):
        """transform vector in unit plane to ellipse plane"""
        # rotation matrix
        C = self.rotation_to_world_frame(xstart, xgoal)
        # scale by major axis r1, minor axis r2.
        r1 = cmax / 2
        d2 = np.dot((xstart - xgoal).T, (xstart - xgoal))
        r2 = np.sqrt(abs(cmax * cmax - d2)) / 2
        L = np.diag([r1, r2])
        # dot rotation with scale to get final matrix
        return np.dot(C, L)

    @staticmethod
    def least_cost(vcosts, vsoln):
        if len(vsoln) == 1:
            return vsoln[0], vcosts[vsoln[0]]
        else:
            idx_cmin = np.argmin(vcosts[vsoln])
            return vsoln[idx_cmin], vcosts[vsoln[idx_cmin]]

    @staticmethod
    def rad2deg(a):
        return a * 180 / np.pi

    def get_ellipse_for_plt(
        self, xstart, xgoal, cmax
    ) -> Tuple[np.ndarray, float, float, float]:
        xcent = (xgoal + xstart) / 2
        # get the rotation
        CL = self.get_ellipse_xform(xstart, xgoal, cmax)
        # apply the rotation to find major axis, minor axis, angle
        a = np.dot(CL, np.array([1, 0]))
        b = np.dot(CL, np.array([0, 1]))
        majax = 2 * np.linalg.norm(a)
        minax = 2 * np.linalg.norm(b)
        ang = self.rad2deg(np.arctan2((a)[1], (a)[0]))
        return xcent, majax, minax, ang

    def plan(self, xstart: np.ndarray, xgoal: np.ndarray):
        """
        Compute a plan from `xstart` to `xgoal`. Using the RRT* Informed algorithm.

        Keep in mind: if the cost function does not correspond to the Euclidean cost,
        this algorithm may undersample the solution space -- therefore, for non R^2
        distance cost functions, it is recommended to use the RRT* algorithm instead.

        The plan is a tree, with the root at `xstart` and a leaf at `xgoal`. If xgoal
        could not be connected to the tree, the leaf nearest to xgoal is considered the
        "goal" leaf.

        Parameters
        ----------
        xstart : np.ndarray
            (2, ) start point
        xgoal : np.ndarray
            (2, ) goal point

        Returns
        -------
        Tuple[nx.DiGraph, int]
            DiGraph of the tree, and the vertex of the goal leaf (if goal could be
            reached) or the closest tree node.
        """
        vsoln = []
        sampled = set()
        points = np.full((self.n, 2), dtype=int, fill_value=self.not_a_point)
        vcosts = np.full((self.n,), fill_value=self.not_a_dist)
        parents, children = {}, defaultdict(list)
        points[0] = xstart
        vcosts[0] = 0
        parents[0] = None
        i, j = 0, 1
        if self.pbar:
            pbar = tqdm(total=self.n)

        while i < self.n:
            if self.pbar:
                pbar.update(1)

            # either sample from free or sample from ellipse depending on vsoln
            if len(vsoln) == 0:
                xnew = self.sample_all_free()
            else:
                vbest, cbest = self.least_cost(vcosts, list(vsoln))
                cbest += r2norm(xgoal - points[vbest])
                xnew = self.sample_ellipse(xstart, xgoal, cbest)
                self.ellipses[j] = self.get_ellipse_for_plt(xstart, xgoal, cbest)

            vnearest = self.near(points, xnew)[0]
            xnearest = points[vnearest]

            nocoll = self.collisionfree(self.og, xnearest, xnew)
            if nocoll and tuple(xnew) not in sampled and j != self.n:
                sampled.add(tuple(xnew))

                # check least cost path to xnew
                vbest = vnearest
                cbest = self.cost(vcosts, points, vbest, xnew)
                vnear = self.within(points, xnew, self.r_rewire)

                for vn in vnear:
                    xn = points[vn]
                    cn = self.cost(vcosts, points, vn, xnew)
                    if cn < cbest:
                        if self.collisionfree(self.og, xn, xnew):
                            vbest = vn
                            cbest = cn

                # store new point
                vnew = j
                points[vnew] = xnew
                vcosts[vnew] = cbest
                # store new edge
                parents[vnew] = vbest
                children[vbest].append(vnew)

                # tree rewire
                for vn in vnear:
                    xn = points[vn]
                    cn = vcosts[vn]
                    cmaybe = self.cost(vcosts, points, vn, xnew)
                    if cmaybe < cn:
                        if self.collisionfree(self.og, xn, xnew):
                            parent = parents[vn]
                            if parent is not None:
                                children[parent].remove(vn)
                                parents[vn] = vnew
                                vcosts[vn] = cmaybe
                # check goal
                if r2norm(xnew - xgoal) < self.r_goal:
                    vsoln.append(vnew)

                j += 1
            i += 1

        # go to goal if possible
        vgoal, children, parents, points, vcosts = self.go2goal(
            vcosts, points, xgoal, j, children, parents
        )

        # build graph
        T = self.build_graph(vgoal, points, parents, vcosts)

        return T, vgoal


if __name__ == "__main__":
    from oggen import perlin_occupancygrid
    import matplotlib.pyplot as plt
    from plots import plot_og, plot_rrt_lines, plot_path, plot_start_goal

    # create occupancy Grid
    w, h = 800, 450
    og = perlin_occupancygrid(w, h)

    # create planner
    rrtplanner = RRTStarInformed(og, n=2000, r_rewire=64, r_goal=12)
    xstart = random_point_og(og)
    xgoal = random_point_og(og)

    T, gv = rrtplanner.make(xstart, xgoal)
    pathlines = rrtplanner.route2gv(T, gv)
    pathlines = rrtplanner.path_points(T, pathlines)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    plot_og(ax, og)
    plot_start_goal(ax, xstart, xgoal)
    plot_rrt_lines(ax, T, color_costs=True)
    plot_path(ax, pathlines)
    plt.show()
