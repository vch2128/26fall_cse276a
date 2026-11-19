#!/usr/bin/env python3
"""HW4 path planners: minimum distance (visibility graph) and maximum safety (Voronoi diagram).

Both planners use the same known map (config/map_hw4.yaml) and the same search (Dijkstra);
they differ in the world representation:

    shortest   VISIBILITY GRAPH. Every obstacle is grown by the robot radius + a margin
               (configuration space, mitre corners), so the robot can be treated as a point.
               Nodes = start, goal and the corners of the grown obstacles; an edge joins two
               nodes that can see each other. The shortest path in the plane around polygons
               only bends at their corners, so Dijkstra on this graph gives the shortest path.

    safest     VORONOI DIAGRAM of points sampled densely on the walls and the obstacle edges.
               Keeping only the edges between samples of DIFFERENT objects leaves the
               generalized Voronoi diagram: every point on it is as far as possible from the
               two nearest objects. Edges closer than the inflation radius are removed, the
               start and the goal are connected to the nearest visible vertices, and Dijkstra
               finds the shortest route along the diagram. The path is then simplified (nearly
               collinear points removed) without losing clearance, so the robot turns less.

    shortest   RRT* (sampling-based alternative, plan_rrtstar): random tree in the same
    (RRT*)     configuration space, with choose-parent + rewiring so the path length converges
               toward the visibility-graph optimum; informed sampling once a path exists;
               line-of-sight shortcutting before driving. Writes path_rrtstar.png (tree +
               convergence vs the visibility-graph length).

hw4_control.py calls plan_visibility / plan_voronoi at start-up. Run this file on its own to
print the comparison table and save path_compare.png (both representations and paths, for the
report) next to the map file unless --out is given.

No ROS imports: runs on a laptop.
    pip install numpy scipy matplotlib pyyaml
    python3 landmark_nav/landmark_nav/planner.py                    # uses ../config/map_hw4.yaml
    python3 landmark_nav/landmark_nav/planner.py --map my_map.yaml --out figures/
"""
import argparse
import heapq
import math
import os

import numpy as np
import yaml

TOL = 1e-6           # m: geometric tolerance (touching an obstacle boundary is allowed)
CHECK_STEP = 0.01    # m: spacing of clearance checks along a segment


# ===================================================================== map / geometry
def rect_polygon(cx, cy, w, h, yaw):
    """Corners of a rotated rectangle, counter-clockwise."""
    c, s = math.cos(yaw), math.sin(yaw)
    local = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return np.array([(cx + c * x - s * y, cy + s * x + c * y) for x, y in local])


def inflate(poly, r):
    """Grow a convex counter-clockwise polygon by r (every edge moved out by r, mitre corners)."""
    n = len(poly)
    lines = []
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        d = (q - p) / np.linalg.norm(q - p)
        normal = np.array([d[1], -d[0]])                  # outward for a CCW polygon
        lines.append((p + r * normal, d))
    out = []
    for i in range(n):                                    # corner i joins edge i-1 and edge i
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        t, _ = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
        out.append(p1 + t * d1)
    return np.array(out)


def point_segment_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def inside_convex(p, poly):
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0:   # right of a CCW edge
            return False
    return True


def segment_enters(a, b, poly, tol=TOL):
    """True if segment a-b passes through the INTERIOR of convex CCW polygon (Cyrus-Beck clipping).
    Running along an edge or touching a corner is allowed."""
    t0, t1 = 0.0, 1.0
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        d = q - p
        normal = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        f0 = np.dot(normal, a - p) + tol                  # < 0: a is strictly inside this edge
        df = np.dot(normal, b - a)
        if abs(df) < 1e-12:
            if f0 >= 0:
                return False
            continue
        t = -f0 / df
        if df < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 >= t1:
            return False
    return t1 - t0 > 1e-9


class World:
    def __init__(self, cfg):
        self.W = float(cfg['world']['width'])
        self.H = float(cfg['world']['height'])
        rob = cfg['robot']
        self.radius = float(rob['radius'])
        self.R = self.radius + float(rob.get('margin', 0.0))          # inflation radius
        s, g = cfg['start'], cfg['goal']
        self.start = np.array([float(s['x']), float(s['y'])])
        self.start_th = math.radians(float(s.get('theta_deg', 0.0)))
        self.goal = np.array([float(g['x']), float(g['y'])])
        self.obstacles = [rect_polygon(float(o['cx']), float(o['cy']), float(o['w']), float(o['h']),
                                       math.radians(float(o.get('yaw_deg', 0.0))))
                          for o in cfg.get('obstacles', [])]
        self.inflated = [inflate(p, self.R) for p in self.obstacles]
        self.landmarks = {str(k): (float(v['x']), float(v['y'])) for k, v in (cfg.get('landmarks') or {}).items()}
        pl = cfg.get('planner') or {}
        self.sample = float(pl.get('voronoi_sample', 0.03))
        self.simplify_tol = float(pl.get('simplify_tol', 0.04))
        for name, p in (('start', self.start), ('goal', self.goal)):
            if self.clearance(p) < self.R:
                raise ValueError(f'{name} {p} is closer than {self.R:.2f} m to a wall or an obstacle')

    def walls(self):
        W, H = self.W, self.H
        c = [np.array(p, float) for p in ((0, 0), (W, 0), (W, H), (0, H))]
        return [(c[i], c[(i + 1) % 4]) for i in range(4)]

    def clearance(self, p):
        """Distance from point p to the nearest wall or obstacle (0 inside an obstacle)."""
        p = np.asarray(p, float)
        d = min(p[0], self.W - p[0], p[1], self.H - p[1])
        for poly in self.obstacles:
            if inside_convex(p, poly):
                return 0.0
            n = len(poly)
            d = min(d, min(point_segment_dist(p, poly[i], poly[(i + 1) % n]) for i in range(n)))
        return float(d)

    def in_free_box(self, p):
        return self.R - TOL <= p[0] <= self.W - self.R + TOL and self.R - TOL <= p[1] <= self.H - self.R + TOL

    def segment_free(self, a, b):
        """Visibility test in the configuration space (grown obstacles, shrunk walls)."""
        return (self.in_free_box(a) and self.in_free_box(b)
                and not any(segment_enters(a, b, poly) for poly in self.inflated))

    def obstacle_clearance(self, p):
        """Distance from point p to the nearest obstacle only (walls ignored)."""
        p = np.asarray(p, float)
        d = math.inf
        for poly in self.obstacles:
            if inside_convex(p, poly):
                return 0.0
            n = len(poly)
            d = min(d, min(point_segment_dist(p, poly[i], poly[(i + 1) % n]) for i in range(n)))
        return float(d)

    def segment_min_clearance(self, a, b, walls=True):
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / CHECK_STEP)) + 1)
        f = self.clearance if walls else self.obstacle_clearance
        return min(f(a + t * (b - a)) for t in np.linspace(0.0, 1.0, n))

    def segment_clear(self, a, b):
        """Every point of a-b at least R from walls and obstacles (exact clearance, round corners)."""
        return self.segment_min_clearance(a, b) >= self.R - TOL


# ===================================================================== search
def dijkstra(adj, src, dst):
    """adj: {node: [(neighbor, cost)]}. Returns the node list src..dst, or None."""
    dist, prev = {src: 0.0}, {}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    if dst not in dist:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


def add_edge(adj, u, v, w):
    adj.setdefault(u, []).append((v, w))
    adj.setdefault(v, []).append((u, w))


# ===================================================================== planners
def plan_visibility(world):
    """Shortest path: visibility graph on the grown obstacles + Dijkstra."""
    nodes = [world.start, world.goal]
    for poly in world.inflated:
        nodes += [c for c in poly if world.in_free_box(c)]
    adj, edges = {}, []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if world.segment_free(nodes[i], nodes[j]):
                w = float(np.linalg.norm(nodes[i] - nodes[j]))
                add_edge(adj, i, j, w)
                edges.append((nodes[i], nodes[j]))
    idx = dijkstra(adj, 0, 1)
    if idx is None:
        raise RuntimeError('visibility graph: no path from start to goal')
    return {'path': [nodes[k] for k in idx], 'nodes': nodes, 'edges': edges}


def boundary_samples(world):
    """Points every `sample` m on each wall and each obstacle edge, with the id of their object."""
    pts, owner = [], []
    segments = [(f'wall{i}', a, b) for i, (a, b) in enumerate(world.walls())]
    for k, poly in enumerate(world.obstacles):
        segments += [(f'obs{k}', poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    for name, a, b in segments:
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / world.sample)))
        for t in np.arange(n) / n:                        # end point = start of the next edge
            pts.append(a + t * (b - a))
            owner.append(name)
    return np.array(pts), owner


def plan_voronoi(world):
    """Safest path: generalized Voronoi diagram of walls + obstacles, Dijkstra along it."""
    from scipy.spatial import Voronoi

    pts, owner = boundary_samples(world)
    vor = Voronoi(pts)
    V = vor.vertices
    ok = {}                                               # vertex index -> usable?

    def usable(i):
        if i not in ok:
            v = V[i]
            ok[i] = (0.0 <= v[0] <= world.W and 0.0 <= v[1] <= world.H
                     and world.clearance(v) >= world.R - TOL)
        return ok[i]

    adj, edges = {}, []
    for (pi, pj), (vi, vj) in zip(vor.ridge_points, vor.ridge_vertices):
        if vi < 0 or vj < 0 or owner[pi] == owner[pj]:    # infinite ridge / inside one object
            continue
        if usable(vi) and usable(vj) and world.segment_clear(V[vi], V[vj]):
            add_edge(adj, vi, vj, float(np.linalg.norm(V[vi] - V[vj])))
            edges.append((V[vi], V[vj]))
    if not adj:
        raise RuntimeError('Voronoi: no edge has enough clearance (obstacle too big / radius too large?)')

    # connect start / goal to the nearest diagram vertices they can reach in a straight line
    graph_nodes = np.array(list(adj.keys()))
    S, G = 'start', 'goal'
    for name, p in ((S, world.start), (G, world.goal)):
        order = graph_nodes[np.argsort(np.linalg.norm(V[graph_nodes] - p, axis=1))]
        linked = 0
        for k in order[:40]:
            if world.segment_clear(p, V[k]):
                add_edge(adj, name, k, float(np.linalg.norm(V[k] - p)))
                edges.append((p, V[k]))
                linked += 1
                if linked == 3:
                    break
        if not linked:
            raise RuntimeError(f'Voronoi: {name} cannot reach the diagram')

    idx = dijkstra(adj, S, G)
    if idx is None:
        raise RuntimeError('Voronoi: no path from start to goal')
    raw = [world.start] + [V[k] for k in idx[1:-1]] + [world.goal]
    return {'path': simplify(world, raw), 'raw': raw, 'edges': edges}


def simplify(world, path):
    """Douglas-Peucker: drop points within simplify_tol of the chord, but only if the chord keeps
    (almost) the clearance of the part of the path it replaces."""
    path = [np.asarray(p, float) for p in path]

    def rec(i, j):
        if j <= i + 1:
            return [path[i]]
        a, b = path[i], path[j]
        devs = [point_segment_dist(path[k], a, b) for k in range(i + 1, j)]
        k = i + 1 + int(np.argmax(devs))
        keep_clear = min(world.clearance(path[m]) for m in range(i, j + 1)) - world.simplify_tol
        if max(devs) <= world.simplify_tol and world.segment_min_clearance(a, b) >= max(keep_clear, world.R - TOL):
            return [a]
        return rec(i, k) + rec(k, j)

    return rec(0, len(path) - 1) + [path[-1]]


# ===================================================================== sampling-based planner
def free_area(world, n=20000, seed=1):
    """Monte-Carlo area (m^2) of the configuration-space free region (for the RRT* radius)."""
    rng = np.random.default_rng(seed)
    lo, hi = world.R, np.array([world.W - world.R, world.H - world.R])
    pts = lo + rng.random((n, 2)) * (hi - lo)
    free = sum(not any(inside_convex(p, poly) for poly in world.inflated) for p in pts)
    return float(np.prod(hi - lo)) * free / n


def plan_rrtstar(world, iters=3000, step=0.25, goal_bias=0.05, informed=True, seed=0, shortcut=True):
    """Shortest path, sampling-based: RRT* (Karaman & Frazzoli 2011), optionally informed
    (Gammell et al. 2014), on the same configuration space as plan_visibility (grown obstacles,
    shrunk walls, the same world.segment_free test), so the two lengths can be compared directly.

    Each iteration: sample a free point (the goal with probability goal_bias; once a path of cost
    c_best exists and informed=True, only inside the ellipse with foci start / goal and major axis
    c_best, since no point outside it can shorten the path), steer from the nearest node at most
    `step` m toward it, then
        choose parent   connect x_new to the node within radius r giving the lowest cost-to-come;
        rewire          re-parent every node within r whose cost drops by going through x_new
                        (the cost change is pushed down its subtree).
    r = min(step, gamma * sqrt(log n / n)), gamma = 2 sqrt(1.5) sqrt(area_free / pi) (the 2-D
    asymptotic-optimality bound). Nodes within `step` of the goal that see it are goal candidates;
    the best path is the cheapest candidate + its straight segment to the goal.

    Returns {'path', 'raw', 'nodes', 'edges', 'history', 'iters', 'first_iter'}:
        raw      the RRT* tree path; path = raw shortcut (line-of-sight) if shortcut=True, which
                 never makes it longer and removes the zig-zag before the robot drives it;
        history  best raw path cost after every iteration (inf before the first solution);
        first_iter  iteration of the first solution.
    """
    rng = np.random.default_rng(seed)
    start, goal = np.asarray(world.start, float), np.asarray(world.goal, float)
    lo, hi = np.array([world.R, world.R]), np.array([world.W - world.R, world.H - world.R])
    gamma = 2.0 * math.sqrt(1.5) * math.sqrt(free_area(world) / math.pi)

    nodes = np.zeros((iters + 1, 2))
    nodes[0] = start
    parent = [-1]
    cost = [0.0]
    children = [[]]
    n = 1
    goal_cand = []                                       # node indices that can see the goal
    best_cost, best_node, first_iter = math.inf, None, None
    history = np.full(iters, math.inf)

    # informed-sampling ellipse frame
    c_min = float(np.linalg.norm(goal - start))
    centre = 0.5 * (start + goal)
    a1 = (goal - start) / c_min
    C = np.array([[a1[0], -a1[1]], [a1[1], a1[0]]])      # rotation: ellipse x axis -> start->goal

    def in_cfree(p):
        return (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]
                and not any(inside_convex(p, poly) for poly in world.inflated))

    def sample():
        if rng.random() < goal_bias:
            return goal.copy()
        while True:
            if informed and best_cost < math.inf:
                r1 = best_cost / 2.0
                r2 = math.sqrt(max(best_cost ** 2 - c_min ** 2, 0.0)) / 2.0
                u = rng.random() * 2 * math.pi
                rad = math.sqrt(rng.random())             # uniform in the unit disk
                p = centre + C @ np.array([r1 * rad * math.cos(u), r2 * rad * math.sin(u)])
            else:
                p = lo + rng.random(2) * (hi - lo)
            if in_cfree(p):
                return p

    def push_cost(k, delta):
        stack = [k]
        while stack:
            j = stack.pop()
            cost[j] += delta
            stack.extend(children[j])

    for it in range(iters):
        x_rand = sample()
        d = np.linalg.norm(nodes[:n] - x_rand, axis=1)
        k_near = int(np.argmin(d))
        if d[k_near] < 1e-9:
            history[it] = best_cost
            continue
        x_near = nodes[k_near]
        x_new = x_near + (x_rand - x_near) * min(1.0, step / d[k_near])
        if not world.segment_free(x_near, x_new):
            history[it] = best_cost
            continue

        r = min(step, gamma * math.sqrt(math.log(n + 1) / (n + 1)))
        dn = np.linalg.norm(nodes[:n] - x_new, axis=1)
        near = [int(j) for j in np.flatnonzero(dn <= r)]
        if k_near not in near:
            near.append(k_near)

        # ---- choose the best parent
        free_to = {}
        best_p, best_c = k_near, cost[k_near] + float(dn[k_near])
        free_to[k_near] = True
        for j in near:
            c = cost[j] + float(dn[j])
            if c < best_c:
                ok = world.segment_free(nodes[j], x_new)
                free_to[j] = ok
                if ok:
                    best_p, best_c = j, c
        new = n
        nodes[new] = x_new
        parent.append(best_p)
        cost.append(best_c)
        children.append([])
        children[best_p].append(new)
        n += 1

        # ---- rewire the neighbourhood through x_new
        for j in near:
            if j == best_p:
                continue
            c = best_c + float(dn[j])
            if c < cost[j] - 1e-12:
                ok = free_to.get(j)
                if ok is None:
                    ok = world.segment_free(x_new, nodes[j])
                if ok:
                    children[parent[j]].remove(j)
                    parent[j] = new
                    children[new].append(j)
                    push_cost(j, c - cost[j])

        # ---- goal connection
        dg = float(np.linalg.norm(goal - x_new))
        if dg <= step and world.segment_free(x_new, goal):
            goal_cand.append(new)
        if goal_cand:
            cands = [(cost[k] + float(np.linalg.norm(goal - nodes[k])), k) for k in goal_cand]
            c, k = min(cands)
            if c < best_cost:
                if first_iter is None:
                    first_iter = it + 1
                best_cost, best_node = c, k
        history[it] = best_cost

    if best_node is None:
        raise RuntimeError(f'RRT*: no path after {iters} iterations (more iters / larger step?)')
    raw = [goal]
    k = best_node
    while k != -1:
        raw.append(nodes[k].copy())
        k = parent[k]
    raw = raw[::-1]
    edges = [(nodes[parent[k]], nodes[k]) for k in range(1, n)]
    path = shortcut_path(world, raw) if shortcut else raw
    return {'path': path, 'raw': raw, 'nodes': nodes[:n].copy(), 'edges': edges,
            'history': history, 'iters': iters, 'first_iter': first_iter}


def shortcut_path(world, path):
    """Greedy line-of-sight shortcutting: from each kept point jump to the farthest later point it
    can see (world.segment_free). Never longer than the input."""
    path = [np.asarray(p, float) for p in path]
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not world.segment_free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def plot_rrt(world, rrt, vis_len, filename):
    """RRT* tree + path, and the best cost vs iteration against the visibility-graph optimum."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1.0, 1.1]})
    ax = axes[0]
    draw_world(ax, world, f'Min distance: RRT* ({rrt["iters"]} iterations, {len(rrt["nodes"])} nodes)')
    ax.add_collection(LineCollection([[a, b] for a, b in rrt['edges']], colors='#c9b3e0', lw=0.5, zorder=1))
    r = np.array(rrt['raw'])
    ax.plot(r[:, 0], r[:, 1], ':', color='#6a3d9a', lw=1.3, zorder=2,
            label=f'RRT* tree path {sum(np.linalg.norm(np.diff(r, axis=0), axis=1)):.3f} m')
    p = np.array(rrt['path'])
    ax.plot(p[:, 0], p[:, 1], '-o', color='#6a3d9a', lw=2.5, ms=5, zorder=3,
            label=f'after shortcutting {sum(np.linalg.norm(np.diff(p, axis=0), axis=1)):.3f} m')
    ax.legend(loc='lower left', fontsize=8)

    ax = axes[1]
    h = rrt['history']
    it = np.arange(1, len(h) + 1)
    ax.plot(it, h, color='#6a3d9a', lw=2, label='RRT* best path cost')
    ax.axhline(vis_len, color='#d0521b', ls='--', lw=1.5, label=f'visibility graph (optimal) {vis_len:.3f} m')
    if rrt['first_iter']:
        ax.axvline(rrt['first_iter'], color='gray', ls=':', lw=1, label=f'first solution (iter {rrt["first_iter"]})')
    fin = h[np.isfinite(h)]
    if len(fin):
        ax.set_ylim(vis_len * 0.98, max(fin[0], vis_len) * 1.05)
    ax.set_xlabel('iteration')
    ax.set_ylabel('path length (m)')
    ax.set_title('Convergence toward the shortest path', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# ===================================================================== outputs
def metrics(world, path):
    length = sum(float(np.linalg.norm(b - a)) for a, b in zip(path, path[1:]))
    clear = min(world.segment_min_clearance(a, b) for a, b in zip(path, path[1:]))
    obs = min(world.segment_min_clearance(a, b, walls=False) for a, b in zip(path, path[1:]))
    heading, turning = world.start_th, 0.0
    for a, b in zip(path, path[1:]):
        h = math.atan2(b[1] - a[1], b[0] - a[0])
        turning += abs(math.atan2(math.sin(h - heading), math.cos(h - heading)))
        heading = h
    return {'length': length, 'clearance': clear, 'gap': clear - world.radius, 'obstacle': obs,
            'waypoints': len(path) - 1, 'turning_deg': math.degrees(turning)}


def draw_world(ax, world, title):
    from matplotlib.patches import Polygon, Rectangle
    ax.add_patch(Rectangle((0, 0), world.W, world.H, fill=False, lw=2, color='black'))
    ax.add_patch(Rectangle((world.R, world.R), world.W - 2 * world.R, world.H - 2 * world.R,
                           fill=False, ls='--', lw=0.8, color='gray'))
    for poly, grown in zip(world.obstacles, world.inflated):
        ax.add_patch(Polygon(grown, closed=True, fill=True, alpha=0.15, color='gray', lw=0))
        ax.add_patch(Polygon(grown, closed=True, fill=False, ls='--', lw=0.8, color='gray'))
        ax.add_patch(Polygon(poly, closed=True, color='dimgray'))
    for name, (x, y) in world.landmarks.items():
        ax.plot(x, y, marker='s', ms=6, color='#7b3fa0', ls='none')
        ax.annotate(name, (x, y), textcoords='offset points', xytext=(3, 3), fontsize=6, color='#7b3fa0')
    ax.plot(*world.start, marker='o', ms=9, color='#2a7a2a', ls='none')
    ax.plot(*world.goal, marker='*', ms=13, color='#c03030', ls='none')
    ax.annotate('start', world.start, textcoords='offset points', xytext=(-10, 8), fontsize=8)
    ax.annotate('goal', world.goal, textcoords='offset points', xytext=(6, -12), fontsize=8)
    ax.set_xlim(-0.15, world.W + 0.15)
    ax.set_ylim(-0.15, world.H + 0.15)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)


def plot(world, vis, vor, m_vis, m_vor, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    short_c, safe_c = '#d0521b', '#1f6fb4'
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    ax = axes[0]
    draw_world(ax, world, 'Min distance: visibility graph + Dijkstra')
    for a, b in vis['edges']:
        ax.plot([a[0], b[0]], [a[1], b[1]], color='#bbbbbb', lw=0.7, zorder=1)
    nodes = np.array(vis['nodes'])
    ax.plot(nodes[:, 0], nodes[:, 1], 'o', ms=3, color='#666666')
    p = np.array(vis['path'])
    ax.plot(p[:, 0], p[:, 1], '-o', color=short_c, lw=2.5, ms=5, zorder=3)

    ax = axes[1]
    draw_world(ax, world, 'Max safety: Voronoi diagram + Dijkstra')
    for a, b in vor['edges']:
        ax.plot([a[0], b[0]], [a[1], b[1]], color='#9cc3e6', lw=0.8, zorder=1)
    r = np.array(vor['raw'])
    ax.plot(r[:, 0], r[:, 1], ':', color=safe_c, lw=1.2, zorder=2)
    p = np.array(vor['path'])
    ax.plot(p[:, 0], p[:, 1], '-o', color=safe_c, lw=2.5, ms=5, zorder=3)

    ax = axes[2]
    draw_world(ax, world, 'Comparison')
    for res, m, c, name in ((vis, m_vis, short_c, 'shortest'), (vor, m_vor, safe_c, 'safest')):
        p = np.array(res['path'])
        ax.plot(p[:, 0], p[:, 1], '-o', color=c, lw=2.5, ms=4,
                label=f'{name}: {m["length"]:.2f} m, obstacle clearance {m["obstacle"]:.2f} m')
    ax.legend(loc='lower left', fontsize=8)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# ===================================================================== main
def default_map():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, '..', 'config', 'map_hw4.yaml')]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(get_package_share_directory('landmark_nav'), 'config', 'map_hw4.yaml'))
    except Exception:
        pass
    return next((os.path.normpath(c) for c in candidates if os.path.isfile(c)), candidates[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--map', default=default_map(), help='map yaml (default: config/map_hw4.yaml)')
    ap.add_argument('--out', default=None, help='output folder (default: the map file folder)')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--rrt-iters', type=int, default=3000, help='RRT* iterations')
    ap.add_argument('--rrt-step', type=float, default=0.25, help='RRT* steer step (m)')
    ap.add_argument('--seed', type=int, default=0, help='RRT* random seed')
    ap.add_argument('--no-informed', action='store_true', help='plain RRT* sampling')
    args = ap.parse_args()

    with open(args.map) as f:
        world = World(yaml.safe_load(f))
    out = args.out or os.path.dirname(os.path.abspath(args.map))
    os.makedirs(out, exist_ok=True)
    print(f'map {args.map}: {world.W:.2f} x {world.H:.2f} m, {len(world.obstacles)} obstacle(s), '
          f'inflation {world.R:.2f} m')

    vis = plan_visibility(world)
    vor = plan_voronoi(world)
    import time
    t_rrt = time.perf_counter()
    rrt = plan_rrtstar(world, iters=args.rrt_iters, step=args.rrt_step, seed=args.seed,
                       informed=not args.no_informed)
    t_rrt = time.perf_counter() - t_rrt
    m_vis, m_vor = metrics(world, vis['path']), metrics(world, vor['path'])
    m_rrt = metrics(world, rrt['path'])
    m_raw = metrics(world, rrt['raw'])

    print(f'\n{"":24s}{"shortest":>12s}{"safest":>12s}{"RRT* raw":>12s}{"RRT*":>12s}')
    rows = [('path length (m)', 'length', '.3f'), ('min clearance (m)', 'clearance', '.3f'),
            ('robot body gap (m)', 'gap', '.3f'), ('obstacle clearance (m)', 'obstacle', '.3f'),
            ('legs', 'waypoints', 'd'),
            ('total turning (deg)', 'turning_deg', '.0f')]
    for label, key, fmt in rows:
        print(f'{label:24s}{format(m_vis[key], fmt):>12s}{format(m_vor[key], fmt):>12s}'
              f'{format(m_raw[key], fmt):>12s}{format(m_rrt[key], fmt):>12s}')
    print(f'RRT*: {args.rrt_iters} iterations in {t_rrt:.1f} s, first solution at iteration '
          f'{rrt["first_iter"]}, raw path {100 * (m_raw["length"] / m_vis["length"] - 1):.1f} % '
          f'longer than the visibility graph')

    for res, label in ((vis, 'shortest (visibility graph)'), (vor, 'safest (Voronoi)'), (rrt, 'shortest (RRT*)')):
        print(f'\n{label}: ' + ' -> '.join(f'({p[0]:.2f}, {p[1]:.2f})' for p in res['path']))
    if not args.no_plot:
        fn = os.path.join(out, 'path_compare.png')
        plot(world, vis, vor, m_vis, m_vor, fn)
        print(f'\nwrote {fn}')
        fn = os.path.join(out, 'path_rrtstar.png')
        plot_rrt(world, rrt, m_vis['length'], fn)
        print(f'wrote {fn}')


if __name__ == '__main__':
    main()
