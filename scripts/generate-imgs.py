"""
Hard benchmark image generator.

Generates 100 question images containing intersecting circles, triangles,
and parallelograms. Ground-truth labels are the number of unique boundary
intersection points across different shapes in each image.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw

WIDTH = 2000
HEIGHT = 1418
DEFAULT_NUM_IMAGES = 100
DEFAULT_MIN_INTERSECTIONS = 0
DEFAULT_MAX_INTERSECTIONS = 15
MIN_INTERSECTION_ANGLE_DEG = 22.0
MIN_ENDPOINT_CLEARANCE = 36.0
MIN_BORDER_CLEARANCE = 44.0
MIN_ANY_VERTEX_CLEARANCE = 26.0
MIN_NEARBY_PRIMITIVE_CLEARANCE = 20.0
MIN_PAIR_INTERSECTION_SEPARATION = 34.0
INTERSECTION_MERGE_TOL = 8.0
MIN_INTERSECTION_SEPARATION = 28.0
LOCAL_INTERSECTION_CLUSTER_RADIUS = 30.0
MAX_LOCAL_NEIGHBORS = 1
MAX_INTERSECTIONS_PER_SHAPE = 6
DISALLOW_CIRCLE_CIRCLE_INTERSECTIONS = True


@dataclass(frozen=True)
class Segment:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    shape_id: int


@dataclass(frozen=True)
class Circle:
    center: Tuple[float, float]
    radius: float
    shape_id: int


@dataclass
class Scene:
    segments: List[Segment]
    circles: List[Circle]
    drawable_shapes: List[Tuple[str, Tuple]]


@dataclass(frozen=True)
class IntersectionCandidate:
    point: Tuple[float, float]
    angle_deg: float
    endpoints: Tuple[Tuple[float, float], ...]
    shape_ids: Tuple[int, ...]


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def _point_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


def _acute_angle_deg(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_th = _dot(v1[0], v1[1], v2[0], v2[1]) / (n1 * n2)
    cos_th = max(-1.0, min(1.0, cos_th))
    return math.degrees(math.acos(abs(cos_th)))


def _point_to_segment_distance(point: Tuple[float, float], seg: Segment) -> float:
    px, py = point
    (x1, y1), (x2, y2) = seg.p1, seg.p2
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom < 1e-9:
        return _point_dist(point, seg.p1)
    t = ((px - x1) * dx + (py - y1) * dy) / denom
    t = max(0.0, min(1.0, t))
    proj = (x1 + t * dx, y1 + t * dy)
    return _point_dist(point, proj)


def _point_to_circle_boundary_distance(point: Tuple[float, float], circle: Circle) -> float:
    return abs(_point_dist(point, circle.center) - circle.radius)


def _segment_to_segment_distance(s1: Segment, s2: Segment) -> float:
    if segment_segment_intersections(s1, s2):
        return 0.0
    return min(
        _point_to_segment_distance(s1.p1, s2),
        _point_to_segment_distance(s1.p2, s2),
        _point_to_segment_distance(s2.p1, s1),
        _point_to_segment_distance(s2.p2, s1),
    )


def _segment_to_circle_boundary_distance(seg: Segment, cir: Circle) -> float:
    points = segment_circle_intersections(seg, cir)
    if points:
        return 0.0
    d1 = _point_dist(seg.p1, cir.center)
    d2 = _point_dist(seg.p2, cir.center)
    center_proj_dist = _point_to_segment_distance(cir.center, seg)
    return min(abs(d1 - cir.radius), abs(d2 - cir.radius), abs(center_proj_dist - cir.radius))


def _circle_to_circle_boundary_distance(c1: Circle, c2: Circle) -> float:
    points = circle_circle_intersections(c1, c2)
    if points:
        return 0.0
    d = _point_dist(c1.center, c2.center)
    return min(abs(d - (c1.radius + c2.radius)), abs(d - abs(c1.radius - c2.radius)))


def _triangle_angles_deg(pts: Sequence[Tuple[float, float]]) -> Tuple[float, float, float]:
    a, b, c = pts

    def angle_at(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> float:
        v1 = _sub(p, q)
        v2 = _sub(r, q)
        return _acute_angle_deg(v1, v2)

    return (angle_at(c, a, b), angle_at(a, b, c), angle_at(b, c, a))


def _triangle_area(pts: Sequence[Tuple[float, float]]) -> float:
    (x1, y1), (x2, y2), (x3, y3) = pts
    return abs(0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)))


def _triangle_is_valid(pts: Sequence[Tuple[float, float]]) -> bool:
    s1 = _point_dist(pts[0], pts[1])
    s2 = _point_dist(pts[1], pts[2])
    s3 = _point_dist(pts[2], pts[0])
    if min(s1, s2, s3) < 120.0:
        return False
    if max(s1, s2, s3) > 520.0:
        return False

    a1, a2, a3 = _triangle_angles_deg(pts)
    if min(a1, a2, a3) < 24.0:
        return False

    if _triangle_area(pts) < 11000.0:
        return False

    return True


def _segment_angle_deg(s1: Segment, s2: Segment) -> float:
    v1 = _sub(s1.p2, s1.p1)
    v2 = _sub(s2.p2, s2.p1)
    return _acute_angle_deg(v1, v2)


def _segment_circle_angle_deg(seg: Segment, circle: Circle, pt: Tuple[float, float]) -> float:
    seg_vec = _sub(seg.p2, seg.p1)
    radius_vec = _sub(pt, circle.center)
    tangent_vec = (-radius_vec[1], radius_vec[0])
    return _acute_angle_deg(seg_vec, tangent_vec)


def _circle_circle_angle_deg(c1: Circle, c2: Circle, pt: Tuple[float, float]) -> float:
    r1 = _sub(pt, c1.center)
    r2 = _sub(pt, c2.center)
    t1 = (-r1[1], r1[0])
    t2 = (-r2[1], r2[0])
    return _acute_angle_deg(t1, t2)


def segment_segment_intersections(s1: Segment, s2: Segment, eps: float = 1e-8) -> List[Tuple[float, float]]:
    p = s1.p1
    r = _sub(s1.p2, s1.p1)
    q = s2.p1
    s = _sub(s2.p2, s2.p1)

    rxs = _cross(r[0], r[1], s[0], s[1])
    qmp = _sub(q, p)

    if abs(rxs) < eps:
        return []

    t = _cross(qmp[0], qmp[1], s[0], s[1]) / rxs
    u = _cross(qmp[0], qmp[1], r[0], r[1]) / rxs

    if eps < t < 1.0 - eps and eps < u < 1.0 - eps:
        return [(p[0] + t * r[0], p[1] + t * r[1])]
    return []


def segment_circle_intersections(seg: Segment, circle: Circle, eps: float = 1e-8) -> List[Tuple[float, float]]:
    (x1, y1), (x2, y2) = seg.p1, seg.p2
    cx, cy = circle.center
    dx, dy = x2 - x1, y2 - y1

    fx, fy = x1 - cx, y1 - cy
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - circle.radius * circle.radius

    disc = b * b - 4 * a * c
    if disc < eps:
        return []

    sqrt_disc = math.sqrt(max(disc, 0.0))
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    out: List[Tuple[float, float]] = []
    for t in (t1, t2):
        if eps < t < 1.0 - eps:
            out.append((x1 + t * dx, y1 + t * dy))
    return out


def circle_circle_intersections(c1: Circle, c2: Circle, eps: float = 1e-8) -> List[Tuple[float, float]]:
    x0, y0 = c1.center
    x1, y1 = c2.center
    r0, r1 = c1.radius, c2.radius

    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)

    if d < eps:
        return []
    if d > r0 + r1 - eps:
        return []
    if d < abs(r0 - r1) + eps:
        return []

    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h_sq = r0 * r0 - a * a
    if h_sq < eps:
        return []

    h = math.sqrt(h_sq)
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d

    rx = -dy * (h / d)
    ry = dx * (h / d)

    p1 = (xm + rx, ym + ry)
    p2 = (xm - rx, ym - ry)
    return [p1, p2]


def unique_points(points: Sequence[Tuple[float, float]], tol: float = 3.0) -> List[Tuple[float, float]]:
    uniq: List[Tuple[float, float]] = []
    for p in points:
        if all(_point_dist(p, q) > tol for q in uniq):
            uniq.append(p)
    return uniq


def _point_is_clear(point: Tuple[float, float], endpoints: Sequence[Tuple[float, float]]) -> bool:
    x, y = point
    if (
        x < MIN_BORDER_CLEARANCE
        or x > WIDTH - MIN_BORDER_CLEARANCE
        or y < MIN_BORDER_CLEARANCE
        or y > HEIGHT - MIN_BORDER_CLEARANCE
    ):
        return False
    for ep in endpoints:
        if _point_dist(point, ep) < MIN_ENDPOINT_CLEARANCE:
            return False
    return True


def _collect_intersection_candidates(scene: Scene) -> Tuple[List[IntersectionCandidate], bool]:
    candidates: List[IntersectionCandidate] = []

    for s1, s2 in combinations(scene.segments, 2):
        if s1.shape_id == s2.shape_id:
            continue
        angle = _segment_angle_deg(s1, s2)
        for pt in segment_segment_intersections(s1, s2):
            candidates.append(
                IntersectionCandidate(
                    point=pt,
                    angle_deg=angle,
                    endpoints=(s1.p1, s1.p2, s2.p1, s2.p2),
                    shape_ids=(s1.shape_id, s2.shape_id),
                )
            )

    for seg in scene.segments:
        for cir in scene.circles:
            if seg.shape_id == cir.shape_id:
                continue
            points = segment_circle_intersections(seg, cir)
            if len(points) == 2 and _point_dist(points[0], points[1]) < MIN_PAIR_INTERSECTION_SEPARATION:
                return [], False
            for pt in points:
                candidates.append(
                    IntersectionCandidate(
                        point=pt,
                        angle_deg=_segment_circle_angle_deg(seg, cir, pt),
                        endpoints=(seg.p1, seg.p2),
                        shape_ids=(seg.shape_id, cir.shape_id),
                    )
                )

    for c1, c2 in combinations(scene.circles, 2):
        if c1.shape_id == c2.shape_id:
            continue
        points = circle_circle_intersections(c1, c2)
        if points and DISALLOW_CIRCLE_CIRCLE_INTERSECTIONS:
            return [], False
        if len(points) == 2 and _point_dist(points[0], points[1]) < MIN_PAIR_INTERSECTION_SEPARATION:
            return [], False
        for pt in points:
            candidates.append(
                IntersectionCandidate(
                    point=pt,
                    angle_deg=_circle_circle_angle_deg(c1, c2, pt),
                    endpoints=(),
                    shape_ids=(c1.shape_id, c2.shape_id),
                )
            )

    return candidates, True


def analyze_scene(scene: Scene) -> Tuple[int, bool]:
    candidates, pairwise_clear = _collect_intersection_candidates(scene)
    if not pairwise_clear:
        return 0, False
    if not candidates:
        return 0, True

    all_vertices = unique_points([s.p1 for s in scene.segments] + [s.p2 for s in scene.segments], tol=1.0)
    raw_points = [c.point for c in candidates]
    for c in candidates:
        if c.angle_deg < MIN_INTERSECTION_ANGLE_DEG:
            return 0, False
        if not _point_is_clear(c.point, c.endpoints):
            return 0, False
        if any(_point_dist(c.point, v) < MIN_ANY_VERTEX_CLEARANCE for v in all_vertices):
            return 0, False

        shape_ids = set(c.shape_ids)
        for seg in scene.segments:
            if seg.shape_id in shape_ids:
                continue
            if _point_to_segment_distance(c.point, seg) < MIN_NEARBY_PRIMITIVE_CLEARANCE:
                return 0, False
        for cir in scene.circles:
            if cir.shape_id in shape_ids:
                continue
            if _point_to_circle_boundary_distance(c.point, cir) < MIN_NEARBY_PRIMITIVE_CLEARANCE:
                return 0, False

    # Count all visible intersections once geometry passes ambiguity checks.
    merged = unique_points(raw_points, tol=INTERSECTION_MERGE_TOL)

    # Reject scenes with intersections that are visually too close together.
    for p1, p2 in combinations(merged, 2):
        if _point_dist(p1, p2) < MIN_INTERSECTION_SEPARATION:
            return len(merged), False

    # Reject local intersection clusters.
    for p in merged:
        neighbors = sum(
            1 for q in merged if p != q and _point_dist(p, q) < LOCAL_INTERSECTION_CLUSTER_RADIUS
        )
        if neighbors > MAX_LOCAL_NEIGHBORS:
            return len(merged), False

    # Reject scenes where too many intersections involve the same shape.
    shape_counts = {}
    for c in candidates:
        for sid in c.shape_ids:
            shape_counts[sid] = shape_counts.get(sid, 0) + 1
    if shape_counts and max(shape_counts.values()) > MAX_INTERSECTIONS_PER_SHAPE:
        return len(merged), False

    # Keep any two unrelated primitives apart to avoid near-edge ambiguity.
    for s1, s2 in combinations(scene.segments, 2):
        if s1.shape_id == s2.shape_id:
            continue
        if not segment_segment_intersections(s1, s2) and _segment_to_segment_distance(
            s1, s2
        ) < MIN_NEARBY_PRIMITIVE_CLEARANCE:
            return len(merged), False

    for seg in scene.segments:
        for cir in scene.circles:
            if seg.shape_id == cir.shape_id:
                continue
            if not segment_circle_intersections(seg, cir) and _segment_to_circle_boundary_distance(
                seg, cir
            ) < MIN_NEARBY_PRIMITIVE_CLEARANCE:
                return len(merged), False

    for c1, c2 in combinations(scene.circles, 2):
        if c1.shape_id == c2.shape_id:
            continue
        if not circle_circle_intersections(c1, c2) and _circle_to_circle_boundary_distance(
            c1, c2
        ) < MIN_NEARBY_PRIMITIVE_CLEARANCE:
            return len(merged), False

    return len(merged), True


def random_point(margin: int = 80) -> Tuple[float, float]:
    return (
        random.uniform(margin, WIDTH - margin),
        random.uniform(margin, HEIGHT - margin),
    )


def make_triangle(shape_id: int) -> Tuple[List[Segment], Tuple[str, Tuple]]:
    pts: List[Tuple[float, float]] = []
    for _ in range(200):
        cx, cy = random_point(200)
        base = random.uniform(150, 290)
        trial_pts = []
        for k in range(3):
            # Keep vertices roughly spread around center to avoid needle triangles.
            ang = random.uniform(0, 2 * math.pi) + (2 * math.pi * k / 3)
            r = random.uniform(base * 0.85, base * 1.1)
            trial_pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        if _triangle_is_valid(trial_pts):
            pts = trial_pts
            break

    if not pts:
        # Deterministic fallback: near-equilateral triangle with mild jitter.
        cx, cy = random_point(220)
        r = random.uniform(170, 230)
        theta = random.uniform(0, 2 * math.pi)
        pts = [
            (cx + r * math.cos(theta), cy + r * math.sin(theta)),
            (cx + r * math.cos(theta + 2 * math.pi / 3), cy + r * math.sin(theta + 2 * math.pi / 3)),
            (cx + r * math.cos(theta + 4 * math.pi / 3), cy + r * math.sin(theta + 4 * math.pi / 3)),
        ]

    segs = [
        Segment(pts[0], pts[1], shape_id),
        Segment(pts[1], pts[2], shape_id),
        Segment(pts[2], pts[0], shape_id),
    ]
    return segs, ("polygon", tuple(pts))


def make_parallelogram(shape_id: int) -> Tuple[List[Segment], Tuple[str, Tuple]]:
    cx, cy = random_point(220)
    w = random.uniform(140, 320)
    h = random.uniform(90, 240)
    angle = random.uniform(0, 2 * math.pi)
    skew = random.uniform(0.25, 0.85)

    ux, uy = math.cos(angle), math.sin(angle)
    vx, vy = -uy, ux

    a = (cx - ux * w / 2 - vx * h * skew / 2, cy - uy * w / 2 - vy * h * skew / 2)
    b = (cx + ux * w / 2 - vx * h * skew / 2, cy + uy * w / 2 - vy * h * skew / 2)
    c = (b[0] + vx * h, b[1] + vy * h)
    d = (a[0] + vx * h, a[1] + vy * h)

    pts = [a, b, c, d]
    segs = [
        Segment(pts[0], pts[1], shape_id),
        Segment(pts[1], pts[2], shape_id),
        Segment(pts[2], pts[3], shape_id),
        Segment(pts[3], pts[0], shape_id),
    ]
    return segs, ("polygon", tuple(pts))


def make_circle(shape_id: int) -> Tuple[List[Segment], List[Circle], Tuple[str, Tuple]]:
    cx, cy = random_point(260)
    r = random.uniform(80, 210)
    return [], [Circle((cx, cy), r, shape_id)], ("circle", (cx, cy, r))


def make_line(shape_id: int) -> Tuple[List[Segment], Tuple[str, Tuple]]:
    cx, cy = random_point(180)
    angle = random.uniform(0, 2 * math.pi)
    length = random.uniform(260, 560)
    dx = 0.5 * length * math.cos(angle)
    dy = 0.5 * length * math.sin(angle)
    p1 = (cx - dx, cy - dy)
    p2 = (cx + dx, cy + dy)
    return [Segment(p1, p2, shape_id)], ("line", (p1, p2))


def build_random_scene(target_intersections: int | None = None) -> Scene:
    if target_intersections is None:
        shape_count = random.randint(5, 8)
    elif target_intersections <= 2:
        shape_count = random.randint(2, 4)
    elif target_intersections <= 6:
        shape_count = random.randint(4, 6)
    elif target_intersections <= 10:
        shape_count = random.randint(5, 7)
    else:
        shape_count = random.randint(9, 13)

    segments: List[Segment] = []
    circles: List[Circle] = []
    draw_shapes: List[Tuple[str, Tuple]] = []

    for sid in range(shape_count):
        # Add standalone lines to diversify geometry.
        t = random.choice(["triangle", "parallelogram", "circle", "line"])
        if t == "triangle":
            segs, drawable = make_triangle(sid)
            segments.extend(segs)
            draw_shapes.append(drawable)
        elif t == "parallelogram":
            segs, drawable = make_parallelogram(sid)
            segments.extend(segs)
            draw_shapes.append(drawable)
        elif t == "circle":
            segs, cirs, drawable = make_circle(sid)
            segments.extend(segs)
            circles.extend(cirs)
            draw_shapes.append(drawable)
        else:
            segs, drawable = make_line(sid)
            segments.extend(segs)
            draw_shapes.append(drawable)

    return Scene(segments=segments, circles=circles, drawable_shapes=draw_shapes)


def generate_scene_in_range(
    min_intersections: int,
    max_intersections: int,
    preferred_target: int | None = None,
    max_attempts: int = 50000,
) -> Tuple[Scene, int]:
    if min_intersections < 0 or max_intersections < min_intersections:
        raise ValueError("Invalid intersection range")
    for _ in range(max_attempts):
        scene = build_random_scene(target_intersections=preferred_target)
        intersections, unambiguous = analyze_scene(scene)
        if unambiguous and min_intersections <= intersections <= max_intersections:
            return scene, intersections
    raise RuntimeError(
        "Could not generate an unambiguous scene in requested intersection range. "
        "Relax constraints or increase max_attempts."
    )


def generate_scene_near_target(
    target: int,
    min_intersections: int,
    max_intersections: int,
) -> Tuple[Scene, int]:
    if target < min_intersections or target > max_intersections:
        raise ValueError("target must be inside the requested range")

    if target == 0:
        for _ in range(5000):
            scene = build_random_scene(target_intersections=0)
            count, ok = analyze_scene(scene)
            if ok and count == 0:
                return scene, count

    # Try exact first, then progressively relax closeness to keep runtime bounded.
    tolerances = [0, 1, 2]
    for tol in tolerances:
        for _ in range(600):
            try:
                scene, count = generate_scene_in_range(
                    min_intersections=min_intersections,
                    max_intersections=max_intersections,
                    preferred_target=target,
                    max_attempts=40,
                )
            except RuntimeError:
                continue
            if abs(count - target) <= tol:
                return scene, count

    raise RuntimeError(f"Could not generate a scene near target={target}.")


def _rotate_and_translate(
    p: Tuple[float, float],
    center: Tuple[float, float],
    angle_rad: float,
    scale: float,
) -> Tuple[float, float]:
    x, y = p
    xr = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    yr = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    return (center[0] + xr * scale, center[1] + yr * scale)


def _grid_centers() -> List[Tuple[float, float]]:
    xs = [260.0, 740.0, 1260.0, 1740.0]
    ys = [240.0, 710.0, 1180.0]
    return [(x, y) for y in ys for x in xs]


def _add_triangle(
    segments: List[Segment],
    drawable_shapes: List[Tuple[str, Tuple]],
    sid: int,
    pts: Sequence[Tuple[float, float]],
) -> None:
    p0, p1, p2 = pts
    segments.append(Segment(p0, p1, sid))
    segments.append(Segment(p1, p2, sid))
    segments.append(Segment(p2, p0, sid))
    drawable_shapes.append(("polygon", tuple(pts)))


def _add_parallelogram(
    segments: List[Segment],
    drawable_shapes: List[Tuple[str, Tuple]],
    sid: int,
    pts: Sequence[Tuple[float, float]],
) -> None:
    p0, p1, p2, p3 = pts
    segments.append(Segment(p0, p1, sid))
    segments.append(Segment(p1, p2, sid))
    segments.append(Segment(p2, p3, sid))
    segments.append(Segment(p3, p0, sid))
    drawable_shapes.append(("polygon", tuple(pts)))


def _count_scene_points(scene: Scene) -> int:
    raw_points: List[Tuple[float, float]] = []
    for s1, s2 in combinations(scene.segments, 2):
        if s1.shape_id == s2.shape_id:
            continue
        raw_points.extend(segment_segment_intersections(s1, s2))
    for seg in scene.segments:
        for cir in scene.circles:
            if seg.shape_id == cir.shape_id:
                continue
            raw_points.extend(segment_circle_intersections(seg, cir))
    for c1, c2 in combinations(scene.circles, 2):
        if c1.shape_id == c2.shape_id:
            continue
        raw_points.extend(circle_circle_intersections(c1, c2))
    return len(unique_points(raw_points, tol=INTERSECTION_MERGE_TOL))


def build_constructive_scene(target: int, rng: random.Random) -> Scene:
    if target < 0:
        raise ValueError("target must be non-negative")
    if target > 20:
        raise ValueError("target too high for constructive template")

    two_point_motifs = target // 2
    one_point_motifs = target % 2
    motif_types = []
    motif_types.extend(["cross_lines"] * one_point_motifs)
    motif_pool = ["circle_secant", "triangle_secant", "parallelogram_secant"]
    for _ in range(two_point_motifs):
        motif_types.append(rng.choice(motif_pool))
    rng.shuffle(motif_types)

    centers = _grid_centers()
    rng.shuffle(centers)
    if len(motif_types) > len(centers):
        raise RuntimeError("Not enough placement cells for requested target.")

    segments: List[Segment] = []
    circles: List[Circle] = []
    drawable_shapes: List[Tuple[str, Tuple]] = []

    sid = 0
    for motif, center in zip(motif_types, centers):
        angle = rng.uniform(-math.pi, math.pi)
        scale = rng.uniform(0.9, 1.1)
        transform = lambda p: _rotate_and_translate(p, center, angle, scale)

        if motif == "cross_lines":
            p1, p2 = transform((-85, -45)), transform((85, 45))
            q1, q2 = transform((-85, 45)), transform((85, -45))
            segments.append(Segment(p1, p2, sid))
            drawable_shapes.append(("line", (p1, p2)))
            sid += 1
            segments.append(Segment(q1, q2, sid))
            drawable_shapes.append(("line", (q1, q2)))
            sid += 1
        elif motif == "circle_secant":
            cxy = transform((0, 0))
            circles.append(Circle(cxy, 56 * scale, sid))
            drawable_shapes.append(("circle", (cxy[0], cxy[1], 56 * scale)))
            sid += 1
            p1, p2 = transform((-100, 28)), transform((100, -28))
            segments.append(Segment(p1, p2, sid))
            drawable_shapes.append(("line", (p1, p2)))
            sid += 1
        elif motif == "triangle_secant":
            tri = [transform((-90, 65)), transform((90, 65)), transform((0, -85))]
            _add_triangle(segments, drawable_shapes, sid, tri)
            sid += 1
            p1, p2 = transform((-110, 5)), transform((110, 5))
            segments.append(Segment(p1, p2, sid))
            drawable_shapes.append(("line", (p1, p2)))
            sid += 1
        else:
            para = [
                transform((-95, -55)),
                transform((25, -55)),
                transform((95, 55)),
                transform((-25, 55)),
            ]
            _add_parallelogram(segments, drawable_shapes, sid, para)
            sid += 1
            p1, p2 = transform((0, -110)), transform((0, 110))
            segments.append(Segment(p1, p2, sid))
            drawable_shapes.append(("line", (p1, p2)))
            sid += 1

    return Scene(segments=segments, circles=circles, drawable_shapes=drawable_shapes)


def _scene_shape_ids(scene: Scene) -> List[int]:
    ids = {s.shape_id for s in scene.segments}
    ids.update(c.shape_id for c in scene.circles)
    return sorted(ids)


def _next_shape_id(scene: Scene) -> int:
    ids = _scene_shape_ids(scene)
    return (ids[-1] + 1) if ids else 0


def _scene_centers(scene: Scene) -> List[Tuple[float, float]]:
    centers: List[Tuple[float, float]] = []
    for shape_type, payload in scene.drawable_shapes:
        if shape_type == "line":
            p1, p2 = payload
            centers.append(((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5))
        elif shape_type == "circle":
            cx, cy, _ = payload
            centers.append((cx, cy))
        else:
            pts = payload
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            centers.append((cx, cy))
    return centers


def _shape_type_counts(scene: Scene) -> dict:
    counts = {"line": 0, "circle": 0, "polygon": 0}
    for shape_type, _ in scene.drawable_shapes:
        counts[shape_type] = counts.get(shape_type, 0) + 1
    return counts


def _scene_diversity_ok(scene: Scene, target: int) -> bool:
    n_shapes = len(scene.drawable_shapes)
    if not (6 <= n_shapes <= 14):
        return False

    counts = _shape_type_counts(scene)
    nonzero = sum(1 for v in counts.values() if v > 0)
    if target >= 15:
        if nonzero < 3:
            return False
        if counts["line"] == 0 or counts["circle"] == 0 or counts["polygon"] == 0:
            return False
    elif nonzero < 2:
        return False

    max_share = max(counts.values()) / float(n_shapes)
    if max_share > 0.65:
        return False

    centers = _scene_centers(scene)
    if not centers:
        return False
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    if (max(xs) - min(xs)) < 520 or (max(ys) - min(ys)) < 360:
        return False
    return True


def _scene_signature(scene: Scene) -> str:
    tokens: List[str] = []
    for shape_type, payload in scene.drawable_shapes:
        if shape_type == "line":
            p1, p2 = payload
            cx = (p1[0] + p2[0]) * 0.5
            cy = (p1[1] + p2[1]) * 0.5
            ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            orient = int(((ang + math.pi) / (2 * math.pi)) * 12) % 12
            size = int(_point_dist(p1, p2) // 80)
        elif shape_type == "circle":
            cx, cy, r = payload
            orient = 0
            size = int(r // 30)
        else:
            pts = payload
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            e0 = _sub(pts[1], pts[0])
            ang = math.atan2(e0[1], e0[0])
            orient = int(((ang + math.pi) / (2 * math.pi)) * 12) % 12
            perimeter = sum(_point_dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))
            size = int(perimeter // 120)
        bx = int(cx // 220)
        by = int(cy // 180)
        tokens.append(f"{shape_type}:{bx}:{by}:{orient}:{size}")
    tokens.sort()
    return "|".join(tokens)


def _build_random_shape(shape_id: int) -> Tuple[List[Segment], List[Circle], Tuple[str, Tuple]]:
    t = random.choice(["triangle", "parallelogram", "circle", "line"])
    if t == "triangle":
        segs, draw = make_triangle(shape_id)
        return segs, [], draw
    if t == "parallelogram":
        segs, draw = make_parallelogram(shape_id)
        return segs, [], draw
    if t == "circle":
        segs, cirs, draw = make_circle(shape_id)
        return segs, cirs, draw
    segs, draw = make_line(shape_id)
    return segs, [], draw


def _scene_with_added_shape(scene: Scene) -> Scene:
    sid = _next_shape_id(scene)
    segs, cirs, draw = _build_random_shape(sid)
    return Scene(
        segments=scene.segments + segs,
        circles=scene.circles + cirs,
        drawable_shapes=scene.drawable_shapes + [draw],
    )


def generate_guided_scene_for_target(
    target: int,
    used_signatures: set,
    max_restarts: int = 120,
) -> Tuple[Scene, int]:
    # Incrementally add shapes, keeping only moves that approach target exactly.
    for _ in range(max_restarts):
        scene = build_random_scene(target_intersections=max(3, target // 2))
        count, ok = analyze_scene(scene)
        if not ok or count > target:
            continue

        stagnation = 0
        while count < target and stagnation < 10:
            best_scene = None
            best_count = count
            candidates = 100 if target >= 16 else 70
            for _ in range(candidates):
                trial = _scene_with_added_shape(scene)
                trial_count, trial_ok = analyze_scene(trial)
                if not trial_ok or trial_count > target:
                    continue
                if trial_count > best_count:
                    best_scene = trial
                    best_count = trial_count
                    if best_count == target:
                        break

            if best_scene is None:
                stagnation += 1
                continue

            scene = best_scene
            count = best_count
            stagnation = 0

        if count != target:
            continue
        if not _scene_diversity_ok(scene, target):
            continue
        sig = _scene_signature(scene)
        if sig in used_signatures:
            continue
        used_signatures.add(sig)
        return scene, count

    raise RuntimeError(f"Could not generate diverse scene for target={target}")


def draw_scene(scene: Scene, out_path: str) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)

    for shape_type, payload in scene.drawable_shapes:
        if shape_type == "polygon":
            pts = payload
            draw.polygon(pts, outline="black", fill=None, width=3)
        elif shape_type == "line":
            p1, p2 = payload
            draw.line([p1, p2], fill="black", width=3)
        else:
            cx, cy, r = payload
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="black", width=3)

    img.save(out_path)


def generate_questions(
    n: int = DEFAULT_NUM_IMAGES,
    output_dir: str = "public/imgs",
    truth_path: str = "truth.txt",
    min_intersections: int = DEFAULT_MIN_INTERSECTIONS,
    max_intersections: int = DEFAULT_MAX_INTERSECTIONS,
    seed: int | None = None,
) -> None:
    if seed is not None:
        random.seed(seed)

    if min_intersections < 0 or max_intersections < min_intersections:
        raise ValueError("Invalid intersection range")

    os.makedirs(output_dir, exist_ok=True)

    span = list(range(min_intersections, max_intersections + 1))
    if not span:
        raise ValueError("Intersection range must contain at least one value")

    # Build near-even targets and randomize order.
    per_bucket = n // len(span)
    remainder = n % len(span)
    targets: List[int] = []
    for v in span:
        targets.extend([v] * per_bucket)
    if remainder:
        extras = span[:]
        random.shuffle(extras)
        targets.extend(extras[:remainder])
    random.shuffle(targets)

    truths: List[int] = []
    used_signatures: set = set()
    for i, target in enumerate(targets, start=1):
        scene, answer = generate_guided_scene_for_target(target=target, used_signatures=used_signatures)
        out_file = os.path.join(output_dir, f"{i}.png")
        draw_scene(scene, out_file)
        truths.append(answer)
        print(f"Generated {out_file}: target={target} intersections={answer}")

    with open(truth_path, "w", encoding="utf-8") as f:
        for v in truths:
            f.write(f"{v}\n")

    print(f"\nDone. Wrote {n} harder question images to '{output_dir}'.")
    print(f"Ground truth saved to '{truth_path}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 100 harder shape-intersection benchmark questions"
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="Number of images/questions to generate (default: 100)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="public/imgs",
        help="Output image directory (default: public/imgs)",
    )
    parser.add_argument(
        "-t",
        "--truth",
        type=str,
        default="truth.txt",
        help="Output truth file path (default: truth.txt)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--min-intersections",
        type=int,
        default=DEFAULT_MIN_INTERSECTIONS,
        help="Minimum intersections per generated image (default: 0)",
    )
    parser.add_argument(
        "--max-intersections",
        type=int,
        default=DEFAULT_MAX_INTERSECTIONS,
        help="Maximum intersections per generated image (default: 15)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_questions(
        n=args.num,
        output_dir=args.output,
        truth_path=args.truth,
        min_intersections=args.min_intersections,
        max_intersections=args.max_intersections,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
