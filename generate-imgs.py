"""
Line Intersection Benchmark Generator

Generates images with random line segments and records the number of intersections.
Each image has between 1 and 6 intersections.
"""

import os
import random
import argparse
from PIL import Image, ImageDraw
from itertools import combinations


# Image dimensions (matching the provided reference image)
WIDTH = 2000
HEIGHT = 1418


def ccw(A, B, C):
    """Check if three points are in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(seg1, seg2):
    """
    Check if two line segments intersect.
    seg1 and seg2 are tuples: ((x1, y1), (x2, y2))
    Returns True if they intersect (not at endpoints), False otherwise.
    """
    A, B = seg1
    C, D = seg2
    
    # Check if segments share an endpoint (we don't count these as intersections)
    if A == C or A == D or B == C or B == D:
        return False
    
    # Check if the segments intersect using CCW algorithm
    if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
        # Additional check: ensure intersection point is within both segments
        return True
    
    return False


def get_intersection_point(seg1, seg2):
    """
    Get the intersection point of two line segments.
    Returns None if they don't intersect.
    """
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Parallel lines
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 < t < 1 and 0 < u < 1:  # Strict inequalities to exclude endpoints
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return (px, py)
    
    return None


def count_intersections(segments):
    """Count the number of intersections between all pairs of segments."""
    count = 0
    for seg1, seg2 in combinations(segments, 2):
        if get_intersection_point(seg1, seg2) is not None:
            count += 1
    return count


def generate_random_segment(margin=50, min_length=150, max_length=600):
    """Generate a random line segment within the image bounds."""
    # Random starting point
    x1 = random.randint(margin, WIDTH - margin)
    y1 = random.randint(margin, HEIGHT - margin)
    
    # Random angle and length
    import math
    angle = random.uniform(0, 2 * math.pi)
    length = random.uniform(min_length, max_length)
    
    x2 = x1 + length * math.cos(angle)
    y2 = y1 + length * math.sin(angle)
    
    # Clamp to image bounds
    x2 = max(margin, min(WIDTH - margin, x2))
    y2 = max(margin, min(HEIGHT - margin, y2))
    
    return ((int(x1), int(y1)), (int(x2), int(y2)))


def generate_segments_with_target_intersections(target_intersections, max_attempts=1000):
    """
    Generate a set of line segments with exactly the target number of intersections.
    Uses a trial-and-error approach with intelligent segment addition/removal.
    """
    min_segments = 2  # Need at least 2 segments for 1 intersection
    max_segments = 10  # Reasonable upper bound
    
    for attempt in range(max_attempts):
        # Start with a random number of segments
        num_segments = random.randint(min_segments, max_segments)
        segments = [generate_random_segment() for _ in range(num_segments)]
        
        current_intersections = count_intersections(segments)
        
        # If we got lucky, return
        if current_intersections == target_intersections:
            return segments
        
        # Try to adjust by adding/removing segments
        for _ in range(100):
            if current_intersections < target_intersections:
                # Try adding a segment
                new_seg = generate_random_segment()
                test_segments = segments + [new_seg]
                new_count = count_intersections(test_segments)
                if new_count <= target_intersections:
                    segments = test_segments
                    current_intersections = new_count
                    if new_count == target_intersections:
                        return segments
            elif current_intersections > target_intersections:
                # Try removing a segment
                if len(segments) > 2:
                    idx = random.randint(0, len(segments) - 1)
                    test_segments = segments[:idx] + segments[idx+1:]
                    new_count = count_intersections(test_segments)
                    if new_count >= target_intersections:
                        segments = test_segments
                        current_intersections = new_count
                        if new_count == target_intersections:
                            return segments
    
    # Fallback: brute force approach
    return brute_force_generate(target_intersections)


def brute_force_generate(target_intersections, max_attempts=5000):
    """Brute force generation - keep trying until we get the target."""
    for _ in range(max_attempts):
        num_segments = random.randint(2, 12)
        segments = [generate_random_segment() for _ in range(num_segments)]
        if count_intersections(segments) == target_intersections:
            return segments
    
    # If still no luck, use a more controlled approach
    return controlled_generate(target_intersections)


def controlled_generate(target_intersections):
    """
    More controlled generation for difficult cases.
    Builds segments one at a time, tracking intersections.
    """
    segments = []
    current_intersections = 0
    
    # Add first segment
    segments.append(generate_random_segment())
    
    max_iterations = 10000
    iteration = 0
    
    while current_intersections != target_intersections and iteration < max_iterations:
        iteration += 1
        
        if current_intersections < target_intersections:
            # Need more intersections - try to add a segment that creates some
            best_segment = None
            best_diff = float('inf')
            
            for _ in range(50):
                new_seg = generate_random_segment()
                test_segments = segments + [new_seg]
                new_count = count_intersections(test_segments)
                diff = abs(new_count - target_intersections)
                
                if diff < best_diff:
                    best_diff = diff
                    best_segment = new_seg
                    if diff == 0:
                        break
            
            if best_segment:
                segments.append(best_segment)
                current_intersections = count_intersections(segments)
        
        else:
            # Too many intersections - remove a segment
            if len(segments) > 2:
                best_idx = None
                best_diff = float('inf')
                
                for idx in range(len(segments)):
                    test_segments = segments[:idx] + segments[idx+1:]
                    new_count = count_intersections(test_segments)
                    diff = abs(new_count - target_intersections)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = idx
                
                if best_idx is not None:
                    segments = segments[:best_idx] + segments[best_idx+1:]
                    current_intersections = count_intersections(segments)
            else:
                # Reset and try again
                segments = [generate_random_segment()]
                current_intersections = 0
    
    return segments


def draw_segments(segments, filename):
    """Draw the segments on a white background and save to file."""
    img = Image.new('RGB', (WIDTH, HEIGHT), 'white')
    draw = ImageDraw.Draw(img)
    
    for seg in segments:
        draw.line([seg[0], seg[1]], fill='black', width=2)
    
    img.save(filename)


def generate_benchmark(n, output_dir='benchmark_output'):
    """Generate n benchmark images with their ground truth."""
    os.makedirs(output_dir, exist_ok=True)
    
    truths = []
    
    for i in range(n):
        # Random target between 1 and 6 intersections
        target = random.randint(1, 6)
        
        print(f"Generating image {i+1}/{n} with {target} intersection(s)...")
        
        segments = generate_segments_with_target_intersections(target)
        actual_intersections = count_intersections(segments)
        
        # Verify we got the right count
        if actual_intersections != target:
            print(f"  Warning: Got {actual_intersections} intersections instead of {target}")
        
        # Save image
        filename = os.path.join(output_dir, f'image_{i+1:04d}.png')
        draw_segments(segments, filename)
        
        truths.append(actual_intersections)
        print(f"  Saved {filename} with {actual_intersections} intersection(s)")
    
    # Save truth file
    truth_file = os.path.join(output_dir, 'truth.txt')
    with open(truth_file, 'w') as f:
        for i, count in enumerate(truths):
            f.write(f"image_{i+1:04d}.png: {count}\n")
    
    print(f"\nGenerated {n} images in '{output_dir}/'")
    print(f"Ground truth saved to '{truth_file}'")
    
    # Print summary
    from collections import Counter
    distribution = Counter(truths)
    print("\nIntersection distribution:")
    for k in sorted(distribution.keys()):
        print(f"  {k} intersection(s): {distribution[k]} images")


def main():
    parser = argparse.ArgumentParser(
        description='Generate line intersection benchmark images'
    )
    parser.add_argument(
        'n', 
        type=int, 
        help='Number of images to generate'
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default='benchmark_output',
        help='Output directory (default: benchmark_output)'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    generate_benchmark(args.n, args.output)


if __name__ == '__main__':
    main()