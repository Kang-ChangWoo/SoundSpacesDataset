"""
Analyze how many samples are filtered by the wall detection logic.

Scans all depth files in the dataset and reports:
  - Total samples per scene
  - Number of wall samples filtered (by the strict AND criteria)
  - Depth statistics for each filtered sample
  - Summary table across all scenes

Usage:
    python explore/analyze_wall_filter.py --dataset_dir /path/to/dataset
    python explore/analyze_wall_filter.py --dataset_dir /path/to/dataset --depth_type erp
    python explore/analyze_wall_filter.py --dataset_dir /path/to/dataset --std_thresh 0.5
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def is_wall_depth(depth, std_thresh=0.3, concentration_thresh=0.85, range_thresh=0.5):
    """Check if depth map represents a wall-facing view (uniform depth).

    Strict AND criteria — ALL conditions must be met.

    Returns:
        is_wall (bool), stats (dict)
    """
    valid = depth[(depth > 0) & ~np.isnan(depth) & ~np.isinf(depth)]
    if len(valid) < 100:
        return False, {}

    valid_ratio = float(len(valid) / depth.size)
    if valid_ratio < 0.3:
        return False, {}

    depth_std = float(np.std(valid))
    depth_mean = float(np.mean(valid))
    depth_median = float(np.median(valid))
    depth_iqr = float(np.percentile(valid, 75) - np.percentile(valid, 25))

    narrow_band = 0.3  # metres
    concentrated = float(np.sum(np.abs(valid - depth_median) < narrow_band) / len(valid))

    stats = {
        'depth_std': round(depth_std, 4),
        'depth_mean': round(depth_mean, 4),
        'depth_median': round(depth_median, 4),
        'depth_iqr': round(depth_iqr, 4),
        'concentrated_ratio': round(concentrated, 4),
        'valid_ratio': round(valid_ratio, 4),
    }

    is_wall = (
        depth_std < std_thresh
        and concentrated > concentration_thresh
        and depth_iqr < range_thresh
    )
    return is_wall, stats


def find_depth_files(scene_dir, depth_type='pinhole'):
    """Return sorted list of (step_idx, depth_path) for a scene."""
    if depth_type == 'erp':
        depth_dir = os.path.join(scene_dir, 'erp_depth')
        prefix = 'erp_depth_'
    else:
        depth_dir = os.path.join(scene_dir, 'pinhole_depth')
        if not os.path.isdir(depth_dir):
            depth_dir = os.path.join(scene_dir, 'pinhole', 'depth')
        prefix = 'pinhole_depth_'

    if not os.path.isdir(depth_dir):
        return []

    results = []
    for f in sorted(os.listdir(depth_dir)):
        if f.startswith(prefix) and f.endswith('.npy'):
            try:
                idx = int(f.replace(prefix, '').replace('.npy', ''))
                results.append((idx, os.path.join(depth_dir, f)))
            except ValueError:
                continue
    return results


def analyze_dataset(dataset_dir, depth_type, std_thresh, concentration_thresh, range_thresh):
    """Scan every depth file and classify wall vs non-wall."""
    scenes = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')
    )

    if not scenes:
        print(f"No scene directories found in {dataset_dir}")
        return

    # Per-scene results
    scene_stats = {}  # scene -> {'total': N, 'wall': N, 'samples': [...]}
    total_all = 0
    wall_all = 0

    for si, scene_id in enumerate(scenes):
        scene_dir = os.path.join(dataset_dir, scene_id)
        depth_files = find_depth_files(scene_dir, depth_type)

        if not depth_files:
            continue

        wall_samples = []
        for step_idx, depth_path in depth_files:
            try:
                depth = np.load(depth_path).astype(np.float32)
            except Exception as e:
                print(f"  Warning: failed to load {depth_path}: {e}")
                continue

            is_wall, stats = is_wall_depth(
                depth,
                std_thresh=std_thresh,
                concentration_thresh=concentration_thresh,
                range_thresh=range_thresh,
            )
            if is_wall:
                wall_samples.append({'step_idx': step_idx, **stats})

        n_total = len(depth_files)
        n_wall = len(wall_samples)
        total_all += n_total
        wall_all += n_wall

        pct = n_wall / n_total * 100 if n_total > 0 else 0
        print(f"  [{si+1}/{len(scenes)}] {scene_id}: {n_wall}/{n_total} wall ({pct:.1f}%)", flush=True)

        scene_stats[scene_id] = {
            'total': n_total,
            'wall': n_wall,
            'samples': wall_samples,
        }

    # ── Print results ──────────────────────────────────────────────
    print("=" * 80)
    print("Wall Filter Analysis")
    print("=" * 80)
    print(f"  Dataset dir  : {dataset_dir}")
    print(f"  Depth type   : {depth_type}")
    print(f"  Thresholds   : std < {std_thresh}m, "
          f"concentration > {concentration_thresh}, "
          f"IQR < {range_thresh}m")
    print(f"  Scenes found : {len(scene_stats)}")
    print()

    # Per-scene table
    header = f"{'Scene':<25} {'Total':>6} {'Wall':>6} {'Filtered%':>10}"
    print(header)
    print("-" * len(header))

    for scene_id in sorted(scene_stats.keys()):
        s = scene_stats[scene_id]
        pct = s['wall'] / s['total'] * 100 if s['total'] > 0 else 0
        marker = " <<<" if pct > 10 else ""
        print(f"{scene_id:<25} {s['total']:>6} {s['wall']:>6} {pct:>9.1f}%{marker}")

    print("-" * len(header))
    pct_all = wall_all / total_all * 100 if total_all > 0 else 0
    print(f"{'TOTAL':<25} {total_all:>6} {wall_all:>6} {pct_all:>9.1f}%")
    print()

    # Detailed list of wall samples (top-N by concentration)
    all_wall = []
    for scene_id, s in scene_stats.items():
        for ws in s['samples']:
            all_wall.append({'scene_id': scene_id, **ws})

    all_wall.sort(key=lambda x: x['concentrated_ratio'], reverse=True)

    if all_wall:
        print(f"Top wall samples (showing up to 30 / {len(all_wall)} total):")
        print(f"  {'Scene':<25} {'Step':>5}  "
              f"{'Std':>7} {'Median':>7} {'IQR':>7} {'Conc%':>7}")
        for w in all_wall[:30]:
            print(f"  {w['scene_id']:<25} {w['step_idx']:>5}  "
                  f"{w['depth_std']:>7.3f} {w['depth_median']:>7.2f} "
                  f"{w['depth_iqr']:>7.3f} {w['concentrated_ratio']:>6.1%}")
    else:
        print("No wall samples detected with current thresholds.")

    print()
    print(f"Summary: {wall_all}/{total_all} samples ({pct_all:.1f}%) "
          f"would be filtered as wall-facing.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze wall-facing sample filtering in the dataset"
    )
    parser.add_argument(
        '--dataset_dir', type=str,
        default=os.environ.get(
            'SOUNDSPACES_DATASET_DIR',
            '/home/rvi-lab/workspace/sound-spaces/dataset'
        ),
        help='Path to the dataset root directory',
    )
    parser.add_argument('--depth_type', type=str, default='pinhole',
                        choices=['pinhole', 'erp'],
                        help='Depth type to scan (default: pinhole)')
    parser.add_argument('--std_thresh', type=float, default=0.3,
                        help='Max depth std for wall (default: 0.3)')
    parser.add_argument('--concentration_thresh', type=float, default=0.85,
                        help='Min concentration ratio for wall (default: 0.85)')
    parser.add_argument('--range_thresh', type=float, default=0.5,
                        help='Max IQR for wall (default: 0.5)')
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Error: dataset directory not found: {args.dataset_dir}")
        sys.exit(1)

    analyze_dataset(
        args.dataset_dir,
        args.depth_type,
        args.std_thresh,
        args.concentration_thresh,
        args.range_thresh,
    )


if __name__ == '__main__':
    main()
