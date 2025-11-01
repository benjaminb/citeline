"""
Plot heatmaps of hitrate@50 and hitrate@100 for grid search results.
For each overlap value, creates heatmaps with min_length on x-axis and increment on y-axis.
"""
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def parse_directory_name(dirname):
    """Extract min_length, increment, and overlap from directory name.

    Args:
        dirname: Directory name like 'gs_min100_inc200_ov0'

    Returns:
        tuple: (min_length, increment, overlap) or None if pattern doesn't match
    """
    pattern = r'gs_min(\d+)_inc(\d+)_ov(\d+)'
    match = re.match(pattern, dirname)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def load_grid_search_results(base_dir):
    """Load all grid search results from the base directory.

    Args:
        base_dir: Path to the gridsearch directory

    Returns:
        dict: Nested dict organized as {overlap: {(min, inc): hitrate_data}}
    """
    base_path = Path(base_dir)
    results = defaultdict(dict)

    # Find all directories matching the pattern
    for dir_path in base_path.glob('gs_min*_inc*_ov*'):
        if not dir_path.is_dir():
            continue

        # Parse directory name
        params = parse_directory_name(dir_path.name)
        if params is None:
            print(f"Skipping {dir_path.name}: doesn't match pattern")
            continue

        min_length, increment, overlap = params

        # Look for results JSON file
        json_file = dir_path / f'results_{dir_path.name}.json'
        if not json_file.exists():
            print(f"Warning: {json_file} not found")
            continue

        # Load the JSON
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            hitrate_at_k = data.get('average_hitrate_at_k', [])
            if len(hitrate_at_k) < 100:
                print(f"Warning: {json_file} has only {len(hitrate_at_k)} k values")
                continue

            # Extract hitrate@25 (index 24), hitrate@50 (index 49) and hitrate@100 (index 99)
            # Note: 0-indexed, so hitrate@1 is at index 0, hitrate@25 is at index 24, etc.
            hitrate_25 = hitrate_at_k[24] if len(hitrate_at_k) > 24 else None
            hitrate_50 = hitrate_at_k[49] if len(hitrate_at_k) > 49 else None
            hitrate_100 = hitrate_at_k[99] if len(hitrate_at_k) > 99 else None

            results[overlap][(min_length, increment)] = {
                'hitrate_25': hitrate_25,
                'hitrate_50': hitrate_50,
                'hitrate_100': hitrate_100,
                'config': data.get('config', {}),
                'all_hitrates': hitrate_at_k
            }

        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    return dict(results)


def create_heatmap_data(results_dict):
    """Convert results dict to 2D arrays suitable for heatmap plotting.

    Args:
        results_dict: Dict with {(min, inc): data} structure

    Returns:
        tuple: (min_values, inc_values, hitrate_25_grid, hitrate_50_grid, hitrate_100_grid)
    """
    if not results_dict:
        return None, None, None, None, None

    # Extract unique min and increment values
    all_keys = list(results_dict.keys())
    min_values = sorted(set(k[0] for k in all_keys))
    inc_values = sorted(set(k[1] for k in all_keys))

    # Create 2D grids for hitrate@25, hitrate@50 and hitrate@100
    hitrate_25_grid = np.full((len(inc_values), len(min_values)), np.nan)
    hitrate_50_grid = np.full((len(inc_values), len(min_values)), np.nan)
    hitrate_100_grid = np.full((len(inc_values), len(min_values)), np.nan)

    # Fill the grids
    for (min_val, inc_val), data in results_dict.items():
        min_idx = min_values.index(min_val)
        inc_idx = inc_values.index(inc_val)

        if data['hitrate_25'] is not None:
            hitrate_25_grid[inc_idx, min_idx] = data['hitrate_25']
        if data['hitrate_50'] is not None:
            hitrate_50_grid[inc_idx, min_idx] = data['hitrate_50']
        if data['hitrate_100'] is not None:
            hitrate_100_grid[inc_idx, min_idx] = data['hitrate_100']

    return min_values, inc_values, hitrate_25_grid, hitrate_50_grid, hitrate_100_grid


def plot_heatmaps(results, overlap, output_dir=None):
    """Create and save heatmaps for a specific overlap value.

    Args:
        results: Results dict for this overlap value
        overlap: The overlap value
        output_dir: Directory to save plots (defaults to current directory)
    """
    min_values, inc_values, hitrate_25_grid, hitrate_50_grid, hitrate_100_grid = create_heatmap_data(results)

    if min_values is None:
        print(f"No data for overlap={overlap}")
        return

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    # Plot hitrate@25
    im1 = ax1.imshow(hitrate_25_grid, aspect='auto', cmap='viridis',
                     interpolation='nearest', origin='lower')
    ax1.set_xlabel('Min Length', fontsize=12)
    ax1.set_ylabel('Increment', fontsize=12)
    ax1.set_title(f'Hitrate@25 (Overlap={overlap})', fontsize=14, fontweight='bold')

    # Set tick labels
    ax1.set_xticks(range(len(min_values)))
    ax1.set_xticklabels(min_values, rotation=45, ha='right')
    ax1.set_yticks(range(len(inc_values)))
    ax1.set_yticklabels(inc_values)

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Hitrate@25', fontsize=11)

    # Add text annotations on heatmap
    for i in range(len(inc_values)):
        for j in range(len(min_values)):
            if not np.isnan(hitrate_25_grid[i, j]):
                text = ax1.text(j, i, f'{hitrate_25_grid[i, j]:.2f}',
                               ha="center", va="center", color="white", fontsize=8)

    # Plot hitrate@50
    im2 = ax2.imshow(hitrate_50_grid, aspect='auto', cmap='viridis',
                     interpolation='nearest', origin='lower')
    ax2.set_xlabel('Min Length', fontsize=12)
    ax2.set_ylabel('Increment', fontsize=12)
    ax2.set_title(f'Hitrate@50 (Overlap={overlap})', fontsize=14, fontweight='bold')

    # Set tick labels
    ax2.set_xticks(range(len(min_values)))
    ax2.set_xticklabels(min_values, rotation=45, ha='right')
    ax2.set_yticks(range(len(inc_values)))
    ax2.set_yticklabels(inc_values)

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Hitrate@50', fontsize=11)

    # Add text annotations on heatmap
    for i in range(len(inc_values)):
        for j in range(len(min_values)):
            if not np.isnan(hitrate_50_grid[i, j]):
                text = ax2.text(j, i, f'{hitrate_50_grid[i, j]:.2f}',
                               ha="center", va="center", color="white", fontsize=8)

    # Plot hitrate@100
    im3 = ax3.imshow(hitrate_100_grid, aspect='auto', cmap='viridis',
                     interpolation='nearest', origin='lower')
    ax3.set_xlabel('Min Length', fontsize=12)
    ax3.set_ylabel('Increment', fontsize=12)
    ax3.set_title(f'Hitrate@100 (Overlap={overlap})', fontsize=14, fontweight='bold')

    # Set tick labels
    ax3.set_xticks(range(len(min_values)))
    ax3.set_xticklabels(min_values, rotation=45, ha='right')
    ax3.set_yticks(range(len(inc_values)))
    ax3.set_yticklabels(inc_values)

    # Add colorbar
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Hitrate@100', fontsize=11)

    # Add text annotations on heatmap
    for i in range(len(inc_values)):
        for j in range(len(min_values)):
            if not np.isnan(hitrate_100_grid[i, j]):
                text = ax3.text(j, i, f'{hitrate_100_grid[i, j]:.2f}',
                               ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()

    # Save figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'heatmap_overlap_{overlap}.png')
    else:
        output_path = f'heatmap_overlap_{overlap}.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent

    # Load all grid search results
    print("Loading grid search results...")
    all_results = load_grid_search_results(script_dir)

    if not all_results:
        print("No results found!")
        return

    # Print summary
    print(f"\nFound results for {len(all_results)} overlap values:")
    for overlap in sorted(all_results.keys()):
        print(f"  Overlap {overlap}: {len(all_results[overlap])} configurations")

    # Create heatmaps for specific overlap values
    print("\nGenerating heatmaps...")
    target_overlaps = [0, 50, 100, 150, 200]
    for overlap in target_overlaps:
        if overlap in all_results:
            print(f"\nProcessing overlap={overlap}...")
            plot_heatmaps(all_results[overlap], overlap, output_dir=script_dir)
        else:
            print(f"\nWarning: No data found for overlap={overlap}")

    print("\nDone!")


if __name__ == "__main__":
    main()
