import numpy as np
from PIL import Image
from tqdm import tqdm
from numba import jit
import argparse

# Presets for different resolutions
PRESETS = {
    '800': {'width': 800, 'height': 800, 'steps': 1_000_000_000},
    '1080p': {'width': 1920, 'height': 1080, 'steps': 10_000_000_000},
    '2k': {'width': 2560, 'height': 1440, 'steps': 15_000_000_000},
    '4k': {'width': 3840, 'height': 2160, 'steps': 20_000_000_000},
}

@jit(nopython=True)
def random_walk(canvas, x, y, r, g, b, steps, width, height):
    """JIT-compiled random walk for maximum speed"""
    for i in range(len(steps)):
        x = min(max(x + steps[i, 0], 0), width - 1)
        y = min(max(y + steps[i, 1], 0), height - 1)
        r = min(max(r + steps[i, 2], 0), 255)
        g = min(max(g + steps[i, 3], 0), 255)
        b = min(max(b + steps[i, 4], 0), 255)
        canvas[y, x, 0] = r
        canvas[y, x, 1] = g
        canvas[y, x, 2] = b
    return x, y, r, g, b

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate a random walk image in 5D space')
parser.add_argument('--resolution', '-r', choices=['800', '1080p', '2k', '4k'], 
                    default='1080p', help='Resolution preset (default: 1080p)')
parser.add_argument('--output', '-o', default='random_walk.png', 
                    help='Output filename (default: random_walk.png)')
args = parser.parse_args()

preset = PRESETS[args.resolution]
width, height = preset['width'], preset['height']
n_steps = preset['steps']

print(f"Generating {width}x{height} image with {n_steps:,} steps...")

# Initialize canvas
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Start in the middle
x, y = width // 2, height // 2
r, g, b = 128, 128, 128

batch_size = 10_000_000

for i in tqdm(range(0, n_steps, batch_size), desc="Random walk"):
    current_batch = min(batch_size, n_steps - i)
    
    steps = np.random.randint(0, 2, size=(current_batch, 5), dtype=np.int8) * 2 - 1
    
    x, y, r, g, b = random_walk(canvas, x, y, r, g, b, steps, width, height)

print(f"Saving to {args.output}...")
Image.fromarray(canvas).save(args.output)
print("Done!")
