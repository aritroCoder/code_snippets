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
    """JIT-compiled random walk for maximum speed - RGB mode"""
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

@jit(nopython=True)
def random_walk_hsv(canvas, x, y, h, s, v, steps, width, height):
    """JIT-compiled random walk in HSV color space"""
    for i in range(len(steps)):
        x = min(max(x + steps[i, 0], 0), width - 1)
        y = min(max(y + steps[i, 1], 0), height - 1)
        h = (h + steps[i, 2]) % 360  # Hue wraps around
        s = min(max(s + steps[i, 3], 0), 100)
        v = min(max(v + steps[i, 4], 0), 100)
        
        # Convert HSV to RGB (simplified for JIT)
        c = v * s / 10000.0
        x_val = c * (1 - abs((h / 60.0) % 2 - 1))
        m = v / 100.0 - c
        
        if h < 60:
            r, g, b = c, x_val, 0
        elif h < 120:
            r, g, b = x_val, c, 0
        elif h < 180:
            r, g, b = 0, c, x_val
        elif h < 240:
            r, g, b = 0, x_val, c
        elif h < 300:
            r, g, b = x_val, 0, c
        else:
            r, g, b = c, 0, x_val
        
        canvas[y, x, 0] = int((r + m) * 255)
        canvas[y, x, 1] = int((g + m) * 255)
        canvas[y, x, 2] = int((b + m) * 255)
    return x, y, h, s, v

@jit(nopython=True)
def random_walk_drift(canvas, x, y, r, g, b, steps, drift_steps, width, height):
    """Random walk with occasional color drift jumps"""
    drift_idx = 0
    for i in range(len(steps)):
        x = min(max(x + steps[i, 0], 0), width - 1)
        y = min(max(y + steps[i, 1], 0), height - 1)
        
        # Regular step
        r = min(max(r + steps[i, 2], 0), 255)
        g = min(max(g + steps[i, 3], 0), 255)
        b = min(max(b + steps[i, 4], 0), 255)
        
        # Apply drift every 100 steps
        if i % 100 == 0 and drift_idx < len(drift_steps):
            r = min(max(r + drift_steps[drift_idx, 0], 0), 255)
            g = min(max(g + drift_steps[drift_idx, 1], 0), 255)
            b = min(max(b + drift_steps[drift_idx, 2], 0), 255)
            drift_idx += 1
        
        canvas[y, x, 0] = r
        canvas[y, x, 1] = g
        canvas[y, x, 2] = b
    return x, y, r, g, b

@jit(nopython=True)
def random_walk_complementary(canvas, x, y, r, g, b, steps, flip_mask, width, height):
    """Random walk with occasional complementary color flips"""
    for i in range(len(steps)):
        x = min(max(x + steps[i, 0], 0), width - 1)
        y = min(max(y + steps[i, 1], 0), height - 1)
        r = min(max(r + steps[i, 2], 0), 255)
        g = min(max(g + steps[i, 3], 0), 255)
        b = min(max(b + steps[i, 4], 0), 255)
        
        # Flip to complementary color based on mask
        if flip_mask[i]:
            r, g, b = 255 - r, 255 - g, 255 - b
        
        canvas[y, x, 0] = r
        canvas[y, x, 1] = g
        canvas[y, x, 2] = b
    return x, y, r, g, b

@jit(nopython=True)
def random_walk_temperature(canvas, x, y, r, g, b, steps, width, height, center_x, center_y):
    """Random walk with temperature-based colors (distance from center)"""
    for i in range(len(steps)):
        x = min(max(x + steps[i, 0], 0), width - 1)
        y = min(max(y + steps[i, 1], 0), height - 1)
        
        # Calculate distance from center
        dx = x - center_x
        dy = y - center_y
        dist = (dx * dx + dy * dy) ** 0.5
        max_dist = ((width/2) ** 2 + (height/2) ** 2) ** 0.5
        temp = int(255 * min(dist / max_dist, 1.0))
        
        # Hot (center) to cold (edges) gradient
        r = 255 - temp
        g = min(max(g + steps[i, 3], 0), 255)
        b = temp
        
        canvas[y, x, 0] = r
        canvas[y, x, 1] = g
        canvas[y, x, 2] = b
    return x, y, r, g, b

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Generate a random walk image in 5D space (2D position + 3D color)',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Color Modes:
  rgb           Classic random walk in RGB color space (default)
                Each color channel moves ±1 independently, creating organic patterns
  
  hsv           Random walk in HSV (Hue-Saturation-Value) color space
                Produces more vibrant, rainbow-like patterns with smooth hue transitions
  
  drift         RGB walk with occasional large color jumps
                Adds dramatic color shifts every ~100 steps for more variation
  
  complementary RGB walk with occasional flips to complementary colors
                Periodically inverts colors (255-R, 255-G, 255-B) for bold contrasts
  
  temperature   Colors based on distance from canvas center
                Hot colors (red) near center, cool colors (blue) at edges

Examples:
  %(prog)s -r 1080p                    # Generate 1080p image with default RGB mode
  %(prog)s -r 4k -m hsv                # Generate 4K image with HSV colors
  %(prog)s -r 2k -m drift -o art.png   # Generate 2K with color drift mode
''')

parser.add_argument('--resolution', '-r', choices=['800', '1080p', '2k', '4k'], 
                    default='800', 
                    help='Resolution preset: 800 (800x800, 1B steps), 1080p (1920x1080, 10B steps), '
                         '2k (2560x1440, 15B steps), 4k (3840x2160, 20B steps) [default: 800]')
parser.add_argument('--mode', '-m', choices=['rgb', 'hsv', 'drift', 'complementary', 'temperature'],
                    default='rgb',
                    help='Color generation mode [default: rgb]')
parser.add_argument('--output', '-o', default='random_walk.png', 
                    help='Output filename [default: random_walk.png]')
args = parser.parse_args()

preset = PRESETS[args.resolution]
width, height = preset['width'], preset['height']
n_steps = preset['steps']

print(f"Generating {width}x{height} image with {n_steps:,} steps using '{args.mode}' mode...")
print(f"Output: {args.output}")

# Initialize canvas
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Start in the middle
x, y = width // 2, height // 2

# Initialize color values based on mode
if args.mode == 'hsv':
    h, s, v = 180, 50, 50  # Mid hue, moderate saturation and value
else:
    r, g, b = 128, 128, 128  # Mid gray

batch_size = 10_000_000

for i in tqdm(range(0, n_steps, batch_size), desc="Random walk"):
    current_batch = min(batch_size, n_steps - i)
    
    steps = np.random.randint(0, 2, size=(current_batch, 5), dtype=np.int8) * 2 - 1
    
    if args.mode == 'rgb':
        x, y, r, g, b = random_walk(canvas, x, y, r, g, b, steps, width, height)
    
    elif args.mode == 'hsv':
        x, y, h, s, v = random_walk_hsv(canvas, x, y, h, s, v, steps, width, height)
    
    elif args.mode == 'drift':
        # Generate drift jumps (larger color shifts)
        drift_count = current_batch // 100
        drift_steps = np.random.randint(-10, 11, size=(drift_count, 3), dtype=np.int8)
        x, y, r, g, b = random_walk_drift(canvas, x, y, r, g, b, steps, drift_steps, width, height)
    
    elif args.mode == 'complementary':
        # Generate random flip mask (0.1% chance per step)
        flip_mask = np.random.random(current_batch) < 0.001
        x, y, r, g, b = random_walk_complementary(canvas, x, y, r, g, b, steps, flip_mask, width, height)
    
    elif args.mode == 'temperature':
        center_x, center_y = width // 2, height // 2
        x, y, r, g, b = random_walk_temperature(canvas, x, y, r, g, b, steps, width, height, center_x, center_y)

print(f"Saving to {args.output}...")
Image.fromarray(canvas).save(args.output)
print("Done!")
