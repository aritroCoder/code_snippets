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

@jit(nopython=True)
def random_walk_psychedelic(canvas, x, y, phase, steps, width, height, center_x, center_y):
    """Psychedelic mode: wave interference patterns creating trippy visuals"""
    for i in range(len(steps)):
        x = min(max(x + steps[i, 0], 0), width - 1)
        y = min(max(y + steps[i, 1], 0), height - 1)
        phase = (phase + steps[i, 2]) % 1000
        
        # Calculate distance and angle from center
        dx = x - center_x
        dy = y - center_y
        dist = (dx * dx + dy * dy) ** 0.5
        angle = np.arctan2(dy, dx)
        
        # Create wave interference patterns
        wave1 = np.sin(dist * 0.05 + phase * 0.01)
        wave2 = np.sin(angle * 5 + phase * 0.02)
        wave3 = np.cos(dist * 0.03 - phase * 0.015)
        
        # Map waves to RGB with phase shifts
        r = int(127.5 + 127.5 * np.sin(wave1 * 3.14 + phase * 0.01))
        g = int(127.5 + 127.5 * np.sin(wave2 * 3.14 + phase * 0.01 + 2.09))
        b = int(127.5 + 127.5 * np.sin(wave3 * 3.14 + phase * 0.01 + 4.18))
        
        canvas[y, x, 0] = r
        canvas[y, x, 1] = g
        canvas[y, x, 2] = b
    return x, y, phase

@jit(nopython=True)
def random_walk_spiral(canvas, x, y, spiral_phase, steps, width, height, center_x, center_y):
    """Spiral mode: rotating color spirals radiating from center"""
    for i in range(len(steps)):
        x = min(max(x + steps[i, 0], 0), width - 1)
        y = min(max(y + steps[i, 1], 0), height - 1)
        spiral_phase = (spiral_phase + 1) % 10000
        
        # Calculate polar coordinates
        dx = x - center_x
        dy = y - center_y
        dist = (dx * dx + dy * dy) ** 0.5
        angle = np.arctan2(dy, dx)
        
        # Create spiraling color pattern
        spiral = (angle * 3 + dist * 0.1 + spiral_phase * 0.001) % (2 * 3.14159)
        
        # Map to vibrant RGB
        hue = (spiral / (2 * 3.14159)) * 360
        if hue < 60:
            r, g, b = 255, int(hue * 4.25), 0
        elif hue < 120:
            r, g, b = int(255 - (hue - 60) * 4.25), 255, 0
        elif hue < 180:
            r, g, b = 0, 255, int((hue - 120) * 4.25)
        elif hue < 240:
            r, g, b = 0, int(255 - (hue - 180) * 4.25), 255
        elif hue < 300:
            r, g, b = int((hue - 240) * 4.25), 0, 255
        else:
            r, g, b = 255, 0, int(255 - (hue - 300) * 4.25)
        
        canvas[y, x, 0] = r
        canvas[y, x, 1] = g
        canvas[y, x, 2] = b
    return x, y, spiral_phase

@jit(nopython=True)
def random_walk_kaleidoscope(canvas, x, y, r, g, b, steps, width, height, center_x, center_y):
    """Kaleidoscope mode: symmetrical mirrored patterns"""
    for i in range(len(steps)):
        x = min(max(x + steps[i, 0], 0), width - 1)
        y = min(max(y + steps[i, 1], 0), height - 1)
        r = min(max(r + steps[i, 2], 0), 255)
        g = min(max(g + steps[i, 3], 0), 255)
        b = min(max(b + steps[i, 4], 0), 255)
        
        # Calculate angle from center
        dx = x - center_x
        dy = y - center_y
        angle = np.arctan2(dy, dx)
        
        # Create 6-fold symmetry (kaleidoscope effect)
        folded_angle = abs((angle % (3.14159 / 3)) - 3.14159 / 6)
        
        # Modulate colors based on symmetry
        symmetry_factor = folded_angle / (3.14159 / 6)
        r_mod = int(r * (0.5 + 0.5 * symmetry_factor))
        g_mod = int(g * (0.5 + 0.5 * (1 - symmetry_factor)))
        b_mod = int(b * (0.5 + 0.5 * np.sin(symmetry_factor * 3.14159)))
        
        canvas[y, x, 0] = r_mod
        canvas[y, x, 1] = g_mod
        canvas[y, x, 2] = b_mod
    return x, y, r, g, b

@jit(nopython=True)
def random_walk_plasma(canvas, x, y, time, steps, width, height):
    """Plasma mode: organic flowing plasma effect"""
    for i in range(len(steps)):
        x = min(max(x + steps[i, 0], 0), width - 1)
        y = min(max(y + steps[i, 1], 0), height - 1)
        time = (time + 1) % 10000
        
        # Multiple sine waves create plasma effect
        plasma1 = np.sin(x * 0.02 + time * 0.001)
        plasma2 = np.sin(y * 0.03 + time * 0.002)
        plasma3 = np.sin((x + y) * 0.015 + time * 0.0015)
        plasma4 = np.sin(np.sqrt(x * x + y * y) * 0.01 + time * 0.001)
        
        plasma = (plasma1 + plasma2 + plasma3 + plasma4) / 4.0
        
        # Map plasma value to psychedelic colors
        r = int(127.5 + 127.5 * np.sin(plasma * 3.14159))
        g = int(127.5 + 127.5 * np.sin(plasma * 3.14159 + 2.09))
        b = int(127.5 + 127.5 * np.sin(plasma * 3.14159 + 4.18))
        
        canvas[y, x, 0] = r
        canvas[y, x, 1] = g
        canvas[y, x, 2] = b
    return x, y, time

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
  
  psychedelic   Wave interference patterns creating trippy, mind-bending visuals
                Multiple sine waves create flowing, organic hallucination-like effects
  
  spiral        Rotating color spirals radiating from center
                Creates mesmerizing vortex patterns with smooth color transitions
  
  kaleidoscope  6-fold symmetrical mirrored patterns
                Symmetry-based color modulation for hypnotic geometric effects
  
  plasma        Organic flowing plasma effect with multiple sine waves
                Classic demo-scene plasma effect with psychedelic color cycling

Examples:
  %(prog)s -r 1080p                    # Generate 1080p image with default RGB mode
  %(prog)s -r 4k -m psychedelic        # Generate 4K with mind-bending wave patterns
  %(prog)s -r 2k -m spiral -o art.png  # Generate 2K with spiral vortex effect
  %(prog)s -m plasma                   # Quick 800x800 plasma effect
''')

parser.add_argument('--resolution', '-r', choices=['800', '1080p', '2k', '4k'], 
                    default='800', 
                    help='Resolution preset: 800 (800x800, 1B steps), 1080p (1920x1080, 10B steps), '
                         '2k (2560x1440, 15B steps), 4k (3840x2160, 20B steps) [default: 800]')
parser.add_argument('--mode', '-m', choices=['rgb', 'hsv', 'drift', 'complementary', 'temperature', 
                                             'psychedelic', 'spiral', 'kaleidoscope', 'plasma'],
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
elif args.mode in ['psychedelic', 'spiral', 'plasma']:
    phase = 0  # Phase for wave-based modes
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
    
    elif args.mode == 'psychedelic':
        center_x, center_y = width // 2, height // 2
        x, y, phase = random_walk_psychedelic(canvas, x, y, phase, steps, width, height, center_x, center_y)
    
    elif args.mode == 'spiral':
        center_x, center_y = width // 2, height // 2
        x, y, phase = random_walk_spiral(canvas, x, y, phase, steps, width, height, center_x, center_y)
    
    elif args.mode == 'kaleidoscope':
        center_x, center_y = width // 2, height // 2
        x, y, r, g, b = random_walk_kaleidoscope(canvas, x, y, r, g, b, steps, width, height, center_x, center_y)
    
    elif args.mode == 'plasma':
        x, y, phase = random_walk_plasma(canvas, x, y, phase, steps, width, height)

print(f"Saving to {args.output}...")
Image.fromarray(canvas).save(args.output)
print("Done!")
