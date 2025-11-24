import math
import numpy as np
import cv2
import os

class IRSceneryGenerator:
    """
    Simulates an Infrared (IR) environment.
    Maps Temperature (Kelvin/Celsius) to Radiance/Pixel Intensity.
    """
    def __init__(self, width=256, height=256, ambient_temp=20.0):
        self.width = width
        self.height = height
        self.ambient_temp = ambient_temp # Celsius
        self.objects = []
        
        # Initialize background with slight thermal gradient (ground usually warmer/cooler than sky)
        self.grid = np.full((height, width), self.ambient_temp, dtype=np.float32)

    def add_object(self, obj_dict):
        """Adds an object dictionary to the scene.

        Expected keys (required): 'x','y','vx','vy','temp','cooling_rate','type'
        Additional shape-specific keys:
          - for 'blob': 'size'
          - for 'rectangle': 'width','height'
          - for 'silhouette': 'scale'
        Optional pulsing keys: 'pulse_amplitude', 'pulse_frequency'
        """
        if 'type' not in obj_dict:
            raise ValueError("Object dictionary must include a 'type' field ('blob','rectangle','silhouette')")
        self.objects.append(obj_dict)

    def _gaussian_blob(self, x, y, size, temp_diff):
        """Creates a heat signature (blob) using 2D Gaussian function."""
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        # decay factor controls how sharp the heat drop-off is
        decay = 2 * (size ** 2)
        blob = temp_diff * np.exp(-((xx - x)**2 + (yy - y)**2) / decay)
        return blob

    def _rectangle(self, x, y, width, height, temp_diff):
        """Create a rectangular heat signature.

        x,y are center coordinates. width,height in pixels. temp_diff is delta T (°C).
        Returns an array with the rectangle contribution (float32).
        """
        rect = np.zeros((self.height, self.width), dtype=np.float32)
        # Compute integer bounds (clamped)
        x0 = int(round(x - width / 2))
        x1 = int(round(x + width / 2))
        y0 = int(round(y - height / 2))
        y1 = int(round(y + height / 2))

        x0_clamped = max(0, x0)
        x1_clamped = min(self.width, x1)
        y0_clamped = max(0, y0)
        y1_clamped = min(self.height, y1)

        if x0_clamped < x1_clamped and y0_clamped < y1_clamped:
            rect[y0_clamped:y1_clamped, x0_clamped:x1_clamped] = temp_diff

        # Soften edges with a small Gaussian to simulate thermal diffusion
        rect = cv2.GaussianBlur(rect, (3, 3), 0)
        return rect

    def _silhouette(self, x, y, scale, temp_diff):
        """Create a simple human-like silhouette heat signature.

        The silhouette is constructed from a head (small gaussian), torso (rectangle), and two legs (rectangles).
        scale controls overall size.
        """
        # base canvas
        body = np.zeros((self.height, self.width), dtype=np.float32)

        # Head: small gaussian above torso
        head_radius = max(3, int(6 * scale))
        head_y = y - int(12 * scale)
        head = self._gaussian_blob(x, head_y, head_radius, temp_diff * 0.9)
        body += head

        # Torso: rectangle
        torso_w = int(10 * scale)
        torso_h = int(18 * scale)
        torso = self._rectangle(x, y - int(2 * scale), torso_w, torso_h, temp_diff)
        body += torso

        # Legs: two narrow rectangles below torso
        leg_w = max(3, int(3 * scale))
        leg_h = int(12 * scale)
        leg_sep = int(4 * scale)
        left_leg = self._rectangle(x - leg_sep, y + int(10 * scale), leg_w, leg_h, temp_diff * 0.95)
        right_leg = self._rectangle(x + leg_sep, y + int(10 * scale), leg_w, leg_h, temp_diff * 0.95)
        body += left_leg + right_leg

        # Mild blur to simulate body heat diffusion
        body = cv2.GaussianBlur(body, (3, 3), 0)
        return body

    def update_physics(self, frame_number=0):
        """Updates object positions and creates the thermal map.

        frame_number is supplied for temperature pulsing calculations.
        """
        # Reset grid to ambient (plus slight random thermal fluctuation in air)
        base_noise = np.random.normal(0, 0.5, (self.height, self.width))
        self.grid = np.full((self.height, self.width), self.ambient_temp, dtype=np.float32) + base_noise

        for obj in self.objects:
            # Update Position with toroidal wrapping (toroidal topology)
            obj['x'] = (obj['x'] + obj['vx']) % self.width
            obj['y'] = (obj['y'] + obj['vy']) % self.height

            # Simulate Temperature Decay (Object cooling down slightly as it moves)
            if obj.get('cooling_rate', 0) > 0:
                obj['temp'] -= obj['cooling_rate']

            # Temperature pulsing (optional)
            pulse_variation = 0.0
            if 'pulse_amplitude' in obj and 'pulse_frequency' in obj:
                # Per-plan: temp_variation = pulse_amplitude × sin(2π × pulse_frequency × frame_number)
                pulse_variation = obj['pulse_amplitude'] * math.sin(2 * math.pi * obj['pulse_frequency'] * frame_number)

            # Delta T = (Object Temp + pulse) - Ambient Temp
            delta_t = (obj['temp'] + pulse_variation) - self.ambient_temp

            # Render heat signature based on object type
            obj_type = obj.get('type', 'blob')
            if obj_type == 'blob':
                size = obj.get('size', 8)
                heat_signature = self._gaussian_blob(obj['x'], obj['y'], size, delta_t)
            elif obj_type == 'rectangle':
                width = obj.get('width', 10)
                height = obj.get('height', 10)
                heat_signature = self._rectangle(obj['x'], obj['y'], width, height, delta_t)
            elif obj_type == 'silhouette':
                scale = obj.get('scale', 1.0)
                heat_signature = self._silhouette(obj['x'], obj['y'], scale, delta_t)
            else:
                # fallback to blob
                size = obj.get('size', 8)
                heat_signature = self._gaussian_blob(obj['x'], obj['y'], size, delta_t)

            # Add heat to grid (Linear superposition)
            self.grid += heat_signature

    def get_thermal_map(self):
        return self.grid

class IRSensor:
    """
    Simulates the camera hardware: Optics, Sensor Noise, and ADC.
    """
    def __init__(self, bit_depth=8):
        self.bit_depth = bit_depth
        self.max_val = (2**bit_depth) - 1

    def apply_optics(self, thermal_map):
        """Simulates atmospheric attenuation and lens diffusion (Blur)."""
        # 1) Blur represents heat diffusion or out-of-focus optics
        blurred = cv2.GaussianBlur(thermal_map, (5, 5), 0)

        # 2) Distance-based attenuation (exponential along y-axis)
        # distance_factor = y / height (0..1). Objects farther (larger y) are dimmer.
        yy = np.arange(0, blurred.shape[0], dtype=np.float32)
        distance_factor = yy / float(blurred.shape[0] - 1)
        # attenuation coefficient: moderate atmospheric conditions
        att_coeff = 0.3
        attenuation_row = np.exp(-att_coeff * distance_factor)
        attenuation_map = np.repeat(attenuation_row[:, np.newaxis], blurred.shape[1], axis=1)

        attenuated = blurred * attenuation_map
        return attenuated

    def apply_noise(self, image_array):
        """Adds sensor noise (Shot noise / Thermal noise)."""
        noise_level = np.random.normal(0, 2.0, image_array.shape) # 2.0 is sigma
        noisy_image = image_array + noise_level
        return noisy_image

    def capture(self, thermal_map, min_temp_range, max_temp_range, frame_number=0):
        """
         digitization: Maps physical temperature to pixel values (0-255).
        """
        # 1. Apply Optical Effects
        optical_img = self.apply_optics(thermal_map)
        
        # 2. Normalization (Mapping Temp to Dynamics Range)
        # Clip values to the sensor's dynamic range
        normalized = (optical_img - min_temp_range) / (max_temp_range - min_temp_range)
        normalized = np.clip(normalized, 0, 1)
        
        # Scale to bit depth (e.g., 255)
        pixel_data = normalized * self.max_val
        
        # 3. Add Sensor Noise
        pixel_data = self.apply_noise(pixel_data)
        
        # 4. Quantization (Convert float to int)
        return np.clip(pixel_data, 0, self.max_val).astype(np.uint8)

# --- MAIN PIPELINE ---
def main():
    # Configuration
    OUTPUT_DIR = "ir_dataset"
    FRAMES = 30
    WIDTH, HEIGHT = 256, 256
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Initialize Simulation
    sim = IRSceneryGenerator(width=WIDTH, height=HEIGHT, ambient_temp=20.0) # 20°C Background
    sensor = IRSensor(bit_depth=8)

    # Define Objects (Diverse set: blobs, silhouette, rectangle)
    # Frame counter for pulsing calculations
    frame_number = 0

    # Object 1: Hot Gaussian Blob (Vehicle Engine)
    sim.add_object({
        'x': 50, 'y': 100,
        'vx': 4, 'vy': 0,
        'type': 'blob',
        'size': 15,
        'temp': 50.0,  # 50°C
        'cooling_rate': 0.1,
        'pulse_amplitude': 1.5,
        'pulse_frequency': 0.5
    })

    # Object 2: Human Silhouette (Walking Person)
    sim.add_object({
        'x': 200, 'y': 50,
        'vx': -2, 'vy': 3,
        'type': 'silhouette',
        'scale': 1.0,
        'temp': 36.5,  # body temperature
        'cooling_rate': 0.0,
        'pulse_amplitude': 0.3,
        'pulse_frequency': 0.25
    })

    # Object 3: Rectangular Equipment (Machinery)
    sim.add_object({
        'x': 128, 'y': 180,
        'vx': 1, 'vy': -2,
        'type': 'rectangle',
        'width': 20, 'height': 30,
        'temp': 42.0,
        'cooling_rate': 0.05
    })

    # Object 4: Cool Gaussian Blob (Small Heat Source)
    sim.add_object({
        'x': 180, 'y': 220,
        'vx': -3, 'vy': -1,
        'type': 'blob',
        'size': 8,
        'temp': 32.0,
        'cooling_rate': 0.0,
        'pulse_amplitude': 2.0,
        'pulse_frequency': 1.0
    })

    print(f"Generating {FRAMES} frames in '{OUTPUT_DIR}'...")

    for i in range(FRAMES):
        # 1. Update Physics (Move objects, calc temperatures) with frame_number for pulsing
        sim.update_physics(frame_number)

        # 2. Get raw temperature grid
        raw_temp_map = sim.get_thermal_map()

        # 3. Sensor Capture (Normalize Temp 15°C to 55°C range to pixels)
        frame = sensor.capture(raw_temp_map, min_temp_range=15.0, max_temp_range=55.0, frame_number=frame_number)

        # 4. Save
        filename = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

        # Increment frame counter
        frame_number += 1

if __name__ == "__main__":
    main()