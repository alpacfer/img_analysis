# EXERCISE 1
import math

a = 10
b = 3

theta = math.atan2(b, a)
# In degrees
theta_deg = math.degrees(theta)
print(theta_deg)


# EXERCISE 2
def camera_b_distance(f, g):
    return f * g / (g - f)


f = 15  # mm

print(camera_b_distance(f, 100)  # 0.1 m
      , camera_b_distance(f, 1000)  # 1 m
      , camera_b_distance(f, 5000)  # 5 m
      , camera_b_distance(f, 15000))  # 15 m

# EXERCISE 3
f = 5  # mm
height = 1800  # mm
g = 5000  # mm
x_ccd = 6.4  # mm
x_pixels = 640  # pixels
y_ccd = 4.8  # mm
y_pixels = 480  # pixels

# Distance from the lens to the inside image
b = camera_b_distance(f, g)
print(b)

# Height of the image
G = height / 2
B = G * b / g
image_height = B * 2
print(image_height)

# Size of a single pixel
x_pixel_size = x_ccd / x_pixels
y_pixel_size = y_ccd / y_pixels
print(x_pixel_size, y_pixel_size)

# Height in pixels of the image
image_height_pixels = image_height / y_pixel_size
print(image_height_pixels)

# Horizontal field of view in degrees
horizontal_fov = 2 * math.atan2(x_ccd / 2, f)
horizontal_fov_deg = math.degrees(horizontal_fov)
print(horizontal_fov_deg)

# Vertical field of view in degrees
vertical_fov = 2 * math.atan2(y_ccd / 2, f)
vertical_fov_deg = math.degrees(vertical_fov)
print(vertical_fov_deg)
