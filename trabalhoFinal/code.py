import numpy as np
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    return np.asarray(image)

def compute_tangent_bitangent(width, height):
    tangents = np.zeros((height, width, 3))
    bitangents = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            tangents[y, x] = np.array([1, 0, 0])
            bitangents[y, x] = np.array([0, 1, 0])
    return tangents, bitangents

def decode_normal_map(normal_map):
    # Normalize the normal map values to range [-1, 1]
    normals = (normal_map / 255.0) * 2.0 - 1.0
    return normals

def calculate_lighting(base_texture, normals, light_dir, tangents, bitangents):
    height, width = normals.shape[:2]
    light_dir = light_dir / np.linalg.norm(light_dir)
    final_color = np.zeros((height, width, 3))
    
    for y in range(height):
        for x in range(width):
            T = tangents[y, x]
            B = bitangents[y, x]
            N = normals[y, x]
            
            # Compute the TBN matrix
            TBN = np.array([T, B, np.cross(T, B)])
            
            # Transform normal from tangent space to world space
            perturbed_normal = np.dot(TBN, N)
            perturbed_normal = perturbed_normal / np.linalg.norm(perturbed_normal)
            
            # Calculate the intensity using Lambertian reflection
            intensity = np.dot(perturbed_normal, -light_dir / np.linalg.norm(light_dir))
            intensity = np.clip(intensity, 0, 1) + 0.1
            #print(x, y, intensity)
            
            # Apply the intensity to the base texture
            final_color[y, x] = base_texture[y, x] * intensity
    
    return final_color

# Load your base texture and normal map
base_texture = load_image('brick_texture.tga')
normal_map = load_image('normal_map.tga')

# Print shapes to debug
print("Base texture shape:", base_texture.shape)
print("Normal map shape:", normal_map.shape)

height, width = base_texture.shape[:2]
tangents, bitangents = compute_tangent_bitangent(width, height)
normals = decode_normal_map(normal_map)

# Print a few normal values to debug
print("Sample normal values:", normals[0, 0], normals[100, 100])

light_dir = np.array([1, 1, -1])
final_color = calculate_lighting(base_texture, normals, light_dir, tangents, bitangents)

# Ensure the final color values are within [0, 255]
final_color = np.clip(final_color, 0, 255)

final_image = Image.fromarray(np.uint8(final_color))
final_image.save('output_with_normal_map.png')
