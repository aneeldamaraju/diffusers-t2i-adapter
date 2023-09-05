import numpy as np
import cv2


import torch
import torch.optim as optim
from random import shuffle
import random

#Generate the three canonical shapes

def generate_centered_square(H, W, square_side_length):
    # Create a blank image
    image = np.zeros((H, W), dtype=np.uint8)
    
    # Determine the center coordinates
    center_x = W // 2
    center_y = H // 2
    
    # Calculate the start and end coordinates for the square
    start_x = center_x - square_side_length // 2
    start_y = center_y - square_side_length // 2
    end_x = start_x + square_side_length
    end_y = start_y + square_side_length
    
    # Set the square region to ones
    image[start_y:end_y, start_x:end_x] = 1
    
    return image

def generate_centered_triangle(H, W, triangle_side_length):
    # Create a blank image
    image = np.zeros((H, W), dtype=np.uint8)
    
    # Determine the center coordinates
    center_x = W // 2
    center_y = H // 2
    
    # Calculate the half side length of the triangle
    half_side_length = triangle_side_length // 2
    
    # Calculate the three vertex coordinates of the triangle
    vertex1 = (center_x, center_y - half_side_length)
    vertex2 = (center_x - half_side_length, center_y + half_side_length)
    vertex3 = (center_x + half_side_length, center_y + half_side_length)
    
    # Draw the triangle on the image
    points = np.array([vertex1, vertex2, vertex3], np.int32)
    cv2.fillPoly(image, [points], 1)
    
    return image

def generate_centered_circle(H, W, circle_radius):
    # Create a blank image
    image = np.zeros((H, W), dtype=np.uint8)
    
    # Determine the center coordinates
    center_x = W // 2
    center_y = H // 2
    
    # Draw the circle on the image
    cv2.circle(image, (center_x, center_y), circle_radius, 1, thickness=-1)
    
    return image



#HARD CODE NUMBER OF IMAGES FOR NOW (can easily change later if needed)


from torch.nn import functional as F
def apply_rotation_and_translation(image_tensor, rotation_angle,translation_x,translation_y):
    # Create the affine transformation matrix
    theta = torch.stack([torch.cos(rotation_angle), -torch.sin(rotation_angle),translation_x,
                         torch.sin(rotation_angle), torch.cos(rotation_angle),translation_y]).reshape(2, 3)
    
    # Apply the rotation transformation using grid_sample
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    grid = F.affine_grid(theta.unsqueeze(0), image_tensor.size())
    rotated_image_tensor = F.grid_sample(image_tensor, grid)
    
    return rotated_image_tensor.squeeze(0)

# Define the overlap function
def compute_overlap(im1, im2, rotation, translation_x, translation_y):
    # Apply rotation and translation transformations to square2
    im2_t = apply_rotation_and_translation(im2, rotation,translation_x,translation_y)
    
    # Compute the element-wise intersection of the two squares
    intersection = torch.multiply(im1, im2_t)
    union = im1+im2_t - intersection
    # Compute the overlap as a percentage of the total area
    overlap = intersection.sum() / im1.sum()
    return overlap,union

# Define the loss function
def loss_function(overlap,min=True):
    # Set the target overlap range
    if min:
      target_min = 0.3
    else:
      target_min = 0
    target_max = 0.45

    # Compute the loss using tensor operations
    loss =  (torch.clamp(overlap, target_min, target_max)-overlap)**2
    return loss

# Set the learning rate and number of optimization steps
def space_masks(shapes):
  '''
  Given a list of initial masks, space them out and decide if they are valid
  '''
  NUM_SHAPES = len(shapes)

  # Initialize the rotation and translation parameters
  init_params = torch.rand([NUM_SHAPES-1,3])*.1
  rotation = init_params[:,0].clone().requires_grad_(True)
  translation_x = init_params[:,1].clone().requires_grad_(True)
  translation_y = init_params[:,2].clone().requires_grad_(True)


  learning_rate = .1
  num_steps = 60

  # Create an optimizer
  optimizer = optim.SGD([rotation, translation_x, translation_y], lr=learning_rate)

  # Optimization loop
  valid = False
  for step in range(num_steps):
      overlaps = []
      optimizer.zero_grad() 
      loss = torch.tensor(0.0)
      for idx in range(NUM_SHAPES - 1):
        if idx == 0:
          overlap,union = compute_overlap(shapes[idx], shapes[idx+1], rotation[idx], translation_x[idx], translation_y[idx])
          loss = loss +  loss_function(overlap)
          overlaps.append(overlap.item())
        else:
          for j in range(idx+1):
            overlap,union = compute_overlap(shapes[j], shapes[idx+1], rotation[idx], translation_x[idx], translation_y[idx])
            loss = loss +  loss_function(overlap,(j-idx==1))
            overlaps.append(overlap.item())
      loss.backward()
      optimizer.step()

      # Print the progress
      # if (step % (num_steps/10) == 0):
        # print(f"Step [{step+1}/{num_steps}], Loss: {loss.item()}, Overlap: {overlaps}")
      if loss < 1e-1:
        valid = True
        # print(f"Complete on step {step}, with overlap {overlaps}")
        break
  out_shapes = [shapes[0]]
  for idx in range(NUM_SHAPES - 1):
    out_shapes.append(apply_rotation_and_translation(shapes[idx+1], rotation[idx],translation_x[idx],translation_y[idx])[0,...].detach())

  return out_shapes,valid


# Next step, make random colors, r g and b for the three shapes and assign them

import torch
import torch.nn.functional as F

def color_mask(binary_mask, color, wave=False):
    """
    Fill a binary mask with the specified color.

    Args:
        binary_mask (torch.Tensor): Binary mask tensor of shape (H, W) where H is the height and W is the width.
        color (str): Candidate color - 'red', 'green', or 'blue'.

    Returns:
        torch.Tensor: Image tensor of shape (3, H, W) with the filled color based on the binary mask.
    """
    # Validate the color input
    valid_colors = ['red', 'green', 'blue']
    if color not in valid_colors:
        raise ValueError("Invalid color. Choose from 'red', 'green', or 'blue'.")

    # Expand the binary mask to have 3 channels for RGB
    mask_rgb = binary_mask.unsqueeze(0).repeat(3, 1, 1)
    
     # Change the type of the images from solid color to wavy
    # Create a grid of size 512x512
    x, y = np.meshgrid(np.arange(512), np.arange(512))

    # Define the wavelength and angles
    wavelength = 64
    angles = [45, 90, 135, 180]

    # Initialize an empty list to store the sinusoid arrays
    sinusoid_arrays = []
    
    angle = angles[random.randint(0, 3)]
    # Generate sinusoids for each angle
    theta = np.radians(angle)

    # Compute the sinusoid
    sinusoid = (np.sin(2 * np.pi * (x * np.cos(theta) + y * np.sin(theta)) / wavelength) + 2)/3

    # Define the color channels
    if color == 'red':
        color_channels = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).view(3, 1, 1)
    elif color == 'green':
        color_channels = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).view(3, 1, 1)
    else:  # color == 'blue'
        color_channels = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).view(3, 1, 1)

    # Fill the mask with the specified color (TODO: add sinusoid to args, include args in this function or something similar)
    if wave:
        filled_image = mask_rgb * color_channels * sinusoid + 1-mask_rgb
    else:
        filled_image = mask_rgb * color_channels + 1-mask_rgb
    return filled_image




def generate_canonical_shapes():
  H = 512
  W = 512
  square_side_length = 150
  triangle_side_length = 150
  circle_radius = 100

  im1 = torch.tensor(generate_centered_square(H, W, square_side_length),dtype = torch.float32)
  im2 = torch.tensor(generate_centered_triangle(H, W, triangle_side_length),dtype = torch.float32)
  im3 = torch.tensor(generate_centered_circle(H, W, circle_radius),dtype = torch.float32)
  shapes = [im1,im2,im3]
  random.shuffle(shapes)
  NUM_SHAPES = 3
  masks, valid = space_masks(shapes)
  if not valid:
    print("invalid!")
  final_out = None
  colors = ['red','green','blue']
  polygons = []
  random.shuffle(colors)
  for idx in np.arange(NUM_SHAPES):
    if final_out == None:
      polygon = color_mask(masks[idx],colors[idx])
      polygons.append(polygon.clone().permute(1,2,0))
      final_out = polygon
    else:
      bool_tensor = masks[idx]>.5
      polygon = color_mask(masks[idx],colors[idx])
      polygons.append(polygon.permute(1,2,0))
      final_out[:,bool_tensor] = polygon[:,bool_tensor]

  final_out = final_out.permute(1,2,0)
  return final_out,[mask.unsqueeze(-1) for mask in masks],polygons



def combine_masks(binary_masks, index):
    '''
    input: binary_masks, size B x Layers x C x H x W
    '''
    
    combined_list = []
    for batch_id in range(binary_masks.shape[0]):
        combined_mask = torch.zeros_like(binary_masks[0,0,...], dtype=torch.uint8)
        for i in range(index):
            combined_mask = torch.logical_or(combined_mask, binary_masks[batch_id,i,...])
        combined_list.append(combined_mask)
    return torch.stack(combined_list)

if __name__ == "__main__":
    pass