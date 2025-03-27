# ---
# title: "Film Negative Converter"
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Film Negative Converter
#
# This notebook converts film negatives to positive images with different conversion presets similar to Negative Lab Pro.

# %%
# Import necessary libraries
import numpy as np
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage import exposure, color, filters

# %% [markdown]
# ## Step 1: Load the Test Image

# %%
# Load the test image
test_img_path = 'test.JPG'
negative = cv2.imread(test_img_path)
negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Display the original negative
plt.figure(figsize=(10, 8))
plt.imshow(negative)
plt.title('Original Film Negative')
plt.axis('off')

# %% [markdown]
# ## Step 2: Film Border Detection and Base Color Extraction

# %%
def detect_film_border(image, threshold_value=20, min_border_width=10, inner_edge_threshold=40):
    """Detect the film border and extract the base color
    
    This function detects both the hard scanner edge and the softer inner edge
    that surrounds the actual image content.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold to separate the film from scanner border (hard edge)
    _, binary_outer = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the binary image
    kernel = np.ones((5, 5), np.uint8)
    binary_outer = cv2.morphologyEx(binary_outer, cv2.MORPH_CLOSE, kernel)
    binary_outer = cv2.morphologyEx(binary_outer, cv2.MORPH_OPEN, kernel)
    
    # Find contours for the outer edge
    contours_outer, _ = cv2.findContours(binary_outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_outer:
        return None, None, None, None
    
    # Get the largest contour (should be the outer film edge)
    largest_contour_outer = max(contours_outer, key=cv2.contourArea)
    
    # Create a mask for the outer film area
    mask_outer = np.zeros_like(gray)
    cv2.drawContours(mask_outer, [largest_contour_outer], 0, 255, -1)
    
    # Extract the outer film area
    x_outer, y_outer, w_outer, h_outer = cv2.boundingRect(largest_contour_outer)
    film_area_outer = image[y_outer:y_outer+h_outer, x_outer:x_outer+w_outer]
    gray_film_area = cv2.cvtColor(film_area_outer, cv2.COLOR_RGB2GRAY)
    
    # Apply a different threshold to detect the inner edge (softer edge)
    _, binary_inner = cv2.threshold(gray_film_area, inner_edge_threshold, 255, cv2.THRESH_BINARY)
    binary_inner = cv2.morphologyEx(binary_inner, cv2.MORPH_CLOSE, kernel)
    binary_inner = cv2.morphologyEx(binary_inner, cv2.MORPH_OPEN, kernel)
    
    # Find contours for the inner edge
    contours_inner, _ = cv2.findContours(binary_inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a border mask (area between outer and inner edge)
    if contours_inner:
        # Get the largest inner contour
        largest_contour_inner = max(contours_inner, key=cv2.contourArea)
        
        # Create a mask for the inner film area
        mask_inner = np.zeros_like(gray_film_area)
        cv2.drawContours(mask_inner, [largest_contour_inner], 0, 255, -1)
        
        # Get the inner film area bounding box
        x_inner, y_inner, w_inner, h_inner = cv2.boundingRect(largest_contour_inner)
        
        # Adjust inner coordinates to be relative to the original image
        x_inner += x_outer
        y_inner += y_outer
        
        # Create a combined mask for visualization
        mask_combined = np.zeros_like(gray)
        cv2.rectangle(mask_combined, (x_outer, y_outer), (x_outer + w_outer, y_outer + h_outer), 128, 2)
        cv2.rectangle(mask_combined, (x_inner, y_inner), (x_inner + w_inner, y_inner + h_inner), 255, 2)
        
        # Create a border mask (between outer and inner edge)
        border_mask = np.zeros_like(gray)
        cv2.rectangle(border_mask, (x_outer, y_outer), (x_outer + w_outer, y_outer + h_outer), 255, -1)
        cv2.rectangle(border_mask, (x_inner, y_inner), (x_inner + w_inner, y_inner + h_inner), 0, -1)
    else:
        # If no inner contour, use the outer contour with a margin
        border_margin = 20
        mask_inner = mask_outer.copy()
        x_inner = x_outer + border_margin
        y_inner = y_outer + border_margin
        w_inner = w_outer - 2 * border_margin
        h_inner = h_outer - 2 * border_margin
        mask_combined = mask_outer
        
        # Create a border mask
        border_mask = np.zeros_like(gray)
        cv2.rectangle(border_mask, (x_outer, y_outer), (x_outer + w_outer, y_outer + h_outer), 255, -1)
        cv2.rectangle(border_mask, (x_inner, y_inner), (x_inner + w_inner, y_inner + h_inner), 0, -1)
    
    # Get the film base color from the border
    border_pixels = image[border_mask > 0]
    if len(border_pixels) > 0:
        film_base_color = np.median(border_pixels, axis=0)
    else:
        film_base_color = np.array([0, 0, 0])
    
    # Return all the necessary information
    return mask_combined, film_base_color, (x_outer, y_outer, w_outer, h_outer), (x_inner, y_inner, w_inner, h_inner)

# %%
# Apply the film border detection
film_mask, film_base_color, film_bbox, inner_bbox = detect_film_border(negative)

# Display the results
if film_mask is not None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image with bounding box
    axes[0].imshow(negative)
    if film_bbox:
        x, y, w, h = film_bbox
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
    axes[0].set_title('Detected Film Area')
    axes[0].axis('off')
    
    # Film mask
    axes[1].imshow(film_mask, cmap='gray')
    axes[1].set_title('Film Mask')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Display the detected film base color
    print(f"Detected film base color (RGB): {film_base_color}")
    
    # Visualize the detected film base color
    color_patch = np.ones((100, 100, 3), dtype=np.uint8)
    color_patch[:, :] = film_base_color
    plt.figure(figsize=(2, 2))
    plt.imshow(color_patch)
    plt.title('Film Base Color')
    plt.axis('off')
else:
    print("Could not detect film border.")

# %% [markdown]
# ## Step 3: Negative to Positive Conversion Algorithm

# %%
def apply_cmyk_adjustments(img, cyan=0, magenta=0, yellow=0, black=0):
    """Apply CMYK adjustments to the image"""
    # Convert values to the 0-1 range adjustments
    c_adj = cyan / 100.0
    m_adj = magenta / 100.0
    y_adj = yellow / 100.0
    k_adj = black / 100.0
    
    # Work with float image
    img_float = img.astype(np.float32) / 255.0
    
    # Apply adjustments to RGB channels based on CMYK model
    # Cyan affects blue and green (reduces red)
    if c_adj != 0:
        img_float[:,:,0] = np.clip(img_float[:,:,0] * (1 - c_adj), 0, 1)
    
    # Magenta affects red and blue (reduces green)
    if m_adj != 0:
        img_float[:,:,1] = np.clip(img_float[:,:,1] * (1 - m_adj), 0, 1)
    
    # Yellow affects red and green (reduces blue)
    if y_adj != 0:
        img_float[:,:,2] = np.clip(img_float[:,:,2] * (1 - y_adj), 0, 1)
    
    # Black affects all channels
    if k_adj != 0:
        img_float = np.clip(img_float * (1 - k_adj), 0, 1)
    
    # Convert back to 8-bit
    return (img_float * 255).astype(np.uint8)

def convert_negative_to_positive(negative_img, film_base_color, preset='standard', 
                                cyan=0, magenta=0, yellow=0, black=0):
    """Convert negative to positive using the film base color as reference"""
    # Make a copy of the image to avoid modifying the original
    img = negative_img.copy().astype(np.float32) / 255.0
    
    # Normalize film base color to 0-1 range
    base_color = film_base_color.astype(np.float32) / 255.0
    
    # Invert the image
    img = 1.0 - img
    
    # Apply different presets
    if preset == 'standard':
        # Standard conversion with auto white balance
        for c in range(3):
            if base_color[c] > 0:
                img[:,:,c] = img[:,:,c] / base_color[c]
        
        # Auto contrast stretch - process each channel separately
        for c in range(3):
            p2, p98 = np.percentile(img[:,:,c], (2, 98))
            img[:,:,c] = exposure.rescale_intensity(img[:,:,c], in_range=(p2, p98), out_range=(0, 1))
    
    elif preset == 'frontier':
        # Frontier conversion - warmer tones, higher contrast
        for c in range(3):
            if base_color[c] > 0:
                img[:,:,c] = img[:,:,c] / base_color[c]
        
        # Warm color balance adjustment - made less blue
        img[:,:,0] *= 1.15  # Increase red more
        img[:,:,1] *= 1.05  # Slightly increase green
        img[:,:,2] *= 0.85  # Decrease blue more
        
        # Higher contrast
        img = exposure.adjust_gamma(img, 1.2)
        
        # Instead of using equalize_adapthist which has strict range requirements,
        # use a combination of contrast stretching and exposure adjustment
        for c in range(3):
            # Ensure values are in [0, 1] range
            img[:,:,c] = np.clip(img[:,:,c], 0, 1)
            # Apply contrast stretching
            p2, p98 = np.percentile(img[:,:,c], (2, 98))
            img[:,:,c] = exposure.rescale_intensity(img[:,:,c], in_range=(p2, p98), out_range=(0, 1))
            # Add a bit more local contrast
            img[:,:,c] = exposure.adjust_sigmoid(img[:,:,c], cutoff=0.5, gain=7)
    
    elif preset == 'noritsu':
        # Noritsu conversion - neutral tones, more saturated
        for c in range(3):
            if base_color[c] > 0:
                img[:,:,c] = img[:,:,c] / base_color[c]
        
        # Adjust color balance - reduce blue cast
        img[:,:,0] *= 1.1   # Increase red
        img[:,:,1] *= 1.05  # Slightly increase green
        img[:,:,2] *= 0.85  # Decrease blue
                
        # Replace equalize_hist with a more controlled approach
        for c in range(3):
            # Ensure values are in [0, 1] range
            img[:,:,c] = np.clip(img[:,:,c], 0, 1)
            # Apply mild contrast enhancement
            p5, p95 = np.percentile(img[:,:,c], (5, 95))
            img[:,:,c] = exposure.rescale_intensity(img[:,:,c], in_range=(p5, p95), out_range=(0, 1))
        
        # Increase saturation
        hsv = color.rgb2hsv(img)
        hsv[:,:,1] *= 1.2  # Increase saturation
        img = color.hsv2rgb(hsv)
    
    elif preset == 'hasselblad':
        # Hasselblad X5 - high dynamic range, clean colors
        for c in range(3):
            if base_color[c] > 0:
                img[:,:,c] = img[:,:,c] / base_color[c]
        
        # Reduce blue cast for Hasselblad preset
        img[:,:,0] *= 1.15  # Increase red
        img[:,:,1] *= 1.05  # Slightly increase green
        img[:,:,2] *= 0.8   # Decrease blue significantly
        
        # Enhanced dynamic range - process each channel separately
        for c in range(3):
            img[:,:,c] = exposure.rescale_intensity(img[:,:,c], in_range=(0.05, 0.95), out_range=(0, 1))
        
        # Fine contrast adjustment
        img = exposure.adjust_sigmoid(img, cutoff=0.5, gain=8)
    
    # Clip values to ensure they're in the valid range
    img = np.clip(img, 0, 1)
    
    # Convert back to 0-255 range
    img = (img * 255).astype(np.uint8)
    
    # Apply any CMYK adjustments
    if any([cyan != 0, magenta != 0, yellow != 0, black != 0]):
        img = apply_cmyk_adjustments(img, cyan, magenta, yellow, black)
    
    return img

# %% [markdown]
# ## Step 4: Apply Different Presets

# %%
# Apply different presets and display results
if film_mask is not None and film_bbox is not None:
    # Extract the film area
    x, y, w, h = film_bbox
    film_area = negative[y:y+h, x:x+w]
    
    # Create different preset conversions
    presets = ['standard', 'frontier', 'noritsu', 'hasselblad']
    converted_images = {}
    
    for preset in presets:
        converted = convert_negative_to_positive(film_area, film_base_color, preset=preset)
        converted_images[preset] = converted
    
    # Display all presets
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, preset in enumerate(presets):
        axes[i].imshow(converted_images[preset])
        axes[i].set_title(f'{preset.capitalize()} Preset')
        axes[i].axis('off')
    
    plt.tight_layout()

# %% [markdown]
# ## Step 5: Interactive Preset Selector with Fine-tuning Controls

# %%
def interactive_converter(preset, contrast=1.0, saturation=1.0, brightness=1.0, highlight_recovery=0.0):
    """Interactive interface for conversion parameters"""
    global negative, film_base_color, inner_bbox
    
    try:
        # Extract the inner film area
        x, y, w, h = inner_bbox
        film_area = negative[y:y+h, x:x+w]
        
        # Convert the negative to positive
        result = convert_negative_to_positive(
            film_area, 
            film_base_color,
            preset=preset,
            contrast=contrast,
            saturation=saturation,
            brightness=brightness,
            highlight_recovery=highlight_recovery
        )
        
        # Display the conversion
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Original image
        axes[0].imshow(film_area)
        axes[0].set_title('Original Negative')
        axes[0].axis('off')
        
        # Converted image
        axes[1].imshow(result)
        axes[1].set_title(f'Converted with {preset.capitalize()} Preset')
        axes[1].axis('off')
        
        # Parameters displayed in title
        plt.suptitle(f"Contrast: {contrast:.1f}, Saturation: {saturation:.1f}, Brightness: {brightness:.1f}, Highlight Recovery: {highlight_recovery:.1f}")
        
        plt.tight_layout()
        return result
    except Exception as e:
        print(f"Error in conversion: {str(e)}")
        return None

# %%
# Create the interactive interface
# Disabled for standalone app - not compatible with Tkinter
# if film_mask is not None and film_bbox is not None:
#     interact(
#         interactive_converter,
#         preset=widgets.Dropdown(
#             options=['standard', 'frontier', 'noritsu', 'hasselblad'],
#             value='standard',
#             description='Preset:'
#         ),
#         contrast=widgets.FloatSlider(min=0.5, max=2.0, step=0.1, value=1.0, description='Contrast:'),
#         saturation=widgets.FloatSlider(min=0.5, max=2.0, step=0.1, value=1.0, description='Saturation:'),
#         brightness=widgets.FloatSlider(min=0.5, max=2.0, step=0.1, value=1.0, description='Brightness:'),
#         highlight_recovery=widgets.FloatSlider(min=0.0, max=1.0, step=0.1, value=0.0, description='Highlight Recovery:')
#     )
# else:
#     print("Cannot create interactive interface: Film area not detected.")

# %% [markdown]
# ## Step 6: Save Final Converted Image

# %%
def save_converted_image(preset='standard', contrast=1.0, saturation=1.0, brightness=1.0, 
                         highlight_recovery=0.0, output_path=None):
    """Save the converted image to a file"""
    global negative, film_base_color, inner_bbox
    
    try:
        # Extract the inner film area
        x, y, w, h = inner_bbox
        film_area = negative[y:y+h, x:x+w]
        
        # Convert the negative to positive
        result = convert_negative_to_positive(
            film_area, 
            film_base_color,
            preset=preset,
            contrast=contrast,
            saturation=saturation,
            brightness=brightness,
            highlight_recovery=highlight_recovery
        )
        
        if output_path is None:
            output_path = f"converted_{preset}.jpg"
        
        # Save using OpenCV
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_bgr)
        
        print(f"Saved converted image to {output_path}")
        
        # Visualize the saved image
        plt.figure(figsize=(10, 8))
        plt.imshow(result)
        plt.title(f'Saved Image: {output_path}')
        plt.axis('off')
        
        return output_path
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return None

# %%
# Example: Save an image with the Frontier preset
# save_converted_image(preset='frontier', contrast=1.2, saturation=1.1, 
#                     output_path='frontier_converted.jpg') 