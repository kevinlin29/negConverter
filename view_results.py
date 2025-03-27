"""
Film Negative Converter - Results Viewer

This script runs the film converter and saves the results as image files.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from film_converter import detect_film_border, convert_negative_to_positive

# Load the test image
test_img_path = 'test.JPG'
negative = cv2.imread(test_img_path)
negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Apply the film border detection with improved inner edge detection
print("Detecting film borders...")
film_mask, film_base_color, film_bbox, inner_bbox = detect_film_border(negative)

if film_mask is not None and inner_bbox is not None:
    # Display the detected borders
    display_img = negative.copy()
    x_outer, y_outer, w_outer, h_outer = film_bbox
    x_inner, y_inner, w_inner, h_inner = inner_bbox
    
    cv2.rectangle(display_img, (x_outer, y_outer), (x_outer + w_outer, y_outer + h_outer), (0, 255, 0), 2)
    cv2.rectangle(display_img, (x_inner, y_inner), (x_inner + w_inner, y_inner + h_inner), (255, 0, 0), 2)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(display_img)
    plt.title("Detected Borders (Green: Outer, Blue: Inner)")
    plt.axis('off')
    plt.savefig('detected_borders.jpg')
    
    # Extract the inner film area (actual image content)
    x, y, w, h = inner_bbox
    film_area = negative[y:y+h, x:x+w]
    
    # Print detected film base color
    print(f"Detected film base color: {film_base_color}")
    
    # Create different preset conversions
    presets = ['standard', 'frontier', 'noritsu', 'hasselblad']
    
    # Process and save each preset
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, preset in enumerate(presets):
        print(f"Applying {preset} preset...")
        # Test with no CMYK adjustments
        converted = convert_negative_to_positive(film_area, film_base_color, preset=preset)
        
        # Display in the subplot
        axes[i].imshow(converted)
        axes[i].set_title(f'{preset.capitalize()} Preset')
        axes[i].axis('off')
        
        # Save individual result
        output_filename = f"converted_{preset}.jpg"
        plt.imsave(output_filename, converted)
        print(f"Saved {output_filename}")
    
    plt.tight_layout()
    plt.savefig('all_presets.jpg')
    plt.show()
    
    # Show an example with CMYK adjustments on the standard preset
    print("\nTesting CMYK adjustments...")
    adjustments = [
        {"name": "No Adjustment", "C": 0, "M": 0, "Y": 0, "K": 0},
        {"name": "More Blue", "C": 20, "M": 0, "Y": -20, "K": 0},
        {"name": "More Warm", "C": -20, "M": 0, "Y": 15, "K": 0},
        {"name": "High Contrast", "C": 0, "M": 15, "Y": 0, "K": 20}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, adj in enumerate(adjustments):
        print(f"Applying adjustment: {adj['name']}...")
        converted = convert_negative_to_positive(
            film_area, film_base_color, preset='standard',
            cyan=adj['C'], magenta=adj['M'], yellow=adj['Y'], black=adj['K']
        )
        
        # Display in the subplot
        axes[i].imshow(converted)
        axes[i].set_title(f"Adjustment: {adj['name']}\nC:{adj['C']} M:{adj['M']} Y:{adj['Y']} K:{adj['K']}")
        axes[i].axis('off')
        
        # Save individual result
        output_filename = f"adjusted_{adj['name'].replace(' ', '_').lower()}.jpg"
        plt.imsave(output_filename, converted)
        print(f"Saved {output_filename}")
    
    plt.tight_layout()
    plt.savefig('cmyk_adjustments.jpg')
    plt.show()
    
    print("All conversions completed!")
else:
    print("Cannot convert image: Film area not detected.") 