"""
Film Negative Converter UI

A streamlined user interface for the film negative converter with:
- Preset selection dropdown
- CMYK adjustment sliders
- Manual film base color selection
"""

import os
import sys
import numpy as np
import cv2
# Add macOS compatibility fixes
if sys.platform == 'darwin':
    # Fix for NSApplication macOSVersion unrecognized selector issue
    os.environ['TK_SILENCE_DEPRECATION'] = '1'
    # Disable interactive mode for all matplotlib operations
    os.environ['MPLBACKEND'] = 'Agg'

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# Import matplotlib only for conversion, not for display
import matplotlib
matplotlib.use('Agg')  # Force Agg backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

from film_converter import detect_film_border, convert_negative_to_positive

def rgb_to_hex(rgb):
    """Convert RGB values to hex color code safely"""
    r, g, b = rgb
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f'#{r:02x}{g:02x}{b:02x}'

class FilmConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Film Negative Converter")
        self.root.geometry("1000x700")
        
        # Variables
        self.input_image_path = None
        self.negative_image = None
        self.film_base_color = None
        self.film_bbox = None
        self.inner_bbox = None
        self.converted_image = None
        self.current_preset = tk.StringVar(value="standard")
        self.is_color_picking = False
        self.display_size = (400, 800)  # Default size until updated
        self.is_processed = False  # Track if the image has been processed
        
        # CMYK adjustment values
        self.cyan_value = tk.IntVar(value=0)
        self.magenta_value = tk.IntVar(value=0)
        self.yellow_value = tk.IntVar(value=0)
        self.black_value = tk.IntVar(value=0)
        
        # Set up the UI
        self._create_ui()
    
    def _create_ui(self):
        # Use a simpler style to avoid macOS issues
        style = ttk.Style()
        style.theme_use('default')
        
        # Create custom styles for buttons with macOS-friendly colors
        style.configure("Process.TButton", font=("Helvetica", 11, "bold"), 
                      foreground="#004080", background="#e1e1e1")
        style.map("Process.TButton", 
                foreground=[('active', '#00254D')],
                background=[('active', '#d0d0d0')])
        style.configure("Regular.TButton", font=("Helvetica", 10))
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title label
        title_label = ttk.Label(main_frame, text="Film Negative Converter", 
                              font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # macOS Compatibility Notice
        if sys.platform == 'darwin':
            compat_label = ttk.Label(main_frame, 
                                  text="macOS Compatibility Mode Active", 
                                  font=("Helvetica", 9, "italic"),
                                  foreground="#666666")
            compat_label.pack(pady=(0, 5))
        
        # Top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Load and save buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="1. Load Image", command=self._load_image, 
                  style="Regular.TButton").pack(side=tk.LEFT, padx=5)
        
        self.process_button = ttk.Button(button_frame, text="2. Process Image", 
                                      command=self._process_image, style="Process.TButton")
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="3. Save Converted", command=self._save_image, 
                  style="Regular.TButton").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Pick Base Color", command=self._toggle_color_picker, 
                  style="Regular.TButton").pack(side=tk.LEFT, padx=5)
        
        # Preset dropdown
        preset_frame = ttk.Frame(control_frame)
        preset_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT, padx=5)
        preset_dropdown = ttk.Combobox(preset_frame, textvariable=self.current_preset, 
                                     values=["standard", "frontier", "noritsu", "hasselblad"],
                                     state="readonly", width=15)
        preset_dropdown.pack(side=tk.LEFT, padx=5)
        # Add a label to indicate that changing presets requires reprocessing
        ttk.Label(preset_frame, text="(requires reprocessing)", 
                font=("Helvetica", 8, "italic")).pack(side=tk.LEFT)
        
        # Image display area with side-by-side layout
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Combined display panel
        self.display_panel = ttk.Label(self.image_frame)
        self.display_panel.pack(fill=tk.BOTH, expand=True)
        self.display_panel.bind("<Button-1>", self._on_click)
        
        # Bottom control panel with CMYK sliders
        cmyk_frame = ttk.LabelFrame(main_frame, text="CMYK Adjustments")
        cmyk_frame.pack(fill=tk.X, pady=5)
        
        # CMYK sliders layout
        sliders_frame = ttk.Frame(cmyk_frame)
        sliders_frame.pack(fill=tk.X, expand=True, pady=5)
        
        # Create CMYK sliders with +/- buttons
        self._create_adjustment_row(sliders_frame, "Cyan", self.cyan_value, 0)
        self._create_adjustment_row(sliders_frame, "Magenta", self.magenta_value, 1)
        self._create_adjustment_row(sliders_frame, "Yellow", self.yellow_value, 2)
        self._create_adjustment_row(sliders_frame, "Black", self.black_value, 3)
        
        # Status bar with film base color display
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        # Use a Frame instead of a Canvas for color display (more compatible)
        self.color_sample = tk.Frame(status_frame, width=30, height=20, relief="sunken", bd=1, bg="#808080")
        self.color_sample.pack(side=tk.LEFT, padx=5)
        
        # Status text
        self.status_var = tk.StringVar(value="Ready. Please load an image.")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.LEFT, expand=True)
    
    def _create_adjustment_row(self, parent, name, var, row):
        # Label
        ttk.Label(parent, text=name, width=10).grid(row=row, column=0, padx=5, pady=2)
        
        # Minus button
        ttk.Button(parent, text="-", width=3, 
                  command=lambda: self._adjust_value(var, -5)).grid(row=row, column=1, padx=5, pady=2)
        
        # Slider
        slider = ttk.Scale(parent, from_=-100, to=100, orient=tk.HORIZONTAL, 
                          variable=var)
        slider.grid(row=row, column=2, sticky="ew", padx=5, pady=2)
        
        # Plus button
        ttk.Button(parent, text="+", width=3,
                  command=lambda: self._adjust_value(var, 5)).grid(row=row, column=3, padx=5, pady=2)
        
        # Value label
        value_label = ttk.Label(parent, textvariable=var, width=5)
        value_label.grid(row=row, column=4, padx=5, pady=2)
        
        # Set column weight
        parent.columnconfigure(2, weight=1)
    
    def _adjust_value(self, var, amount):
        """Adjust a CMYK value by the given amount (+/- 5)"""
        current = var.get()
        var.set(max(-100, min(100, current + amount)))
        # Don't automatically update preview anymore
        # self._update_preview()
    
    def _load_image(self):
        """Load a film negative image"""
        file_path = filedialog.askopenfilename(
            title="Select Film Negative Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.input_image_path = file_path
            self.status_var.set(f"Loading image: {os.path.basename(file_path)}")
            self.root.update()
            
            # Load and process the image
            self.negative_image = cv2.imread(file_path)
            if self.negative_image is None:
                raise ValueError("Could not read the image file")
            
            # Convert to RGB for display
            self.negative_image = cv2.cvtColor(self.negative_image, cv2.COLOR_BGR2RGB)
            
            # Detect film borders
            self.status_var.set("Detecting film borders...")
            self.root.update()
            
            try:
                # Try to detect borders automatically
                _, self.film_base_color, self.film_bbox, self.inner_bbox = detect_film_border(self.negative_image)
                
                # Update color sample
                self._update_color_sample()
                
                # Set is_color_picking to False
                self.is_color_picking = False
            except Exception as e:
                # If automatic detection fails, use the whole image
                h, w = self.negative_image.shape[:2]
                self.film_bbox = (0, 0, w, h)
                self.inner_bbox = (0, 0, w, h)
                self.film_base_color = np.array([128, 128, 128])  # Default gray
                
                # Inform user to pick a base color manually
                self.status_var.set("Auto detection failed. Please pick a base color manually.")
                self.is_color_picking = True
            
            # Display the original without processing
            self._display_original()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set(f"Error: {str(e)}")
    
    def _display_original(self):
        """Display only the original negative image"""
        if self.negative_image is None:
            return
            
        # Resize the image for display
        h, w = self.negative_image.shape[:2]
        max_height = 400
        scale = max_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize the image
        resized_img = cv2.resize(self.negative_image, (new_w, new_h))
        
        # Create a display image with labels
        display_img = resized_img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_img, "Original - Click 'Process' to convert", (10, 30), font, 0.7, (255, 255, 255), 2)
        
        # Store the display size for reference when clicking
        self.display_size = (new_h, new_w)
        
        # Convert to PhotoImage
        img = Image.fromarray(display_img)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update panel
        self.display_panel.configure(image=img_tk)
        self.display_panel.image = img_tk  # Keep a reference
        
        self.status_var.set("Image loaded. Press 'Process' to convert the negative.")
    
    def _process_image(self):
        """Process the loaded image with current settings"""
        if self.negative_image is None:
            messagebox.showinfo("Info", "Please load an image first")
            return
            
        try:
            # Update the button and status
            self.process_button.configure(text="Processing...")
            self.root.config(cursor="watch")  # Change cursor to wait
            self.status_var.set("Processing image... Please wait.")
            self.root.update()
            
            # Extract the inner film area
            x, y, w, h = self.inner_bbox
            film_area = self.negative_image[y:y+h, x:x+w]
            
            # Apply conversion with current settings
            preset = self.current_preset.get()
            cyan = self.cyan_value.get()
            magenta = self.magenta_value.get()
            yellow = self.yellow_value.get()
            black = self.black_value.get()
            
            # Process the image
            self.converted_image = convert_negative_to_positive(
                film_area, self.film_base_color, preset=preset,
                cyan=cyan, magenta=magenta, yellow=yellow, black=black
            )
            
            # Display the processed image
            self._display_side_by_side()
            
            # Update the status
            self.process_button.configure(text="2. Process Image")
            self.root.config(cursor="")  # Reset cursor
            self.status_var.set(f"Processing complete! Used preset: {preset} | C: {cyan} M: {magenta} Y: {yellow} K: {black}")
            self.is_processed = True
        except Exception as e:
            self.process_button.configure(text="2. Process Image")
            self.root.config(cursor="")  # Reset cursor
            messagebox.showerror("Error", f"Error processing image: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def _display_side_by_side(self):
        """Display original and converted images side by side"""
        if self.negative_image is None or self.converted_image is None:
            return
            
        # Create a side-by-side display
        h1, w1 = self.negative_image.shape[:2]
        h2, w2 = self.converted_image.shape[:2]
        
        # Calculate the unified display size (with consistent aspect ratio)
        max_height = 400
        scale1 = max_height / h1
        scale2 = max_height / h2
        
        new_w1 = int(w1 * scale1)
        new_h1 = int(h1 * scale1)
        new_w2 = int(w2 * scale2)
        new_h2 = int(h2 * scale2)
        
        # Resize both images
        negative_display = cv2.resize(self.negative_image, (new_w1, new_h1))
        converted_display = cv2.resize(self.converted_image, (new_w2, new_h2))
        
        # Create a combined image with a divider
        combined_width = new_w1 + new_w2 + 2  # +2 for the divider
        combined_height = max(new_h1, new_h2)
        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Insert the images
        combined_image[0:new_h1, 0:new_w1] = negative_display
        combined_image[0:new_h2, new_w1+2:] = converted_display
        
        # Add a vertical divider
        combined_image[:, new_w1:new_w1+2] = [200, 200, 200]  # Light gray divider
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_image, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_image, f"Converted", (new_w1 + 10, 30), font, 0.7, (255, 255, 255), 2)
        
        # Store the display size for reference when clicking
        self.display_size = (combined_height, combined_width)
        
        # Convert to PhotoImage
        img = Image.fromarray(combined_image)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update panel
        self.display_panel.configure(image=img_tk)
        self.display_panel.image = img_tk  # Keep a reference
    
    def _toggle_color_picker(self):
        """Toggle the color picker mode"""
        if self.negative_image is None:
            messagebox.showinfo("Info", "Please load an image first")
            return
        
        self.is_color_picking = not self.is_color_picking
        
        if self.is_color_picking:
            self.status_var.set("Color picker active: Click on the image to select film base color")
        else:
            self.status_var.set("Color picker deactivated")
    
    def _on_click(self, event):
        """Handle image clicks for color picking"""
        if not self.is_color_picking or self.negative_image is None:
            return
        
        # Get the click position relative to the image
        # Calculate which side of the display was clicked
        if event.x < self.display_size[1] // 2:  # Left side (original image)
            # Convert from display coordinates to original image coordinates
            display_h, display_w = self.display_size
            left_display_w = display_w // 2 - 1  # Account for divider
            
            img_h, img_w = self.negative_image.shape[:2]
            
            x_ratio = img_w / left_display_w
            y_ratio = img_h / display_h
            
            x = int(event.x * x_ratio)
            y = int(event.y * y_ratio)
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            
            # Get the color at the clicked position
            self.film_base_color = self.negative_image[y, x].copy()
            
            # Update the color sample
            self._update_color_sample()
            
            # Don't automatically process, just update the display
            # self._update_preview()
            
            # Provide feedback
            self.status_var.set(f"Base color selected: RGB {self.film_base_color}. Click 'Process' to convert.")
            
            # Turn off color picking mode
            self.is_color_picking = False
    
    def _update_color_sample(self):
        """Update the film base color sample in the UI"""
        if self.film_base_color is None:
            return
        
        try:
            # Convert RGB values to hex format and set background
            hex_color = rgb_to_hex(self.film_base_color)
            self.color_sample.config(bg=hex_color)
        except Exception as e:
            # If there's any issue, use a default color
            self.color_sample.config(bg="#808080")
    
    def _save_image(self):
        """Save the converted image"""
        if self.converted_image is None:
            if not self.is_processed:
                messagebox.showinfo("Info", "Please process the image before saving.")
            else:
                messagebox.showwarning("Warning", "No converted image to save!")
            return
        
        # Get save path
        original_filename = os.path.basename(self.input_image_path)
        preset = self.current_preset.get()
        default_filename = f"converted_{preset}_{original_filename}"
        
        save_path = filedialog.asksaveasfilename(
            title="Save Converted Image",
            defaultextension=".jpg",
            initialfile=default_filename,
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            # Use PIL to save the image instead of matplotlib
            img = Image.fromarray(self.converted_image)
            img.save(save_path)
            self.status_var.set(f"Image saved to: {save_path}")
            messagebox.showinfo("Success", f"Image saved to: {save_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Could not save image: {str(e)}")
            self.status_var.set(f"Error saving image: {str(e)}")

def main():
    root = tk.Tk()
    app = FilmConverterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 