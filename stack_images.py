# import os
# from PIL import Image, ImageDraw, ImageFont

# def stack_images_vertically_with_labels(image_paths, labels, output_path, font_path=None, font_size=20, spacing=10):
#     # Load images
#     images = [Image.open(img_path) for img_path in image_paths]
    
#     # Ensure there are images to process
#     if not images:
#         raise ValueError("No images found in the specified directory.")
    
#     # Ensure the number of labels matches the number of images
#     if len(images) != len(labels):
#         raise ValueError("The number of labels must match the number of images")
    
#     # Load the font
#     if font_path is not None:
#         try:
#             font = ImageFont.truetype(font_path, font_size)
#         except OSError:
#             print(f"Font '{font_path}' not found. Using default font.")
#             font = ImageFont.load_default()
#     else:
#         font = ImageFont.load_default()
    
#     # Calculate the maximum label width after rotation
#     max_label_width = 0
#     for label in labels:
#         label_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), label, font=font)
#         label_width, label_height = label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1]
#         rotated_label_width = label_height  # Width after 90 degrees rotation
#         if rotated_label_width > max_label_width:
#             max_label_width = rotated_label_width
        
#     max_label_width *= 2
    
#     # Calculate the total width and height for the final stacked image
#     widths, heights = zip(*(img.size for img in images))
#     total_width = max(widths) + max_label_width
#     total_height = sum(heights) + (spacing * (len(images) - 1))
    
#     # Create a new image with a white background
#     stacked_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
#     # Initialize the drawing context
#     draw = ImageDraw.Draw(stacked_image)
    
#     # Paste each image into the stacked image and draw the label
#     y_offset = 0
#     for img, label in zip(images, labels):
#         # Calculate the size of the label
#         label_bbox = draw.textbbox((0, 0), label, font=font)
#         label_width, label_height = label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1]
        
#         # Create a new image for the label
#         label_image = Image.new('RGB', (label_width, label_height), (255, 255, 255))
#         label_draw = ImageDraw.Draw(label_image)
#         label_draw.text((0, 0), label, font=font, fill=(0, 0, 0))
        
#         # Rotate the label
#         label_image = label_image.rotate(90, expand=True)
        
#         # Center the label vertically next to the image
#         rotated_label_width, rotated_label_height = label_image.size
#         img_width, img_height = img.size
#         y_centered = y_offset + (img_height - rotated_label_height) // 2
        
#         # Paste the label and the image into the stacked image
#         stacked_image.paste(label_image, (0, y_centered))
#         stacked_image.paste(img, (max_label_width, y_offset))
        
#         # Update the y offset for the next image, including the spacing
#         y_offset += img_height + spacing
    
#     # Save the final stacked image
#     stacked_image.save(output_path)

# # Example usage
# directory = "qualitative_results"
# image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
# labels = [os.path.splitext(f)[0] for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# ordering = ["baseline", "hybrid downsample", "hybrid downsample ReLU", "hybrid", "linformer", "swin", "mamba", "transformer++"]
# assert len(image_paths) == len(labels)
# image_paths = [image_paths[labels.index(o)] for o in ordering]
# labels = ordering

# if not image_paths:
#     raise ValueError("No image files found in the specified directory.")

# output_path = 'stacked_image.jpg'
# # font_path = 'arial.ttf'  # Update this to the correct path if needed
# font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf"

# font_size = 40  # Adjust the font size as needed
# spacing = 20  # Adjust the spacing as needed
# stack_images_vertically_with_labels(image_paths, labels, output_path, font_path=font_path, font_size=font_size, spacing=spacing)








import os
from PIL import Image, ImageDraw, ImageFont

def stack_images_vertically_with_labels(image_paths, labels, output_path, font_path=None, font_size=20, spacing=10):
    # Load images
    images = [Image.open(img_path) for img_path in image_paths]
    
    # Ensure there are images to process
    if not images:
        raise ValueError("No images found in the specified directory.")
    
    # Ensure the number of labels matches the number of images
    if len(images) != len(labels):
        raise ValueError("The number of labels must match the number of images")
    
    # Load the font
    if font_path is not None:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            print(f"Font '{font_path}' not found. Using default font.")
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    
    # Calculate the maximum label dimensions after rotation
    max_label_width = 0
    for label in labels:
        label_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), label, font=font)
        label_width, label_height = label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1]
        rotated_label_width = label_height  # Width after 90 degrees rotation
        if rotated_label_width > max_label_width:
            max_label_width = rotated_label_width
    
    max_label_width += 20  # Add some padding to ensure no part of the label is cut off
    
    # Calculate the total width and height for the final stacked image
    widths, heights = zip(*(img.size for img in images))
    total_width = max(widths) + max_label_width
    total_height = sum(heights) + (spacing * (len(images) - 1))
    
    # Create a new image with a white background
    stacked_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # Initialize the drawing context
    draw = ImageDraw.Draw(stacked_image)
    
    # Paste each image into the stacked image and draw the label
    y_offset = 0
    for img, label in zip(images, labels):
        # Calculate the size of the label
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_height, label_width = label_bbox[2] - label_bbox[0], label_bbox[3] - label_bbox[1]
        
        # Create a new image for the label with padding
        label_image = Image.new('RGB', (label_height + 20, label_width + 20), (255, 255, 255))
        label_draw = ImageDraw.Draw(label_image)
        label_draw.text((10, 10), label, font=font, fill=(0, 0, 0))  # Add padding before rotating
        
        # Rotate the label
        label_image = label_image.rotate(90, expand=True)
        
        # Center the label vertically next to the image
        rotated_label_width, rotated_label_height = label_image.size

        img_width, img_height = img.size
        y_centered = y_offset + (img_height - rotated_label_height) // 2
        
        # Paste the label and the image into the stacked image
        stacked_image.paste(img, (max_label_width, y_offset))
        stacked_image.paste(label_image, (0, y_centered))

        # Update the y offset for the next image, including the spacing
        y_offset += img_height + spacing
    
    # Save the final stacked image
    stacked_image.save(output_path)

# Example usage
directory = "qualitative_results"
image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
labels = [os.path.splitext(f)[0] for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

ordering = ["baseline", "hybrid downsample", "hybrid downsample ReLU", "hybrid", "linformer", "swin", "mamba", "transformer++"]
assert len(image_paths) == len(labels)
image_paths = [image_paths[labels.index(o)] for o in ordering]
labels = ordering

if not image_paths:
    raise ValueError("No image files found in the specified directory.")

output_path = 'stacked_image.jpg'
# font_path = 'arial.ttf'  # Update this to the correct path if needed
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf"

font_size = 40  # Adjust the font size as needed
spacing = 20  # Adjust the spacing as needed
stack_images_vertically_with_labels(image_paths, labels, output_path, font_path=font_path, font_size=font_size, spacing=spacing)
