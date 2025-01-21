from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io

app = Flask(__name__)

# Function to apply filters
def apply_filters(image, filter_type):
    if filter_type == 'dramatic':
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)
    elif filter_type == 'vivid':
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(2.0)
    elif filter_type == 'mono':
        return image.convert('L')
    elif filter_type == 'retro':
        # Simple retro effect (add sepia)
        np_img = np.array(image)
        np_img = np_img // 2
        return Image.fromarray(np_img)
    elif filter_type == 'polaroid':
        # Add polaroid effect
        np_img = np.array(image)
        np_img = np_img * [1.2, 1, 0.8]  
        np_img = np_img.clip(0, 255)
        return Image.fromarray(np_img.astype(np.uint8))
    elif filter_type == 'vintage':
        # Simple vintage filter
        np_img = np.array(image)
        np_img = np_img * [0.9, 0.7, 0.5]
        return Image.fromarray(np_img.clip(0, 255).astype(np.uint8))
    elif filter_type == 'blackwhite':
        return image.convert('L')
    elif filter_type == 'sunset':
        # Simple sunset filter
        np_img = np.array(image)
        np_img = np_img * [1, 0.6, 0.3]  # Add a warm sunset tint
        return Image.fromarray(np_img.clip(0, 255).astype(np.uint8))
    return image

# Function to apply effects
def apply_effects(image, effect_type):
    if effect_type == 'duotone':
        image = image.convert('RGB')
        np_img = np.array(image)
        np_img = np_img // 2  # Simple duotone effect by reducing RGB values
        return Image.fromarray(np_img)
    elif effect_type == 'blur':
        return image.filter(ImageFilter.GaussianBlur(5))
    return image

# Function to adjust image properties (brightness, contrast, etc.)
def adjust_image(image, params):
    # Brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(params['brightness'])
    
    # Contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(params['contrast'])
    
    # Saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(params['saturation'])

    # Temperature adjustment (just adding a slight tint for simplicity)
    temperature = params.get('temperature', 0)
    if temperature != 0:
        np_img = np.array(image)
        np_img = np_img + temperature 
        np_img = np_img.clip(0, 255)
        image = Image.fromarray(np_img.astype(np.uint8))
    
    # Tint adjustment
    tint = params.get('tint', 0)
    if tint != 0:
        np_img = np.array(image)
        np_img[:, :, 1] = np_img[:, :, 1] * (1 + tint)  
        np_img = np_img.clip(0, 255)
        image = Image.fromarray(np_img.astype(np.uint8))

    # Highlights and Shadows (could use a more advanced method)
    highlights = params.get('highlights', 1)
    if highlights != 1:
        np_img = np.array(image)
        np_img[np_img > 180] = np_img[np_img > 180] * highlights  
        np_img = np_img.clip(0, 255)
        image = Image.fromarray(np_img.astype(np.uint8))
    
    shadows = params.get('shadows', 1)
    if shadows != 1:
        np_img = np.array(image)
        np_img[np_img < 50] = np_img[np_img < 50] * shadows  
        np_img = np_img.clip(0, 255)
        image = Image.fromarray(np_img.astype(np.uint8))
    
    # Whites and Blacks adjustment
    whites = params.get('whites', 1)
    blacks = params.get('blacks', 1)
    if whites != 1 or blacks != 1:
        np_img = np.array(image)
        np_img = np_img * [whites, whites, whites]  # Apply whites adjustment
        np_img = np_img - (np_img * (1 - blacks))  # Apply blacks adjustment
        np_img = np_img.clip(0, 255)
        image = Image.fromarray(np_img.astype(np.uint8))

    # Vibrance (increase saturation in low-saturation areas)
    vibrance = params.get('vibrance', 1)
    if vibrance != 1:
        np_img = np.array(image)
        np_img = np_img * vibrance  
        np_img = np_img.clip(0, 255)
        image = Image.fromarray(np_img.astype(np.uint8))

    # Sharpness
    sharpness = params.get('sharpness', 1)
    if sharpness != 1:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
    
    # Clarity (increases midtone contrast)
    clarity = params.get('clarity', 1)
    if clarity != 1:
        np_img = np.array(image)
        np_img = np_img * clarity  
        np_img = np_img.clip(0, 255)
        image = Image.fromarray(np_img.astype(np.uint8))

    # Vignette (add dark corners)
    vignette = params.get('vignette', 0)
    if vignette != 0:
        np_img = np.array(image)
        center = np_img.shape[1] // 2, np_img.shape[0] // 2
        y, x = np.ogrid[:np_img.shape[0], :np_img.shape[1]]
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2
        mask = np.exp(-mask / (2 * (vignette * 10)))
        np_img = np_img * mask[..., None]  
        np_img = np_img.clip(0, 255)
        image = Image.fromarray(np_img.astype(np.uint8))
    
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    image = Image.open(file)
    image.save('static/uploaded_image.jpg')
    return jsonify({"message": "Image uploaded successfully"})

@app.route('/edit', methods=['POST'])
def edit_image():
    params = request.json
    image = Image.open('static/uploaded_image.jpg')

    # Apply filters
    if params.get('filter'):
        image = apply_filters(image, params['filter'])
    
    # Apply effects
    if params.get('effect'):
        image = apply_effects(image, params['effect'])
    
    # Adjust image properties
    image = adjust_image(image, params)

    # Save the modified image
    image.save('static/edited_image.jpg')

    # Return the image URL
    return jsonify({"image_url": "/static/edited_image.jpg"})

if __name__ == '__main__':
    app.run(debug=True)
