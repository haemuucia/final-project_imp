import os
import shutil
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
from scipy.signal import wiener

app = Flask(__name__)

# Set the upload folder path
upload_folder = os.path.join('static', 'uploads')
output_folder = os.path.join('static', 'processed')

# Create directories if they don't exist
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

app.config['UPLOAD'] = upload_folder
app.config['PROCESSED'] = output_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")  # Main page

@app.route("/aura")
def aura():
    return render_template("aura.html")

@app.route("/filters")
def filters():
    return render_template("filters.html")

@app.route("/feature_matching")
def feature_matching():
    return render_template("feature_matching.html")

@app.route("/enhancement")
def enhancement():
    return render_template("enhancement.html")

@app.route("/compress", methods=["GET", "POST"])
def compress():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("compress.html", error="No file uploaded.")

        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            # Compress the image
            image = cv2.imread(filepath)
            compressed_image_path = os.path.join(app.config['PROCESSED'], f"compressed_{filename}")
            
            # Resize image for compression (adjust the compression factor as needed)
            compression_factor = 0.5  # Resize factor (50% reduction)
            width = int(image.shape[1] * compression_factor)
            height = int(image.shape[0] * compression_factor)
            resized_image = cv2.resize(image, (width, height))

            # Save the compressed image
            cv2.imwrite(compressed_image_path, resized_image)

            # Return the processed image
            compressed_image_url = f"/static/processed/compressed_{filename}"
            return render_template("compress.html", original_image=f"/static/uploads/{filename}", compressed_image=compressed_image_url, download_link=compressed_image_url)

        else:
            return render_template("compress.html", error="Invalid file type.")
    
    return render_template("compress.html")

def erosion(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.erode(binary_image, kernel)


def dilation(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(binary_image, kernel)


def opening(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)


def closing(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)


def skeletonization(binary_image):
    return cv2.ximgproc.thinning(binary_image)


def boundary_extraction(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    temp = cv2.dilate(binary_image, kernel)
    return cv2.subtract(temp, binary_image)


@app.route("/binary", methods=['GET', 'POST'])
def binary():
    if request.method == 'POST':
        if 'img' in request.files:
            file = request.files['img']
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD'], filename)
                file.save(filepath)

                image = cv2.imread(filepath)
                if image is None:
                    return jsonify({'error': 'Failed to read the image'}), 400
                
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

                operation = request.form.get('operation')

                if operation == 'erosion':
                    processed_image = erosion(binary_image)
                elif operation == 'dilation':
                    processed_image = dilation(binary_image)
                elif operation == 'opening':
                    processed_image = opening(binary_image)
                elif operation == 'closing':
                    processed_image = closing(binary_image)
                elif operation == 'skeleton':
                    processed_image = skeletonization(binary_image)
                elif operation == 'boundary':
                    processed_image = boundary_extraction(binary_image)
                else:
                    return jsonify({'error': 'Invalid operation'}), 400

                if processed_image is None:
                    return jsonify({'error': 'Failed to process the image'}), 400

                processed_filename = f"{operation}_{filename}"
                processed_image_path = os.path.join(app.config['PROCESSED'], processed_filename)
                cv2.imwrite(processed_image_path, processed_image)

                return jsonify({'processed_image_url': f"/static/processed/{processed_filename}"}), 200
        
        return jsonify({'error': 'No image uploaded'}), 400

    return render_template('binary.html')


# For blending images, route changed to '/blending'
@app.route("/blending", methods=['GET', 'POST'])
def blending():
    if request.method == 'POST':
        if 'file1' in request.files and 'file2' in request.files:  # For blending images
            file1 = request.files['file1']
            file2 = request.files['file2']
            alpha = float(request.form['alpha'])

            # Open the images
            img1 = Image.open(file1)
            img2 = Image.open(file2)

            # Ensure both images are the same size
            img1 = img1.resize((800, 600))
            img2 = img2.resize((800, 600))

            # Blend the images
            blended = Image.blend(img1, img2, alpha)

            # Save the blended image
            blended_path = 'static/blended_image.png'
            blended.save(blended_path)

            return render_template('blending.html', blended_image_url=blended_path)

    return render_template('blending.html')

@app.route('/crop', methods=['GET', 'POST'])
def crop_file():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files.get('img')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD'], filename)
            file.save(filepath)

            # Get user input for cropping width and height
            try:
                width = int(request.form.get('width', 100))  # Crop width
                height = int(request.form.get('height', 100))  # Crop height

                # Process the image (crop)
                processed_img_path = crop_image(filepath, width, height)

                # Check which button was clicked (Crop or Save)
                if 'save' in request.form:
                    # Save the processed image
                    save_folder = os.path.join(app.config['UPLOAD'], 'saved')
                    os.makedirs(save_folder, exist_ok=True)  # Ensure the folder exists
                    saved_path = os.path.join(save_folder, f"cropped_{filename}")
                    shutil.copy(processed_img_path, saved_path)  # Save the cropped image

                    # Inform the user about the save action
                    return render_template('crop.html', img=processed_img_path, message=f"Image saved to {saved_path}")

                # Render the processed image for "Crop"
                return render_template('crop.html', img=processed_img_path)

            except ValueError:
                return "Invalid input. Please ensure all fields are filled correctly.", 400

    return render_template('crop.html')

@app.route("/filters", methods=["GET", "POST"])
def filters_app():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("filters.html", error="No file uploaded.")
        
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            filter_option = request.form.get("filter")
            image = cv2.imread(filepath)

            original_image_path = f"static/uploads/{filename}"
            processed_image_path = os.path.join(app.config['PROCESSED'], f"processed_{filename}")
            
            processed_image_url = f"/static/processed/processed_{filename}"


            # Existing filters...
            if filter_option == "Wiener":
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_image = gray_image.astype(np.float32) / 255.0
                restored_image = wiener(gray_image, (5, 5))
                restored_image = np.clip(restored_image * 255, 0, 255).astype(np.uint8)
                cv2.imwrite(processed_image_path, restored_image)

            elif filter_option == "Gaussian":
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
                cv2.imwrite(processed_image_path, blurred_image)
            
            else:
                processed_image_url = original_image_path

            return render_template("filters.html", 
                                   original_image=original_image_path, 
                                   processed_image=processed_image_url)
        else:
            return render_template("filters.html", error="Invalid file type.")
    
    return render_template("filters.html")

# Feature matching page
@app.route("/feature_matching", methods=["GET", "POST"])
def feature_matching_page():
    if request.method == "POST":
        if "file1" not in request.files or "file2" not in request.files:
            return render_template("feature_matching.html", error="Both images must be uploaded.")
        
        file1 = request.files["file1"]
        file2 = request.files["file2"]
        
        if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
            filename1 = secure_filename(file1.filename)
            filename2 = secure_filename(file2.filename)
            filepath1 = os.path.join(upload_folder, filename1)
            filepath2 = os.path.join(upload_folder, filename2)
            
            file1.save(filepath1)
            file2.save(filepath2)
            
            match_method = request.form.get("method", "SIFT")
            matched_image_path = feature_matching(filepath1, filepath2, method=match_method)
            matched_image_url = f"static/uploads/{matched_image_path.split('/')[-1]}"
            
            return render_template("feature_matching.html", matched_image_url=matched_image_url)
        else:
            return render_template("feature_matching.html", error="Invalid file type.")
    
    return render_template("feature_matching.html")

# Feature Matching Function: SIFT or ORB
def feature_matching(img1_path, img2_path, method="SIFT"):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if method == "SIFT":
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
    else:  # ORB method
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

    # Brute Force Matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_path = "static/uploads/matched_image.jpg"
    cv2.imwrite(match_path, img_matches)
    return match_path


# For border addition, route changed to '/uploads'
@app.route("/uploads", methods=['GET', 'POST'])
def uploads():
    if request.method == 'POST':
        if 'Save Image' in request.form:
            processed_path = request.form.get('processed_image_path')
            if processed_path:
                full_path = os.path.join(app.root_path, processed_path)
                if os.path.exists(full_path):
                    return send_file(full_path, 
                                     as_attachment=True,
                                     download_name=f"bordered_{os.path.basename(processed_path)}")
            return "No processed image to save", 400

        if 'img' in request.files:
            file = request.files.get('img')
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD'], filename)
                file.save(filepath)

                try:
                    border_thickness = int(request.form.get('thickness', 1))
                    border_type = request.form.get('border_type', 'solid')
                    border_color = request.form.get('border_color', '#000000')

                    if not (border_color.startswith('#') and len(border_color) == 7):
                        return "Invalid color format. Please provide a hex color like #RRGGBB.", 400

                    border_color_rgb = hex_to_rgb(border_color)
                    bgr_color = (border_color_rgb[2], border_color_rgb[1], border_color_rgb[0])

                    processed_img_path = add_border(filepath, border_thickness, border_type, bgr_color)
                    relative_path = processed_img_path

                    return render_template('uploads.html', 
                                           img=relative_path,
                                           processed_image_path=relative_path)

                except ValueError:
                    return "Invalid input. Please ensure all fields are filled correctly.", 400

    return render_template('uploads.html')

# Helper function to convert hex color to RGB (if not already defined)
def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def add_border(image_path, thickness, border_type, border_color):
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    if border_type == 'solid':
        bordered_img = cv2.copyMakeBorder(
            img, thickness, thickness, thickness, thickness,
            cv2.BORDER_CONSTANT, value=border_color
        )
    elif border_type == 'dotted':
        bordered_img = img.copy()
        for i in range(0, width, 2 * thickness):
            cv2.line(bordered_img, (i, 0), (i, thickness), border_color, thickness)
            cv2.line(bordered_img, (i, height - thickness), (i, height), border_color, thickness)
        for i in range(0, height, 2 * thickness):
            cv2.line(bordered_img, (0, i), (thickness, i), border_color, thickness)
            cv2.line(bordered_img, (width - thickness, i), (width, i), border_color, thickness)

    processed_filename = f'bordered_{os.path.basename(image_path)}'
    processed_filepath = os.path.join(app.config['PROCESSED'], processed_filename)
    cv2.imwrite(processed_filepath, bordered_img)

    return os.path.join('static', 'processed', processed_filename)

def crop_image(image_path, width, height):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Get the dimensions of the image
    img_height, img_width, _ = img.shape

    # Calculate x and y to center the crop
    x = max(0, (img_width - width) // 2)  # Center horizontally
    y = max(0, (img_height - height) // 2)  # Center vertically

    # Ensure the crop dimensions are within the image size
    x_end = min(x + width, img_width)
    y_end = min(y + height, img_height)

    # Crop the image
    cropped_img = img[y:y_end, x:x_end]

    # Save the cropped image
    processed_filename = f'cropped_{os.path.basename(image_path)}'
    processed_filepath = os.path.join(app.config['PROCESSED'], processed_filename)
    cv2.imwrite(processed_filepath, cropped_img)

    return os.path.join('static', 'processed', processed_filename)

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

@app.route('/bg_replacement', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'Save Image' in request.form.values():
            # Handle save image request
            if 'processed_image_path' in request.form:
                processed_path = request.form['processed_image_path']
                return send_file(processed_path, as_attachment=True)
            return "No processed image to save", 400

        # Get the uploaded image
        file = request.files.get('img')
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD'], filename)
            file.save(filepath)

            # Get the background image for replacement
            bg_file = request.files.get('bg_img')
            if bg_file:
                bg_filename = secure_filename(bg_file.filename)
                bg_filepath = os.path.join(app.config['UPLOAD'], bg_filename)
                bg_file.save(bg_filepath)

            # Perform background replacement
            try:
                processed_img_path = replace_background(filepath, bg_filepath)
                return render_template('bg_replacement.html', 
                                    img=processed_img_path,
                                    processed_image_path=os.path.join(app.config['PROCESSED'], 
                                                                    os.path.basename(processed_img_path)))
            except Exception as e:
                return f"Error processing the image: {str(e)}", 500

    return render_template('bg_replacement.html')

def replace_background(image_path, bg_image_path):
    # Read input images
    image = cv2.imread(image_path)
    bg_image = cv2.imread(bg_image_path)

    # Resize background to match the input image size
    bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))

    # Convert image to a 2D array of pixels (flattening)
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    # Define criteria and apply KMeans clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2  # We want to segment into 2 clusters (foreground and background)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert center back into 8 bit values
    center = np.uint8(center)
    segmented_image = center[label.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    # Identify the background cluster
    label_counts = np.bincount(label.flatten())
    background_cluster = np.argmax(label_counts)

    # Create a mask for the foreground
    mask = (label.flatten() == background_cluster)
    mask = mask.reshape(image.shape[:2])

    # Replace the background with the new background image
    result = image.copy()
    result[mask] = bg_image[mask]

    # Save the result image
    processed_filename = f'background_replaced_{os.path.basename(image_path)}'
    processed_filepath = os.path.join(app.config['PROCESSED'], processed_filename)
    cv2.imwrite(processed_filepath, result)

    return os.path.join('static', 'processed', processed_filename)

@app.route("/overlay", methods=['GET', 'POST'])
def overlay():
    if request.method == 'POST':
        if 'image1' not in request.files or 'image2' not in request.files:
            return "Please upload two images.", 400

        image1 = request.files['image1']
        image2 = request.files['image2']

        if image1 and allowed_file(image1.filename) and image2 and allowed_file(image2.filename):
            image1_filename = secure_filename(image1.filename)
            image2_filename = secure_filename(image2.filename)

            image1_path = os.path.join(app.config['UPLOAD_FOLDER'], image1_filename)
            image2_path = os.path.join(app.config['UPLOAD_FOLDER'], image2_filename)

            image1.save(image1_path)
            image2.save(image2_path)

            opacity = float(request.form.get('opacity', 0.5))
            position_x = int(request.form.get('positionX', 0))
            position_y = int(request.form.get('positionY', 0))

            try:
                base_image = Image.open(image1_path).convert("RGBA")  
                overlay_image = Image.open(image2_path).convert("RGBA")  
                overlay_image = overlay_image.resize(base_image.size, Image.ANTIALIAS)

                # Opacity
                overlay_image.putalpha(int(255 * opacity))

                # Merge 2 pic
                base_image.paste(overlay_image, (position_x, position_y), overlay_image)

                # Save overlay result
                result_filename = f"overlay_result_{image1_filename}"
                result_path = os.path.join(app.config['PROCESSED_FOLDER'], result_filename)
                base_image.save(result_path)

                # Render halaman dengan hasil overlay
                return render_template('overlay_result.html', result_image=result_path)

            except Exception as e:
                return f"Error processing the overlay: {str(e)}", 500
        else:
            return "Invalid file type. Only PNG, JPG, JPEG are allowed.", 400

    return render_template("overlay.html")

if __name__ == '__main__':
    app.run(debug=True, port=8001)
