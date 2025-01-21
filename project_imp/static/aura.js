let uploadedImage = null;
let canvas = null;
let ctx = null;

function previewImage() {
    const fileInput = document.getElementById('uploadImage');
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();

        // Read the file as a data URL
        reader.onload = function (e) {
            const imagePreview = document.getElementById('imagePreview');
            imagePreview.src = e.target.result;  
            imagePreview.style.display = 'block';  
            uploadedImage = e.target.result;  

            // Initialize the canvas for drawing
            canvas = document.createElement('canvas');
            ctx = canvas.getContext('2d');
            const img = new Image();
            img.onload = function () {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
            };
            img.src = uploadedImage;
        };

        reader.readAsDataURL(file);  
    }
}

function uploadImage() {
    const fileInput = document.getElementById('uploadImage');
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.message === "Image uploaded successfully") {
            console.log('Image uploaded');
        }
    })
    .catch(error => {
        console.error("Error uploading image:", error);
    });
}

function updateImage() {
    const brightness = document.getElementById('brightness').value;
    const contrast = document.getElementById('contrast').value;
    const saturation = document.getElementById('saturation').value;
    const temperature = document.getElementById('temperature').value;
    const tint = document.getElementById('tint').value;
    const highlights = document.getElementById('highlights').value;
    const shadows = document.getElementById('shadows').value;
    const whites = document.getElementById('whites').value;
    const blacks = document.getElementById('blacks').value;
    const vibrance = document.getElementById('vibrance').value;
    const sharpness = document.getElementById('sharpness').value;
    const clarity = document.getElementById('clarity').value;
    const vignette = document.getElementById('vignette').value;
    const filter = document.getElementById('filter').value;
    const effect = document.getElementById('effect').value;

    // Apply the image adjustments using canvas context
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const img = new Image();
    img.onload = function () {
        // Apply basic adjustments
        ctx.filter = `
            brightness(${brightness}) 
            contrast(${contrast}) 
            saturate(${saturation}) 
            hue-rotate(${temperature}deg)
            sepia(${tint})`;

        // Apply highlight, shadow, whites, blacks, vibrance, sharpness, clarity, vignette adjustments
        ctx.filter += `
            contrast(${highlights}) 
            brightness(${shadows}) 
            sepia(${whites}) 
            grayscale(${blacks}%) 
            saturate(${vibrance}) 
            blur(${sharpness}px) 
            opacity(${clarity}) 
            vignette(${vignette}%)`;

        // Apply selected filter
        if (filter === 'dramatic') {
            ctx.filter += ' contrast(1.5)';
        } else if (filter === 'vivid') {
            ctx.filter += ' saturate(2)';
        } else if (filter === 'mono') {
            ctx.filter += ' grayscale(100%)';
        } else if (filter === 'retro') {
            ctx.filter += ' sepia(1)';
        } else if (filter === 'polaroid') {
            ctx.filter += ' brightness(1.3) contrast(1.2) sepia(0.8)';
        } else if (filter === 'vintage') {
            ctx.filter += ' sepia(0.6) contrast(1.2)';
        } else if (filter === 'blackwhite') {
            ctx.filter += ' grayscale(100%)';
        } else if (filter === 'sunset') {
            ctx.filter += ' contrast(1.5) brightness(1.2)';
        }

        // Apply selected effect
        if (effect === 'blur') {
            ctx.filter += ' blur(5px)';
        } else if (effect === 'duotone') {
            ctx.filter += ' sepia(1)';
        } else if (effect === 'shadows') {
            ctx.filter += ' drop-shadow(10px 10px 10px rgba(0,0,0,0.5))';
        }

        ctx.drawImage(img, 0, 0);

        // Update the image preview
        const updatedImage = canvas.toDataURL();
        document.getElementById('imagePreview').src = updatedImage;
    };
    img.src = uploadedImage;  
}

function applyChanges() {
    updateImage();
}

function saveImage() {
    const link = document.createElement('a');
    link.href = canvas.toDataURL(); 
    link.download = 'edited_image.png';  
    link.click();  
}
