const uploadFileInput = document.getElementById('upload-file');
const uploadedImage = document.getElementById('uploaded-image');

uploadFileInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      uploadedImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
});
