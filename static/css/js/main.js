document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clearButton');
    const predictButton = document.getElementById('predictButton');
    const predictionSpan = document.getElementById('prediction');
    const fileInput = document.getElementById('fileInput');
    const uploadedImage = document.getElementById('uploadedImage');
    const uploadedImageContainer = document.getElementById('uploadedImageContainer');
    
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    
    // Set up canvas
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'black';
    
    // Fill canvas with white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Clear canvas
    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        predictionSpan.textContent = '-';
        uploadedImage.style.display = 'none';
    }
    
    // Drawing event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = [
            e.offsetX || e.touches[0].clientX - canvas.offsetLeft, 
            e.offsetY || e.touches[0].clientY - canvas.offsetTop
        ];
    }
    
    function draw(e) {
        if (!isDrawing) return;
        
        let x, y;
        if (e.type === 'mousemove') {
            x = e.offsetX;
            y = e.offsetY;
        } else if (e.type === 'touchmove') {
            const rect = canvas.getBoundingClientRect();
            x = e.touches[0].clientX - rect.left;
            y = e.touches[0].clientY - rect.top;
        }
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        
        [lastX, lastY] = [x, y];
    }
    
    function stopDrawing() {
        isDrawing = false;
    }
    
    function handleTouch(e) {
        e.preventDefault();
        if (e.type === 'touchstart') {
            startDrawing(e);
        } else if (e.type === 'touchmove') {
            draw(e);
        }
    }
    
    // File import functionality
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            
            reader.onload = function(event) {
                const img = new Image();
                img.onload = function() {
                    // Clear canvas and draw imported image
                    ctx.fillStyle = 'white';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // Scale and center the image
                    const scale = Math.min(
                        canvas.width / img.width, 
                        canvas.height / img.height
                    );
                    const scaledWidth = img.width * scale;
                    const scaledHeight = img.height * scale;
                    const x = (canvas.width - scaledWidth) / 2;
                    const y = (canvas.height - scaledHeight) / 2;
                    
                    ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
                };
                img.src = event.target.result;
                
                // Show uploaded image preview
                uploadedImage.src = event.target.result;
                uploadedImage.style.display = 'block';
            };
            
            reader.readAsDataURL(file);
        }
    });
    
    // Button event listeners
    clearButton.addEventListener('click', clearCanvas);
    predictButton.addEventListener('click', predict);
    
    async function predict() {
        let imageData;
        
        // Check if there's an uploaded image
        if (uploadedImage.src && uploadedImage.style.display !== 'none') {
            // Use uploaded image
            imageData = uploadedImage.src;
        } else {
            // Use canvas drawing
            imageData = canvas.toDataURL('image/png');
        }
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData }),
            });
            
            if (!response.ok) {
                throw new Error('Prediction failed');
            }
            
            const data = await response.json();
            predictionSpan.textContent = data.prediction;
            
        } catch (error) {
            console.error('Error:', error);
            predictionSpan.textContent = 'Error';
        }
    }
    
    // Clear canvas on load
    clearCanvas();
});
