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
    
    // Debugging: Console log to verify script is running
    console.log("JavaScript loaded successfully!");

    // Set up canvas with explicit size and styling
    canvas.width = 280;
    canvas.height = 280;
    canvas.style.width = '280px';
    canvas.style.height = '280px';
    
    // Explicit drawing context setup
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Comprehensive drawing function
    function drawLine(x1, y1, x2, y2) {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        console.log(`Drawing from (${x1},${y1}) to (${x2},${y2})`);
    }

    // Event handlers with extensive logging
    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        
        // Determine coordinates based on event type
        const rect = canvas.getBoundingClientRect();
        let x, y;
        
        if (e.type.startsWith('mouse')) {
            x = e.clientX - rect.left;
            y = e.clientY - rect.top;
        } else if (e.type.startsWith('touch')) {
            x = e.touches[0].clientX - rect.left;
            y = e.touches[0].clientY - rect.top;
        }
        
        lastX = x;
        lastY = y;
        
        console.log(`Start drawing at (${x},${y})`);
    }

    function draw(e) {
        e.preventDefault();
        if (!isDrawing) return;
        
        const rect = canvas.getBoundingClientRect();
        let x, y;
        
        if (e.type.startsWith('mouse')) {
            x = e.clientX - rect.left;
            y = e.clientY - rect.top;
        } else if (e.type.startsWith('touch')) {
            x = e.touches[0].clientX - rect.left;
            y = e.touches[0].clientY - rect.top;
        }
        
        drawLine(lastX, lastY, x, y);
        
        lastX = x;
        lastY = y;
    }

    function stopDrawing(e) {
        e.preventDefault();
        isDrawing = false;
        console.log("Stopped drawing");
    }

    // Add event listeners with multiple event types
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events
    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDrawing, { passive: false });

    // Clear canvas function
    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        predictionSpan.textContent = '-';
        uploadedImage.style.display = 'none';
        console.log("Canvas cleared");
    }

    // File import with extensive logging
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        console.log("File selected:", file);

        if (file) {
            const reader = new FileReader();
            
            reader.onload = function(event) {
                console.log("File read successful");
                const img = new Image();
                
                img.onload = function() {
                    console.log("Image loaded successfully");
                    // Clear canvas
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
                    
                    // Show uploaded image preview
                    uploadedImage.src = event.target.result;
                    uploadedImage.style.display = 'block';
                    console.log("Image drawn on canvas");
                };
                
                img.onerror = function() {
                    console.error("Error loading image");
                };
                
                img.src = event.target.result;
            };
            
            reader.onerror = function() {
                console.error("Error reading file");
            };
            
            reader.readAsDataURL(file);
        }
    });

    // Prediction function with extensive error handling
    async function predict() {
        let imageData;
        
        try {
            // Check if there's an uploaded image
            if (uploadedImage.src && uploadedImage.style.display !== 'none') {
                // Use uploaded image
                imageData = uploadedImage.src;
                console.log("Using uploaded image for prediction");
            } else {
                // Use canvas drawing
                imageData = canvas.toDataURL('image/png');
                console.log("Using canvas image for prediction");
            }

            console.log("Image data length:", imageData.length);
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData }),
            });
            
            console.log("Prediction request sent");
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error("Prediction failed:", errorText);
                throw new Error('Prediction failed: ' + errorText);
            }
            
            const data = await response.json();
            console.log("Prediction result:", data);
            
            predictionSpan.textContent = data.prediction;
            
        } catch (error) {
            console.error('Prediction Error:', error);
            predictionSpan.textContent = 'Error: ' + error.message;
        }
    }

    // Button event listeners
    clearButton.addEventListener('click', clearCanvas);
    predictButton.addEventListener('click', predict);
    
    // Initial canvas clear
    clearCanvas();
    
    console.log("Canvas and prediction setup complete!");
});
