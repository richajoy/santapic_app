import io
import os
import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageEnhance, ImageFilter
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
from diffusers import StableDiffusionPipeline
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import requests
import base64
import logging

app = FastAPI(
    title="Santa Photo Generator API",
    description="API for generating Christmas photos with Santa",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Check for CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def download_sam_checkpoint():
    """Downloads the SAM checkpoint if it doesn't exist."""
    checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    if not os.path.exists("sam_vit_h_4b8939.pth"):
        print("Downloading SAM checkpoint...")
        response = requests.get(checkpoint_url, stream=True)
        with open("sam_vit_h_4b8939.pth", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete!")

# Initialize SAM
print("Initializing SAM...")
sam_checkpoint = "sam_vit_h_4b8939.pth"
download_sam_checkpoint()
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# Initialize Stable Diffusion
print("Loading Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
pipe = pipe.to(device)

class ErrorResponse(BaseModel):
    error: str

class SuccessResponse(BaseModel):
    message: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/", tags=["Health Check"])
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Santa Photo Generator API is running"}

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        # Load and process the uploaded image
        image_data = await file.read()
        logger.info("Image received")
        
        user_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        logger.info(f"Image loaded successfully. Size: {user_image.size}")

        # Generate colorful Santa scene
        santa_prompt = """
        A vibrant professional studio portrait photograph of Santa Claus sitting on an 
        ornate deep red velvet and gold trimmed chair on the left side. Santa wears a 
        rich red and white suit with golden details, has a natural white beard, rosy 
        cheeks, and a warm friendly smile. Beautifully wrapped Christmas presents in 
        red, gold, and green surround his chair. A stunning Christmas tree with warm 
        golden lights and colorful ornaments is in the background. Professional studio 
        lighting with warm tones, shot on Canon EOS R5, photorealistic, vivid colors, 
        ultra HD quality.
        """
        
        negative_prompt = """
        black and white, monochrome, grayscale, cartoon, anime, illustration, painting,
        drawing, artwork, 3d render, deformed, distorted, disfigured, bad anatomy,
        blurry, low quality, oversaturated, overexposed, bad lighting
        """

        # Generate Santa scene with color-focused settings
        santa_image = pipe(
            prompt=santa_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.5,
            num_inference_steps=50,
            width=1024,
            height=1024,
        ).images[0].convert("RGB")

        # Extract people with SAM
        image_array = np.array(user_image)
        masks = mask_generator.generate(image_array)
        
        if not masks:
            raise HTTPException(status_code=400, detail="No person detected in the image")
        
        # Filter masks by size and position
        valid_masks = []
        image_area = image_array.shape[0] * image_array.shape[1]
        
        for mask in masks:
            # Calculate relative size
            size_ratio = mask['area'] / image_area
            
            # Only consider masks that are significant in size (adjust thresholds as needed)
            if size_ratio > 0.05:  # Mask must be at least 5% of image
                valid_masks.append(mask)
        
        if not valid_masks:
            raise HTTPException(status_code=400, detail="No suitable person detected")
        
        # Create combined mask for all detected people
        combined_mask = np.zeros_like(image_array[:,:,0], dtype=np.uint8)
        for mask in valid_masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask['segmentation'].astype(np.uint8))
        
        # Refine mask edges
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.GaussianBlur(combined_mask * 255, (5,5), 0)
        
        # Convert to PIL mask
        mask_pil = Image.fromarray(combined_mask)
        
        # Extract people
        user_rgba = user_image.convert("RGBA")
        extracted_people = Image.new("RGBA", user_image.size, (0, 0, 0, 0))
        extracted_people.paste(user_rgba, (0, 0), mask=mask_pil)
        
        # Get tight bounding box
        bbox = extracted_people.getbbox()
        if bbox:
            # Add padding
            padding = 40  # Increased padding
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(user_image.size[0], bbox[2] + padding)
            y2 = min(user_image.size[1], bbox[3] + padding)
            extracted_people = extracted_people.crop((x1, y1, x2, y2))
        
        # Size calculations for composition
        w, h = santa_image.size
        
        # Calculate target size while maintaining aspect ratio
        target_height = int(h * 0.75)  # 75% of scene height
        person_aspect = extracted_people.size[0] / extracted_people.size[1]
        target_width = int(target_height * person_aspect)
        
        # Resize extracted people
        extracted_people = extracted_people.resize(
            (target_width, target_height),
            Image.LANCZOS
        )
        
        # Position calculation
        pos_x = int(w * 0.45)  # Slightly right of center
        pos_y = int(h * 0.2)   # Adjust vertical position to match Santa's level
        
        # Create final composite
        composite = santa_image.copy().convert("RGBA")
        
        # Add subtle shadow
        shadow = Image.new('RGBA', extracted_people.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rectangle(
            [0, target_height-20, target_width, target_height],
            fill=(0, 0, 0, 50)
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))
        composite.alpha_composite(shadow, (pos_x, pos_y + 10))
        
        # Add extracted people
        composite.alpha_composite(extracted_people, (pos_x, pos_y))
        
        # Final color adjustments
        final_image = composite.convert("RGB")
        
        # Enhance colors
        enhancer = ImageEnhance.Color(final_image)
        final_image = enhancer.enhance(1.1)  # Slightly boost colors
        
        enhancer = ImageEnhance.Contrast(final_image)
        final_image = enhancer.enhance(1.05)  # Slight contrast boost
        
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(1.02)  # Slight brightness boost
        
        # Save with high quality
        out_path = "final_card.jpg"
        final_image.save(out_path, quality=95)
        
        with open(out_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        return JSONResponse({
            "message": "Image processed successfully",
            "image_base64": f"data:image/jpeg;base64,{encoded_string}"
        })
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status", tags=["System"])
async def get_models_status():
    """Get the status of loaded AI models"""
    return {
        "sam_model": "loaded" if 'sam' in globals() else "not_loaded",
        "stable_diffusion": "loaded" if 'pipe' in globals() else "not_loaded",
        "device": device
    }

# Example cURL command for documentation
@app.get("/docs/curl", tags=["Documentation"])
async def get_curl_example():
    """Get example cURL command for API usage"""
    return {
        "curl_example": """
        curl -X 'POST' \
          'http://your-api-url/process-image' \
          -H 'accept: application/json' \
          -H 'Content-Type: multipart/form-data' \
          -F 'file=@your-image.jpg'
        """
    }

# Example Python code for documentation
@app.get("/docs/python", tags=["Documentation"])
async def get_python_example():
    """Get example Python code for API usage"""
    return {
        "python_example": """
import requests

url = "http://your-api-url/process-image"
files = {
    'file': ('image.jpg', open('path/to/your/image.jpg', 'rb'), 'image/jpeg')
}

response = requests.post(url, files=files)
if response.status_code == 200:
    data = response.json()
    # The base64 encoded image will be in data['image_base64']
    print("Success:", data['message'])
else:
    print("Error:", response.json()['detail'])
        """
    }

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI application using uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",  # Allows external access
        port=8000,       # Port number
        workers=1,       # Number of worker processes
        log_level="info"
    )
