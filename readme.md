# Santa Photo Generator 🎅

An AI-powered application that combines your family photos with generated Santa scenes to create magical Christmas portraits. The application uses Stable Diffusion to generate a high-quality Santa scene and seamlessly combines it with your uploaded photo.

## 🌟 Features

- Generates a professional Santa scene with Christmas decorations
- Preserves people from your original photo
- Combines both images seamlessly
- Places extracted people next to Santa
- Adds proper lighting and shadows
- High-quality image processing

## 🛠️ Technology Stack

- **FastAPI**: Web framework for the API
- **Stable Diffusion**: For generating Santa scenes
- **SAM (Segment Anything Model)**: For precise person extraction
- **PIL (Pillow)**: For image processing and composition
- **CUDA**: GPU acceleration support

## 📋 Prerequisites

- Python 3.10+
- CUDA-capable GPU
- 8GB+ VRAM recommended
- 16GB+ RAM recommended

## 🔧 Installation

1. Install required packages:

```bash
pip install torch torchvision
pip install diffusers==0.21.4
pip install transformers
pip install accelerate
pip install fastapi uvicorn python-multipart
pip install pillow
```

## 🚀 Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### API Endpoint

#### Process Image

```bash
POST /process-image
```

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/process-image" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@family_photo.jpg"
```

**Response:**
```json
{
    "message": "Image processed successfully",
    "image_base64": "base64_encoded_image_data"
}
```

## 🎨 Key Features

### Santa Scene Generation
- Professional studio-quality Santa scene
- Warm lighting and Christmas decorations
- Christmas tree and presents in background
- Consistent high-quality output

### Person Extraction
- Precise segmentation of people from original photos
- Maintains original quality and details
- Preserves family poses and expressions

### Image Composition
- Seamless blending of extracted people with Santa scene
- Professional positioning and scaling
- Shadow effects for natural grounding
- Color balance and enhancement

## ⚠️ Limitations

- Image generating is not perfect. Code base is in works.
Input images should have clear subjects
- Processing time depends on GPU capabilities
- Requires specific versions of dependencies

## 🔧 Configuration

Key parameters in the image processing pipeline:

```python
# Santa generation parameters
guidance_scale=8.0
num_inference_steps=75
width=1024
height=1024

# Image composition parameters
target_height = int(h * 0.75)  # 75% of scene height
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📝 License

This project is licensed under the MIT License.

## 🔮 Future Improvements

- [ ] Add multiple Santa scene options
- [ ] Improve blending algorithms
- [ ] Add batch processing capability
- [ ] Optimize processing speed
- [ ] Add more customization options