# Christmas Card Generator 🎄

An AI-powered application that transforms your family photos into magical Christmas cards using state-of-the-art image generation models and adds festive messages.

## 🌟 Features

- Transform regular photos into Christmas-themed scenes
- Preserve original people and poses while adding festive backgrounds
- Generate beautiful Christmas messages
- High-quality image processing with depth awareness
- Easy-to-use REST API interface

## 🛠️ Technology Stack

- **FastAPI**: Modern web framework for building APIs
- **PyTorch**: Deep learning framework
- **Diffusers**: Stable Diffusion and ControlNet implementation
- **Transformers**: For depth estimation and text generation
- **Pillow**: Image processing
- **CUDA**: GPU acceleration support

## 📋 Prerequisites

- Python 3.10+
- CUDA-capable GPU with at least 8GB VRAM
- 16GB+ RAM recommended

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/christmas-card-generator.git
cd christmas-card-generator
```

2. Install dependencies:
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
pip install transformers
pip install diffusers==0.21.4
pip install accelerate
pip install fastapi uvicorn python-multipart
pip install pillow
pip install transformers[torch]
pip install opencv-python
```

3. Download required models (optional - will download automatically on first run):
```bash
huggingface-cli download lllyasviel/control_v11f1p_sd15_depth
huggingface-cli download runwayml/stable-diffusion-v1-5
```

## 🚀 Usage

### Running the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### API Endpoints

#### Create Christmas Card
```bash
POST /create-christmas-card
```

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/create-christmas-card" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@family_photo.jpg"
```

**Response:**
```json
{
    "message": "Christmas card created successfully",
    "christmas_message": "Your festive message here",
    "image_base64": "base64_encoded_image_data"
}
```

## 🎨 Customization

### Modifying Christmas Messages

You can customize the Christmas messages by editing the `generate_christmas_message` method in the `ChristmasCardGenerator` class.

### Adjusting Image Generation

The following parameters can be tuned in the `create_christmas_card` method:
- `strength`: Controls the intensity of the transformation (0.0 to 1.0)
- `guidance_scale`: Controls how closely the image follows the prompt
- `num_inference_steps`: Controls the quality of the generation

## 🔍 Technical Details

The application uses:
- ControlNet with depth awareness for maintaining spatial consistency
- DDIM scheduler for stable image generation
- GPU memory optimization techniques
- Automatic image resizing for optimal processing

## ⚠️ Limitations

- Maximum input image size is limited to 768px (automatically resized)
- Processing time varies based on GPU capabilities
- Requires significant GPU memory

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for the base image generation
- [ControlNet](https://github.com/lllyasviel/ControlNet) for improved image control
- [Hugging Face](https://huggingface.co/) for model hosting and libraries

## 📞 Support

For issues and feature requests, please use the GitHub issue tracker.

## 🔮 Future Improvements

- [ ] Add more Christmas card templates
- [ ] Implement batch processing
- [ ] Add more customization options for text placement
- [ ] Improve processing speed
- [ ] Add more festive effects options