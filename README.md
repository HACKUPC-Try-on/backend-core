# What's this repo?
This repo is a usefull backend for an innovative project we've worked on in our journey through HackUPC 2024.
By using lots of Inditex's data, we've created a new way to try on clothes based on a recommendation system that works with embeddings.

# Technologies used
- [Python](https://www.python.org/)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything](https://segment-anything.com/)
- [Faiss](https://ai.meta.com/tools/faiss/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [CLIP](https://openai.com/index/clip)
- [SigLIP](https://huggingface.co/docs/transformers/main/en/model_doc/siglip)

# How to set it up?
```bash
# Clone the repo
git clone https://github.com/HACKUPC-Try-on/backend-core

# Make sure to have poetry and Python 3.10 running and install the dependencies
poetry install --no-root

# Install Grounding DINO
mkdir -p packages && cd packages
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
poetry run python setup.py install

# Weights
cd ../../
mkdir -p weights/dino && cd weights/dino
cp ../../packages/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py .
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

cd ..
mkdir -p weights/sam && cd weights/sam
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Run the API!
poetry run python uvicorn main:app
```
That's it! Our API is running on localhost, on port 8000!

# Are you a developer?
If you want to contribute to our project, feel free to do!
Make sure to check out our `CONTRIBUTING.md` file