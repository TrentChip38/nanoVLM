//Gonna just put all the instructions of everything I do for Local laptop Nano VLM

git clone https://github.com/huggingface/nanoVLM.git
cd nanoVLM

python3 -m venv .venv
.venv\Scripts\activate


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # For CUDA-enabled GPUs
pip install transformers accelerate safetensors pillow
(Adjust cu118 if you have a different CUDA version or remove --index-url if you are not using a GPU.)

pip install torch numpy torchvision pillow datasets huggingface-hub transformers wandb accelerate safetensors

#Python
from models.vision_language_model import VisionLanguageModel
model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM")


Using the generate.py file it can already work with pre-trained models
Need to download more things to make it work offline

pip install huggingface-hub
hf download lusxvr/nanoVLM --local-dir checkpoint

#Camera--------------
pip install opencv-python

