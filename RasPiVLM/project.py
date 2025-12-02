import argparse
import torch
from PIL import Image
import subprocess
import os
import time
import pyttsx3

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from an image with nanoVLM")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF."
    )
    parser.add_argument(
        "--hf_model", type=str, default="lusxvr/nanoVLM-222M",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    parser.add_argument("--prompt", type=str, default="What is this?",
                        help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=1,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=80,
                        help="Maximum number of tokens per output")
    return parser.parse_args()

def take_picture(filename):
    dest_dir = 'pictures'
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    subprocess.run(['rpicam-still', '-t', '5000', '-o', dest_path], check=True)
    return dest_path

def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Loading weights from: {source}")
    model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()

    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    template = f"Question: {args.prompt} Answer:"

    #engine = pyttsx3.init()
    #engine.say("PI VLM Model Starting")
    #engine.runAndWait()
    with open("out.txt", "w") as outfile:
        for i in range(5):
            filename = f"test_{i+1:03d}.png"
            image_path = take_picture(filename)
            print(f"\n[{i+1}/100] Picture saved to {image_path}")

            encoded = tokenizer.batch_encode_plus([template], return_tensors="pt")
            tokens = encoded["input_ids"].to(device)

            img = Image.open(image_path).convert("RGB")
            img_t = image_processor(img).unsqueeze(0).to(device)

            print("Input:", args.prompt, "\nOutputs:")
            for j in range(args.generations):
                gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens)
                out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
                print(f"  >> Generation {j+1}: {out}")
                outfile.write(f"Picture {filename}, Generation {j+1}: {out}\n")
                #engine.say(out)
                #engine.runAndWait()

            # Optional: wait a bit between captures
            time.sleep(1)

if __name__ == "__main__":
    main()
