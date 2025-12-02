import argparse
import torch
from PIL import Image
import time

from camera import capture_image  # Import the capture_image function
from speak import speak           # <-- Add this import

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from an image with nanoVLM")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint",
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF."
    )
    parser.add_argument(
        "--hf_model", type=str, default="lusxvr/nanoVLM-230M-8k",
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    parser.add_argument("--prompt", type=str, default="What do you see?",
                        help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=1,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=300,
                        help="Maximum number of tokens per output")
    parser.add_argument("--measure_vram", action="store_true",
                        help="Measure and display VRAM usage during model loading and generation")
    parser.add_argument("--num_images", type=int, default=20,
                        help="Number of images to capture and process")
    return parser.parse_args()

def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    source = args.checkpoint
    print(f"Loading weights from: {source}")

    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Getting Model")
    model = VisionLanguageModel.from_pretrained(source).to(device)
    print(f"VRAM?")
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.synchronize()
        model_vram_bytes = torch.cuda.memory_allocated(device)
        model_vram_mb = model_vram_bytes / (1024 ** 2)
        print(f"VRAM used after loading model: {model_vram_mb:.2f} MB")
    print(f"Tokenizer")
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens, model.cfg.lm_chat_template)
    resize_to_max_side_len = False
    if hasattr(model.cfg, "resize_to_max_side_len"):
        resize_to_max_side_len = model.cfg.resize_to_max_side_len
    image_processor = get_image_processor(model.cfg.max_img_size, model.cfg.vit_img_size, resize_to_max_side_len)

    for i in range(args.num_images):
        image_path = f"pictures/image{i}.png"
        print(f"\n--- Capturing image {i+1}/{args.num_images} ---")
        success = capture_image(output_filename=image_path)
        if not success:
            print(f"Skipping image {i}")
            continue

        print(f"Opening Image {image_path}")
        img = Image.open(image_path).convert("RGB")
        processed_image, splitted_image_ratio = image_processor(img)
        if not hasattr(tokenizer, "global_image_token") and splitted_image_ratio[0]*splitted_image_ratio[1] == len(processed_image) - 1:
            processed_image = processed_image[1:]

        image_string = get_image_string(tokenizer, [splitted_image_ratio], model.cfg.mp_image_token_length)
        print(f"Generating text for image {i}")
        messages = [{"role": "user", "content": image_string + args.prompt}]
        encoded_prompt = tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True)
        tokens = torch.tensor(encoded_prompt).to(device)
        img_t = processed_image.to(device)

        print("\nInput:\n ", args.prompt, "\nOutput:")
        for j in range(args.generations):
            gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens)
            out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

            if args.measure_vram and torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_vram_bytes = torch.cuda.max_memory_allocated(device)
                peak_vram_mb = peak_vram_bytes / (1024 ** 2)
                current_vram_bytes = torch.cuda.memory_allocated(device)
                current_vram_mb = current_vram_bytes / (1024 ** 2)
                print(f"  >> Generation {j+1}: {out}")
                print(f"     VRAM - Peak: {peak_vram_mb:.2f} MB, Current: {current_vram_mb:.2f} MB")
            else:
                print(f"  >> Generation {j+1}: {out}")
            speak(out)
        time.sleep(0.5)  # Optional: small delay between captures

if __name__ == "__main__":
    main()