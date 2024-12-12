import os
import argparse  # Import argparse for command-line argument parsing
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from transformers.generation.streamers import TextIteratorStreamer
from PIL import Image
import torch
import warnings
import csv
from datetime import datetime
import torch.nn.functional as F  # For softmax function
from accelerate import Accelerator
import subprocess
import signal
import atexit
os.environ['TMPDIR'] = '/data/guan3/tmp'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Confirm the setting
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
accelerator = Accelerator()
print(f"Using device: {accelerator.device}")
print("Active CUDA devices after initialization:")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}, Memory allocated: {torch.cuda.memory_allocated(i)}")

device = accelerator.device

# Ensure we run in GPU mode, skipping bitsandbytes if GPU is not available
if torch.cuda.is_available():
    import bitsandbytes as bnb
    #device = 'cuda'
    # device_ids = list(range(torch.cuda.device_count()))  # Get all available GPU devices
    # print(f"Using GPUs: {device_ids}")
else:
    print("Running on CPU, skipping bitsandbytes...")
    device = 'cpu'

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message="The installed version of bitsandbytes was compiled without GPU support.")

repo_id = "liuhaotian/llava-v1.6-vicuna-13b"
cache_dir = "/data/guan3/LLaVA/model_zoo/llava_model"

# Define cleanup function
def close_model_and_free_memory():
    print("Cleaning up GPU memory...")
    del model, tokenizer, image_processor
    torch.cuda.empty_cache()  # Clear GPU memory cache
    accelerator.free_memory()  # Free memory managed by Accelerator
    sys.exit(0)  # Exit the program



# Signal handler for Ctrl-C and termination
def signal_handler(sig, frame):
    print("Interrupt received, shutting down gracefully...")
    close_model_and_free_memory()

# Register the signal handler for SIGINT and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Register the cleanup function to run on program exit
atexit.register(close_model_and_free_memory)

def load_model_and_processor():
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        repo_id, model_name="llava-v1.6-vicuna-13b", model_base=None, load_8bit=False, load_4bit=False)

    # Use Hugging Face Accelerate to manage resources
    model, tokenizer, image_processor = accelerator.prepare(model, tokenizer, image_processor)

    return model, tokenizer, image_processor

# Save the original model weights
def save_original_weights(model):
    with torch.no_grad():
        if device != 'cpu':
            original_weights = {name: param.clone().cpu() for name, param in model.named_parameters()}
            torch.cuda.empty_cache()  # Free GPU memory
        else:
            original_weights = {name: param.clone() for name, param in model.named_parameters()}
    return original_weights

# Restore the model weights from the original state
def restore_original_weights(model, original_weights):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(original_weights[name])

# Image preprocessing
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

# Clean model output
def clean_output(output):
    return output.replace("</s>", "").strip()

# Modified for multi-GPU inference
def inference_with_probabilities(model, tokenizer, image_processor, image_path, prompt, top_p=1.0, temperature=1.0, max_tokens=20, top_k=10):
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()

    image_data = load_image(image_path)
    image_tensor = image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].float().to(device)

    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    # Check if the model is wrapped in DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # Access the original model

    try:
        with torch.no_grad():
            # Step 1: Generate tokens using top_p and temperature
            generated_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=True,
                top_p=top_p,              # Use top_p nucleus sampling
                temperature=temperature,  # Adjust temperature
                max_new_tokens=max_tokens
            )
            
            # Step 2: Pass generated tokens back to the model to get logits
            outputs = model(input_ids=generated_ids, images=image_tensor)
            logits = outputs.logits  # Extract the logits for the generated tokens

            # Step 3: Convert logits to probabilities
            probabilities = F.softmax(logits, dim=-1)

            # Step 4: Get top-k predictions from the last step
            top_probabilities, top_indices = torch.topk(probabilities[:, -1, :], k=10, dim=-1)

            # Decode the top tokens to human-readable words
            top_words = [tokenizer.decode(idx.item()) for idx in top_indices.squeeze()]
            top_probs = top_probabilities.squeeze().tolist()

            # Print the top predictions and their probabilities
            print("Top predictions:")
            for word, prob in zip(top_words, top_probs):
                print(f"Word: {word}, Probability: {prob}")

            return top_words, top_probs

    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def inference(model, tokenizer, image_processor, image_path, prompt, top_p=1.0, temperature=1.0, max_tokens=20):
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()

    # Preprocess the image and move to the correct device
    image_data = load_image(image_path)
    image_tensor = image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].float().to(device).half()
    image_tensor = accelerator.prepare(image_tensor)  # Let Accelerate handle device management

    # Prepare the prompt and tokenize input
    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    input_ids = accelerator.prepare(input_ids)  # Ensure input_ids are prepared for multi-GPU

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=300.0)

    try:
        with torch.no_grad():  # Prevent gradients from being calculated (save memory)
            # Use model.generate with tensors prepared by Accelerate
            generated_ids = model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=False,
                # temperature=temperature,
                # top_p=top_p,
                max_new_tokens=max_tokens,
                streamer=streamer,
                use_cache=True
            )

            # Convert streamed results into a string
            result = ""
            for new_text in streamer:
                result += new_text
                # if new_text.endswith(stop_str):
                #     break

        # Clean up: delete tensors and free GPU memory
        del image_tensor, input_ids, generated_ids
        torch.cuda.empty_cache()  # Explicitly free the GPU cache
        accelerator.free_memory() 
        return clean_output(result)

    except Exception as e:
        print(f"Error during inference: {e}")
        # Clean up memory in case of an error
        del image_tensor, input_ids
        torch.cuda.empty_cache()
        return None

# Sequential image processing
def process_images_and_infer(model, tokenizer, image_processor, image_files, prompt, i, j, noise_std, modification_percentage=1):
    # Create the 'result' directory if it doesn't exist
    result_dir = f"./result/noise{noise_std}_mod{modification_percentage}"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_filename = os.path.join(result_dir, f"layer{i}_to_layer{j}_noise{noise_std}_percentage_{modification_percentage}_inf_{timestamp}.csv")
    
    # Open the CSV file for writing
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(["Image File", "Inference Result"])
        
        for image_file in image_files:
            full_image_path = os.path.join("./PNT_images", image_file)
            result = inference(model, tokenizer, image_processor, full_image_path, prompt)

            # Process results for each image
            image_name = os.path.basename(full_image_path)
            print(f"Result for {image_name}: {result}")
                
            # Write the result to the CSV file
            csv_writer.writerow([image_name, result])
            # Free up GPU memory after each image
            del result
            torch.cuda.empty_cache()
    print(f"Inference results saved to {csv_filename}")

#Sequential image processing with probabilities
def process_images_and_infer_probabilities(model, tokenizer, image_processor, image_files, prompt, i, j, noise_std, modification_percentage=1):
    # Create the 'result' directory if it doesn't exist
    result_dir = f"./result/noise{noise_std}_mod{modification_percentage}"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_filename = os.path.join(result_dir, f"layer{i}_to_layer{j}_noise{noise_std}_percentage_{modification_percentage}_inf_{timestamp}.csv")
    
    # Open the CSV file for writing
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(["Image File", "Top 10 Words", "Probabilities"])
        
        for image_file in image_files:
            full_image_path = os.path.join("./PNT_images", image_file)
            top_words, top_probs = inference_with_probabilities(model, tokenizer, image_processor, full_image_path, prompt)

            # Process results for each image
            image_name = os.path.basename(full_image_path)
            print(f"Result for {image_name}: {top_words} with probabilities {top_probs}")
                
            # Write the result to the CSV file
            csv_writer.writerow([image_name, ', '.join(top_words), ', '.join(map(str, top_probs))])
            del image_tensor, input_ids, generated_ids, outputs, logits, probabilities, top_probabilities, top_indices
            torch.cuda.empty_cache()

            return top_words, top_probs
    print(f"Inference results saved to {csv_filename}")

def modify_one_layer_percentage(model, layer_idx, noise_std=0.1, modification_percentage=0.2, seed=42):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Modify only parameters within the specific layer index that belong to model.layers.*
            if f'layers.{layer_idx}.' in name and 'vision_tower' not in name:
                # Ensure all tensors are on the same device as param
                device = param.device
                
                # Determine the total number of elements in the parameter tensor
                num_elements = param.numel()
                
                # Calculate the number of elements to modify (based on the percentage)
                num_to_modify = int(num_elements * modification_percentage)
                
                # Generate random indices for the elements to modify
                indices_to_modify = torch.randperm(num_elements, device=device)[:num_to_modify]  # Moved to device
                
                # Flatten the parameter tensor for easy indexing
                param_flat = param.view(-1)
                
                # Create noise tensor of the same size as the number of elements to modify
                noise = torch.randn(num_to_modify, device=device) * noise_std  # Moved to device
                
                # Add noise to the selected elements
                param_flat[indices_to_modify] += noise
                
                print(f"Modified {num_to_modify} elements in {name}")


def modify_one_layers_in_steps_percentage(model, tokenizer, image_processor, image_files, noise_std=0.1,modification_percentage=0.2):
    # Save the original model weights
    if device != 'cpu':
        torch.cuda.empty_cache()  # Free GPU memory
    original_weights = save_original_weights(model)

    # Get the number of layers in the language model (excluding vision layers)
    language_layers = [name for name, _ in model.named_parameters() if 'layers' in name and 'vision_tower' not in name]
    num_layers = len(set(name.split('.')[1] for name in language_layers if 'layers' in name))
    print(f"Total number of layers: {num_layers}")
    num_layers = 40
    # Loop through model layers and modify them one layer at a time
    for i in range(num_layers):
        print(f"\nModifying layer: {i}")

        # Modify all sub-parameters in the specified layer
        modify_one_layer_percentage(model, i, noise_std, modification_percentage)

        # Run inference after modifying the layer
        prompt = "Name the picture using only one word. Do not use more than one word."

        # Use process_images_and_infer_top10_first_token to get probablities or process_images_and_infer to not
        process_images_and_infer_top10_first_token(model, tokenizer, image_processor, image_files, prompt, i, i, noise_std, modification_percentage)

        # Restore original weights after inference
        restore_original_weights(model, original_weights)
        print(f"Restored original weights after processing layer {i}")


import torch.nn.functional as F


def process_images_and_infer_top10_first_token(model, tokenizer, image_processor, image_files, prompt, i, j, noise_std,
                                               modification_percentage=1):
    # Create the 'result' directory if it doesn't exist
    result_dir = f"./result/noise{noise_std}_mod{modification_percentage}"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_filename = os.path.join(result_dir,
                                f"layer{i}_to_layer{j}_noise{noise_std}_percentage_{modification_percentage}_inf_{timestamp}.csv")

    # Open the CSV file for writing
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(["Image File", "Inference Result", "Top 10 Words", "Top 10 Probabilities"])

        for image_file in image_files:
            full_image_path = os.path.join("./PNT_images", image_file)
            result, top_words, top_probs = inference_top10_first_token(model, tokenizer, image_processor,
                                                                       full_image_path, prompt)

            # Process results for each image
            image_name = os.path.basename(full_image_path)
            print(f"Result for {image_name}: {result}")
            print(f"Top 10 words for first token: {top_words}")
            print(f"Top 10 probabilities for first token: {top_probs}")

            # Write the result and top 10 probabilities to the CSV file
            csv_writer.writerow([image_name, result, ', '.join(top_words), ', '.join(map(str, top_probs))])

            # Free up GPU memory after each image
            del result, top_words, top_probs
            torch.cuda.empty_cache()

    print(f"Inference results saved to {csv_filename}")

def inference_top10_first_token(model, tokenizer, image_processor, image_path, prompt, max_tokens=5):
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()

    # Preprocess the image and move to the correct device
    image_data = load_image(image_path)
    image_tensor = image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].float().to(device).half()
    image_tensor = accelerator.prepare(image_tensor)  # Let Accelerate handle device management

    # Prepare the prompt and tokenize input
    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    input_ids = accelerator.prepare(input_ids)  # Ensure input_ids are prepared for multi-GPU

    try:
        with torch.no_grad():  # Prevent gradients from being calculated (save memory)
            # Generate tokens with up to 5 tokens but capture logits of the first token only
            generated_output = model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_scores=True,  # To capture logits directly
                return_dict_in_generate=True
            )

            # Get the logits for the first generated token
            first_token_logits = generated_output.scores[0].squeeze()  # Extract scores for the first generated token
            probabilities = F.softmax(first_token_logits, dim=-1)

            # Get the top 10 tokens and their probabilities
            top_probabilities, top_indices = torch.topk(probabilities, k=10, dim=-1)
            top_words = [tokenizer.decode(idx.item()).strip() for idx in top_indices]
            top_probs = top_probabilities.tolist()

            # Convert generated tokens to text, stopping at the end token if it appears
            result = tokenizer.decode(generated_output.sequences[0], skip_special_tokens=True).split('</s>')[0].strip()

            return result, top_words, top_probs

    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None, None



# Main program
if __name__ == "__main__":
    # Parse command-line arguments for noise level
    parser = argparse.ArgumentParser(description="Run LLaVA inference with noise modification.")
    parser.add_argument('--noise_std', type=float, default=0.1, help='Standard deviation of noise to be added to model layers')
    parser.add_argument('--modification_percentage', type=float, default=0.2, help='Percentage of elements in the layer to modify')
    args = parser.parse_args()

    # Load model and processor
    model, tokenizer, image_processor = load_model_and_processor()  
    image_path = "./PNT_images/"
    image_files = [f for f in os.listdir(image_path) if f.endswith('.png')]

    image_files.sort(key=lambda f: int(f.split('img')[-1].split('.png')[0]))

    # Run the modification and inference steps
    modify_one_layers_in_steps_percentage(model, tokenizer, image_processor, image_files, noise_std=args.noise_std, modification_percentage=args.modification_percentage)

    # Call this at the end of your script or on interruption
    close_model_and_free_memory()