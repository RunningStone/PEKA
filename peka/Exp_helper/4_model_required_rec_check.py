import argparse
import sys
import os
import torch
import timm
from huggingface_hub import login
from torchvision import transforms
from dotenv import load_dotenv

print(f"🤖 running 3_model_required_rec_check.py")
# Load environment variables
proj_path = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
sys.path.append(proj_path)
sys.path.append(proj_path + "/PEKA/")
load_dotenv(dotenv_path=f"{proj_path}/PEKA/.env")
hf_token = os.getenv("HF_TOKEN")
print(f"⭐️ proj_path: {proj_path}")
print(f"⭐️ hf_token: {hf_token}")

if __name__ == "__main__":  
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compute resource check with optional LoRA.")
    parser.add_argument("--model_name", type=str, default="hf-hub:bioptimus/H-optimus-0", help="Name of the model.")
    parser.add_argument("--input_size", type=int, nargs=3, default=[3, 224, 224], help="Input size as [C, H, W].")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA parameter R.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter.")
    parser.add_argument("--target_modules", type=str, nargs='+', default=["qkv", "proj"], help="LoRA target modules.")
    parser.add_argument("--cuda_ram_test", action="store_true", help="Run CUDA RAM test.")
    parser.add_argument("--run_lora_test", action="store_true", help="Run LoRA test.")

    args = parser.parse_args()
    print(f"🤖 apply huggingface login...")
    login(token=hf_token)
    print(f"🤖 load model...")
    # Load model
    model = timm.create_model(args.model_name, pretrained=True, init_values=1e-5, dynamic_img_size=False)
    model.to("cuda")
    model.eval()
    print(f"🤖 define transform...")
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
    ])
    print(f"🤖 print trainable parameters in default full-parameters FT ...")
    # Function to print model parameters
    from peka.Model.utils import print_trainable_parameters
    print_trainable_parameters(model)
    print(f"Feature dimensions: {model(torch.randn(args.input_size).unsqueeze(0).to('cuda')).shape}")
    print(f"🤖 execute RAM tests ...")
    
    # Execute RAM tests if specified
    input = torch.randn(*args.input_size)
    input = transforms.ToPILImage()(input)
    if args.cuda_ram_test:
        print("1) Running CUDA RAM Test: Inference")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.inference_mode():
                features = model(transform(input).unsqueeze(0).to("cuda"))
        del features
        torch.cuda.empty_cache()
        print("Inference done.")

        print("Running CUDA RAM Test: Train")
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features = model(transform(input).unsqueeze(0).to("cuda"))
        del features
        torch.cuda.empty_cache()
        print("Train done.")

    if args.run_lora_test:
        print("2) Running LoRA test:")
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.target_modules,
                            lora_dropout=0.05, bias="none", task_type="SEQ_CLS", modules_to_save=["classifier"])
        lora_model = get_peft_model(model, config)
        print(f"🤖 print trainable parameters in LoRA FT ...")
        print_trainable_parameters(lora_model)
        torch.cuda.empty_cache()
        print("LoRA test done.")