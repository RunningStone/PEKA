import torch
import torch.nn as nn
from peft import LoraConfig, AdaLoraConfig, HRAConfig, BoneConfig, get_peft_model
from typing import Optional, Union

class HistoPath_AlignmentModel(nn.Module):
    def __init__(self, encoder:nn.Module, translate_model:nn.Module, 
                lora_paras: Union[LoraConfig, AdaLoraConfig, HRAConfig, BoneConfig, None]=None):
        super(HistoPath_AlignmentModel, self).__init__()

        # load H-optimus-0 model output_dim 1536
        self.encoder = encoder
        # add linear layer to map feature dimension from 1536 to 512
        self.translate_model = translate_model

        if lora_paras is not None:
            # get num_blocks
            num_blocks = len(self.encoder.blocks)
            # set lora_paras target_modules
            lora_paras.target_modules = [f"blocks.{i}.attn.qkv" for i in range(num_blocks)]
            # apply PEFT
            self.encoder = get_peft_model(self.encoder, lora_paras)
            print(f"apply {type(lora_paras).__name__}")


    def forward(self, img_input_batch, gene_expression_input_batch=None):
        """
        image_patch: PIL Image or Tensor, size (3, H, W)
        gene_expression_input: input to scBERT, depends on scBERT implementation
        """
        with torch.cuda.amp.autocast():
            hoptimus_features = self.encoder(img_input_batch)
        hpred_features = self.translate_model(hoptimus_features)

        return hpred_features

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x):
        return self.mlp(x)