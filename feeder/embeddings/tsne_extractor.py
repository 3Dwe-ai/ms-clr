import torch
import torchlight
from torchlight import import_class
import argparse
import yaml
import random
import numpy as np
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from processor.processor import Processor

class EmbeddingExtractor(Processor):
    def __init__(self, config_path, device='cuda:0'):
        # Load configuration
        self.load_config(config_path)
        self.arg.save_log = False
        self.arg.print_log = False
        self.arg.use_gpu = device == 'cuda:0'
        
        # Initialize the environment
        self.init_environment()
        
        # Load the pretrained model
        self.load_model()
        
        # Move model to the specified device
        self.device = device
        self.model.to(self.device)

    def load_config(self, config_path):
        # Load the configuration file
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Set arguments from the config file
        self.arg = argparse.Namespace(**config)
        
    def load_model(self):
        # Load model architecture
        self.model = self.io.load_model(self.arg.model, **self.arg.model_args)
        
        # Load pretrained weights
        self.model.load_state_dict(torch.load(self.arg.weights))
        
    
        