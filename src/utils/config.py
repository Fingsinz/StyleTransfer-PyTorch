import yaml
from typing import Tuple
import torch
import torch.optim as optim

from data.image_dataset import ImageDataset, data_transform

from .recorder import Recorder

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

config = load_config("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_swanlab = config['swanlab'].get('enable', False)
vgg_version = config['vgg_version'] if config['vgg_version'] else 16

def get_attention(config=config) -> str:
    return config['attention']
    
def get_base(config=config) -> int:
    return config['base']
    
def get_epochs(training_config=config['training']) -> int:
    return training_config.get('epochs', 10)

def get_lr(training_config=config['training']) -> float:
    return training_config.get('lr', 1e-3)

def get_style_interval(training_config=config['training']) -> int:
    return training_config.get('style_interval', 20)

def get_training_weight(training_config=config['training']) -> list[float]:
    return [training_config.get('style_weight', 50),
            training_config.get('content_weight', 1),
            training_config.get('tv_weight', 1e-6)]

def get_record_per_epochs(training_config=config['training']) -> int:
    return training_config.get('record_per_epochs', 5)

def get_training_batch(training_config=config['training']) -> int:
    return training_config.get('batch_size', 8)

def get_test_batch(test_config=config['test']) -> int:
    return test_config.get('test_batch', 4)

def get_data(data_config=config['data']) -> Tuple[ImageDataset, ImageDataset]:
    content_dataset = ImageDataset(data_config['content_dir'], transform=data_transform)
    style_dataset = ImageDataset(data_config['style_dir'], transform=data_transform)
    return content_dataset, style_dataset

def get_num_workers(data_config=config['data']) -> int:
    return data_config.get('num_workers', 4)

def get_model_save(model_config=config['model_save_path']) -> str:
    return model_config

def get_inference_model_path(inference_config=config['inference']) -> list[str]:
    return [inference_config['metanet'], inference_config['transformnet']]

def get_output_path(inference_config=config['inference']) -> str:
    return inference_config.get('output', "../output/")

def print_training_config(recorder: Recorder, config=config):
    recorder.log("[CONFIG]")
    recorder.log(f"\tSwanlab: {is_swanlab}")
    recorder.log(f"\tDevice: {device}")
    recorder.log(f"\tVGG: {vgg_version}")    
    recorder.log(f"\tAttention: {config['attention']}")
    recorder.log(f"\tBase: {config['base']}")
    
    for key, value in config['training'].items():
        recorder.log(f"\t{key}: {value}")
        
    recorder.log("[DATA]")
    recorder.log(f"\tContent: {config['data']['content_dir']}")
    recorder.log(f"\tStyle: {config['data']['style_dir']}")
    
    recorder.log("[Test]")
    recorder.log(f"\tBatch: {config['test']['test_batch']}")

def print_inference_config(config=config):
    print("[CONFIG]")
    print(f"\tDevice: {device}")
    print(f"\tVGG: {vgg_version}")    
    print(f"\tAttention: {config['attention']}")
    print(f"\tBase: {config['base']}")
    
    for key, value in config['inference'].items():
        print(f"\t{key}: {value}")
