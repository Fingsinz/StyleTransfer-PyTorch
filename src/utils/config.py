import yaml
import torch
import torch.optim as optim

from data.image_dataset import ImageDataset, data_transform

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

config = load_config("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_swanlab = config['swanlab'].get('enable', False)
vgg_version = config['vgg_version'] if config['vgg_version'] else 16
    
def get_attention(training_config=config['attention']):
    return [training_config.get('channel_attention', True),
            training_config.get('spatial_attention', True)]
    
def get_epochs(training_config=config['training']):
    return training_config.get('epochs', 10)

def get_lr(training_config=config['training']):
    return training_config.get('lr', 1e-3)

def get_style_interval(training_config=config['training']):
    return training_config.get('style_interval', 20)

def get_training_weight(training_config=config['training']):
    return [training_config.get('style_weight', 50),
            training_config.get('content_weight', 1),
            training_config.get('tv_weight', 1e-6)]

def get_record_per_epochs(training_config=config['training']):
    return training_config.get('record_per_epochs', 5)

def get_training_batch(training_config=config['training']):
    return training_config.get('batch_size', 8)

def get_test_batch(test_config=config['test']):
    return test_config.get('test_batch_size', 4)

def get_base(training_config=config['training']):
    return training_config.get('base', 32)

def get_data(data_config=config['data']):
    content_dataset = ImageDataset(data_config['content_dir'], transform=data_transform)
    style_dataset = ImageDataset(data_config['style_dir'], transform=data_transform)
    return content_dataset, style_dataset

def get_num_workers(data_config=config['data']):
    return data_config.get('num_workers', 4)

def get_model_save(model_config=config['model_save_path']):
    return model_config
