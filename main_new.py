#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import pickle as pkl
import logging
import argparse
import warnings
import tensorflow as tf
from Network import Network
from networks.DAMI_new import DAMI
from data_loader_new import DataLoader
from utility import *

# Set random seeds for reproducibility
RANDOM_SEED = 7
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CONFIG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')

def setup_logging(args):
    """Setup logging configuration"""
    logger = logging.getLogger("Tensorflow")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' ')[:3])

    # Create log directory if it doesn't exist
    os.makedirs(args.log_path, exist_ok=True)

    # Set up log file name
    log_path = os.path.join(
        args.log_path,
        f"{args.model_name}.{args.data_name}.{args.phase}{args.suffix}.{args.mode}.{args.ways}.{now_time}.log"
    )

    if os.path.exists(log_path):
        os.remove(log_path)

    if args.log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def load_configs(data_loader, args):
    """Load data and model configurations"""
    data_config_path = os.path.join(CONFIG_ROOT, 'data', f'config.{args.data_name}.json')
    model_config_path = os.path.join(CONFIG_ROOT, 'model', f'config.{args.model_name}.json')
    
    data_config = data_loader.load_config(data_config_path)
    model_config = data_loader.load_config(model_config_path)
    
    return data_config, model_config

def get_network(model_name, memory, vocab, model_config, logger):
    """Initialize the appropriate network model"""
    if model_name == 'network':
        network = Network(memory=memory, vocab=vocab)
    elif model_name == 'dami':
        network = DAMI(memory=memory, vocab=vocab, config_dict=model_config)
    else:
        logger.error(f"Model {model_name} not found. Please check the model name.")
        raise ValueError(f"Model {model_name} not found. Please check the model name.")
    
    return network

def train(network, data_generator, keep_prob, epochs, data_name,
          mode='train', batch_size=20, nb_classes=2, shuffle=True,
          is_val=True, is_test=True, save_best=True, ways='crf'):
    """Train the network using the specified method"""
    if ways == 'crf':
        network.train_crf(
            data_generator=data_generator,
            keep_prob=keep_prob,
            epochs=epochs,
            data_name=data_name,
            mode=mode,
            batch_size=batch_size,
            nb_classes=nb_classes,
            shuffle=shuffle,
            is_val=is_val,
            is_test=is_test,
            save_best=save_best
        )
    elif ways == 'sup':
        network.train_sup(
            data_generator=data_generator,
            keep_prob=keep_prob,
            epochs=epochs,
            data_name=data_name,
            mode=mode,
            batch_size=batch_size,
            nb_classes=nb_classes,
            shuffle=shuffle,
            is_val=is_val,
            is_test=is_test,
            save_best=save_best
        )
    elif ways == 'dami':
        network.train(
            data_generator=data_generator,
            keep_prob=keep_prob,
            epochs=epochs,
            data_name=data_name,
            mode=mode,
            batch_size=batch_size,
            nb_classes=nb_classes,
            shuffle=shuffle,
            is_val=is_val,
            is_test=is_test,
            save_best=save_best
        )
    else:
        raise ValueError(f"Invalid training method: {ways}. Please check the 'ways' parameter.")

def main():
    start_t = time.time()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser('Tensorflow')
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict')
    parser.add_argument('--data_name', default='makeup', help='Data name to use')
    parser.add_argument('--model_name', default='dami', help='Model name to use')
    parser.add_argument('--model_path', default='none', help='Model path to load')
    parser.add_argument('--memory', default='0.', help='GPU memory to use')
    parser.add_argument('--gpu', default='0', help='GPU to use')
    parser.add_argument('--log_path', default='./networks/logs/', help='Path for log files')
    parser.add_argument('--suffix', default='.128', help='Suffix for log differentiation')
    parser.add_argument('--mode', default='train', help='Mode fold to try')
    parser.add_argument('--ways', default='dami', help='Training method to use')
    parser.add_argument('--use_pretrain', default='1', help='Whether to use pretrained model')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args)
    logger.info(f'Random seed: {RANDOM_SEED}')
    logger.info(f'Running with args: {args}')
    
    # Initialize data loader
    data_loader = DataLoader(data_name=args.data_name)
    
    # Load configurations
    logger.info('Loading dataset and vocabulary...')
    data_config, model_config = load_configs(data_loader, args)
    logger.info(f'Data config: {data_config}')
    logger.info(f'Model config: {model_config}')
    
    # Extract configuration parameters
    model_name = model_config['model_name']
    batch_size = model_config['batch_size']
    epochs = model_config['epochs']
    keep_prob = model_config['keep_prob']
    mode = args.mode
    is_val = model_config['is_val']
    is_test = model_config['is_test']
    save_best = model_config['save_best']
    shuffle = model_config['shuffle']
    data_name = data_config['data_name']
    nb_classes = data_config['nb_classes']
    
    # Load vocabulary
    vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', data_name, 'vocab.pkl')
    with open(vocab_path, 'rb') as fp:
        vocab = pkl.load(fp)
    
    # Initialize network
    memory = float(args.memory)
    logger.info(f"Memory in train: {memory}")
    network = get_network(model_name, memory, vocab, model_config, logger)
    
    # Configure network
    network.set_nb_words(min(vocab.size(), data_config['nb_words']) + 1)
    network.set_data_name(data_name)
    network.set_name(f"{model_name}{args.suffix}train")
    network.set_from_model_config(model_config)
    network.set_from_data_config(data_config)
    
    # Select data generator
    if 'sup' in args.ways:
        logger.info('Using data_generator_sup')
        data_generator = data_loader.data_generator_sup
    elif args.ways == 'crf':
        logger.info('Using data_generator_crf')
        data_generator = data_loader.data_generator_crf
    elif args.ways == 'dami':
        logger.info('Using data_generator_m')
        data_generator = data_loader.data_generator_m
    else:
        raise ValueError(f"Invalid data generator: {args.ways}. Please check the 'ways' parameter.")
    
    # Build network graph
    network.build_graph()
    logger.info(f'Network configuration: {network.__dict__}')
    
    # Train or evaluate
    if args.phase == 'train':
        train(network, data_generator, keep_prob, epochs, data_name,
              mode=mode, batch_size=batch_size, nb_classes=nb_classes, shuffle=shuffle,
              is_val=is_val, is_test=is_test, save_best=save_best, ways=args.ways)
    else:
        logger.error(f"Invalid phase: {args.phase}. Please use 'train' or 'evaluate'.")
        raise ValueError(f"Invalid phase: {args.phase}. Please use 'train' or 'evaluate'.")
    
    # Log completion time
    elapsed_time = int(time.time()) - start_t
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60
    logger.info(f'The whole program took: {hours}h: {minutes}m: {seconds}s')
    print("DONE!")

if __name__ == '__main__':
    main() 