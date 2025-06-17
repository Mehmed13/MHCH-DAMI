import os
import sys
import json
import argparse
import pickle as pkl
import random
import logging
from tqdm import tqdm
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore', category=FutureWarning)

class DataLoader:
    def __init__(self, mode='test', data_name='makeup', log_path=None):
        """
        Constant variable declaration and configuration.
        """
        # Get the base directory (where MHCH-DAMI is located)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set data directory paths relative to base_dir
        if data_name == 'clothing':
            dataset_folder_name = 'data/clothing'
        elif data_name == 'makeup':
            dataset_folder_name = 'data/makeup'
        else:
            raise ValueError("Please confirm the correct data name you entered.")

        # Construct full paths
        self.vocab_save_path = os.path.join(base_dir, dataset_folder_name, 'vocab.pkl')
        self.train_path = os.path.join(base_dir, dataset_folder_name, 'train.pkl')
        self.val_path = os.path.join(base_dir, dataset_folder_name, 'eval.pkl')
        self.test_path = os.path.join(base_dir, dataset_folder_name, 'test.pkl')

        # Print paths for debugging
        print(f"Data paths:")
        print(f"Base dir: {base_dir}")
        print(f"Vocab path: {self.vocab_save_path}")
        print(f"Train path: {self.train_path}")
        print(f"Val path: {self.val_path}")
        print(f"Test path: {self.test_path}")

        # Setup logging
        self.logger = logging.getLogger("Data Preprocessing")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if log_path:
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Initialize data containers
        self.role_list = []
        self.pos_list = []
        self.dialogues_ids_list = []
        self.dialogues_len_list = []
        self.dialogues_sent_len_list = []
        self.label_list = []
        self.senti_list = []
        self.tf_list = []

        self.mode = mode

    def load_pkl_data(self, mode='train'):
        """Load data from pickle files"""
        path_map = {
            'train': self.train_path,
            'eval': self.val_path,
            'test': self.test_path
        }
        
        load_path = path_map.get(mode)
        if not load_path:
            raise ValueError(f"{mode} mode not exists, please check it.")

        if not os.path.exists(load_path):
            raise ValueError(f"{load_path} not exists, please generate it firstly.")
            
        with open(load_path, 'rb') as fin:
            # Load data in sequence
            self.dialogues_ids_list = pkl.load(fin)
            self.role_list = pkl.load(fin)
            self.tf_list = pkl.load(fin)
            self.pos_list = pkl.load(fin)
            self.senti_list = pkl.load(fin)
            self.dialogues_sent_len_list = pkl.load(fin)
            self.dialogues_len_list = pkl.load(fin)
            self.label_list = pkl.load(fin)
            
        self.logger.info(f"Load variable from {load_path} successfully!")

    @staticmethod
    def load_config(config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as fp:
            return json.load(fp)

    @staticmethod
    def cut_sent_len(lens_list, max_len=50):
        """Cut sentence lengths to max_len"""
        return [[min(l, max_len) for l in lens] for lens in lens_list]

    def _prepare_batch(self, x1, x2, x3, label_list, sent_len, dia_len, i, batch_size, nb_classes=None):
        """Prepare a single batch of data"""
        batch_x1 = pad_sequences(x1[i:i + batch_size], maxlen=30, padding='post', truncating='post', dtype='float32')
        batch_x2 = pad_sequences(x2[i:i + batch_size], maxlen=30, padding='post', truncating='post', dtype='float32')
        batch_x3 = x3[i:i + batch_size]
        batch_sent_len = pad_sequences(sent_len[i:i + batch_size], maxlen=30, padding='post', truncating='post', dtype='int32')
        batch_dia_len = dia_len[i:i + batch_size]
        
        if nb_classes:
            labels_padded = pad_sequences(label_list[i:i + batch_size], maxlen=30, padding='post', truncating='post', dtype='int32', value=0)
            batch_labels = to_categorical(labels_padded, nb_classes)
            return batch_x1, batch_x2, batch_x3, batch_labels, batch_sent_len, batch_dia_len
        else:
            labels_padded = pad_sequences(label_list[i:i + batch_size], maxlen=30, padding='post', truncating='post', dtype='int32', value=0)
            return batch_x1, batch_x2, batch_x3, labels_padded, batch_sent_len, batch_dia_len

    def data_generator_sup(self, data_name='makeup', mode='test', batch_size=32, shuffle=True, nb_classes=2, epoch=0):
        """Supervised learning data generator"""
        print('Using data_generator_sup')
        self.load_pkl_data(mode=mode)
        
        x1 = self.dialogues_ids_list
        x2 = self.role_list
        x3 = self.senti_list
        label_list = self.label_list
        sent_len = self.cut_sent_len(self.dialogues_sent_len_list)
        dia_len = self.dialogues_len_list

        if shuffle or mode == 'train':
            list_pack = list(zip(x1, x2, x3, label_list, sent_len, dia_len))
            random.seed(epoch + 7)
            random.shuffle(list_pack)
            x1[:], x2[:], x3[:], label_list[:], sent_len[:], dia_len[:] = zip(*list_pack)

        for i in tqdm(range(0, len(dia_len), batch_size), desc="Processing:"):
            yield self._prepare_batch(x1, x2, x3, label_list, sent_len, dia_len, i, batch_size, nb_classes)

    def data_generator_crf(self, data_name='makeup', mode='test', batch_size=32, shuffle=True, nb_classes=2, epoch=0):
        """CRF data generator"""
        print('Using data_generator_crf')
        self.load_pkl_data(mode=mode)
        
        x1 = self.dialogues_ids_list
        x2 = self.role_list
        x3 = self.senti_list
        label_list = self.label_list
        sent_len = self.cut_sent_len(self.dialogues_sent_len_list)
        dia_len = self.dialogues_len_list

        if shuffle or mode == 'train':
            list_pack = list(zip(x1, x2, x3, label_list, sent_len, dia_len))
            random.seed(epoch + 7)
            random.shuffle(list_pack)
            x1[:], x2[:], x3[:], label_list[:], sent_len[:], dia_len[:] = zip(*list_pack)

        for i in tqdm(range(0, len(dia_len), batch_size), desc="Processing:"):
            yield self._prepare_batch(x1, x2, x3, label_list, sent_len, dia_len, i, batch_size)

    def data_generator_m(self, data_name='makeup', mode='test', batch_size=32, shuffle=True, nb_classes=2, epoch=0):
        """Multi-task data generator"""
        print('Using data_generator_m')
        self.load_pkl_data(mode=mode)
        
        x1 = self.dialogues_ids_list
        x2 = self.role_list
        x3 = self.senti_list
        tf_list = self.tf_list
        pos_list = self.pos_list
        label_list = self.label_list
        sent_len = self.cut_sent_len(self.dialogues_sent_len_list)
        dia_len = self.dialogues_len_list

        if shuffle or mode == 'train':
            list_pack = list(zip(x1, x2, x3, tf_list, pos_list, label_list, sent_len, dia_len))
            random.seed(epoch + 7)
            random.shuffle(list_pack)
            x1[:], x2[:], x3[:], tf_list[:], pos_list[:], label_list[:], sent_len[:], dia_len[:] = zip(*list_pack)

        for i in tqdm(range(0, len(dia_len), batch_size), desc="Processing:"):
            batch_x1, batch_x2, batch_x3, batch_labels, batch_sent_len, batch_dia_len = self._prepare_batch(
                x1, x2, x3, label_list, sent_len, dia_len, i, batch_size, nb_classes
            )
            batch_tf = tf_list[i:i + batch_size]
            # Pad position list with proper dimensions
            batch_pos = pos_list[i:i + batch_size]
            # Initialize with shape [batch_size, dia_max_len, sent_max_len, pos_dim]
            batch_paded_pos = np.zeros((len(batch_pos), 30, 50, 52), dtype=np.int32)
            
            for j, dialogue in enumerate(batch_pos):
                for k, sentence in enumerate(dialogue[:30]):  # Truncate to max 30 sentences
                    if k < len(dialogue):
                        # Convert sentence to numpy array first
                        sentence_array = np.array(sentence)
                        # Pad or truncate each position to match pos_dim
                        for m, pos in enumerate(sentence[:50]):  # Truncate to max 50 positions per sentence
                            pos_array = np.array(pos) if isinstance(pos, (list, np.ndarray)) else np.array([pos])
                            if len(pos_array) > 52:
                                batch_paded_pos[j, k, m, :] = pos_array[:52]
                            else:
                                batch_paded_pos[j, k, m, :len(pos_array)] = pos_array
                                # Rest is already zeros from initialization
            
            yield batch_x1, batch_x2, batch_x3, batch_tf, batch_paded_pos, batch_labels, batch_sent_len, batch_dia_len 