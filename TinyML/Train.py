# Name: Sagnik Sinha
# Email: ssinha348@gatech.edu
# High-Level Overview:
# This code trains and validates an image captioning model based on the Show, Attend and Tell architecture,
# which uses an encoder-decoder framework with attention. The encoder extracts image features using a convolutional
# neural network (CNN), and the decoder generates captions word by word using an RNN with attention over the encoder's output.

# Imports and Setup
# Purpose: Imports libraries and modules for deep learning, data processing, evaluation, and file handling.
# CUDA_LAUNCH_BLOCKING: Ensures proper CUDA debugging by synchronizing kernel launches.
# gc: Helps manage memory by explicitly invoking garbage collection.

import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models_resnet_quantized import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf
from nltk.translate.gleu_score import corpus_gleu
from nltk.translate.nist_score import corpus_nist
from nltk.translate.ribes_score import corpus_ribes
from torchvision.models import ResNet101_Weights
from nltk.corpus import wordnet as wn
import gc
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from rouge_score import rouge_scorer
import pandas as pd
import openpyxl

# Defines paths and hyperparameters for training and model architecture.
# Data parameters
data_folder_path = 'C:/Users/sagni/Documents/Personal Files/CS7643/Group project/Show, Attend And Tell/media_flickr8k/ssd/caption data'  # folder with data files saved by create_input_files.py
data_file_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
embedding_dim = 512  # dimension of word embeddings
attention_layer_dim = 512  # dimension of attention linear layers
decoder_hidden_dim = 512  # dimension of decoder RNN
dropout_rate = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
total_epochs = 1  # number of epochs to train for (if early stopping is not triggered)
epochs_since_last_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
data_loader_workers = 0 #os.cpu_count()  # for data-loading; right now, only 1 works with h5py
encoder_learning_rate = 1e-4  # learning rate for encoder if fine-tuning
decoder_learning_rate = 4e-4  # learning rate for decoder
gradient_clip = 5.  # clip gradients at an absolute value of
alpha_c_value = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4_score = 0.  # BLEU-4 score right now
print_frequency = 300  # print training/validation stats every __ batches
fine_tune_encoder_flag = False  # fine-tune encoder?
checkpoint_path = None  # path to checkpoint, None if none

# Purpose: Coordinates model training and validation.
# Loads the vocabulary map (word_map) that maps words to indices.
def main():
    """
    Training and validation.
    """

    global best_bleu4_score, epochs_since_last_improvement, checkpoint_path, start_epoch, fine_tune_encoder_flag, data_file_name, word_map
    metrics_columns = ['Epoch', 'Train Loss', 'Train Top-5 Accuracy', 'Val Loss', 'Val Top-5 Accuracy', 'bleu', 'chrf', 'gleu', 'nist', 'ribes']
    metrics_dataframe = pd.DataFrame(columns=metrics_columns)

    # Read word map
    word_map_file_path = os.path.join(data_folder_path, 'WORDMAP_' + data_file_name + '.json')
    with open(word_map_file_path, 'r') as j:
        word_map = json.load(j)

    # If no checkpoint is provided, initializes the encoder, decoder, and optimizers.
    # Otherwise, loads the model state and optimizer configurations from a saved checkpoint.
    if checkpoint_path is None:
        decoder_model = DecoderWithAttention(attention_dim=attention_layer_dim,
                                             embed_dim=embedding_dim,
                                             decoder_dim=decoder_hidden_dim,
                                             vocab_size=len(word_map),
                                             dropout=dropout_rate)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_model.parameters()),
                                             lr=decoder_learning_rate)
        encoder_model = Encoder()
        encoder_model.fine_tune(fine_tune_encoder_flag)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_model.parameters()),
                                             lr=encoder_learning_rate) if fine_tune_encoder_flag else None

    else:
        checkpoint_data = torch.load(checkpoint_path)
        start_epoch = checkpoint_data['epoch'] + 1
        epochs_since_last_improvement = checkpoint_data['epochs_since_improvement']
        best_bleu4_score = checkpoint_data['bleu-4']
        decoder_model = checkpoint_data['decoder']
        decoder_optimizer = checkpoint_data['decoder_optimizer']
        encoder_model = checkpoint_data['encoder']
        encoder_optimizer = checkpoint_data['encoder_optimizer']
        if fine_tune_encoder_flag and encoder_optimizer is None:
            encoder_model.fine_tune(fine_tune_encoder_flag)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder_model.parameters()),
                                                 lr=encoder_learning_rate)

    # Move to GPU, if available
    gc.collect()
    decoder_model = decoder_model.to(device)
    encoder_model = encoder_model.to(device)

    # Loss function
    loss_function = nn.CrossEntropyLoss().to(device)

    # Sets up dataloaders for training and validation.
    # Custom dataloaders
    normalization_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
    
    train_data_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder_path, data_file_name, 'TRAIN', transform=transforms.Compose([normalization_transform])),
        batch_size=batch_size, shuffle=True, num_workers=data_loader_workers, pin_memory=True)

    val_data_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder_path, data_file_name, 'VAL', transform=transforms.Compose([normalization_transform])),
        batch_size=batch_size, shuffle=True, num_workers=data_loader_workers, pin_memory=True)

    # Implements a training loop, adjusting learning rates if there's no improvement over time.
    for epoch in range(start_epoch, total_epochs):
        
        # Decay learning rate if no improvement for 8 consecutive epochs
        if epochs_since_last_improvement == 20:
            break
        if epochs_since_last_improvement > 0 and epochs_since_last_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder_flag:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train_loss, train_top5_accuracy = train_epoch(train_loader=train_data_loader,
                                                      encoder=encoder_model,
                                                      decoder=decoder_model,
                                                      criterion=loss_function,
                                                      encoder_optimizer=encoder_optimizer,
                                                      decoder_optimizer=decoder_optimizer,
                                                      epoch=epoch)

        # One epoch's validation
        recent_bleu4, val_loss, val_top5_accuracy, chrf, gleu, nist, ribes = validate_epoch(val_loader=val_data_loader,
                                                                                        encoder=encoder_model,
                                                                                        decoder=decoder_model,
                                                                                        criterion=loss_function)

        # Check if there was an improvement
        is_best_model = recent_bleu4 > best_bleu4_score
        best_bleu4_score = max(recent_bleu4, best_bleu4_score)

        new_metrics = pd.DataFrame([{
                        'Epoch':epoch, 'Train Loss':train_loss, 'Train Top-5 Accuracy':train_top5_accuracy, 'Val Loss':val_loss, 'Val Top-5 Accuracy':val_top5_accuracy, 'bleu':recent_bleu4, 'chrf': chrf, 'gleu':gleu, 'nist':nist, 'ribes':ribes
                    }])

        metrics_dataframe = pd.concat([metrics_dataframe, new_metrics], ignore_index=True)
        if not is_best_model:
            epochs_since_last_improvement += 1
            print(f"\nEpochs since last improvement: {epochs_since_last_improvement}\n")
        else:
            epochs_since_last_improvement = 0

        # Save checkpoint
        save_checkpoint(data_file_name, epoch, epochs_since_last_improvement, encoder_model, decoder_model, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best_model)
        
    print('---------------Summary of epochs---------------')
    print(metrics_dataframe)
    metrics_dataframe.to_excel('C:/Users/sagni/Documents/Personal Files/CS7643/Group project/Show, Attend And Tell/training_metrics.xlsx', index=False)
