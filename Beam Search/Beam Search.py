# Sample Code
# python caption.py --img "C:/Users/sagni/Documents/Personal Files/CS7643/Group project/Show, Attend And Tell/media_flickr8k/ssd/caption data/Flickr8k_Dataset/12830823_87d2654e31.jpg" --model "C:/Users/sagni/Documents/Personal Files/CS7643/Group project/Show, Attend And Tell/BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar" --word_map "C:/Users/sagni/Documents/Personal Files/CS7643/Group project/Show, Attend And Tell/media_flickr8k/ssd/caption data/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json" --beam_size=1

import os
os.chdir("C:/Users/sagni/Documents/Personal Files/CS7643/Group project/Show, Attend And Tell/")
import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from matplotlib.pyplot import imread
from cv2 import resize as imresize
from PIL import Image
import io
from torch.serialization import add_safe_globals
from models_resnet import Encoder, DecoderWithAttention as Decoder  # Replace with actual module

# Define device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_caption_beam_search(image_encoder, text_decoder, image_path, word_to_index_map, beam_width=3):
    """
    Generates a caption for an image using beam search with attention.

    :param image_encoder: The image encoder model
    :param text_decoder: The text decoder model with attention mechanism
    :param image_path: Path to the input image
    :param word_to_index_map: A dictionary mapping words to indices
    :param beam_width: The number of sequences to consider at each decoding step
    :return: Best caption sequence and attention weights for visualization
    """

    beam_width = beam_width
    vocab_size = len(word_to_index_map)

    # Step 1: Load and preprocess the image
    image = imread(image_path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]  # Convert grayscale to RGB
        image = np.concatenate([image, image, image], axis=2)  # Replicate channels
    image = imresize(image, (256, 256))  # Resize image to 256x256
    image = image.transpose(2, 0, 1)  # Change image shape to (C, H, W)
    image = image / 255.  # Normalize pixel values to [0, 1]
    image_tensor = torch.FloatTensor(image).to(device)

    # Define normalization parameters for pre-trained model
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_transform = transforms.Compose([normalize])

    # Apply transformation (standardize image)
    image_tensor = image_transform(image_tensor)  # (3, 256, 256)

    # Step 2: Pass image through encoder
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension: (1, 3, 256, 256)
    encoder_output = image_encoder(image_tensor)  # (1, enc_image_size, enc_image_size, encoder_dim)
    
    # Get encoding dimensions
    enc_image_size = encoder_output.size(1)
    encoder_dim = encoder_output.size(3)

    # Flatten the encoder output to (1, num_pixels, encoder_dim)
    encoder_output = encoder_output.view(1, -1, encoder_dim)
    num_pixels = encoder_output.size(1)

    # Step 3: Initialize beam search variables
    encoder_output = encoder_output.expand(beam_width, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store the top beam_width previous words at each step; initialize with <start> token
    previous_words = torch.LongTensor([[word_to_index_map['<start>']]] * beam_width).to(device)

    # Store sequences and scores
    sequences = previous_words  # (k, 1)
    sequence_scores = torch.zeros(beam_width, 1).to(device)  # (k, 1)
    attention_weights = torch.ones(beam_width, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their attention weights, and scores
    completed_sequences = list()
    completed_attention_weights = list()
    completed_scores = list()

    # Step 4: Start decoding loop
    step = 1
    hidden_state, cell_state = text_decoder.init_hidden_state(encoder_output)

    while True:
        # Step 4a: Get word embeddings and apply attention mechanism
        word_embeddings = text_decoder.embedding(previous_words).squeeze(1)  # (s, embed_dim)
        attention_output, alpha = text_decoder.attention(encoder_output, hidden_state)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        # Step 4b: Apply gating mechanism and update attention output
        gating_scalar = text_decoder.sigmoid(text_decoder.f_beta(hidden_state))  # (s, encoder_dim)
        attention_output = gating_scalar * attention_output

        # Step 4c: Decode the combined input (word embedding + attention output)
        hidden_state, cell_state = text_decoder.decode_step(torch.cat([word_embeddings, attention_output], dim=1), (hidden_state, cell_state))  # (s, decoder_dim)

        # Step 4d: Compute the logits (scores) for each word in the vocabulary
        logits = text_decoder.fc(hidden_state)  # (s, vocab_size)
        logits = F.log_softmax(logits, dim=1)

        # Add previous scores to logits
        scores = sequence_scores.expand_as(logits) + logits  # (s, vocab_size)

        if step == 1:
            # For the first step, all beams have the same score (since same previous words, hidden states)
            sequence_scores, word_indices = scores[0].topk(beam_width, 0, True, True)  # (s)
        else:
            # Unroll and find top scores and their indices
            sequence_scores, word_indices = scores.view(-1).topk(beam_width, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of words
        prev_word_indices = (word_indices // vocab_size).long()  # Ensure LongTensor for indexing
        next_word_indices = (word_indices % vocab_size).long()  # Ensure LongTensor for indexing

        # Step 4e: Update sequences and attention weights
        sequences = torch.cat([sequences[prev_word_indices], next_word_indices.unsqueeze(1)], dim=1)  # (s, step+1)
        attention_weights = torch.cat([attention_weights[prev_word_indices], alpha[prev_word_indices].unsqueeze(1)], dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Step 4f: Check which sequences are complete (i.e., have reached <end>)
        incomplete_indices = [i for i, next_word in enumerate(next_word_indices) if next_word != word_to_index_map['<end>']]
        complete_indices = list(set(range(len(next_word_indices))) - set(incomplete_indices))

        if len(complete_indices) > 0:
            completed_sequences.extend(sequences[complete_indices].tolist())
            completed_attention_weights.extend(attention_weights[complete_indices].tolist())
            completed_scores.extend(sequence_scores[complete_indices])
        beam_width -= len(complete_indices)  # Decrease beam width as we complete sequences

        # Proceed with incomplete sequences
        if beam_width == 0:
            break
        sequences = sequences[incomplete_indices]
        attention_weights = attention_weights[incomplete_indices]
        hidden_state = hidden_state[prev_word_indices[incomplete_indices]]
        cell_state = cell_state[prev_word_indices[incomplete_indices]]
        encoder_output = encoder_output[prev_word_indices[incomplete_indices]]
        sequence_scores = sequence_scores[incomplete_indices].unsqueeze(1)
        previous_words = next_word_indices[incomplete_indices].unsqueeze(1)

        # Break after 50 steps to avoid infinite loops
        if step > 50:
            break
        step += 1

    # Step 5: Select the best sequence and its attention weights
    best_sequence_index = completed_scores.index(max(completed_scores))
    best_sequence = completed_sequences[best_sequence_index]
    best_attention_weights = completed_attention_weights[best_sequence_index]

    return best_sequence, best_attention_weights


def visualize_caption_with_attention(image_path, caption_sequence, attention_weights, index_to_word_map, smooth_attention=True):
    """
    Visualizes the generated caption with attention weights at each word.

    :param image_path: Path to the image being captioned
    :param caption_sequence: The generated caption sequence
    :param attention_weights: Attention weights for each word
    :param index_to_word_map: Reverse word map (index to word)
    :param smooth_attention: Whether to smooth the attention weights or not
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    # Convert indices to words
    words = [index_to_word_map[ind] for ind in caption_sequence]

    # Step 1: Create subplots for each word in the caption
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        
        # Apply smoothing to attention weights if necessary
        current_attention = attention_weights[t, :]
        if smooth_attention:
            attention_map = skimage.transform.pyramid_expand(current_attention.numpy(), upscale=24, sigma=8)
        else:
            attention_map = skimage.transform.resize(current_attention.numpy(), [14 * 24, 14 * 24])

        if t == 0:
            plt.imshow(attention_map, alpha=0)
        else:
            plt.imshow(attention_map, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    
    plt.show()


if __name__ == '__main__':
    os.chdir("C:/Users/sagni/Documents/Personal Files/CS7643/Group project/Show, Attend And Tell")
    parser = argparse.ArgumentParser(description='Generate Caption using Show, Attend, and Tell with Beam Search')

    parser.add_argument('--img', '-i', help='Path to image')
    parser.add_argument('--model', '-m', help='Path to model')
    parser.add_argument('--word_map', '-wm', help='Path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='Beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='Do not smooth attention overlay')

    args = parser.parse_args()

    if args.model is None:
        raise ValueError("Model path (--model) must be specified.")

    if not os.path.exists(args.model):
        raise ValueError(f"Model file not found at {args.model}")

    # Load the model
    with open(args.model, 'rb') as f:
        buffer = io.BytesIO(f.read())

    add_safe_globals([Encoder, Decoder])
    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word to index)
    with open(args.word_map, 'r') as j:
        word_to_index_map = json.load(j)

    index_to_word_map = {v: k for k, v in word_to_index_map.items()}  # Reverse map (index to word)

    # Step 6: Generate caption using beam search
    caption_sequence, attention_weights = generate_caption_beam_search(encoder, decoder, args.img, word_to_index_map, args.beam_size)
    attention_weights = torch.FloatTensor(attention_weights)

    # Step 7: Visualize caption and attention
    visualize_caption_with_attention(args.img, caption_sequence, attention_weights, index_to_word_map, args.smooth)
