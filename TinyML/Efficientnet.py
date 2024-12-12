# Name: Sagnik Sinha
# Email: ssinha348@gatech.edu
# High-Level Explanation:
# This code defines a neural network system for image captioning using PyTorch. It includes three main components:
# Encoder:
# Uses a pre-trained ResNet-101 model to extract image features and outputs a reduced, fixed-size feature map for each input image.
# Attention Mechanism:
# Dynamically determines which parts of the image are most relevant for generating each word in the caption.
# Decoder with Attention:
# Combines the image features and the previously generated words to predict the next word in the caption, using an LSTM and the attention mechanism.
# This system is trained to take an image as input and produce a descriptive caption as output.

# Block-by-Block Explanation:
# Imports and Device Setup
# Imports PyTorch (torch), neural network modules (torch.nn), and a ResNet-101 model from torchvision.
# Sets the computation device (GPU if available, otherwise CPU).
# Encoder Class:
# Initializes a ResNet-101 model pre-trained on ImageNet.
# Removes the last layers (used for classification) and keeps the convolutional layers to extract feature maps.
# Applies adaptive pooling to ensure consistent output size regardless of the input image size.
# Adds an option to fine-tune certain layers of ResNet.
# Attention Class:
# Computes attention weights, which determine how much focus to put on different parts of the image during caption generation.
# Uses linear layers and a softmax function to calculate weights.
# Decoder with Attention:
# Predicts captions word-by-word:
# Embeds input words using an embedding layer.
# Initializes hidden and cell states of an LSTM based on image features.
# At each time step, uses attention to focus on relevant image regions, then updates the LSTM states and predicts the next word.
# Includes functions to initialize weights, load pre-trained embeddings, and control whether embeddings are trainable.

import torch
from torch import nn
import torchvision
from torchvision.models import ResNet101_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        efficientnet = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)  # Load pre-trained EfficientNet-B0 model
        modules = list(efficientnet.children())[:-2]  # Remove the classification layers (fully connected and pooling)
        self.resnet = nn.Sequential(*modules) #Adds an adaptive pooling layer to resize output to a fixed size.

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune() #Calls a method to enable or disable fine-tuning of certain layers.

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # Passes the image through the ResNet convolutional layers. (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # Applies pooling to ensure a fixed output size. (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # Rearranges dimensions for compatibility with the decoder. (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out 

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters(): #Freezes all ResNet layers by default.
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]: #Allows fine-tuning of layers 5 and later (higher-level features).
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        #Defines linear layers to transform encoder outputs and decoder states.
        #Uses a softmax layer to compute attention weights.
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # Computes attention weights by combining image features (att1) and decoder state (att2).
        # Applies softmax to get normalized weights (alpha).
        # Computes a weighted sum of image features for the decoder.
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=1280, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        # Combines an attention mechanism, word embeddings, and an LSTM for decoding.
        # Includes a fully connected layer to map LSTM outputs to vocabulary scores.
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        # Prepares image features and captions for decoding.
        # Sorts captions by length for efficient processing.
        # Initializes LSTM states using image features.

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            # At each time step, computes attention-weighted image features and updates the LSTM state.
            # Predicts the next word in the caption.
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    
class DecoderWithoutAttention(nn.Module):
    """
    Decoder without attention.
    """
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=1280, dropout=0.5):
        """
        :param embed_dim: size of word embeddings
        :param decoder_dim: size of decoder's RNN hidden state
        :param vocab_size: size of vocabulary
        :param encoder_dim: size of encoded image features
        :param dropout: dropout probability
        """
        super(DecoderWithoutAttention, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        
        # Embedding layer for captions
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # RNN for decoding
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        
        # Fully connected layer for generating word probabilities
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        # Linear layers to initialize LSTM's hidden and cell states
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize parameters with uniform distribution for better convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Initialize the LSTM's hidden state and cell state.
        
        :param encoder_out: encoded image features, a tensor of size (batch_size, encoder_dim)
        :return: hidden state and cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)  # Average features across pixels
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward pass for decoding.
        
        :param encoder_out: encoded image features, (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, (batch_size, max_caption_length)
        :param caption_lengths: lengths of captions, (batch_size, 1)
        :return: predictions, sorted encoded captions, decode lengths, and sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        
        # Flatten image features
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim).mean(dim=1)  # (batch_size, encoder_dim)
        
        # Sort input data by caption lengths (descending order)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Embed the captions
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        
        # Initialize LSTM states
        h, c = self.init_hidden_state(encoder_out)
        
        # Decode lengths: don't decode <end> token
        decode_lengths = (caption_lengths - 1).tolist()
        
        # Prepare tensors for predictions
        max_length = max(decode_lengths)
        predictions = torch.zeros(batch_size, max_length, self.vocab_size).to(encoder_out.device)
        
        # Decode word by word
        for t in range(max_length):
            batch_size_t = sum([l > t for l in decode_lengths])  # Number of sequences not yet finished
            embeddings_t = embeddings[:batch_size_t, t, :]  # Current timestep embeddings
            h, c = self.decode_step(torch.cat([embeddings_t, encoder_out[:batch_size_t]], dim=1), (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout_layer(h))  # Vocabulary scores
            predictions[:batch_size_t, t, :] = preds
        
        return predictions, encoded_captions, decode_lengths, sort_ind