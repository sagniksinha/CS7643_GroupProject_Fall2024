# Name: Sagnik Sinha
# Email: ssinha348@gatech.edu

import os
os.chdir("C:/Users/sagni/Documents/Personal Files/CS7643/Group project/Show, Attend And Tell")
import numpy as np
import h5py
import json
import torch
from matplotlib.pyplot import imread
from cv2 import resize as imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

def create_input_files(data_name, json_file_path, images_dir, num_captions_per_image, min_word_frequency, output_dir, max_caption_length=100):
    """
    Generates input files for training, validation, and test data.

    :param data_name: name of the dataset ('coco', 'flickr8k', 'flickr30k')
    :param json_file_path: path to the JSON file with image splits and captions
    :param images_dir: directory containing the images
    :param num_captions_per_image: number of captions to sample per image
    :param min_word_frequency: words occurring less frequently than this threshold are grouped as <unk>s
    :param output_dir: directory to save output files
    :param max_caption_length: max allowed length of a caption
    """

    assert data_name in {'coco', 'flickr8k', 'flickr30k'}

    # Load data from the JSON file
    with open(json_file_path, 'r') as json_file:
        dataset_info = json.load(json_file)

    # Initialize lists for image paths and captions for each split
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_frequency = Counter()

    # Process images and their captions
    for image in dataset_info['images']:
        captions = []
        for sentence in image['sentences']:
            word_frequency.update(sentence['tokens'])
            if len(sentence['tokens']) <= max_caption_length:
                captions.append(sentence['tokens'])

        if len(captions) == 0:
            continue

        image_path = os.path.join(images_dir, image['filepath'], image['filename']) if data_name == 'coco' else os.path.join(images_dir, image['filename'])

        # Categorize images based on their split
        if image['split'] in {'train', 'restval'}:
            train_image_paths.append(image_path)
            train_image_captions.append(captions)
        elif image['split'] in {'val'}:
            val_image_paths.append(image_path)
            val_image_captions.append(captions)
        elif image['split'] in {'test'}:
            test_image_paths.append(image_path)
            test_image_captions.append(captions)

    # Ensure the data is correctly matched between images and captions
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Build the word map for the dataset
    valid_words = [word for word, freq in word_frequency.items() if freq > min_word_frequency]
    word_map = {word: index + 1 for index, word in enumerate(valid_words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base filename for output files
    base_filename = f"{data_name}_{num_captions_per_image}_cap_per_img_{min_word_frequency}_min_word_freq"

    # Save the word map to a JSON file
    with open(os.path.join(output_dir, f'WORDMAP_{base_filename}.json'), 'w') as wordmap_file:
        json.dump(word_map, wordmap_file)

    # Sample captions for each image and store the data in HDF5 and JSON files
    seed(123)
    for image_paths, image_captions, split_name in [(train_image_paths, train_image_captions, 'TRAIN'),
                                                   (val_image_paths, val_image_captions, 'VAL'),
                                                   (test_image_paths, test_image_captions, 'TEST')]:
        hdf5_file_path = os.path.join(output_dir, f"{split_name}_IMAGES_{base_filename}.hdf5")
        with h5py.File(hdf5_file_path, 'a') as hdf5_file:
            # Add the number of captions per image as metadata
            hdf5_file.attrs['captions_per_image'] = num_captions_per_image

            # Check if 'images' dataset exists and delete it if necessary
            if 'images' in hdf5_file:
                print(f"Dataset 'images' exists in {hdf5_file_path}. Deleting and recreating.")
                del hdf5_file['images']

            # Create dataset for images in HDF5 file
            images_dataset = hdf5_file.create_dataset('images', (len(image_paths), 3, 256, 256), dtype='uint8')

            print(f"\nProcessing {split_name} images and captions...\n")

            encoded_captions = []
            caption_lengths = []

            for i, path in enumerate(tqdm(image_paths)):
                # Sample captions for the image
                if len(image_captions[i]) < num_captions_per_image:
                    captions = image_captions[i] + [choice(image_captions[i]) for _ in range(num_captions_per_image - len(image_captions[i]))]
                else:
                    captions = sample(image_captions[i], k=num_captions_per_image)

                # Ensure correct number of captions
                assert len(captions) == num_captions_per_image

                # Read and process the image
                img = imread(path)
                if len(img.shape) == 2:  # Convert grayscale to RGB
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)  # Change image shape to (3, 256, 256)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to the HDF5 file
                images_dataset[i] = img

                # Process and encode each caption
                for caption in captions:
                    encoded_caption = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in caption] + [word_map['<end>']] + [word_map['<pad>']] * (max_caption_length - len(caption))

                    # Record caption length
                    caption_length = len(caption) + 2

                    encoded_captions.append(encoded_caption)
                    caption_lengths.append(caption_length)

            # Sanity check to ensure data consistency
            assert images_dataset.shape[0] * num_captions_per_image == len(encoded_captions) == len(caption_lengths)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_dir, f"{split_name}_CAPTIONS_{base_filename}.json"), 'w') as captions_file:
                json.dump(encoded_captions, captions_file)

            with open(os.path.join(output_dir, f"{split_name}_CAPLENS_{base_filename}.json"), 'w') as lengths_file:
                json.dump(caption_lengths, lengths_file)


def initialize_embeddings(embedding_tensor):
    """
    Initializes the embedding tensor with values from a uniform distribution.

    :param embedding_tensor: tensor to initialize with values
    """
    bias = np.sqrt(3.0 / embedding_tensor.size(1))
    torch.nn.init.uniform_(embedding_tensor, -bias, bias)


def load_pretrained_embeddings(embedding_file, word_map):
    """
    Loads pretrained word embeddings for the words in the word map.

    :param embedding_file: file containing word embeddings in GloVe format
    :param word_map: word map containing the vocabulary
    :return: embeddings tensor, embedding dimension
    """
    # Determine the dimension of the embeddings
    with open(embedding_file, 'r') as file:
        embedding_dim = len(file.readline().split(' ')) - 1

    vocabulary = set(word_map.keys())

    # Create tensor for embeddings
    embeddings_tensor = torch.FloatTensor(len(vocabulary), embedding_dim)
    initialize_embeddings(embeddings_tensor)

    print("\nLoading word embeddings...")
    for line in open(embedding_file, 'r'):
        line = line.split(' ')

        word = line[0]
        embedding = list(map(lambda value: float(value), filter(lambda token: token and not token.isspace(), line[1:])))

        # Skip word if it's not in the vocabulary
        if word not in vocabulary:
            continue

        embeddings_tensor[word_map[word]] = torch.FloatTensor(embedding)

    return embeddings_tensor, embedding_dim


def clip_gradient_norm(optimizer, gradient_clip_value):
    """
    Clips gradients to avoid gradient explosion.

    :param optimizer: optimizer whose gradients should be clipped
    :param gradient_clip_value: value to clip gradients to
    """
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-gradient_clip_value, gradient_clip_value)


def save_model_checkpoint(dataset_name, epoch, epochs_since_improvement, encoder_model, decoder_model, encoder_optimizer, decoder_optimizer, bleu_score, best_model):
    """
    Saves the model checkpoint.

    :param dataset_name: dataset name used for training
    :param epoch: current epoch
    :param epochs_since_improvement: epochs since last improvement in BLEU score
    :param encoder_model: the encoder model
    :param decoder_model: the decoder model
    :param encoder_optimizer: optimizer for the encoder
    :param decoder_optimizer: optimizer for the decoder
    :param bleu_score: current BLEU score
    :param best_model: whether this checkpoint is the best so far
    """
    checkpoint_state = {'epoch': epoch,
                        'epochs_since_improvement': epochs_since_improvement,
                        'bleu-4': bleu_score,
                        'encoder': encoder_model,
                        'decoder': decoder_model,
                        'encoder_optimizer': encoder_optimizer,
                        'decoder_optimizer': decoder_optimizer}
    checkpoint_filename = f'checkpoint_{dataset_name}.pth.tar'
    torch.save(checkpoint_state, checkpoint_filename)

    if best_model:
        torch.save(checkpoint_state, f'BEST_{checkpoint_filename}')


class MetricTracker:
    """
    Tracks the recent, average, sum, and count of a specific metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, count=1):
        self.val = value
        self.sum += value * count
        self.count += count
        self.avg = self.sum / self.count


def reduce_learning_rate(optimizer, decay_factor):
    """
    Reduces the learning rate by a specified factor.

    :param optimizer: optimizer to adjust the learning rate for
    :param decay_factor: factor by which the learning rate is reduced
    """
    print("\nReducing learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_factor
    print(f"Updated learning rate: {optimizer.param_groups[0]['lr']:.6f}\n")


def compute_top_k_accuracy(scores, targets, k):
    """
    Computes top-k accuracy based on predicted and true labels.

    :param scores: predicted scores
    :param targets: true labels
    :param k: top-k accuracy to compute
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, predicted_indices = scores.topk(k, 1, True, True)
    correct_predictions = predicted_indices.eq(targets.view(-1, 1).expand_as(predicted_indices))
    total_correct = correct_predictions.view(-1).float().sum()
    return total_correct.item() * (100.0 / batch_size)
