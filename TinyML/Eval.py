# Name: Sagnik Sinha
# Email: ssinha348@gatech.edu

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

# Parameters
data_directory = '/media/ssd/caption data'  # folder with data files saved by create_input_files.py
dataset_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint_file = '../BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_path = '/media/ssd/caption data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model checkpoint
checkpoint_data = torch.load(checkpoint_file)
decoder_model = checkpoint_data['decoder']
decoder_model = decoder_model.to(device)
decoder_model.eval()
encoder_model = checkpoint_data['encoder']
encoder_model = encoder_model.to(device)
encoder_model.eval()

# Load word map (word2ix)
with open(word_map_path, 'r') as json_file:
    word_mapping = json.load(json_file)
reverse_word_mapping = {v: k for k, v in word_mapping.items()}
vocab_size = len(word_mapping)

# Normalization transform for input images
normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

def evaluate_model(beam_size):
    """
    Evaluates the model using beam search for caption generation.

    :param beam_size: number of beams to use for caption generation
    :return: BLEU-4 score
    """
    # DataLoader setup for test data
    data_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_directory, dataset_name, 'TEST', transform=transforms.Compose([normalize_transform])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # Lists to store true captions (references) and predicted captions (hypotheses)
    references = []
    hypotheses = []

    # Iterate through each image in the test dataset
    for i, (image, captions, caption_lengths, all_captions) in enumerate(
            tqdm(data_loader, desc="Evaluating Beam Size " + str(beam_size))):

        k = beam_size  # number of beams

        # Move image to GPU if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode image using the encoder model
        encoder_output = encoder_model(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_image_size = encoder_output.size(1)
        encoder_dim = encoder_output.size(3)

        # Flatten the encoder output
        encoder_output = encoder_output.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_output.size(1)

        # Expand encoder output for beam search
        encoder_output = encoder_output.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Initialize the first word as <start>
        prev_words = torch.LongTensor([[word_mapping['<start>']]] * k).to(device)  # (k, 1)

        # Initialize sequences and scores
        sequences = prev_words  # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and their scores
        completed_sequences = []
        completed_scores = []

        # Initialize hidden state for decoder
        step = 1
        h, c = decoder_model.init_hidden_state(encoder_output)

        # Beam search loop
        while True:

            embeddings = decoder_model.embedding(prev_words).squeeze(1)  # (s, embed_dim)

            attention_weights, _ = decoder_model.attention(encoder_output, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder_model.sigmoid(decoder_model.f_beta(h))  # gating scalar, (s, encoder_dim)
            attention_weights = gate * attention_weights

            h, c = decoder_model.decode_step(torch.cat([embeddings, attention_weights], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder_model.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add the previous top-k scores to the current scores
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # Find top-k words and their scores
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices
            prev_word_indices = top_k_words // vocab_size  # (s)
            next_word_indices = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            sequences = torch.cat([sequences[prev_word_indices], next_word_indices.unsqueeze(1)], dim=1)  # (s, step+1)

            # Identify incomplete and complete sequences
            incomplete_indices = [index for index, next_word in enumerate(next_word_indices) if next_word != word_mapping['<end>']]
            complete_indices = list(set(range(len(next_word_indices))) - set(incomplete_indices))

            # Save complete sequences
            if len(complete_indices) > 0:
                completed_sequences.extend(sequences[complete_indices].tolist())
                completed_scores.extend(top_k_scores[complete_indices])
            k -= len(complete_indices)  # Update the beam size

            # Proceed with incomplete sequences
            if k == 0:
                break
            sequences = sequences[incomplete_indices]
            h = h[prev_word_indices[incomplete_indices]]
            c = c[prev_word_indices[incomplete_indices]]
            encoder_output = encoder_output[prev_word_indices[incomplete_indices]]
            top_k_scores = top_k_scores[incomplete_indices].unsqueeze(1)
            prev_words = next_word_indices[incomplete_indices].unsqueeze(1)

            # Stop if we reach a maximum number of steps
            if step > 50:
                break
            step += 1

        # Select the sequence with the highest score
        best_sequence_idx = completed_scores.index(max(completed_scores))
        best_sequence = completed_sequences[best_sequence_idx]

        # Prepare references (true captions) by removing special tokens
        true_captions = all_captions[0].tolist()
        true_captions_cleaned = list(
            map(lambda c: [word for word in c if word not in {word_mapping['<start>'], word_mapping['<end>'], word_mapping['<pad>']}],
                true_captions))  # Remove special tokens
        references.append(true_captions_cleaned)

        # Prepare hypotheses (predicted captions) by removing special tokens
        predicted_caption = [word for word in best_sequence if word not in {word_mapping['<start>'], word_mapping['<end>'], word_mapping['<pad>']}]
        hypotheses.append(predicted_caption)

        # Ensure the number of references and hypotheses match
        assert len(references) == len(hypotheses)

    # Compute BLEU-4 score
    bleu4_score = corpus_bleu(references, hypotheses)

    return bleu4_score


if __name__ == '__main__':
    beam_width = 1
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_width, evaluate_model(beam_width)))
