import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
class EncoderVGG(nn.Module):
    def __init__(self):
        super(EncoderVGG, self).__init__()
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad_(False)
        
        normal_modules = list(model.features.children())+list(model.avgpool.children())
        classifier_modules=list(model.classifier.children())[:-3]
        self.conv_model = nn.Sequential(*normal_modules)
        self.classifier_model=nn.Sequential(*classifier_modules)
#         self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.conv_model(images)
        features = features.view(features.size(0),-1)
        features = self.classifier_model(features)
        return features
    

# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(DecoderRNN, self).__init__()
        
#         # Set the hidden size for init_hidden
#         self.hidden_size = hidden_size
        
#         # Set the device
#         self.device = device
        
#         # Embedded layer
#         self.embed = nn.Embedding(vocab_size, embed_size)
        
#         # LSTM layer
#         self.lstm = nn.LSTM(input_size=embed_size,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             batch_first= True,
#                             dropout = 0)
        
#         # Fully Connected layer
#         self.fc = nn.Linear(hidden_size, vocab_size)
        
#     def init_hidden(self, batch_size):
#         return (torch.zeros(batch_size, batch_size, self.hidden_size, device = device),
#                 torch.zeros(batch_size, batch_size, self.hidden_size, device = device))
    
#     def forward(self, features, captions):
        
#         # Initialize the hidden state
#         self.hidden = self.init_hidden(features.shape[0])# features is of shape (batch_size, embed_size)
#         print("Features shape: ", features.shape)
#         print("Captions shape: ", captions.shape)
        
        
#         # Embedding the captions
#         print("Captions shape: ", captions.shape)
#         print("Captions: ", captions)
#         embedded = self.embed(captions[:,:-1])
#         # print(embedded.shape)
#         # print(features.unsqueeze(1).shape)
        
#         embedded = torch.cat((features.unsqueeze(1), embedded), dim=1)
#         # print(embedded.shape)
        
#         # LSTM
#         lstm_out, self.hidden = self.lstm(embedded, self.hidden)
        
#         # Functional component
#         out = self.fc(lstm_out)
#         return out

#     def sample(self, inputs, states=None, max_len=20):
#         " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
#         # Initialize the hidden state
#         hidden = self.init_hidden(inputs.shape[0])# features is of shape (batch_size, embed_size)
        
#         out_list = list()
#         word_len = 0
        
#         with torch.no_grad():
#             while word_len < max_len:
#                 lstm_out, hidden = self.lstm(inputs, hidden)
#                 out = self.fc(lstm_out)
#                 out = out.squeeze(1)
#                 out = out.argmax(dim=1)
#                 # print(out)
#                 out_list.append(out.item())
                
#                 inputs = self.embed(out.unsqueeze(0))
                
#                 word_len += 1
#                 if out == 1:
#                     break
        
#         return out_list
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,batch_size):
        ''' Initialize the layers of this model.'''
        super().__init__()
        self.hidden_size = hidden_size    
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=embed_size, \
                            hidden_size=hidden_size, # LSTM hidden units 
                            num_layers=1, # number of LSTM layer
                            bias=True, # use bias weights b_ih and b_hh
                            batch_first=True,  # input & output will have batch size as 1st dimension
                            dropout=0, # Not applying dropout 
                            bidirectional=False, # unidirectional LSTM
                           )
        
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)                     
        self.batch_size=batch_size
        # initialize the hidden state
        # self.hidden = self.init_hidden()
        
    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))

    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """
        
        # Discard the <end> word to avoid predicting when <end> is the input of the RNN
#         captions = captions[:, :-1]     
        input_captions=[caption[:-1] for caption in captions]


        input_captions=nn.utils.rnn.pad_sequence(input_captions,batch_first=True)
#         print(input_captions.size())
        input_captions_lengths=[len(input_caption)+1 for input_caption in input_captions]
        embeddings = self.word_embeddings(input_captions) # embeddings new shape : (batch_size, captions length - 1, embed_size)
        
        # Initialize the hidden state
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        self.hidden = self.init_hidden(self.batch_size) 
                
        # Create embedded word vectors for each word in the captions
        

        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) # embeddings new shape : (batch_size, caption length, embed_size)

        embeddings=nn.utils.rnn.pack_padded_sequence(embeddings,lengths=input_captions_lengths,batch_first=True,enforce_sorted=False)
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden) # lstm_out shape : (batch_size, caption length, hidden_size)
        lstm_out=nn.utils.rnn.pad_packed_sequence(lstm_out,batch_first=True)

        #Fully connected layer
        outputs = self.linear(lstm_out[0]) # outputs shape : (batch_size, caption length, vocab_size)

        return outputs
    
    