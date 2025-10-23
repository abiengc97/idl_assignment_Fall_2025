import torch
import random
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import wandb
import torchaudio.transforms as tat
from torchaudio.models.decoder import cuda_ctc_decoder
import Levenshtein

from sklearn.metrics import accuracy_score
import gc

import glob

import zipfile
from tqdm.auto import tqdm
import os
import datetime


import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

Name = "Abiencherian" # Write your name here
import yaml
with open("/home/agcheria/idl_assignment_Fall_2025_1/HW3/HW3P2/config.yaml") as file:
    config = yaml.safe_load(file)
    
print(config)

BATCH_SIZE = config["batch_size"] # Define batch size from config
root = "/home/agcheria/idl_assignment_Fall_2025_1/HW3/HW3P2/11785-hw3p2" # Specify the directory to your root based on your environment: Google Colab, Kaggle, or PSC

# ARPABET PHONEME MAPPING
# DO NOT CHANGE

CMUdict_ARPAbet = {
    "" : " ",
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH"   : "h", "Z" :
     "z", "K" : "k", "CH": "C", "W" : "w",
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}


CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())

PHONEMES = CMUdict[:-2] #To be used for mapping original transcripts to integer indices
LABELS = ARPAbet[:-2] #To be used for mapping predictions to strings

OUT_SIZE = len(PHONEMES) # Number of output classes
print("Number of Phonemes:", OUT_SIZE)

# Indexes of BLANK and SIL phonemes
BLANK_IDX=CMUdict.index('')
SIL_IDX=CMUdict.index('[SIL]')

print("Index of Blank:", BLANK_IDX)
print("Index of [SIL]:", SIL_IDX)

test_mfcc = f"{root}/train-clean-100/mfcc/103-1240-0000.npy"
test_transcript = f"{root}/train-clean-100/transcript/103-1240-0000.npy"

mfcc = np.load(test_mfcc)
transcript = np.load(test_transcript)[1:-1] #Removed [SOS] and [EOS]

print("MFCC Shape:", mfcc.shape)
print("\nMFCC:\n", mfcc)
print("\nTranscript shape:", transcript.shape)

print("\nOriginal Transcript:\n", transcript)

# map the loaded transcript (from phonemes representation) to corresponding labels representation
mapped_transcript = [CMUdict_ARPAbet[k] for k in transcript]
print("\nTranscript mapped from PHONEMES representation to LABELS representation:\n", mapped_transcript)

# Mapping list of PHONEMES to list of Integer indexes
map = {k: i for i, k in enumerate(PHONEMES)}
print("\nMapping list of PHONEMES to list of Integer indexes:\n", map)

class AudioDataset(torch.utils.data.Dataset):

    # For this homework, we give you full flexibility to design your data set class.
    # Hint: The data from HW1 is very similar to this HW

    #TODO
    def __init__(self, root, partition="train-clean-100", train=True, freq_mask_param=10, time_mask_param=10):
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''

        self.PHONEMES = PHONEMES
        self.subset = config['subset']
        self.train = train  # Flag to control augmentation
        self.freq_masking = tat.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = tat.TimeMasking(time_mask_param=time_mask_param)

        # TODO
        # Define the directories containing MFCC and transcript files
        self.mfcc_dir = os.path.join(root, partition, 'mfcc')
        self.transcript_dir = os.path.join(root, partition, 'transcript')

        # List all files in the directories. Remember to sort the files
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir)  )
        self.transcript_files = sorted(os.listdir(self.transcript_dir))

        # Compute size of data subset
        subset_size = int(self.subset * len(self.mfcc_files))

        # Select subset of data to use
        self.mfcc_files = self.mfcc_files[:subset_size]
        self.transcript_files = self.transcript_files[:subset_size]

        assert(len(self.mfcc_files) == len(self.transcript_files))

        #TODO
        # WHAT SHOULD THE LENGTH OF THE DATASET BE?
        self.length = len(self.mfcc_files)

        #TODO
        # CREATE AN ARRAY TO STORE ALL PROCESSED MFCCS AND TRANSCRIPTS
        # LOAD ALL MFCCS AND CORRESPONDING TRANSCRIPTS AND DO THE NECESSARY PRE-PROCESSING
          # HINTS:
          # WHAT NORMALIZATION TECHNIQUE DID YOU USE IN HW1? CAN WE USE IT HERE?
          # REMEMBER TO REMOVE [SOS] AND [EOS] FROM TRANSCRIPTS
        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''
        self.mfccs = []
        self.transcripts = []
        for i in tqdm(range(len(self.mfcc_files))):

            # TODO: Load a single mfcc. Hint: Use numpy
            mfcc             = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[i]))
            # TODO: Do Cepstral Normalization of mfcc along the Time Dimension (Think about the correct axis)
            mfccs_normalized = (mfcc - np.mean(mfcc, axis=0)) /( np.std(mfcc, axis=0)+1e-6)

            # Convert mfcc to tensor
            mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)

            # TODO: Load the corresponding transcript
            # Remove [SOS] and [EOS] from the transcript
            # (Is there an efficient way to do this without traversing through the transcript?)
            # Note that SOS will always be in the starting and EOS at end, as the name suggests.
            transcript = np.load(f'{self.transcript_dir}/{self.transcript_files[i]}')[1:-1]
            # The available phonemes in the transcript are of string data type
            # But the neural network cannot predict strings as such.
            # Hence, we map these phonemes to integers

            # TODO: Map the phonemes to their corresponding list indexes in self.phonemes
            transcript_indices = [self.PHONEMES.index(phoneme) for phoneme in transcript]
            # Now, if an element in the transcript is 0, it means that it is 'SIL' (as per the above example)

            # Convert transcript to tensor
            transcript_indices = torch.tensor(transcript_indices, dtype=torch.int64)

            # Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfccs_normalized)
            self.transcripts.append(transcript_indices)
        

        #TODO
        # HOW CAN WE REPRESENT PHONEMES? CAN WE CREATE A MAPPING FOR THEM?
        # HINT: TENSORS CANNOT STORE NON-NUMERICAL VALUES OR STRINGS
        self.map = {k: i for i, k in enumerate(PHONEMES)}
        



    def __len__(self):

        '''
        TODO: What do we return here?
        '''

        return self.length


    def __getitem__(self, ind):

        '''
        TODO: RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.

        '''

        # Use preloaded and normalized tensors from __init__
        return self.mfccs[ind], self.transcripts[ind]


    def collate_fn(self,batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''

        # Extract batch of input MFCCs and batch of output transcripts separately
        batch_mfcc = [item[0] for item in batch]
        batch_transcript = [item[1] for item in batch]

        # Store original lengths of the MFCCS and transcripts in the batches
        lengths_mfcc = [item.shape[0] for item in batch_mfcc]
        lengths_transcript = [item.shape[0] for item in batch_transcript]

        # Apply SpecAugment BEFORE padding, and only on training data
        if self.train:
            aug_mfcc = []
            for mfcc in batch_mfcc:
                # mfcc shape: (T, F) -> need (1, F, T) for torchaudio transforms
                mfcc_aug = mfcc.T.unsqueeze(0)  # (1, F, T)
                mfcc_aug = self.time_masking(self.freq_masking(mfcc_aug)).squeeze(0)  # (F, T)
                aug_mfcc.append(mfcc_aug.T)  # back to (T, F)
            batch_mfcc = aug_mfcc

        # Pad the MFCC sequences and transcripts
        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        # Note: (resulting shape of padded MFCCs: [batch, time, freq])
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)

        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)
    
    
    # TODO
# Food for thought -> Do you need to apply transformations in this test dataset class?
class AudioDatasetTest(torch.utils.data.Dataset):
    def __init__(self, root, partition="test-clean"):
        self.root = root
        self.partition = partition
        self.mfcc_dir = os.path.join(root, partition, 'mfcc')
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        self.length = len(self.mfcc_files)
        self.mfccs = []
        for mfcc_file in self.mfcc_files:
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_file))
            mfcc=(mfcc-np.mean(mfcc, axis=0))/(np.std(mfcc, axis=0) + 1e-6)
            self.mfccs.append(torch.tensor(mfcc, dtype=torch.float32))
        self.map = {k: i for i, k in enumerate(PHONEMES)}
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, ind):
        mfcc = self.mfccs[ind]
        return mfcc
    
    def collate_fn(self, batch):
        '''
        Collate function for test dataset to handle variable-length MFCC sequences.
        Returns padded MFCCs and their lengths.
        '''
        # Extract batch of MFCCs
        batch_mfcc = batch
        
        # Store original lengths of the MFCCs
        lengths_mfcc = [item.shape[0] for item in batch_mfcc]
        
        # Pad the MFCC sequences
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        
        # Return padded features and actual lengths
        return batch_mfcc_pad, torch.tensor(lengths_mfcc)
    
    


# To free up ram
import gc
gc.collect()

# Create objects for the dataset classes
# Progressive augmentation curriculum: start with light augmentation, increase later
freq_mask = config.get('freq_mask_param', 10)
time_mask = config.get('time_mask_param', 10)
train_data = AudioDataset(root=root, partition="train-clean-100", train=True, freq_mask_param=freq_mask, time_mask_param=time_mask)
val_data = AudioDataset(root=root, partition="dev-clean", train=False)
test_data = AudioDatasetTest(root=root, partition="test-clean")

# Do NOT forget to pass in the collate function as an argument while creating the dataloader
train_loader = DataLoader(train_data,num_workers=4, pin_memory=True, shuffle=True, batch_size=config['batch_size'], collate_fn=train_data.collate_fn)

val_loader = DataLoader(val_data, num_workers=0, pin_memory=True, shuffle=False, batch_size=config['batch_size'], collate_fn=val_data.collate_fn)

test_loader = DataLoader(test_data, num_workers=4, pin_memory=True, shuffle=False, batch_size=config['batch_size'], collate_fn=test_data.collate_fn)

print("Batch size: ", config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

# sanity check
for data in train_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, ly.shape)
    break


torch.cuda.empty_cache()

class Network(nn.Module):

    def __init__(self):

        super(Network, self).__init__()

        # TODO: Adding some sort of embedding layer or feature extractor might help performance.
        # You can use CNNs as Embedding layer to extract features. Keep in mind the Input dimensions and expected dimension of Pytorch CNN.
        # Food for thought -> What type of Conv layers can be used here?
        #                  -> What should be the size of input channels to the first layer?
        self.embedding = nn.Conv1d(in_channels=config['input_size'], out_channels=config['embed_size'], kernel_size=3, stride=1, padding=1)

        # TODO : look up the documentation. You might need to pass some additional parameters.
        self.lstm = nn.LSTM(input_size = config['input_size'], hidden_size = config['embed_size'], num_layers = 1) #TODO

        self.classification = nn.Sequential(
            nn.Linear(config['embed_size']  , 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, OUT_SIZE),
            #TODO: Linear layer with in_features from the lstm module above and out_features = OUT_SIZE
        )


        self.logSoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x, lx):   
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.classification(x)
        x = self.logSoftmax(x)
        return x
        #TODO
        # The forward function takes 2 parameter inputs here. Why?
        # Refer to the handout for hints
        
torch.cuda.empty_cache()

model = Network().to(device)
# Check to stay below 20 MIL Parameter limit
# assert sum(p.numel() for p in model.parameters() if p.requires_grad) < 20_000_000, "Exceeds 20 MIL params. Any submission made to Kaggle with this model will be flagged as an AIV."

print(model)
        
class Permute(torch.nn.Module):
    '''
    Used to transpose/permute the dimensions of an MFCC tensor.
    '''
    def forward(self, x):
        return x.transpose(1, 2)
    
class pBLSTM(torch.nn.Module):

    '''
    Pyramidal BiLSTM
    Read the write up/paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
        (Tip: Write down the shapes and understand)
        (i) How should  you deal with odd/even length input?
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.blstm = nn.LSTM(
            input_size=input_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x_packed): # x_packed is a PackedSequence

        # TODO: Pad Packed Sequence
        x_unpacked, x_lens = pad_packed_sequence(x_packed, batch_first=True)

        # TODO: Call self.trunc_reshape() which downsamples the time steps of x and increases the feature dimensions as mentioned above
        # self.trunc_reshape will return 2 outputs. What are they? Think about what quantites are changing.
        x, x_lens = self.trunc_reshape(x_unpacked, x_lens)
        # TODO: Pack Padded Sequence. What output(s) would you get?
        x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False) 
        # TODO: Pass the sequence through bLSTM
        x_packed, _ = self.blstm(x_packed)
        
        # What do you return?

        return x_packed

    def trunc_reshape(self, x, x_lens):

        # TODO: If you have odd number of timesteps, how can you handle it? (Hint: You can exclude them)
        # Truncate to even number of timesteps
        B, T, F = x.shape
        if T % 2 == 1:
            x = x[:, :-1, :]
            x_lens = x_lens - 1
        
        # TODO: Reshape x. When reshaping x, you have to reduce number of timesteps by a downsampling factor while increasing number of features by the same factor
        x = x.reshape(B, T // 2, 2 * F)

        
        # TODO: Reduce lengths by the same downsampling factor
        x_lens = x_lens // 2

        return x, x_lens

class LSTMWrapper(torch.nn.Module):
    '''
    Used to get only output of lstm, not the hidden states.
    '''
    def __init__(self, lstm):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm

    def forward(self, x):
        output, _ = self.lstm(x)
        return output
    
class Encoder(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size):
        super(Encoder, self).__init__()


        # TODO: You can use CNNs as Embedding layer to extract features. Keep in mind the Input dimensions and expected dimension of Pytorch CNN.
        # Food for thought -> What type of Conv layers can be used here?
        #                  -> What should be the size of input channels to the first layer?
        self.embedding = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3, stride=1, padding=1)

        # TODO:
        self.BLSTMs = nn.LSTM(
            # TODO: Look up the documentation. You might need to pass some additional parameters.
            input_size=128,
            hidden_size=encoder_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
          )

        self.pBLSTMs = torch.nn.Sequential( # How many pBLSTMs are required?
            # TODO: Fill this up with pBLSTMs - What should the input_size be?
            # Hint: You are downsampling timesteps by a factor of 2, upsampling features by a factor of 2 and the LSTM is bidirectional)
            # Optional: Dropout/Locked Dropout after each pBLSTM (Not needed for early submission)
            # https://github.com/salesforce/awd-lstm-lm/blob/dfd3cb0235d2caf2847a4d53e1cbd495b781b5d2/locked_dropout.py#L5
            # ...
            pBLSTM(input_size=2*encoder_hidden_size, hidden_size=encoder_hidden_size),  # 512 input (2*256)
            pBLSTM(input_size=2*encoder_hidden_size, hidden_size=encoder_hidden_size),  # 512 input (2*256)
            pBLSTM(input_size=2*encoder_hidden_size, hidden_size=encoder_hidden_size),  # 512 input (2*256)
            
            # ...

        )

    def forward(self, x, x_lens):
        # Where are x and x_lens coming from? The dataloader

        # TODO: Call the embedding layer
        x=x.transpose(1, 2)
        x=self.embedding(x)
        x=x.transpose(1, 2)

        # TODO: Pack Padded Sequence
        x_packed=pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        # TODO: Pass Sequence through the Bi-LSTM layer
        x_packed, _ = self.BLSTMs(x_packed)
        # TODO: Pass Sequence through the pyramidal Bi-LSTM layer
        for layer in self.pBLSTMs:
            x_packed = layer(x_packed)
        # TODO: Pad Packed Sequence

        # Remember the number of output(s) each function returns
        
        # Pack the sequence before passing to pBLSTMs (lengths need to be on CPU)
        x_unpacked, _ = pad_packed_sequence(x_packed, batch_first=True)

        # After 3 pBLSTMs, the time dimension is reduced by 8×
        x_lens = x_lens // (2 ** len(self.pBLSTMs))
        # Clamp to ensure lengths are at least 1 (avoid errors with very short sequences)
        x_lens = torch.clamp(x_lens, min=1)

        return x_unpacked, x_lens
    
    
class Decoder(torch.nn.Module):

    def __init__(self, embed_size, output_size= 41):
        super().__init__()

        self.mlp = torch.nn.Sequential(

            Permute(),
            torch.nn.BatchNorm1d(2 * embed_size),
            Permute(),

            #TODO define your MLP arch. Refer HW1P2
            #Use Permute Block before and after BatchNorm1d() to match the size
            #Now you can stack your MLP layers
            nn.Linear(2 * embed_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_size),
        )

        self.softmax = torch.nn.LogSoftmax(dim=2)


    def forward(self, encoder_out):

        #TODO: Call your MLP

        #TODO: Think about what should be the final output of the decoder for classification
        out = self.mlp(encoder_out)
        out = self.softmax(out)
        return out
    
class ASRModel(torch.nn.Module):

    def __init__(self, input_size, embed_size= 192, output_size= len(PHONEMES)):
        super().__init__()

        # Initialize encoder and decoder
        self.encoder        = Encoder(input_size, embed_size)
        self.decoder        = Decoder(embed_size, output_size)


    def forward(self, x, lengths_x):

        encoder_out, encoder_lens   = self.encoder(x, lengths_x)
        decoder_out                 = self.decoder(encoder_out)

        return decoder_out, encoder_lens
    
model = ASRModel(
    input_size  = config['input_size'],
    embed_size  = config['embed_size'],
    output_size = len(PHONEMES)
).to(device)

# Check to stay below 20 MIL Parameter limit
# assert sum(p.numel() for p in model.parameters() if p.requires_grad) < 20_000_000, "Exceeds 20 MIL params. Any submission made to Kaggle with this model will be flagged as an AIV."

summary(model, input_data=[x.to(device), lx.to(device)])
checkpoint = torch.load('/home/agcheria/idl_assignment_Fall_2025_1/HW3/HW3P2/checkpoints_best/checkpoint-best-model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# TODO: Define CTC loss as the criterion. How would the losses be reduced?
criterion = nn.CTCLoss(
    blank=BLANK_IDX,      # your "" token index
    reduction="mean",
    zero_infinity=True
)
# CTC Loss: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
# Refer to the handout for hints

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=1e-2
) #TODO: What goes in here?

# TODO: Declare the decoder. Use the PyTorch Cuda CTC Decoder to decode phonemes
# CTC Decoder: https://pytorch.org/audio/2.1/generated/torchaudio.models.decoder.cuda_ctc_decoder.html
decoder = cuda_ctc_decoder(
    tokens=LABELS,                         # same class order as the network output
    nbest=1,
    beam_size=config["train_beam_width"],
    blank_skip_threshold=0.95
) #TODO

# TODO:
steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config["learning_rate"],
    epochs=config["epochs"],
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,
    anneal_strategy="cos"
)

# Mixed Precision, if you need it
scaler = torch.cuda.amp.GradScaler()


@torch.no_grad()
def decode_prediction(output, output_lens, decoder, PHONEME_MAP = LABELS):

    # Look at docs for CUDA_CTC_DECODER for more info on how it was used here:
    # https://pytorch.org/audio/main/tutorials/asr_inference_with_cuda_ctc_decoder_tutorial.html
    output = output.contiguous()
    output_lens = output_lens.to(torch.int32).contiguous()
    beam_results = decoder(output, output_lens) #lengths - list of lengths

    pred_strings = []

    for i in range(len(beam_results)):
        # Robustly handle different decoder return types
        hyp0 = beam_results[i][0]
        
        # Try attribute access first, then dict-like access
        tokens = getattr(hyp0, "tokens", None)
        if tokens is None:
            tokens = hyp0.get("tokens", None) if hasattr(hyp0, "get") else None
        
        # Convert to list if it's a tensor
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        
        # Map the sequence of indices to actual phoneme LABELS and join them into a string
        pred_strings.append("".join(PHONEME_MAP[t] for t in tokens))
    return pred_strings

def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP= LABELS): # y - sequence of integers

    dist            = 0
    batch_size      = label.shape[0]

    pred_strings    = decode_prediction(output, output_lens, decoder, PHONEME_MAP)

    for i in range(batch_size):
        # Truncate labels by their true lengths (not padded length)
        Li = int(label_lens[i].item())
        lab = label[i, :Li].tolist()
        label_string = "".join(PHONEME_MAP[t] for t in lab)
        pred_string = pred_strings[i]

        dist += Levenshtein.distance(pred_string, label_string)

    # Average the distance over the batch
    dist /= batch_size # Think about why we are doing this
    return dist

torch.cuda.empty_cache()
gc.collect()


# test code to check shapes

model.eval()
for i, data in enumerate(val_loader, 0):
    x, y, lx, ly = data
    x, y = x.to(device), y.to(device)
    lx, ly = lx.to(device), ly.to(device)
    h, lh = model(x, lx)
    print(h.shape)
    h = torch.permute(h, (1, 0, 2))
    print(h.shape, y.shape)
    loss = criterion(h, y, lh, ly)
    print(loss)

    print(calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh.to(device), ly, decoder, LABELS))

    del x, y, lx, ly, h, lh, loss
    torch.cuda.empty_cache()

    break



# Use wandb? Resume Training?
USE_WANDB = config['wandb']

RESUME_LOGGING = False # Set this to true if you are resuming training from a previous run

# Create your wandb run
run_name = '{RUN_NAME}_checkpoint_submission'.format(RUN_NAME=config['RUN_NAME'])

# If you are resuming an old run
if USE_WANDB:

    wandb.login(key="") #TODO

    if RESUME_LOGGING:
        run = wandb.init(
            id     = "", ### Insert specific run id here if you want to resume a previous run
            resume = "must", ### You need this to resume previous runs
            project = "hw3p2-ablations", ### Project should be created in your wandb
            settings = wandb.Settings(_service_wait=300)
        )


    else:
        run = wandb.init(
            name    = run_name, ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit  = True, ### Allows reinitalizing runs when you re-run this cell
            project = "hw3p2-ablations", ### Project should be created in your wandb account
            config  = config ### Wandb Config for your run
        )

        ### Save your model architecture as a string with str(model)
        model_arch  = str(model)
        ### Save it in a txt file
        arch_file   = open("model_arch.txt", "w")
        file_write  = arch_file.write(model_arch)
        arch_file.close()

        ### log it in your wandb run with wandb.save()
        wandb.save('model_arch.txt')
# Train function
def train_model(model, train_loader, criterion, optimizer):

    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    total_loss = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)
        lx, ly = lx.to(device), ly.to(device)

        with torch.cuda.amp.autocast():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() # Update tqdm bar

        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16
        
        # Step scheduler per batch for OneCycleLR
        scheduler.step()

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close() # You need this to close the tqdm bar

    return total_loss / len(train_loader)


# Eval function
def validate_model(model, val_loader, decoder, phoneme_map= LABELS):

    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    vdist = 0

    for i, data in enumerate(val_loader):

        x, y, lx, ly = data
        x, y = x.to(device), y.to(device)
        lx, ly = lx.to(device), ly.to(device)

        with torch.inference_mode():
            h, lh = model(x, lx)
            h = torch.permute(h, (1, 0, 2))
            loss = criterion(h, y, lh, ly)

        total_loss += loss.item()
        vdist += calculate_levenshtein(torch.permute(h, (1, 0, 2)), y, lh.to(device), ly, decoder, phoneme_map)

        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))), dist="{:.04f}".format(float(vdist / (i + 1))))

        batch_bar.update()

        del x, y, lx, ly, h, lh, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    total_loss = total_loss/len(val_loader)
    val_dist = vdist/len(val_loader)
    return total_loss, val_dist
def save_model(model, optimizer, scheduler, metric, epoch, path):
    torch.save(
        {'model_state_dict'         : model.state_dict(),
         'optimizer_state_dict'     : optimizer.state_dict(),
         'scheduler_state_dict'     : scheduler.state_dict() if scheduler is not None else {},
         metric[0]                  : metric[1],
         'epoch'                    : epoch},
         path
    )
    print(f"✓ Checkpoint saved locally to: {path}")

def load_model(path, model, optimizer= None, scheduler= None, metric='valid_dist'):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch   = checkpoint['epoch']
    metric  = checkpoint[metric]

    print("\nResuming training from epoch:", epoch)
    print('----------------------------------------\n')
    print("Epochs left: ", config['epochs'] - epoch)
    print("Optimizer: \n", optimizer)
    print("Current Schedueler T_cur:", scheduler.T_cur)

    print("Best Val Dist:", metric)

    return [model, optimizer, scheduler, epoch, metric]

# Instantiate variables used in training loop
last_epoch_completed = 0
best_lev_dist = float("inf")

# ==================== RESUME TRAINING WITH STRONGER AUGMENTATION ====================
# Phase 1: Train with freq_mask_param=10, time_mask_param=10
# Phase 2: Set RESUME_TRAINING=True, update config.yaml with freq_mask_param=20, time_mask_param=20
#          Then run again to continue training with stronger augmentation
RESUME_TRAINING = False # Set this to True when resuming

#     checkpoint_path = ''
#     checkpoint = load_model(checkpoint_path, model, optimizer, scheduler, metric='valid_dist')

#     last_epoch_completed = checkpoint[3]
#     best_lev_dist = checkpoint[4]

# Set up checkpoint directories and WanDB logging watch
checkpoint_root = os.path.join(os.getcwd(), 'checkpoints')
os.makedirs(checkpoint_root, exist_ok=True)
wandb.watch(model, log="all")

checkpoint_best_model_filename = 'checkpoint-best-model.pth'
checkpoint_last_epoch_filename = 'checkpoint-last-epoch.pth'
epoch_model_path = os.path.join(checkpoint_root, checkpoint_last_epoch_filename)
best_model_path = os.path.join(checkpoint_root, checkpoint_best_model_filename)

# # WanDB log watch
# if config['wandb']:
#   wandb.watch(model, log="all")


# # Clear RAM for storage before you start training
# torch.cuda.empty_cache()
# gc.collect()

# TODO: Please complete the training loop

for epoch in range(last_epoch_completed, config['epochs']):

    print("\nEpoch: {}/{}".format(epoch + 1, config['epochs']))

    curr_lr = optimizer.param_groups[0]['lr']

    train_loss = train_model(model, train_loader, criterion, optimizer)
    valid_loss, valid_dist = validate_model(model, val_loader, decoder)

    # NOTE: OneCycleLR is stepped per batch inside train_model, not here

    print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
    print("\tVal Dist {:.04f}\t Val Loss {:.04f}".format(valid_dist, valid_loss))

    if config['wandb']:
        wandb.log({
            'train_loss': train_loss,
            'valid_dist': valid_dist,
            'valid_loss': valid_loss,
            'lr': curr_lr
    })

    # Save last epoch model locally
    save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, epoch_model_path)
    if config['wandb']:
        wandb.save(epoch_model_path)  # Upload to wandb

    # Save best model when validation improves
    if valid_dist <= best_lev_dist:
        best_lev_dist = valid_dist
        save_model(model, optimizer, scheduler, ['valid_dist', valid_dist], epoch, best_model_path)
        if config['wandb']:
            wandb.save(best_model_path)  # Upload to wandb
        print(f"🎯 New best model! (dist: {valid_dist:.4f})")

# You may find it interesting to explore Wandb Artifacts to version your models

# Finish Wandb run
if config['wandb']:
    run.finish()
# load checkpoint

#TODO: Make predictions

# Follow the steps below:
# 1. Create a new object for CUDA_CTC_DECODER with larger number of beams (why larger?)
# 2. Get prediction string by decoding the results of the beam decoder


# Decoder (modest beam; aggressive blank skip for speed)
test_decoder = cuda_ctc_decoder(
    tokens=LABELS,
    nbest=1,
    beam_size=7,  
    blank_skip_threshold=0.95
)

model.eval()
torch.backends.cudnn.benchmark = True  # speed up conv/LSTM for fixed shapes

results = []
with torch.inference_mode():
    for i, (x, lx) in enumerate(tqdm(test_loader)):
        x  = x.to(device, non_blocking=True)
        lx = lx.to(device, non_blocking=True)

        h, lh = model(x, lx)                    # h: (B, T, C) log-probs; lh: (B,)
        
        preds = decode_prediction(h, lh, test_decoder, LABELS)
        results.extend(preds)
        
        # Clear memory after each batch
        del x, lx, h, lh
        torch.cuda.empty_cache()      
        
if results:
    df = pd.DataFrame({
        'index': range(len(results)), 'label': results
    })

data_dir = "submission.csv"
df.to_csv(data_dir, index = False)