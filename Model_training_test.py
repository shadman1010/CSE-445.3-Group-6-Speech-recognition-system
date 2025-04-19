pip install librosa numpy pandas torch torchaudio tensorflow keras soundfile
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MFCC
import pandas as pd

# === Configuration ===
# Make sure this points to your mounted Drive path, not a web URL
CSV_PATH = "/content/drive/MyDrive/speech recognition/archive/speech_emotions.csv"  # e.g. "/content/drive/MyDrive/datasets/data.csv"
FILES_ROOT = "/content/drive/MyDrive/speech recognition/archive/files"   # e.g. "/content/drive/MyDrive/datasets/files"

# Verify CSV exists
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Please ensure the file is in Drive and the path is correct.")

# Audio preprocessing parameters
sample_rate = 16000
n_mfcc = 40
mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)

# === Build character vocabulary ===
from collections import OrderedDict

def build_vocab(transcripts):
    char2idx = OrderedDict({"<blank>": 0})
    for text in transcripts:
        for ch in text.lower():
            if ch not in char2idx:
                char2idx[ch] = len(char2idx)
    return char2idx

# === please work Dataset ===
class DriveAudioDataset(Dataset):
    def __init__(self,
                 csv_path,
                 files_root,
                 code_col='set_id',
                 text_col='text',
                 filename_col=None,
                 vocab=None):
        # Load CSV with default comma delimiter
        self.df = pd.read_csv(csv_path)

        # Ensure required columns
        missing = [c for c in (code_col, text_col) if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. Available columns: {list(self.df.columns)}"
            )

        # Prepare lists
        self.codes = self.df[code_col].astype(str).tolist()
        self.texts = self.df[text_col].astype(str).tolist()
        self.files_root = files_root

        # Build file paths
        if filename_col and filename_col in self.df.columns:
            # Use explicit filenames from CSV
            self.file_paths = [
                os.path.join(files_root, code, fname)
                for code, fname in zip(self.codes, self.df[filename_col].astype(str))
            ]
        else:
            # Infer: for each row, pick the i-th file in its code folder (sorted)
            # Count how many times each code has appeared so far
            counters = {}
            self.file_paths = []
            for code in self.codes:
                dir_path = os.path.join(files_root, code)
                if code not in counters:
                    counters[code] = 0
                # List and sort files once
                if counters[code] == 0:
                    files = sorted(os.listdir(dir_path))
                    setattr(self, f"_files_{code}", files)
                files = getattr(self, f"_files_{code}")
                idx = counters[code]
                if idx >= len(files):
                    raise ValueError(f"Not enough files for code {code}: expected at least {idx+1}, found {len(files)}")
                self.file_paths.append(os.path.join(dir_path, files[idx]))
                counters[code] += 1

        # Build or reuse vocabulary
        self.char2idx = vocab if vocab else build_vocab(self.texts)
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        transcript_text = self.texts[idx].lower()

        # Load audio
        waveform, sr = torchaudio.load(file_path)
        # Convert multi-channel to mono if needed
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if necessary
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        # Extract MFCC features: output shape [channel, n_mfcc, time]
        mfcc_tensor = mfcc_transform(waveform)
        # Merge channel dimension and transpose to [time, features]
        mfcc = mfcc_tensor.mean(dim=0).transpose(0, 1)

        # Tokenize transcript
        token_ids = [self.char2idx.get(ch, 0) for ch in transcript_text]
        return mfcc, torch.tensor(token_ids, dtype=torch.long)

# === Collate Function ===
def collate_fn(batch):
    waveforms, transcripts = zip(*batch)
    input_lengths = [w.size(0) for w in waveforms]
    target_lengths = [t.size(0) for t in transcripts]

    padded_waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True).transpose(1, 2)
    padded_transcripts = nn.utils.rnn.pad_sequence(transcripts, batch_first=True)
    return padded_waveforms, padded_transcripts, torch.tensor(input_lengths), torch.tensor(target_lengths)

# === DataLoader ===
dataset = DriveAudioDataset(CSV_PATH, FILES_ROOT)
vocab = dataset.char2idx
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# === Model ===
class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)

# === Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeechRecognitionModel(n_mfcc, 256, len(vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

# === Training Loop ===
def train_epoch(data_loader):
    model.train()
    total_loss = 0.0
    for waveforms, transcripts, input_lengths, target_lengths in data_loader:
        waveforms, transcripts = waveforms.to(device), transcripts.to(device)
        input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)
        optimizer.zero_grad()
        outputs = model(waveforms).permute(1, 0, 2)
        loss = ctc_loss(outputs, transcripts, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Avg loss: {total_loss / len(data_loader):.4f}")

# === Execute Training ===
if __name__ == '__main__':
    for epoch in range(1, 11):
        print(f"Epoch {epoch}")
        train_epoch(loader)

