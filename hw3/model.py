import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as transforms

from constants import SAMPLE_RATE, N_MELS, N_FFT, F_MAX, F_MIN, HOP_SIZE


class LogMelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.melspectrogram = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_SIZE, f_min=F_MIN, f_max=F_MAX, n_mels=N_MELS, normalized=False)
    
    def forward(self, audio):
        batch_size = audio.shape[0]
        
        # alignment correction to match with pianoroll
        # pretty_midi.get_piano_roll use ceil, but torchaudio.transforms.melspectrogram uses
        # round when they convert the input into frames.
        padded_audio = nn.functional.pad(audio, (N_FFT // 2, 0), 'constant')
        mel = self.melspectrogram(audio)[:, :, 1:]
        mel = mel.transpose(-1, -2)
        mel = th.log(th.clamp(mel, min=1e-9))
        return mel



class ConvStack(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit):
        super().__init__()

        # shape of input: (batch_size * 1 channel * frames * input_features)
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(cnn_unit, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(cnn_unit, cnn_unit * 2, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((cnn_unit * 2) * (n_mels // 4), fc_unit), # batch size * frames * fc_unit
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2) #change the dimension (batch * channel * frame * features --> batch * frame * channel * features)
        # flatten -2 is summing the dimension of 2nd thing from back
        x = self.fc(x)
        return x


class Transcriber(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()

        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_fc = nn.Linear(fc_unit, 88)
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_fc = nn.Linear(fc_unit, 88)

    def forward(self, audio):
        mel = self.melspectrogram(audio)

        x = self.frame_conv_stack(mel)  # (B x T x C) : Batch size x Time x Channel size
        frame_out = self.frame_fc(x)

        x = self.onset_conv_stack(mel)  # (B x T x C)
        onset_out = self.onset_fc(x)
        return frame_out, onset_out

# Question 1 : Implement LSTM based Model
class Transcriber_RNN(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()
        
        self.frame_bilstm = nn.LSTM(input_size=N_MELS, hidden_size=88, num_layers=2, bidirectional=True, batch_first=True)
        self.frame_fc = nn.Linear(2*88, 88)
        
        self.onset_bilstm = nn.LSTM(input_size=N_MELS, hidden_size=88, num_layers=2, bidirectional=True, batch_first=True)
        self.onset_fc = nn.Linear(2*88, 88)

    def forward(self, audio):
        mel = self.melspectrogram(audio)

        x, _ = self.frame_bilstm(mel)
        frame_out = self.frame_fc(x)

        x, _ = self.onset_bilstm(mel)
        onset_out = self.onset_fc(x)
        return frame_out, onset_out

# Question 2 : Implement CNN-RNN(CRNN) Model
class Transcriber_CRNN(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()

        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_bilstm = nn.LSTM(input_size=fc_unit, hidden_size=88, num_layers=2, bidirectional=True, batch_first=True)
        self.frame_fc = nn.Linear(2*88, 88)
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_bilstm = nn.LSTM(input_size=fc_unit, hidden_size=88, num_layers=2, bidirectional=True, batch_first=True)
        self.onset_fc = nn.Linear(2*88, 88)

    def forward(self, audio):
        mel = self.melspectrogram(audio)

        x = self.frame_conv_stack(mel) 
        x, _ = self.frame_bilstm(x) 
        frame_out = self.frame_fc(x)

        x = self.onset_conv_stack(mel)
        x, _ = self.onset_bilstm(x)
        onset_out = self.onset_fc(x)

        return frame_out, onset_out

# Question 3 : Implement Onsets-and-Frames Model, which have interconnection between         
class Transcriber_ONF(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()
        self.frame_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.frame_middle_fc = nn.Linear(fc_unit, 88)
        self.frame_bilstm = nn.LSTM(input_size=2*88, hidden_size=88, num_layers=2, bidirectional=True, batch_first=True)
        self.frame_fc = nn.Linear(2*88, 88)
        
        self.onset_conv_stack = ConvStack(N_MELS, cnn_unit, fc_unit)
        self.onset_bilstm = nn.LSTM(input_size=fc_unit, hidden_size=88, num_layers=2, bidirectional=True, batch_first=True)
        self.onset_fc = nn.Linear(2*88, 88)

    def forward(self, audio):
        mel = self.melspectrogram(audio)
        # onset
        onset_out = self.onset_conv_stack(mel) # batch * frame * fc_unit
        onset_out, _ = self.onset_bilstm(onset_out) # batch * frame * (fc_unit x 2)
        onset_out = self.onset_fc(onset_out) # batch * frame * piano_roll

        # frame
        frame_middle = self.frame_conv_stack(mel)  
        frame_middle = self.frame_middle_fc(frame_middle)
        # concatenate
        onset_add = th.sigmoid(onset_out.clone().detach())
        frame_middle = th.cat([frame_middle, onset_add], dim=2)
        frame_middle, _ = self.frame_bilstm(frame_middle)
        frame_out = self.frame_fc(frame_middle)

        return frame_out, onset_out

# Question 4 : Implement uni directional LSTM based Model
class Transcriber_udRNN(nn.Module):
    def __init__(self, cnn_unit, fc_unit):
        super().__init__()
        self.melspectrogram = LogMelSpectrogram()
        
        self.frame_unilstm = nn.LSTM(input_size=N_MELS, hidden_size=2*88, num_layers=2, bidirectional=False, batch_first=True)
        self.frame_fc = nn.Linear(2*88, 88)
        
        self.onset_unilstm = nn.LSTM(input_size=N_MELS, hidden_size=2*88, num_layers=2, bidirectional=False, batch_first=True)
        self.onset_fc = nn.Linear(2*88, 88)

    def forward(self, audio):
        mel = self.melspectrogram(audio)

        x, _ = self.frame_unilstm(mel)
        frame_out = self.frame_fc(x)

        x, _ = self.onset_unilstm(mel)
        onset_out = self.onset_fc(x)
        return frame_out, onset_out