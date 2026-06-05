import torch
import torch.nn.functional as F
import os, math
import torch.nn as nn
import numpy as np
import torchaudio.transforms as T

class NCL2NLC(torch.nn.Module):
    def __init__(self):
        super(NCL2NLC, self).__init__()

    def forward(self, input):
        """
        input : Tensor of shape (batch size, T, Cin)

        Outputs a Tensor of shape (batch size, Cin, T).
        """

        return input.transpose(1,2)

class AmplitudeToDb(torch.nn.Module):
    def __init__(self, ref=1.0, amin=1e-7):
        super(AmplitudeToDb, self).__init__()
        self.ref = ref
        self.amin = amin

    def forward(self, input):
        input = torch.clamp(input, min=self.amin)
        return 10.0 * (torch.log10(input) - torch.log10(torch.tensor(self.ref, device=input.device, requires_grad=False)))

class SelectMels(torch.nn.Module):
    def __init__(self, n_mel_select):
        super(SelectMels, self).__init__()
        self.n_mel_select = n_mel_select

    def forward(self, input):
        return input[:,:self.n_mel_select,:]

def compute_criterion(
    inputs, 
    targets, 
    lengths, 
    mask=None, 
    word_embedding=None, 
    loss_mask=None, 
    lang_mask=None,
    phoneme_lang_mask=None,
    calculate_accuracy=False,
    loss_type="bce",):
        losses = { "words": None }
        accuracies = { "words": None }
        word_labels = targets["word_labels"]
        tar = targets['words']
        # Word Loss
        time_Dependency_Criterion= (10, 0.05)
        word_out = inputs["word"]

        if loss_type == "mse":
            inp = F.tanh(word_out)
            loss = nn.MSELoss(reduction='none')(inp, tar)
        elif loss_type == "bce":
            loss = nn.BCEWithLogitsLoss(reduction='none')(word_out, tar)
            inp = F.sigmoid(word_out)
        elif loss_type == "cosine":
            loss = 1 - F.cosine_similarity(inp, tar, dim=-1)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        if mask is not None: loss *= mask.unsqueeze(-1)
        losses["words"] = loss.sum() / mask.sum()
                
        # Word Accuracy
        if calculate_accuracy:
            cor, tot = 0., 0.
            for length, inp_, word_label in zip(lengths, inp, word_labels):
                inp_ = inp_.detach()

                cs_array = torch.cosine_similarity(
                    word_embedding[:,None,:].to(inp_.device).repeat(1, inp_.shape[0],1),
                    inp_[None,:,:].repeat(word_embedding.shape[0],1,1),
                    dim=2)
                cs_array[:,length:] = cs_array[:,[length - 1]]
                cs_array = cs_array.cpu().numpy()
                target_Index = torch.unique(word_label[:length]).cpu().numpy().tolist()
                target_Index = [ti for ti in target_Index if ti!= -1]
                assert len(target_Index) == 1, "More than one target index found in word label: {}".format(target_Index)
                target_Array = cs_array[target_Index[0]]
                other_Max_Array = np.max(np.delete(cs_array, target_Index, 0), axis=0)
                
                #Time dependent RT
                time_Dependency_Check_Array_with_Criterion = target_Array > other_Max_Array + time_Dependency_Criterion[1]
                time_Dependency_Check_Array_Sustainment = target_Array > other_Max_Array
                rt_time = -1
                for cycle in range(target_Array.shape[0] - time_Dependency_Criterion[0]):
                    if all(np.hstack([
                        time_Dependency_Check_Array_with_Criterion[cycle:cycle + time_Dependency_Criterion[0]],
                        time_Dependency_Check_Array_Sustainment[cycle + time_Dependency_Criterion[0]:]
                        ])):
                        rt_time = cycle
                        break
                cor += 1 if rt_time != -1 else 0
                tot += 1 
            print(f'Word accuracy: {cor}/{tot} = {cor/tot:.4f}')
            accuracies["words"] = torch.tensor(1.0, device=inputs["word"].device) * cor / tot
        else:
            accuracies["words"] = torch.tensor(-1.0, device=inputs["word"].device)

        loss = [losses[key]*loss_mask[key] for key in losses.keys() if losses[key] is not None]
        return sum(loss), losses, accuracies

class baselineModel(torch.nn.Module):
    def __init__(self, config):
        super(baselineModel, self).__init__() 
        self.rnn_phone = torch.nn.LSTM(
            input_size=config.inp_size, 
            hidden_size=config.rnn_hidden_size,
            num_layers=1,
            batch_first=True) 
        self.word_linear = torch.nn.Linear(config.rnn_hidden_size, config.pretrained_word_embeddings_dim) 
        self.is_cuda = torch.cuda.is_available()

    def forward(self, data, lengths, h0=None, return_word_rnn=False, training=True):
        rnn_phone_output = self.rnn_phone(data[0])[0]  
        word_out = self.word_linear(rnn_phone_output)
         
        return {
            "rnn_phone": rnn_phone_output,
            "word": word_out,
        }
    
    def criterion(self, **kwargs):
        return compute_criterion(
            **kwargs
            ) 

class BibaselineModel(torch.nn.Module):
    def __init__(self, config):
        super(BibaselineModel, self).__init__()
        self.rnn_phone = torch.nn.LSTM(
            input_size=config.inp_size, 
            hidden_size=320,
            num_layers=1,
            batch_first=True,
            bidirectional=True) 
        self.word_linear = torch.nn.Linear(640, config.pretrained_word_embeddings_dim) 
        self.is_cuda = torch.cuda.is_available()

    def forward(self, data, lengths, h0=None, return_word_rnn=False, training=True):
        rnn_phone_output = self.rnn_phone(data[0])[0]  
        word_out = self.word_linear(rnn_phone_output)
         
        return {
            "rnn_phone": rnn_phone_output,
            "word": word_out,
        }
    
    def criterion(self, **kwargs):
        return compute_criterion(
            **kwargs
            ) 
               
class LSTMModel(torch.nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__() 
        self.spec_augment = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=35), # masks up to 15 frequency bins
            T.TimeMasking(time_mask_param=35)  # masks up to 35 time steps
        )  
        self.rnn_phone = torch.nn.LSTM(
            input_size=config.inp_size, 
            hidden_size=config.rnn_hidden_size,
            num_layers=1,  
            batch_first=True)
        self.rnn_word = torch.nn.LSTM(
            input_size=config.rnn_hidden_size, 
            hidden_size=config.rnn_hidden_size,
            num_layers=1,
            batch_first=True)
        
        self.word_linear = torch.nn.Linear(
            config.rnn_hidden_size, 
            config.pretrained_word_embeddings_dim) 
        self.is_cuda = torch.cuda.is_available()
        
    def forward(self, data, lengths, h0=None, return_word_rnn=False, training=True):
        if training:
            inp = self.spec_augment(data[0])
        else:
            inp = data[0]
        rnn_phone_output = self.rnn_phone(inp)[0] 
        rnn_word_output = self.rnn_word(rnn_phone_output)[0]
 
        word_out = self.word_linear(rnn_word_output)
         
        return {
            "rnn_phone": rnn_phone_output,
            "rnn_word": rnn_word_output,
            "word": word_out,
        }
        
    def criterion(self, **kwargs):
        return compute_criterion(
            **kwargs
            )

class BiLSTMModel(torch.nn.Module):
    def __init__(self, config):
        super(BiLSTMModel, self).__init__() 
        self.spec_augment = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=35), # masks up to 15 frequency bins
            T.TimeMasking(time_mask_param=35)  # masks up to 35 time steps
        )  
        self.rnn_phone = torch.nn.LSTM(
            input_size=config.inp_size, 
            hidden_size=128,
            num_layers=1,  
            batch_first=True,
            bidirectional=True)
        self.rnn_word = torch.nn.LSTM(
            input_size=256, 
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        
        self.word_linear = torch.nn.Linear(
            512, 
            config.pretrained_word_embeddings_dim) 
        self.is_cuda = torch.cuda.is_available()
        
    def forward(self, data, lengths, h0=None, return_word_rnn=False, training=True):
        if training:
            inp = self.spec_augment(data[0])
        else:
            inp = data[0]
        rnn_phone_output = self.rnn_phone(inp)[0] 
        rnn_word_output = self.rnn_word(rnn_phone_output)[0]
 
        word_out = self.word_linear(rnn_word_output)
         
        return {
            "rnn_phone": rnn_phone_output,
            "rnn_word": rnn_word_output,
            "word": word_out,
        }
        
    def criterion(self, **kwargs):
        return compute_criterion(
            **kwargs
            ) 
# ============================================================================
# Additional Model Architectures
# ============================================================================

class CausalConv1dWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1dWrapper, self).__init__()
        
        # Calculate padding needed for causality
        padding = (kernel_size - 1) * dilation
        
        # Create the convolution layer with full padding
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        
        # Store parameters for trimming
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
    
    def forward(self, x):
        # Apply convolution with full padding
        x = self.conv(x)
        
        # Trim the right side to ensure causality
        trim_amount = (self.kernel_size - 1) * self.dilation
        if trim_amount > 0:
            x = x[:, :, :-trim_amount]
        return x
    
class CausalCNNModel(nn.Module):
    """
    Version using CausalConv1dWrapper for cleaner code.
    Updated to match CNNModel structure: 4 blocks of increasing dilation + 1 mixer.
    """
    def __init__(self, config):
        super(CausalCNNModel, self).__init__()
        self.spec_augment = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=35), # masks up to 15 frequency bins
            T.TimeMasking(time_mask_param=35)  # masks up to 35 time steps
        ) 
        
        self.is_cuda = torch.cuda.is_available() 
        # Block 1: Local features (no dilation)
        # Input/64 -> 64 -> 64
        self.conv1_block = nn.Sequential(
            CausalConv1dWrapper(config.inp_size, 64, kernel_size=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            CausalConv1dWrapper(64, 64, kernel_size=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        
        # Block 2: Dilation 2
        # 64 -> 128 -> 128
        self.conv2_block = nn.Sequential(
            CausalConv1dWrapper(64, 128, kernel_size=5, dilation=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            CausalConv1dWrapper(128, 128, kernel_size=5, dilation=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        
        # Block 3: Dilation 4
        # 128 -> 256 -> 256
        self.conv3_block = nn.Sequential(
            CausalConv1dWrapper(128, 256, kernel_size=7, dilation=4, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            CausalConv1dWrapper(256, 256, kernel_size=7, dilation=4, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # Block 4: Dilation 8
        # 256 -> 256 -> 256
        self.conv4_block = nn.Sequential(
            CausalConv1dWrapper(256, 256, kernel_size=7, dilation=8, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            CausalConv1dWrapper(256, 256, kernel_size=7, dilation=8, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # Block 5: Mixer (Kernel 1)
        self.conv5_block = nn.Sequential(
            CausalConv1dWrapper(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
                
        # Output layers
        self.word_linear = nn.Linear(256, config.pretrained_word_embeddings_dim)

        
    def forward(self, data, lengths, h0=None, training=None):
        if training is None:
            training = self.training

        if training:
            inp = self.spec_augment(data[0])
        else:
            inp = data[0]
            
        # Input shape: (batch, time, freq)
        x = inp.transpose(1, 2)  # -> (batch, freq, time)
        
        # Apply causal convolutions (no trimming needed with wrapper)
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.conv3_block(x)
        x = self.conv4_block(x)
        x = self.conv5_block(x)
        
        # Apply dropout and transpose 
        x = x.transpose(1, 2)  # -> (batch, time, channels)
        
        # Generate outputs
        word_out = self.word_linear(x)
        
        return {
            "word": word_out,
        }
        
        # Apply dropout (none here) and transpose 
        x = x.transpose(1, 2)  # -> (batch, time, channels)
        
        # Generate outputs
        word_out = self.word_linear(x)
        
        return {
            "word": word_out,
        }

    def criterion(self, **kwargs):
        return compute_criterion(
            **kwargs
            ) 

class CNNModel(nn.Module):
    """
    Ordinary CNN model with same architecture as CausalCNNModel but using standard convolutions
    (non-causal, centered window).
    """
    def __init__(self, config):
        super(CNNModel, self).__init__()
        self.spec_augment = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=35),
            T.TimeMasking(time_mask_param=35)
        ) 
        
        self.is_cuda = torch.cuda.is_available() 
        
        # Block 1: Local features (no dilation)
        # Input -> 64 -> 64
        self.conv1_block = nn.Sequential(
            nn.Conv1d(config.inp_size, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        
        # Block 2: Dilation 2
        # 64 -> 128 -> 128
        self.conv2_block = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        
        # Block 3: Dilation 4
        # 128 -> 256 -> 256
        self.conv3_block = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, padding=12, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=7, padding=12, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # Block 4: Dilation 8
        # 256 -> 256 -> 256 (Reduced from 512)
        self.conv4_block = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, padding=24, dilation=8),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=7, padding=24, dilation=8),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # Final mixing
        # 256 -> 256 (Reduced from 512)
        self.conv5_block = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
                
        # Output layers
        self.word_linear = nn.Linear(256, config.pretrained_word_embeddings_dim)

        
    def forward(self, data, lengths, h0=None, training=True):
        if training:
            inp = self.spec_augment(data[0])
        else:
            inp = data[0]
            
        # Input shape: (batch, time, freq)
        x = inp.transpose(1, 2)  # -> (batch, freq, time)
        
        # Apply dilated convolutions
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.conv3_block(x)
        x = self.conv4_block(x)
        x = self.conv5_block(x)
        
        # Apply dropout 
        x = x.transpose(1, 2)  # -> (batch, time, channels)
        
        # Generate outputs
        word_out = self.word_linear(x)
        
        return {
            "word": word_out,
        }

    def criterion(self, **kwargs):
        return compute_criterion(
            **kwargs
            )

class CausalRCNNModel(nn.Module):
    """
    Alternative implementation using the CausalConv1dWrapper from previous implementation.
    This provides cleaner code by encapsulating the causal logic in the wrapper.
    """
    def __init__(self, config):
        super(CausalRCNNModel, self).__init__()
        self.spec_augment = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=35), # masks up to 15 frequency bins
            T.TimeMasking(time_mask_param=35)  # masks up to 35 time steps
        ) 
        self.is_cuda = torch.cuda.is_available() 
        self.rnn_hidden_size = config.rnn_hidden_size
 
        # Causal convolutional layers with increasing dilation
                # Causal convolutional layers using wrapper
        self.convnet = nn.Sequential(
            CausalConv1dWrapper(256, 64, kernel_size=3, dilation=1,bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(64, 64, kernel_size=3, dilation=1,bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(64, 128, kernel_size=5, dilation=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(128, 128, kernel_size=5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(128, 128, kernel_size=7, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(128, 256, kernel_size=7, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        ) 
         
        self.receptive_field = 25
        
        # Recurrent layer 
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=config.rnn_hidden_size,
            batch_first=True,
            num_layers=2
        )
        
        self.out_dim = config.rnn_hidden_size 
        
        # Output layers 
        self.word_linear = nn.Linear(self.out_dim, config.pretrained_word_embeddings_dim)
        
    def forward(self, data, lengths, h0=None, training=True):
        if training:
            inp = self.spec_augment(data[0])
        else:
            inp = data[0]
            
        # Input shape: (batch, time, freq)
        x = inp.transpose(1, 2)  # -> (batch, freq, time)
        
        # Apply causal convolutions (trimming handled in wrapper)
        cnn_out = self.convnet(x)
        cnn_out = cnn_out.transpose(1, 2)  # -> (batch, time, channels)
        
        # Apply RNN - unidirectional for causality
        rnn_out, (hidden, cell) = self.rnn(cnn_out, h0) 
        
        # Generate outputs
        word_out = self.word_linear(rnn_out)
        
        return {
            "cnn_out": cnn_out,
            "rnn_phone": rnn_out,
            "word": word_out, 
        }

    def criterion(self, **kwargs):
        return compute_criterion(
            **kwargs
            ) 

class RCNNModel(nn.Module):
    """
    Ordinary RCNN model with same architecture as CausalRCNNModel but using standard convolutions
    (non-causal, centered window).
    """
    def __init__(self, config):
        super(RCNNModel, self).__init__()
        self.spec_augment = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=35), # masks up to 15 frequency bins
            T.TimeMasking(time_mask_param=35)  # masks up to 35 time steps
        ) 
        self.is_cuda = torch.cuda.is_available() 
        self.rnn_hidden_size = config.rnn_hidden_size
 
        # Convolutional layers
        self.convnet = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(128, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(128, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(128, 256, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        ) 
         
        self.receptive_field = 25
        
        # Recurrent layer - Unidirectional
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=config.rnn_hidden_size,
            batch_first=True,
            num_layers=2
        )
        
        self.out_dim = config.rnn_hidden_size 
        
        # Output layers 
        self.word_linear = nn.Linear(self.out_dim, config.pretrained_word_embeddings_dim)
        
    def forward(self, data, lengths, h0=None, training=True):
        if training:
            inp = self.spec_augment(data[0])
        else:
            inp = data[0]
            
        # Input shape: (batch, time, freq)
        x = inp.transpose(1, 2)  # -> (batch, freq, time)
        
        # Apply standard convolutions
        cnn_out = self.convnet(x)
        cnn_out = cnn_out.transpose(1, 2)  # -> (batch, time, channels)
        
        # Apply RNN
        rnn_out, (hidden, cell) = self.rnn(cnn_out, h0) 
        
        # Generate outputs
        word_out = self.word_linear(rnn_out)
        
        return {
            "cnn_out": cnn_out,
            "rnn_phone": rnn_out,
            "word": word_out, 
        }

    def criterion(self, **kwargs):
        return compute_criterion(
            **kwargs
            ) 

class CausalRCNNModelWithDilation(nn.Module):
    """
    Advanced version with dilated causal convolutions for larger receptive field.
    """
    def __init__(self, config):
        super(CausalRCNNModelWithDilation, self).__init__()
        self.spec_augment = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=35), # masks up to 15 frequency bins
            T.TimeMasking(time_mask_param=35)  # masks up to 35 time steps
        ) 
        self.is_cuda = torch.cuda.is_available() 
        self.rnn_hidden_size = config.rnn_hidden_size
 
        # Causal convolutional layers with increasing dilation
                # Causal convolutional layers using wrapper
        self.convnet = nn.Sequential(
            CausalConv1dWrapper(256, 64, kernel_size=3, dilation=1,bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(64, 64, kernel_size=3, dilation=1,bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(64, 128, kernel_size=5, dilation=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(128, 128, kernel_size=5, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(128, 128, kernel_size=7, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            CausalConv1dWrapper(128, 256, kernel_size=7, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        ) 
         
        self.receptive_field = 25
        
        # Recurrent layer 
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=config.rnn_hidden_size,
            batch_first=True,
            num_layers=2
        )
        
        self.out_dim = config.rnn_hidden_size 
        
        # Output layers 
        self.word_linear = nn.Linear(self.out_dim, config.pretrained_word_embeddings_dim)
    
    def forward(self, data, lengths, h0=None, training=True):
        if training:
            inp = self.spec_augment(data[0])
        else:
            inp = data[0]
            
        # Input shape: (batch, time, freq)
        x = inp.transpose(1, 2)  # -> (batch, freq, time)
        
        # Apply dilated causal convolutions
        cnn_out = self.convnet(x)
        cnn_out = cnn_out.transpose(1, 2)  # -> (batch, time, channels)
        
        # Apply RNN
        rnn_out, (hidden, cell) = self.rnn(cnn_out, h0) 
        
        # Generate outputs
        word_out = self.word_linear(rnn_out)
        
        return {
            "cnn_out": cnn_out,
            "rnn_out": rnn_out,
            "word": word_out, 
            "receptive_field": self.receptive_field
        }

    def criterion(self, **kwargs):
        return compute_criterion(
            **kwargs
            ) 

class CausalTransformerModel(torch.nn.Module):
    """
    Transformer-based model for acoustic feature processing.
    Supports both bidirectional and causal (unidirectional) modes.
    """
    def __init__(self, config, causal=False):
        super(CausalTransformerModel, self).__init__() 
        self.spec_augment = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=35),
            T.TimeMasking(time_mask_param=35)
        ) 
        self.is_cuda = torch.cuda.is_available() 
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.inp_size, max_len=5000)
        
        # Transformer encoder with more flexible configuration
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=config.inp_size,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        ) 
        
        # Output layers 
        self.word_linear = torch.nn.Linear(config.inp_size, config.pretrained_word_embeddings_dim)
 
        # Cache for causal mask (for efficiency)
        self.causal_mask_cache = {}
    
    def generate_causal_mask(self, seq_len, device):
        """Generate a causal mask for transformer attention."""
        # Check cache first
        if seq_len in self.causal_mask_cache:
            return self.causal_mask_cache[seq_len]
        
        # Create upper triangular mask (True = masked)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Cache and return
        self.causal_mask_cache[seq_len] = mask.to(device)
        return self.causal_mask_cache[seq_len]
    
    def forward(self, data, lengths, h0=None, training=True):
        if training:
            inp = self.spec_augment(data[0])
        else:
            inp = data[0]
        
        # Add positional encoding
        x = self.pos_encoder(inp)
        
        # Create attention mask for padding
        src_key_padding_mask = None
        if lengths is not None:
            max_len = x.shape[1]
            # Create mask: True for padding positions
            src_key_padding_mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        
        # Generate causal mask 
        attn_mask = None 
        seq_len = x.size(1)
        attn_mask = self.generate_causal_mask(seq_len, device=x.device)
        
        # Transformer encoder with optional causal masking
        transformer_out = self.transformer_encoder(
            x, 
            mask=attn_mask,  # Causal mask 
            src_key_padding_mask=src_key_padding_mask  # Padding mask
        ) 
        
        # Output projection
        word_out = self.word_linear(transformer_out)
        
        return {
            "word": word_out, 
        }
        
    def criterion(self, **kwargs):
        return compute_criterion(
            **kwargs
            ) 

class TransformerModel(torch.nn.Module):
    """
    Standard Transformer-based model for acoustic feature processing (non-causal / bidirectional).
    """
    def __init__(self, config):
        super(TransformerModel, self).__init__() 
        self.spec_augment = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=35),
            T.TimeMasking(time_mask_param=35)
        ) 
        self.is_cuda = torch.cuda.is_available() 
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.inp_size, max_len=5000)
        
        # Transformer encoder with more flexible configuration
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=config.inp_size,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        ) 
        
        # Output layers 
        self.word_linear = torch.nn.Linear(config.inp_size, config.pretrained_word_embeddings_dim)

        # Cache for causal mask (for efficiency)
        self.causal_mask_cache = {}
        
    def generate_causal_mask(self, seq_len, device):
        """Generate a causal mask for transformer attention."""
        # Check cache first
        if seq_len in self.causal_mask_cache:
            return self.causal_mask_cache[seq_len]
        
        # Create upper triangular mask (True = masked)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Cache and return
        self.causal_mask_cache[seq_len] = mask.to(device)
        return self.causal_mask_cache[seq_len]
    
    def forward(self, data, lengths, h0=None, training=True, causal_inference=False):
        if training:
            inp = self.spec_augment(data[0])
        else:
            inp = data[0]
        
        # Add positional encoding
        x = self.pos_encoder(inp)
        
        # Create attention mask for padding
        src_key_padding_mask = None
        if lengths is not None:
            max_len = x.shape[1]
            # Create mask: True for padding positions
            src_key_padding_mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        
        if causal_inference:
            # Generate causal mask 
            attn_mask = None 
            seq_len = x.size(1)
            attn_mask = self.generate_causal_mask(seq_len, device=x.device)
            
        # Transformer encoder without causal masking (fully bidirectional attention)
        transformer_out = self.transformer_encoder(
            x, 
            mask=attn_mask if causal_inference else None,  # Causal mask if inference
            src_key_padding_mask=src_key_padding_mask  # Padding mask only
        ) 
        
        # Output projection
        word_out = self.word_linear(transformer_out)
        
        return {
            "word": word_out, 
        }
        
    def criterion(self, **kwargs):
        return compute_criterion(
            **kwargs
            ) 
        
class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding for Transformer models.
    Provides position information to the transformer encoder.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Create position indices
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[: , :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, time, features)
        return x + self.pe[:x.shape[1], :].unsqueeze(0)

class CausalConvTransformerModel(nn.Module):
    """
    Causal Hybrid Convolutional-Transformer model.
    Only uses current and past information (no future lookahead).
    """
    def __init__(self, config):
        super(CausalConvTransformerModel, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.inp_size = config.inp_size
        
        # Causal multi-scale CNN front-end
        # Using CausalConv1d with left padding only
        self.conv_layers = nn.Sequential(
            CausalConv1dWrapper(config.inp_size, 128, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # For stride=2 with causal convolution, we need to ensure proper padding
            CausalConv1dWrapper(128, 256, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Causal positional encoding
        self.pos_encoder = PositionalEncoding(256, max_len=5000)
        
        # Causal Transformer encoder with masked attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )
        
        # Output layers 
        self.word_linear = nn.Linear(256, config.pretrained_word_embeddings_dim)
        
        # Cache for causal mask (for efficiency)
        self.causal_mask_cache = {}
        
        if self.is_cuda:
            self.cuda()
    
    def create_causal_mask(self, seq_len, device):
        """Create a causal mask for transformer attention"""
        # Create a lower triangular mask (causal mask)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def forward(self, data, lengths, h0=None, training=True):
        # CNN feature extraction with causal convolutions
        x = data[0].transpose(1, 2)  # -> (batch, freq, time)
        x = self.conv_layers(x).transpose(1, 2)  # -> (batch, time, channels)
        
        # Add causal positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask - combine causal mask with padding mask
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Causal mask for transformer
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Padding mask for variable length sequences
        src_key_padding_mask = None
        if lengths is not None:
            max_len = x.shape[1]
            src_key_padding_mask = torch.arange(max_len, device=device)[None, :] >= lengths[:, None]
        
        # Transformer processing with causal masking
        transformer_out = self.transformer_encoder(
            x, 
            mask=causal_mask,  # Causal attention mask
            src_key_padding_mask=src_key_padding_mask
        )
        
        word_out = self.word_linear(transformer_out)
        
        return {
            "word": word_out,
            "conv_out": x,
            "transformer_out": transformer_out,
        }
        
    def criterion(self, **kwargs):
        return compute_criterion(**kwargs)

class ConvTransformerModel(nn.Module):
    """
    Hybrid Convolutional-Transformer model (non-causal / bidirectional).
    """
    def __init__(self, config):
        super(ConvTransformerModel, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.inp_size = config.inp_size
        
        # Multi-scale CNN front-end (standard convolutions)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.inp_size, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Stride 2
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(256, max_len=5000)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )
        
        # Output layers 
        self.word_linear = nn.Linear(256, config.pretrained_word_embeddings_dim)
        
        # Cache for causal mask (for efficiency)
        self.causal_mask_cache = {}
        
        if self.is_cuda:
            self.cuda()
    
    def generate_causal_mask(self, seq_len, device):
        """Generate a causal mask for transformer attention."""
        # Check cache first
        if seq_len in self.causal_mask_cache:
            return self.causal_mask_cache[seq_len]
        
        # Create upper triangular mask (True = masked)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Cache and return
        self.causal_mask_cache[seq_len] = mask.to(device)
        return self.causal_mask_cache[seq_len]
    
    def forward(self, data, lengths, h0=None, training=True, causal_inference=False):
        # CNN feature extraction
        x = data[0].transpose(1, 2)  # -> (batch, freq, time)
        x = self.conv_layers(x).transpose(1, 2)  # -> (batch, time, channels)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask - padding mask only
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        src_key_padding_mask = None
        if lengths is not None:
            max_len = x.shape[1]
            src_key_padding_mask = torch.arange(max_len, device=device)[None, :] >= lengths[:, None] 
        
        if causal_inference:
            # Generate causal mask 
            attn_mask = None 
            seq_len = x.size(1)
            attn_mask = self.generate_causal_mask(seq_len, device=device)
            
        transformer_out = self.transformer_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask,
            mask=attn_mask if causal_inference else None
        )
        
        word_out = self.word_linear(transformer_out)
        
        return {
            "word": word_out,
            "conv_out": x,
            "transformer_out": transformer_out,
        }
        
    def criterion(self, **kwargs):
        return compute_criterion(**kwargs)

