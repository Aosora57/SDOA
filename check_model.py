import torch

# Load the pre-trained model
model = torch.load('net_attention_50.pkl', map_location=torch.device('cpu'))

# Print the model architecture
print(model)
# import torch
# import pandas as pd
#
# # Load the training data
# training_data = torch.load('training_data.pt', weights_only=True)
# signal = training_data['signal'].numpy().flatten()
# doa = training_data['doa'].numpy().flatten()
# ref_sp = training_data['ref_sp'].numpy().flatten()
#
# # Ensure all arrays have the same length
# min_length = min(len(signal), len(doa), len(ref_sp))
# training_df = pd.DataFrame({
#     'signal': signal[:min_length],
#     'doa': doa[:min_length],
#     'ref_sp': ref_sp[:min_length]
# })
# training_df.to_csv('training_data.csv', index=False)
# print("Training data saved to training_data.csv")
#
# # Load the validation data
# validation_data = torch.load('validation_data.pt', weights_only=True)
# noisy_signals = validation_data['noisy_signals'].numpy().flatten()
# signal = validation_data['signal'].numpy().flatten()
# doa = validation_data['doa'].numpy().flatten()
# ref_sp = validation_data['ref_sp'].numpy().flatten()
#
# # Ensure all arrays have the same length
# min_length = min(len(noisy_signals), len(signal), len(doa), len(ref_sp))
# validation_df = pd.DataFrame({
#     'noisy_signals': noisy_signals[:min_length],
#     'signal': signal[:min_length],
#     'doa': doa[:min_length],
#     'ref_sp': ref_sp[:min_length]
# })
# validation_df.to_csv('validation_data.csv', index=False)
# print("Validation data saved to validation_data.csv")


''' version 1 net.pkl
spectrumModule(
  (in_layer): Linear(in_features=32, out_features=64, bias=False)
  (mod): Sequential(
    (0): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(2,), bias=False, padding_mode=circular)
    (1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(2,), bias=False, padding_mode=circular)
    (4): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(2,), bias=False, padding_mode=circular)
    (7): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU()
    (9): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(2,), bias=False, padding_mode=circular)
    (10): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU()
    (12): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(2,), bias=False, padding_mode=circular)
    (13): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(2,), bias=False, padding_mode=circular)
    (16): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (17): ReLU()
  )
  (out_layer): Linear(in_features=64, out_features=32, bias=False)
)
'''

''' version 2  net_attention_50.pkl
spectrumModule(
  (in_layer): Linear(in_features=32, out_features=64, bias=False)
  (mod): Sequential(
    (0): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=same, bias=False)
    (1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.5, inplace=False)
    (4): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=same, bias=False)
    (5): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): Dropout(p=0.5, inplace=False)
    (8): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=same, bias=False)
    (9): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
    (11): Dropout(p=0.5, inplace=False)
    (12): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=same, bias=False)
    (13): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): Dropout(p=0.5, inplace=False)
    (16): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=same, bias=False)
    (17): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (18): ReLU()
    (19): Dropout(p=0.5, inplace=False)
    (20): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=same, bias=False)
    (21): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (22): ReLU()
    (23): Dropout(p=0.5, inplace=False)
  )
  (attention): Attention(
    (Wa): Linear(in_features=2, out_features=16, bias=True)
    (Ua): Linear(in_features=16, out_features=2, bias=True)
    (output_layer): Linear(in_features=2, out_features=64, bias=True)
  )
  (out_layer): Linear(in_features=64, out_features=32, bias=False)
)
'''