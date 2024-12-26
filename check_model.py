import torch

# Load the pre-trained model
model = torch.load('net_attention_resnet_50.pkl', map_location=torch.device('cpu'))

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

''' version 3 net_attention_resnet_50.pkl
  model = torch.load('net_attention_resnet_50.pkl', map_location=torch.device('cpu'))
spectrumModule(
  (in_layer): Linear(in_features=32, out_features=64, bias=False)
  (mod): Sequential(
    (0): ResidualBlock(
      (conv1): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn2): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): ResidualBlock(
      (conv1): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn2): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): ResidualBlock(
      (conv1): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn2): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (3): ResidualBlock(
      (conv1): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn2): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (4): ResidualBlock(
      (conv1): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn2): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (5): ResidualBlock(
      (conv1): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn2): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (attention): Attention(
    (Wa): Linear(in_features=2, out_features=16, bias=True)
    (Ua): Linear(in_features=16, out_features=2, bias=True)
    (output_layer): Linear(in_features=2, out_features=64, bias=True)
  )
  (out_layer): Linear(in_features=64, out_features=32, bias=False)
)

'''




''' version 4 net_attention_resnet_50.pkl
spectrumModule(
  (in_layer): Sequential(
    (0): Linear(in_features=32, out_features=64, bias=True)
    (1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.1, inplace=False)
  )
  (blocks): ModuleList(
    (0-5): 6 x ImprovedResidualBlock(
      (conv1): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv1d(2, 2, kernel_size=(3,), stride=(1,), padding=(1,))
      (bn2): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (se): SEBlock(
        (avg_pool): AdaptiveAvgPool1d(output_size=1)
        (fc): Sequential(
          (0): Linear(in_features=2, out_features=0, bias=False)
          (1): ReLU(inplace=True)
          (2): Linear(in_features=0, out_features=2, bias=False)
          (3): Sigmoid()
        )
      )
      (relu): ReLU(inplace=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (attention): MultiHeadAttention(
    (query): Linear(in_features=32, out_features=32, bias=True)
    (key): Linear(in_features=32, out_features=32, bias=True)
    (value): Linear(in_features=32, out_features=32, bias=True)
    (out): Linear(in_features=32, out_features=32, bias=True)
  )
  (out_layer): Sequential(
    (0): Linear(in_features=64, out_features=32, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=32, out_features=32, bias=True)
  )
)

'''