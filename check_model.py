# import torch
#
# # Load the pre-trained model
# model = torch.load('pretrained/net.pkl', map_location=torch.device('cpu'))
#
# # Print the model architecture
# print(model)
import torch
import pandas as pd

# Load the training data
training_data = torch.load('training_data.pt', weights_only=True)
signal = training_data['signal'].numpy().flatten()
doa = training_data['doa'].numpy().flatten()
ref_sp = training_data['ref_sp'].numpy().flatten()

# Ensure all arrays have the same length
min_length = min(len(signal), len(doa), len(ref_sp))
training_df = pd.DataFrame({
    'signal': signal[:min_length],
    'doa': doa[:min_length],
    'ref_sp': ref_sp[:min_length]
})
training_df.to_csv('training_data.csv', index=False)
print("Training data saved to training_data.csv")

# Load the validation data
validation_data = torch.load('validation_data.pt', weights_only=True)
noisy_signals = validation_data['noisy_signals'].numpy().flatten()
signal = validation_data['signal'].numpy().flatten()
doa = validation_data['doa'].numpy().flatten()
ref_sp = validation_data['ref_sp'].numpy().flatten()

# Ensure all arrays have the same length
min_length = min(len(noisy_signals), len(signal), len(doa), len(ref_sp))
validation_df = pd.DataFrame({
    'noisy_signals': noisy_signals[:min_length],
    'signal': signal[:min_length],
    'doa': doa[:min_length],
    'ref_sp': ref_sp[:min_length]
})
validation_df.to_csv('validation_data.csv', index=False)
print("Validation data saved to validation_data.csv")