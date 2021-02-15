# Naive imlementation of a SimpleRNN in numpy

import numpy as np

timesteps = 100
input_features = 32
output_features = 64

# Random inputs: standard normal distrib with SD=0.1
st_dev = 0.1
inputs = np.random.randn(timesteps, input_features) * st_dev
state_t = np.zeros((output_features,))

W = np.random.randn(output_features, input_features) * st_dev
U = np.random.randn(output_features, output_features) * st_dev
b = np.random.randn(output_features,) * st_dev

output_sequence = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    output_sequence.append(output_t)
    state_t = output_t

final_output_sequence = np.concatenate(output_sequence, axis=0)
final_output_sequence = final_output_sequence.reshape(timesteps, output_features)
print(final_output_sequence.shape)
