# ECG Delineation with ConvBiLSTM model

## Data
Lobachevsky University Electrocardiography Database (LUDB)

## Preprocessing
#### Denoising
Using Discrete Wavelet Transform with biorthogonal3.3 mother function, decomposition level 7, soft SURE threshold.
#### Normalization
Lower bound: 0 and Upper bound: 1
#### Segmentation
Each beat, from P-wave to P-wave.