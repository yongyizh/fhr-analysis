"""PANNet -- a frozen PANNs Cnn14 (AudioSet) under a small trainable head.

Predicts per-frame fetal beat activity from the raw 4 kHz abdomen fiber waveforms. Same Task
seam as funet, ssnet and tslnet, so it shares common's training loop, checkpointing, config
handling and Optuna search.

See ``pann.model`` for the two ideas that make an AudioSet clip classifier work as a dense
per-frame beat model: feeding 4 kHz audio through a 32 kHz filterbank on purpose, and pooling
frequency without ever pooling time.
"""
