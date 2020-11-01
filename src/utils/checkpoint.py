import torch

def save_checkpoint_feature(checkpoint_path, idx_max_length, max_length, idx_write):

  state = {'idx_max_length': idx_sampling,
           'max_length': max_length,
           'idx_write': idx_sample_regime_start, 
          }

  torch.save(state, checkpoint_path)