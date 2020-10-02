import os
import torch


def save_checkpoint(checkpoint_path, start_epoch, encoder, decoder,
                    encoder_best, decoder_best, caption_optimizer, split_no):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'start_epoch': start_epoch,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'caption_optimizer': caption_optimizer.state_dict(),
             'split_no': split_no,
           }

    # filename = os.path.join(checkpoint_path, 'checkpoint_' + model_name + '.pth.tar')
    torch.save(state, checkpoint_path)
   