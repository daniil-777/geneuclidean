## pointplanenet_resnet authors: Daniil Emtsev and Dusan Svilarkovic
from decoder.decoder import DecoderRNN, My_attention, MyDecoderWithAttention


decoder_dict = {
    'lstm': DecoderRNN,
    'lstm_attention': My_attention
}


