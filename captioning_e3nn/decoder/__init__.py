from decoder.decoder import DecoderRNN, My_attention, MyDecoderWithAttention


decoder_dict = {
    'lstm': DecoderRNN,
    'lstm_attention': MyDecoderWithAttention
}


