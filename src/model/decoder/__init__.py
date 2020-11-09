from src.model.decoder.decoder import DecoderRNN, My_attention, MyDecoderWithAttention
from src.model.decoder.decoder_vis import MyDecoderWithAttention_Vis

decoder_dict = {
    'lstm': DecoderRNN,
    'lstm_attention': MyDecoderWithAttention,
    'lstm_attention_vis': MyDecoderWithAttention_Vis
}


