from functools import partial
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from se3cnn.non_linearities.rescaled_act import Softplus
from se3cnn.point.kernel import Kernel
from se3cnn.point.operations import NeighborsConvolution
from se3cnn.point.radial import CosineBasisModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_Length = 245


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers.
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = MAX_Length
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decodes shapes feature vectors and generates SMILES."""
        # print("captions shape initial", captions.shape)
        embeddings = self.embed(
            captions
        )  # shape [batch_size, padded_length, embed_size]
        # print("shape emb", embeddings.shape)
        # print("features emb", features.shape)
        embeddings = torch.cat(
            (features.unsqueeze(1), embeddings), 1
        )  # shape [batch_size, padded_length + 1, embed_size]
        # print("shape embeddings", embeddings.shape)
        packed = pack_padded_sequence(
            embeddings, lengths, batch_first=True
        )  # shape [packed_length, embed_size]
        # print("packed shape", packed.data.shape)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])  # shape [packed_length, vocab_size]
        # print("shape outputs", outputs.shape)
        return outputs

    def sample(self, features, states=None):
        """Samples SMILES tockens for given  features (Greedy search).
        """

        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


    def sample_prob(self, features, states=None):
        """Samples SMILES tockens for given shape features (probalistic picking)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            # print("outputs shape,", outputs.shape)
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)

                # Probabilistic sample tokens
                if probs.is_cuda:
                    probs_np = probs.data.cpu().numpy()
                else:
                    probs_np = probs.data.numpy()
                    # print("shape probs_np", probs_np.shape)

                rand_num = np.random.rand(probs_np.shape[0])
                # print("shape rand_num", rand_num.shape)
                iter_sum = np.zeros((probs_np.shape[0],))
                tokens = np.zeros(probs_np.shape[0], dtype=np.int)
            
                for i in range(probs_np.shape[1]):
                    c_element = probs_np[:, i]
                    iter_sum += c_element
                    valid_token = rand_num < iter_sum
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))
                    tokens[update_indecies] = i

                # put back on the GPU.
                if probs.is_cuda:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
                else:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)))
            # print("shape predicted", predicted.shape)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        # print("shape sampled_ids", len(sampled_ids))
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids




class My_attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(My_attention, self).__init__()
        self.encoder_att = nn.Linear(
            encoder_dim, attention_dim
        )  # linear layer to transform encoded pocket
        self.decoder_att = nn.Linear(
            decoder_dim, attention_dim
        )  # linear layer to transform decoder's output
        self.full_att = nn.Linear(
            attention_dim, 1
        )  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
       

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(
            encoder_out
        )  # (batch_size, num_pixels, attention_dim) or (batch_size, attention_dim) - check again!
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(
            2
        )  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(
            dim=1
        )  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


# this is under construction (sampling part)


class MyDecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        vocab_path,
        encoder_dim=512,
        dropout=0.5,
        device=DEVICE,
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(MyDecoderWithAttention, self).__init__()
        self.device = device
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seg_length = MAX_Length
        self.attention = My_attention(
            encoder_dim, decoder_dim, attention_dim
        )  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim, bias=True
        )  # decoding LSTMCell
        self.init_h = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(
            encoder_dim, decoder_dim
        )  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(
            decoder_dim, encoder_dim
        )  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(
            decoder_dim, vocab_size
        )  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        self.vocab_path = vocab_path
        with open(self.vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
 
    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # mean_encoder_out = encoder_out.mean(dim=1)
        mean_encoder_out = encoder_out
        # print("shape mean enc out", mean_encoder_out.shape)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths, device=DEVICE):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        # num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below

        # TODO - adjust list of lengthes to tensor
        caption_lengths = torch.tensor(caption_lengths).view(-1, 1) #uncomment for attention!!!
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(
            dim=0, descending=True
        )
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(
            encoded_captions
        )  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)  ??

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths).tolist()  # maybe just caption_lengths
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(
            device
        )
        # alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding

        # we have already initialised first hidden state. At the every stage (for t in range)
        # we would update current hidden states (we have a batch of them) and add a prediction
        # at the all vectors who are more in length than the current value. So, we put a predicted score at the decoded
        # sequence at the t-th place (we put an array of vocab length there)
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            # print("shape encoder_out att", encoder_out[:batch_size_t].shape)
            # print("shape h att", h[:batch_size_t].shape)
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            gate = self.sigmoid(
                self.f_beta(h[:batch_size_t])
            )  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            # print("real emb shape", embeddings.shape)
            # print("shape emb help", embeddings[:batch_size_t, t, :].shape)
            # print("shape aw help", attention_weighted_encoding.shape)
            h, c = self.decode_step(    
                torch.cat(
                    [embeddings[:batch_size_t, t, :], attention_weighted_encoding],
                    dim=1,
                ),
                (h[:batch_size_t], c[:batch_size_t]),
            )  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

            # alphas[:batch_size_t, t, :] = alpha
        scores = pack_padded_sequence(
            predictions, decode_lengths, batch_first=True
        )  #!!! shape [padded_length, voc] do that like with simple version
        return scores.data  # , encoded_captions, decode_lengths, alphas, sort_ind

    def sample_prob_1(self, features, states=None, device=DEVICE):
        """Samples SMILES tockens for given  features (Greedy search).
        """
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
        h, c = self.init_hidden_state(features)
        # features = features.long()
        sampled_ids = []
        
        features = features.squeeze(0)
        features = features.unsqueeze(1)
        print("shape feat", features.shape)
        inputs = features.long()
        for i in range(self.max_seg_length):
            embeddings = self.embedding(inputs).squeeze(
                1
            )  # (s, embed_dim)  ?why should we alos use it???
            print("embed shape", embeddings.shape)
            print("h shape", h.shape)
            features = features.permute(1,0)
            print("feat shape before att", features.shape)
            awe, alpha = self.attention(
                features, h
            )  # (s, encoder_dim), (s, num_pixels) - we give to Attention the same features
            print("awe shape", awe.shape)

            # alpha = alpha.view(
            #     -1, enc_image_size, enc_image_size
            # )  # (s, enc_image_size, enc_image_size)

            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            # s is a batch_size_t since we do not have a batch of images, we have just one image
            # and we want to find several words.
            h, c = self.decode_step(
                torch.cat([embeddings, awe], dim=1), (h, c)
            )  # (s, decoder_dim)

            scores = self.fc(h)  # (s, vocab_size)
            # predicted = scores.max(1)[1]  # check that
            #  k_prev_words = (
            #     predicted  # now we have predicted word and give it to the next lastm
            # )
            if i == 0:
                predicted = scores.max(1)[1]
            else:
                probs = F.softmax(scores, dim=1)

                # Probabilistic sample tokens
                if probs.is_cuda:
                    probs_np = probs.data.cpu().numpy()
                else:
                    probs_np = probs.data.numpy()
                    # print("shape probs_np", probs_np.shape)

                rand_num = np.random.rand(probs_np.shape[0])
                # print("shape rand_num", rand_num.shape)
                iter_sum = np.zeros((probs_np.shape[0],))
                tokens = np.zeros(probs_np.shape[0], dtype=np.int)
            
                for i in range(probs_np.shape[1]):
                    c_element = probs_np[:, i]
                    iter_sum += c_element
                    valid_token = rand_num < iter_sum
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))
                    tokens[update_indecies] = i

                # put back on the GPU.
                if probs.is_cuda:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
                else:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)))
            # print("shape predicted", predicted.shape)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        # print("shape sampled_ids", len(sampled_ids))
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


 
    

    def sample_prob(self, features, states=None):
        """Samples SMILES tockens for given  features (Greedy search)."""
        k = 1
        k_prev_words = torch.LongTensor([[self.vocab.word2idx['<start>']]] * k).to(self.device) 
        h, c = self.init_hidden_state(features)

        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)  ?why should we alos use it???

            awe, alpha = self.attention(features, h)  # (s, encoder_dim), (s, num_pixels) - we give to Attention the same features

            # alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
            
            gate = self.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            #s is a batch_size_t since we do not have a batch of images, we have just one image
            # and we want to find several words. 
            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.fc(h)  # (s, vocab_size)
            predicted = scores.max(1)[1] #check that
            k_prev_words = predicted #now we have predicted word and give it to the next lastm
            # scores = F.log_softmax(scores, dim=1)
            # h = h[i] #we have the only word - no sense to have index of h (h dim - [1, decoder_dim])
            # c = c[prev_word_inds[incomplete_inds]]
            # encoder_out = encoder_out[prev_word_inds[incomplete_inds]] - we give to Attention the same features
            sampled_ids.append(predicted)
        sampled_ids = torch.stack(sampled_ids, 1) 
        return sampled_ids

    def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
        """
        Reads an image and captions it with beam search.

        :param encoder: encoder model
        :param decoder: decoder model
        :param image_path: path to image
        :param word_map: word map
        :param beam_size: number of sequences to consider at each decode-step
        :return: caption, weights for visualization
        """

        k = beam_size
        vocab_size = len(word_map)

        # Read image and process
        img = imread(image_path)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = imresize(img, (256, 256))
        img = img.transpose(2, 0, 1)
        img = img / 255.
        img = torch.FloatTensor(img).to(device)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([normalize])
        image = transform(img)  # (3, 256, 256)

        # Encode
        image = image.unsqueeze(0)  # (1, 3, 256, 256)
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
        
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)  ?why should we alos use it???

            awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
            
            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe
            #s is a batch_size_t since we do not have a batch of images, we have just one image
            # and we want to find several words. 
            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)
            #!!!!!!!!!!!!!!!!!!!# choose the highest score here
            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                            next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]

        return seq, alphas