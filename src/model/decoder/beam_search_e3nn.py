    def sample_beam_search(self, features, states=None):
        """
        Reads an image and captions it with beam search.

        :param encoder: encoder model
        :param decoder: decoder model
        :param image_path: path to image
        :param word_map: word map
        :param beam_size: number of sequences to consider at each decode-step
        :return: caption, weights for visualization
        """
        print("feat shape init", features.shape)
        k = self.beam_size
        vocab_size = len(self.vocab)

        # # We'll treat the problem as having a batch size of k
        shape_1 = features.shape[0]
        shape_2 = features.shape[1]
        # inputs = features.unsqueeze(1)
        inputs  = features.expand(k, shape_2) ##? check tomorrow!!!
        inputs = inputs.unsqueeze(1)
        # encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.vocab.word2idx['<start>']]] * k).to(self.device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(self.device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        # seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        # complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
    
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            if step == 1:
                h, states = self.lstm(inputs, states)
            else:
               # h, states = self.lstm(inputs, (h, states))
                h, states = self.lstm(inputs, states)
           # h, states = self.lstm(inputs,  states)
            print("states usual")
            scores = self.linear(h.squeeze(1))
            scores = F.softmax(scores, dim=1)
           # scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
           # print("scores", scores)
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
              top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
              top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
           # print("pred next word", top_k_scores)
            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            # seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
            #                     dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                            next_word != self.vocab.word2idx['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                # complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
           # print("h", h)
            h = h[prev_word_inds[incomplete_inds]]
           # print("incomp ind", incomplete_inds)
           # print("states first shape", states[0].shape)
           # print("ind for states", prev_word_inds[incomplete_inds])
            states = (states[0][:, prev_word_inds[incomplete_inds]], states[1][:, prev_word_inds[incomplete_inds]])
           # print("shape states in end", states[0].shape)
           # print("states in the end")
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            inputs = self.embed(k_prev_words)
            # Break if things have been going on too long
            if step > MAX_Length:
                break
            step += 1
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        print(complete_seqs)
        return complete_seqs