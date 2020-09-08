def train_loop(loader, encoder, decoder, caption_optimizer, split_no, epoch, total_step):
    encoder.train()
    decoder.train()
    for i, (features, geometry, captions, lengths) in enumerate(loader):
        # Set mini-batch dataset
        # features = torch.tensor(features)
        features = features.to(device)
        # features = torch.tensor(features)
        # print("type features", type(features))
        geometry = geometry.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        
        caption_optimizer.zero_grad()
        # print("targets", targets)
        # Forward, backward and optimize
        feature = encoder(geometry, features)
        
        # lengths = torch.tensor(lengths).view(-1, 1) uncomment for attention!!!
        # print("shape lengthes", lengths.shape)
        outputs = decoder(feature, captions, lengths)
        # print("outputs", outputs)
        loss = criterion(outputs, targets)
        # scheduler.step(loss)
        
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        caption_optimizer.step()  #!!! figure out whether we should leave that 

        name = "training_loss_" + str(split_no + 1)
        writer.add_scalar(name, loss.item(), epoch)

        # writer.add_scalar("training_loss", loss.item(), epoch)
        log_file_tensor.write(str(loss.item()) + "\n")
        log_file_tensor.flush()
        handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
        fb_mem_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        mem = fb_mem_info.used >> 20
        print('GPU memory usage: ', mem)
        writer.add_scalar('val/gpu_memory', mem, epoch)
        # Print log info
        if i % log_step == 0:
            result = "Split [{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}".format(
                split_no, epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())
            )
            print(result)
            log_file.write(result + "\n")
            log_file.flush()

        # loss is a real crossentropy loss
        #
        # Save the model checkpoints
        if (i + 1) % save_step == 0:
            torch.save(
                decoder.state_dict(),
                os.path.join(
                    model_path, "decoder-{}-{}-{}.ckpt".format(split_no, epoch + 1, i + 1)
                ),
            )
            torch.save(
                encoder.state_dict(),
                os.path.join(
                    model_path, "encoder-{}-{}-{}.ckpt".format(split_no, epoch + 1, i + 1)
                ),
            )
    log_file_tensor.write("\n")
    log_file_tensor.flush()


def train_loop_mask(loader, encoder, decoder, caption_optimizer, split_no, epoch, total_step):
    encoder.train()
    decoder.train()
    for i, (features, geometry, masks, captions, lengths) in enumerate(loader):
        # Set mini-batch dataset

        features = features.to(device)
        geometry = geometry.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        
        caption_optimizer.zero_grad()
        # Forward, backward and optimize
        feature = encoder(geometry, features, masks)
        outputs = decoder(feature, captions, lengths)

        loss = criterion(outputs, targets)
        # scheduler.step(loss)
        
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        caption_optimizer.step()  #!!! figure out whether we should leave that 

        name = "training_loss_" + str(split_no + 1)
        writer.add_scalar(name, loss.item(), epoch)

        # writer.add_scalar("training_loss", loss.item(), epoch)
        log_file_tensor.write(str(loss.item()) + "\n")
        log_file_tensor.flush()
        handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
        fb_mem_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        mem = fb_mem_info.used >> 20
        print('GPU memory usage: ', mem)
        writer.add_scalar('val/gpu_memory', mem, epoch)
        # Print log info
        if i % log_step == 0:
            result = "Split [{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}".format(
                split_no, epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item())
            )
            print(result)
            log_file.write(result + "\n")
            log_file.flush()

        # loss is a real crossentropy loss
        #
        # Save the model checkpoints
        if (i + 1) % save_step == 0:
            torch.save(
                decoder.state_dict(),
                os.path.join(
                    model_path, "decoder-{}-{}-{}.ckpt".format(split_no, epoch + 1, i + 1)
                ),
            )
            torch.save(
                encoder.state_dict(),
                os.path.join(
                    model_path, "encoder-{}-{}-{}.ckpt".format(split_no, epoch + 1, i + 1)
                ),
            )
    log_file_tensor.write("\n")
    log_file_tensor.flush()