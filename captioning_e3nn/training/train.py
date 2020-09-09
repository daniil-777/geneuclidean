DATA_PATH = os.path.realpath(os.path.dirname(__file__))
N_SPLITS = 5

# args = str(sys.argv[1])
# print(args)

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')

args = parser.parse_args()


cfg = config.load_config(args.config, 'configurations/config_lab/default.yaml')


# configuration = utils.parse_configuration()

# model params
num_epochs = cfg['model_params']['num_epochs']
batch_size = cfg['model_params']['batch_size']
learning_rate = cfg['model_params']['learning_rate']
num_workers = cfg['model_params']['num_workers']

# training params
protein_dir = cfg['training_params']['image_dir']
caption_path = cfg['training_params']['caption_path']
log_step = cfg['training_params']['log_step']
save_step = cfg['training_params']['save_step']
vocab_path = cfg['preprocessing']['vocab_path']

#output files
savedir = cfg['output_parameters']['savedir']
tesnorboard_path = savedir
model_path = os.path.join(savedir, "models")
log_path = os.path.join(savedir, "logs")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Encoder, Decoder = config.get_model(cfg, device=device)

nparameters_enc = sum(p.numel() for p in Encoder.parameters())
nparameters_dec = sum(p.numel() for p in Decoder.parameters())
print(Encoder)
print('Total number of parameters: %d' % (nparameters_enc + nparameters_dec))

if not os.path.exists(log_path):
    os.makedirs(log_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

    
test_idx_file = open(os.path.join(log_path, "test_idx.txt"), "w")
log_file = open(os.path.join(log_path, "log.txt"), "w")
log_file_tensor = open(os.path.join(log_path, "log_tensor.txt"), "w")
writer = SummaryWriter(tesnorboard_path)



# Load vocabulary wrapper
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)



class Trainer():
    def __init__(cfg):
        # model params
        self.num_epochs = cfg['model_params']['num_epochs']
        self.batch_size = cfg['model_params']['batch_size']
        self.learning_rate = cfg['model_params']['learning_rate']
        self.num_workers = cfg['model_params']['num_workers']

        # training params
        self.protein_dir = cfg['training_params']['image_dir']
        self.caption_path = cfg['training_params']['caption_path']
        self.log_step = cfg['training_params']['log_step']
        self.save_step = cfg['training_params']['save_step']
        self.vocab_path = cfg['preprocessing']['vocab_path']

        #output files
        self.savedir = cfg['output_parameters']['savedir']
        self.tesnorboard_path = savedir
        self.model_path = os.path.join(savedir, "models")
        self.log_path = os.path.join(savedir, "logs")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #log files 
        self.test_idx_file = open(os.path.join(log_path, "test_idx.txt"), "w")
        self.log_file = open(os.path.join(log_path, "log.txt"), "w")
        self.log_file_tensor = open(os.path.join(log_path, "log_tensor.txt"), "w")
        self.writer = SummaryWriter(tesnorboard_path)
        self.Encoder, self.Decoder = config.get_model(cfg, device=device)
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

    def train_loop(self,loader, encoder, decoder, caption_optimizer, split_no, epoch):
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
            feature = encoder(features, geometry, masks)
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

    def train_epochs(self):
        # get indexes of all complexes and "nick names"
        # Load vocabulary wrapper

        featuriser = Pdb_Dataset(cfg, vocab=vocab)
        # data_ids, data_names = utils._get_refined_data()
        files_refined = os.listdir(protein_dir)
        # data_ids = np.array([i for i in range(len(files_refined) - 3)])
        data_ids = np.array([i for i in range(20)])

        #cross validation
        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        my_list = list(kf.split(data_ids))
        test_idx = []
        # output memory usage
        py3nvml.nvmlInit()
        for split_no in range(N_SPLITS):
            train_id, test_id = my_list[split_no]
            train_data = data_ids[train_id]
            test_data = data_ids[test_id]
            with open(os.path.join(savedir, 'test_idx_' + str(split_no)), 'wb') as fp:
                pickle.dump(test_data, fp)
            
            test_idx.append(test_data)
            test_idx_file.write(str(test_data) + "\n")
            test_idx_file.flush()

            feat_train = [featuriser[data] for data in train_data]
            
            loader_train = DataLoader(feat_train, batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=num_workers,
                                        collate_fn=collate_fn_masks,)
            # loader_train = config.get_loader(cfg, feat_train, batch_size, num_workers,)

            total_step = len(loader_train)
            print("total_step", total_step)
            encoder = self.Encoder
            decoder = self.Decoder

            criterion = nn.CrossEntropyLoss()
            # params_encoder = filter(lambda p: p.requires_grad, encoder.parameters())

            caption_params = list(decoder.parameters()) + list(encoder.parameters())
            caption_optimizer = torch.optim.Adam(caption_params, lr=learning_rate)

            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(caption_optimizer, 'min')
            for epoch in range(num_epochs):
                # config.get_train_loop(cfg, loader_train, encoder, decoder,caption_optimizer, split_no, epoch, total_step)
                #if add masks everywhere call just train_loop
                train_loop_mask(cfg, loader_train, encoder, decoder,caption_optimizer, split_no, epoch, total_step, writer, log_file, log_file_tensor)



