import os
import yaml
from torchvision import transforms
from encoder import encoder_dict
from decoder import decoder_dict
from torch.utils.data import DataLoader
import data_loader
from data_loader import collate_fn, collate_fn_masks

# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    encoder, decoder = get_model_captioning(
        cfg, device=device)
    return encoder, decoder


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


def get_model_captioning(cfg, device=None, **kwargs):
    r''' Returns the model instance.

    Args:
        cfg (yaml object): the config file
        device (PyTorch device): the PyTorch device
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    decoder = decoder_dict[decoder](
       **decoder_kwargs
    ).to(device).double()

    encoder = encoder_dict[encoder](
        **encoder_kwargs
    ).to(device).double()

    # model = models.PCGN(decoder, encoder)
    # model = model.to(device)
    return encoder, decoder


def get_trainer(model, optimizer, cfg, device, **kwargs):
    r''' Returns the trainer instance.

    Args:
        model (nn.Module): PSGN model
        optimizer (PyTorch optimizer): The optimizer that should be used
        cfg (yaml object): the config file
        device (PyTorch device): the PyTorch device
    '''
    input_type = cfg['data']['input_type']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')

    trainer = training.Trainer(
        model, optimizer, device=device, input_type=input_type,
        vis_dir=vis_dir
    )
    return trainer

def get_loader(cfg, feat_train, batch_size, num_workers):
    if(cfg['preprocessing']['mask'] == True):
        loader = DataLoader(feat_train, batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn_masks,)
    else:
        loader = DataLoader(feat_train, batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn,) 
    return loader

def get_train_loop(cfg, loader_train, encoder, decoder,caption_optimizer, split_no, epoch, total_step):
    if(cfg['preprocessing']['mask'] == True):
        train_loop_mask(loader_train, encoder, decoder,caption_optimizer, split_no, epoch, total_step)
    else:
        train_loop(loader_train, encoder, decoder,caption_optimizer, split_no, epoch, total_step)

#maybe uncomment later
# def get_collate_fn(cfg):
#     if(cfg['preprocessing']['collate_fn'] == 'masks'):
#         collate = data_loader.collate_fn()
#     else:
#         collate = data_loader.collate_fn_masks()

def get_generator(model, cfg, device, **kwargs):
    r''' Returns the generator instance.

    Args:
        cfg (yaml object): the config file
        device (PyTorch device): the PyTorch device
    '''
    generator = generation.Generator3D(model, device=device)
    return generator


def get_data_fields(mode, cfg, **kwargs):
    r''' Returns the data fields.

    Args:
        mode (string): The split that is used (train/val/test)
        cfg (yaml object): the config file
    '''
    with_transforms = cfg['data']['with_transforms']
    pointcloud_transform = data.SubsamplePointcloud(
        cfg['data']['pointcloud_target_n'])

    fields = {}
    fields['pointcloud'] = data.PointCloudField(
        cfg['data']['pointcloud_file'], pointcloud_transform,
        with_transforms=with_transforms
    )

    if mode in ('val', 'test'):
        pointcloud_chamfer_file = cfg['data']['pointcloud_chamfer_file']
        if pointcloud_chamfer_file is not None:
            fields['pointcloud_chamfer'] = data.PointCloudField(
                pointcloud_chamfer_file
            )

    return fields
