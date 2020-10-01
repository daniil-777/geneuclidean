# from encoder.encoder_resnet import ResnetPointnet, Encoder_Resnet_after_se3ACN, Encoder_Resnet_feat_geom_se3ACN, Encoder_Resnet_geom_se3ACN,  Encoder_Resnet

# from encoder.se3cnn import Encoder_se3ACN, Encoder_se3ACN_Fast
from encoder.e3nn import Network, ResNetwork, OutputScalarNetwork, OutputMLPNetwork
from encoder.e3nn_vis import Network_Vis
from encoder.binding_e3nn import Binding_Network

encoder_dict = {
    'network1': Network,
    'network1_vis': Network_Vis,
    'resnetnetwork1': ResNetwork,
    'OutputScalarNetwork': OutputScalarNetwork,
    'OutputMLPNetwork': OutputMLPNetwork,
    'binding_e3nn': Binding_Network
}