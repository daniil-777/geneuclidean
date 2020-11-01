from model.encoder.encoder_resnet import ResnetPointnet, Encoder_Resnet_after_se3ACN, Encoder_Resnet_feat_geom_se3ACN, Encoder_Resnet_geom_se3ACN,  Encoder_Resnet
from model.encoder.pointnet_e3nn import PointNet_Geo_AllNetwork
# from encoder.se3cnn import Encoder_se3ACN, Encoder_se3ACN_Fast
from model.encoder.e3nn import Network, ResNetwork, OutputScalarNetwork, OutputMLPNetwork, Bio_Network
from model.encoder.e3nn_vis import Network_Vis
from model.encoder.binding_e3nn import Binding_Network
from model.encoder.e3nn_res import ResNetwork
from model.encoder.pointnet_e3nn import PointNetAllNetwork
from model.encoder.e3nn_att import AttentionE3nn
from model.encoder.bio_e3nn import Bio_All_Network, Bio_Local_Network

encoder_dict = {
    'network1': Network,
    'network1_vis': Network_Vis,
    'OutputScalarNetwork': OutputScalarNetwork,
    'OutputMLPNetwork': OutputMLPNetwork,
    'binding_e3nn': Binding_Network,
    'e3nn_res': ResNetwork,
    'pointnetall': PointNetAllNetwork,
    'att_e3nn': AttentionE3nn,
    'se3cnn_resnet_after_se3cnn': Encoder_Resnet_after_se3ACN,
    'se3cnn_geo_feat_resnet': Encoder_Resnet_feat_geom_se3ACN,
    'pointnet_geo': PointNet_Geo_AllNetwork,
    'bio_net': Bio_All_Network,
    'bio_local_net': Bio_Local_Network
}