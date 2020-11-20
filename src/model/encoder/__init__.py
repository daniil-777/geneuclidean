from src.model.encoder.encoder_resnet import ResnetPointnet, Encoder_Resnet_after_se3ACN, Encoder_Resnet_feat_geom_se3ACN, Encoder_Resnet_geom_se3ACN,  Encoder_Resnet
from src.model.encoder.pointnet_e3nn import PointNet_Geo_AllNetwork
# from encoder.se3cnn import Encoder_se3ACN, Encoder_se3ACN_Fast
from src.model.encoder.e3nn import Network, ResNetwork, OutputScalarNetwork, OutputMLPNetwork, Bio_Network
from src.model.encoder.e3nn_vis import Network_Vis
from src.model.encoder.binding_e3nn import Binding_Network
from src.model.encoder.e3nn_res import ResNetwork
from src.model.encoder.pointnet_e3nn import PointNetAllNetwork
from src.model.encoder.e3nn_att import AttentionE3nn
from src.model.encoder.bio_e3nn import Bio_All_Network, Bio_All_Network_no_batch, Bio_Vis_All_Network, Bio_Local_Network, ResNet_Bio_ALL_Network, ResNet_Bio_Local_Network, Concat_Bio_Local_Network
from src.model.encoder.bio_e3nn_res import ResNet_Out_Local_Network


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
    'bio_net_no_bn': Bio_All_Network_no_batch,
    'bio_vis_net': Bio_Vis_All_Network,
    'bio_local_net': Bio_Local_Network,
    'resnet_bio_net': ResNet_Bio_ALL_Network,
    'resnet_bio_local_net': ResNet_Bio_Local_Network,
    'concat_bio_local_net': Concat_Bio_Local_Network,
    'res_out_local_net': ResNet_Out_Local_Network
}