## pointplanenet_resnet authors: Daniil Emtsev and Dusan Svilarkovic
# from encoder.encoder_resnet import ResnetPointnet, Encoder_Resnet_after_se3ACN, Encoder_Resnet_feat_geom_se3ACN, Encoder_Resnet_geom_se3ACN,  Encoder_Resnet

from encoder.se3cnn import Encoder_se3ACN, Encoder_se3ACN_Fast
# from encoder.e3nn import Network, ResNetwork, OutputScalarNetwork, OutputMLPNetwork

encoder_dict = {
    # 'resnet': Encoder_Resnet,
    'se3cnn': Encoder_se3ACN,
}
#     'se3cnnfast': Encoder_se3ACN_Fast,
#     'se3cnn_geo_resnet': Encoder_Resnet_geom_se3ACN,
#     'se3cnn_geo_feat_resnet': Encoder_Resnet_feat_geom_se3ACN,
#     'se3cnn_resnet_after_se3cnn': Encoder_Resnet_after_se3ACN,
#     'network1': Network,
#     'resnetnetwork1': ResNetwork,
#     'OutputScalarNetwork': OutputScalarNetwork,
#     'OutputMLPNetwork': OutputMLPNetwork
# }