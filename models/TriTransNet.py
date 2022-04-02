import torch
import torch.nn as nn
import torchvision.models as models
from .ResNet import ResNet50
from network.vit_seg_modeling import VisionTransformer as ViT_seg
from network.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import numpy as np
from options import config


class CA_Enhance(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA_Enhance, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes // 2, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, depth):
        x = torch.cat((rgb, depth), dim=1)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        depth = depth.mul(self.sigmoid(out))
        return depth

class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class CA_SA_Enhance(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA_SA_Enhance, self).__init__()

        self.self_CA_Enhance = CA_Enhance(in_planes)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, rgb, depth):
        x_d = self.self_CA_Enhance(rgb, depth)
        sa = self.self_SA_Enhance(x_d)
        depth_enhance = depth.mul(sa)
        return depth_enhance


class TriTransNet(nn.Module):
    def __init__(self, channel=32):
        super(TriTransNet, self).__init__()

        self.resnet = ResNet50('rgb')
        self.resnet_depth = ResNet50('rgbd')

        self.config = config
        config_vit = CONFIGS_ViT_seg[config.vit_name]

        self.net = ViT_seg(config_vit, img_size=config.img_size).cuda()
        self.net.load_from(weights=np.load(config_vit.pretrained_path))

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.CA_SA_Enhance_0 = CA_SA_Enhance(128)
        self.CA_SA_Enhance_1 = CA_SA_Enhance(512)
        self.CA_SA_Enhance_2 = CA_SA_Enhance(1024)
        self.CA_SA_Enhance_3 = CA_SA_Enhance(2048)
        self.CA_SA_Enhance_4 = CA_SA_Enhance(4096)

        self.T_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.T_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.T_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.up_conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.c_conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.up_conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.c_conv4_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.up_conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.c_conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.deconv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2
        )
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2
        )
        self.deconv_5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2
        )

        self.deconv_layer_3_2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.deconv_layer_4_2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.deconv_layer_5_2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.deconv_layer_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2
        )
        self.deconv_layer_4_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2
        )
        self.deconv_layer_5_1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2
        )

        self.predict_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
            )
        self.predict_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )
        self.predict_layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )
        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)

        x_d = self.CA_SA_Enhance_0(x, x_depth)
        x = x + x_d

        x1 = self.resnet.layer1(x)
        x1_depth = self.resnet_depth.layer1(x_depth)

        x1_d = self.CA_SA_Enhance_1(x1, x1_depth)
        x1 = x1 + x1_d

        x2 = self.resnet.layer2(x1)
        x2_depth = self.resnet_depth.layer2(x1_depth)

        x2_d = self.CA_SA_Enhance_2(x2, x2_depth)
        x2 = x2 + x2_d

        x3_1 = self.resnet.layer3_1(x2)
        x3_1_depth = self.resnet_depth.layer3_1(x2_depth)

        x3_d = self.CA_SA_Enhance_3(x3_1, x3_1_depth)
        x3_1 = x3_1 + x3_d

        x4_1 = self.resnet.layer4_1(x3_1)
        x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)

        x4_d = self.CA_SA_Enhance_4(x4_1, x4_1_depth)
        x4_1 = x4_1 + x4_d


        x2_t = self.T_layer2(x2)
        x3_1_t = self.T_layer3(x3_1)
        x4_1_t = self.T_layer4(x4_1)

        x3_1_u = self.up_conv3_1(self.upsample2(x3_1_t))
        c3_1_u = torch.cat((x3_1_u, x2_t), dim=1)
        x3_1_u = self.c_conv3_1(c3_1_u)

        x4_1_u_0 = self.up_conv4_1(self.upsample2(x4_1_t))
        c4_1_u = torch.cat((x4_1_u_0, x3_1_t), dim=1)
        x4_1_u_1 = self.c_conv4_1(c4_1_u)
        x4_1_u_2 = self.up_conv4_2(self.upsample2(x4_1_u_1))
        c4_2_u = torch.cat((x4_1_u_2, x2_t), dim=1)
        x4_1_u = self.c_conv4_2(c4_2_u)

        h3_c = self.net(x2_t)
        h4_c = self.net(x3_1_u)
        h5_c = self.net(x4_1_u)
        feature_3, feature_4, feature_5 = x2_t, x3_1_u, x4_1_u

        h3_c_c = torch.cat((h3_c, feature_3), 1)
        h3_f = self.deconv_3(h3_c_c)

        h4_c_c = torch.cat((h4_c, feature_4), 1)
        h4_f = self.deconv_4(h4_c_c)

        h5_c_c = torch.cat((h5_c, feature_5), 1)
        h5_f = self.deconv_5(h5_c_c)

        h_3_2c = torch.cat((h3_f, x1), 1)
        h_3_2f = self.deconv_layer_3_2(h_3_2c)

        h_3_1c = torch.cat((h_3_2f, x), 1)
        h_3_1f = self.deconv_layer_3_1(h_3_1c)
        y1 = self.predict_layer_3(h_3_1f)

        h_4_2c = torch.cat((h4_f, x1), 1)
        h_4_2f = self.deconv_layer_4_2(h_4_2c)

        h_4_1c = torch.cat((h_4_2f, x), 1)
        h_4_1f = self.deconv_layer_4_1(h_4_1c)
        y2 = self.predict_layer_4(h_4_1f)

        h_5_2c = torch.cat((h5_f, x1), 1)
        h_5_2f = self.deconv_layer_5_2(h_5_2c)

        h_5_1c = torch.cat((h_5_2f, x), 1)
        h_5_1f = self.deconv_layer_5_1(h_5_1c)
        y3 = self.predict_layer_5(h_5_1f)

        y = y1 + y2 + y3
        return y, y1, y2, y3

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k == 'conv1.weight':
                all_params[k] = torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)
