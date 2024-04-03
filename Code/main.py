# The Complete  code will be upload in the soon!!!



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
            padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Interaction(nn.Module):
    def __init__(self, in_dim, dim):
        super(Interaction_low, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1)

        self.fuse = ACM(dim)

        self.unet = RefUnet(dim, dim // 2)

        self.cov3d1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
            nn.ReLU())

    def weighting(self, x5, seeds):
        B, C, H5, W5 = x5.size()
        weighted_maps = F.conv2d(x5, weight=seeds)
        weighted_maps = weighted_maps.mean(1).view(B, -1)
        return weighted_maps

    def forward(self, seed, high, low):
        merge = []
        # 0 Low-level feature dimension reduction
        low  = self.conv1(low)
        # 2 Low-level endowed semantics
        low_cormap = self.weighting(F.normalize(low, dim=1), seed)
        low_weighted = low * low_cormap + low

        # 3 high-level features prevent dilution
        high_cormap = self.weighting(F.normalize(high, dim=1), seed)
        high_weighted = high * high_cormap + high

        # 4 Filtered noise
        fuse, cc = self.fuse(high_weighted, low_weighted)
        merge.append(fuse)

        # 5 Multilevel, multi-scale, diversity
        f_m = self.unet(cc)
        merge.append(f_m)

        en_merge = []
        for m in merge:
            en_merge.append(m.unsqueeze(2))

        com = torch.cat(en_merge, dim=2)

        com = self.cov3d1(com).squeeze(2)

        return com


class GWENet(nn.Module):
    def __init__(self, cfg, model_name='BOSS-V'):
        super(BOSS, self).__init__()
        self.cfg = cfg
        self.model_name = model_name

        if self.model_name == 'BOSS-V':
            self.encoder = VGG()
            self.pools1 = nn.MaxPool2d(kernel_size=2, stride=2)

        else:
            print("UNDEFINED BACKBONE NAME.")

        self.conv5_1 = conv3x3(512, 64)
        self.bn5_1 = nn.BatchNorm2d(64)

        self.conv_att1 = convformer(6, [64, 64, 64, 64, 64, 64])

        self.fuse1 = co_att(64)

        self.glof = AttLayer(64)

        self.inter1 = Interaction(512, 64)
        self.inter2 = Interaction(512, 64)
        self.inter3 = Interaction(256, 64)
        self.inter4 = Interaction(128, 64)
        self.inter5 = Interaction(64, 64)

        self.cls1 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.cls3 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.cls4 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.cls5 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(5 , 1, kernel_size=1)


    def forward(self, x, shape=None):

        input = x

        # backbone
        features = self.encoder(x)
        x1 = features[0]    # 352
        x2 = features[1]    # 176
        x3 = features[2]    # 88
        x4 = features[3]    # 44
        x5 = features[4]    # 22

        # att downsample
        x6 = F.relu(self.bn5_1(self.conv5_1(self.pools1(x5))))    # 11
        x6 = self.conv_att1(x6)

        # deep sematic with location
        outputs = []
        df1 = self.fuse1(x6)

        # global weight
        weight = self.glof(x6)

        # decoder
        dx5 = F.interpolate(df1, size=x5.size()[2:], mode='bilinear', align_corners=True)
        dx5 = self.inter1(weight,dx5,x5)
        outputs.insert(0, dx5)
        dx4 = F.interpolate(dx5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        dx4 = self.inter2(weight,dx4,x4)
        outputs.insert(0, dx4)
        dx3 = F.interpolate(dx4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        dx3 = self.inter3(weight, dx3, x3)
        outputs.insert(0, dx3)
        dx2 = F.interpolate(dx3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        dx2 = self.inter4(weight, dx2, x2)
        outputs.insert(0, dx2)
        dx1 = F.interpolate(dx2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        dx1 = self.inter5(weight, dx1, x1)
        outputs.insert(0, dx1)

        if shape is None:
            shape = input.size()[2:]

        pred1 = F.interpolate(self.cls1(outputs[0]), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.cls2(outputs[1]), size=shape, mode='bilinear')
        pred3 = F.interpolate(self.cls3(outputs[2]), size=shape, mode='bilinear')
        pred4 = F.interpolate(self.cls4(outputs[3]), size=shape, mode='bilinear')
        pred5 = F.interpolate(self.cls5(outputs[4]), size=shape, mode='bilinear')

        outpred = [pred1,pred2,pred3,pred4,pred5]
        pred0 = self.out_conv(torch.concat(outpred, dim=1))

        return pred0,pred1,pred2,pred3,pred4,pred5 



if __name__ == '__main__':
    import torch

    with torch.cuda.device(0):
        net = BOSS(cfg=None)
        net.eval()
        image = torch.randn(8, 3, 352, 352)
        out = net(image)
        print(out[0].shape)
