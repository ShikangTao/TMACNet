import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):  
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
    
class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = SiLU() 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class CAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CAM, self).__init__()
        # c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1 * 2, c1, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.cv2 = Conv(c1, c1, 1)
        # self.cv3 = Conv(c1, c2, 1, 1)  # act=FReLU(c2)
        # self.m = nn.Sequential(*[CAM(c_, c_, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return x * (nn.Sigmoid()(self.cv1(torch.cat([self.maxpool(x),self.avgpool(x)], 1))) + nn.Sigmoid()(self.cv2(x)))

class Bottleneck(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))    

class C2f(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e) 
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class SPPF(nn.Module):
    # SPP结构，5、9、13最大池化核的最大池化。
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ =  x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MutualAttention(nn.Module):
    def __init__(self, c1, c2, c3):
        super(MutualAttention, self).__init__()
        self.gap = torch.nn.AdaptiveMaxPool2d((c1, c2))
        self.se = SEAttention(channel = c3, reduction= 8)
        # self.conv1 = Conv(c3 * 2, c3, 1)
        # self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # self.zip = nn.Conv2d(c3 * 2, c3, kernel_size=3, stride=1, padding=1)
    def forward(self, x, x2):
        x_se = self.se(x)
        x2_se = self.se(x2)

        x_gap = self.gap(x_se)
        x2_gap = self.gap(x2_se)

        x_weight = torch.sigmoid(x_gap)
        x2_weight = torch.sigmoid(x2_gap)

        x_mutual = x_se * x2_weight
        x2_mutual = x2_se * x_weight

        x_out = x + x_mutual
        x2_out = x2 + x2_mutual

        return x_out, x2_out

        # x_sum_t1 = torch.cat([x, x2],1)
        # x_sum_t1_se = self.se(x_sum_t1)
        # x_out = self.conv1(x_sum_t1 * x_sum_t1_se)

        # x_sum_t2 = torch.cat([x2, x],1)
        # x_sum_t2_se = self.se(x_sum_t2)
        # x2_out = self.conv1(x_sum_t2 * x_sum_t2_se)
        # return x_out, x2_out

class Backbone(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是3, 640, 640
        #-----------------------------------------------#
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)
        
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )

        self.mutual_attention_layer2 = MutualAttention(base_channels * 4, base_channels * 4, base_channels)
        self.mutual_attention_layer3 = MutualAttention(base_channels * 2, base_channels * 2, base_channels * 2)
        self.mutual_attention_layer4 = MutualAttention(base_channels, base_channels, base_channels * 4)
        self.mutual_attention_layer5 = MutualAttention(base_channels // 2, base_channels // 2, base_channels * 8)

        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x, x2):
        x = self.stem(x)
        x2 = self.stem(x2)

        x, x2 = self.mutual_attention_layer2(x, x2)

        x = self.dark2(x)
        x2 = self.dark2(x2)

        x, x2 = self.mutual_attention_layer3(x, x2)

        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        x2 = self.dark3(x2)
        feat1_2 = x2

        x, x2 = self.mutual_attention_layer4(x, x2)
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        x2 = self.dark4(x2)
        feat2_2 = x2

        x, x2 = self.mutual_attention_layer5(x, x2)
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        x2 = self.dark5(x2)
        feat3_2 = x2

        return feat1, feat2, feat3, feat1_2, feat2_2, feat3_2
