import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone


class PanopticFPN(nn.Module):
    def __init__(self, args):
        super(PanopticFPN, self).__init__()
        self.backbone = backbone.__dict__[args.arch](pretrained=args.pretrain)
        self.decoder  = FPNDecoder(args)
        self.sfa=SelfAtten()
        
    def forward(self, x):
        feats = self.backbone(x)
        outs  = self.decoder(feats) 
        outs=self.sfa(outs)
        
        return outs 

class FPNDecoder(nn.Module):
    def __init__(self, args):
        super(FPNDecoder, self).__init__()
        if args.arch == 'resnet18':
            mfactor = 1
            out_dim = 128 
        else:
            mfactor = 4
            out_dim = 256

        self.layer4 = nn.Conv2d(512*mfactor//8, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(512*mfactor//4, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(512*mfactor//2, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer1 = nn.Conv2d(512*mfactor, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        o1 = self.layer1(x['res5'])
        o2 = self.upsample_add(o1, self.layer2(x['res4']))
        o3 = self.upsample_add(o2, self.layer3(x['res3']))
        o4 = self.upsample_add(o3, self.layer4(x['res2']))

        return o4

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y 

class SelfAtten(nn.Module):
    def __init__(self,in_dim=128):
        super(SelfAtten,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.FyO = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        #print(x.size())
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out=self.FyO(out)
        out = out + x
        return out


