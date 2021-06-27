import torch
import torch.nn as nn
import torch.nn.functional as F

# From figure 2 in paper
class DilatedResidualLayer(torch.nn.Module):
    def __init__(self, dilation_factor, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        # padding = dilation_factor to keep size 
        self.block =  nn.Sequential(  nn.Conv1d(in_channels, out_channels, 3, padding=dilation_factor, dilation=dilation_factor),
                                      nn.ReLU(inplace = True),
                                      nn.Conv1d(out_channels, out_channels, 1)
                                      )
    def forward(self,x, mask):
        return (self.block(x) + x )* mask[:, 0:1, :]

    

class TCN(torch.nn.Module):
    def __init__(self, num_layers = 10, num_f_maps=64, dim = 2048, num_classes = 48):
        super(TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # linear increasing of number dilation using i+1
            layer = DilatedResidualLayer(i+1, num_f_maps, num_f_maps)
            self.layers.append(layer)
        self.last_conv = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x , mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out,mask)
        out = self.last_conv(out)  * mask[:, 0:1, :]
        return out
    
    
class MultiStageTCN(torch.nn.Module):
    def __init__(self, num_stages = 4, num_layers = 10, num_f_maps=64, dim = 2048, num_classes = 48):
        super(MultiStageTCN, self).__init__()
        self.first_TCN = TCN(num_layers, num_f_maps, dim, num_classes)
        self.TCNs = nn.ModuleList()
        for i in range(num_layers-1):
            tcn =  TCN(num_layers, num_f_maps, num_classes + dim, num_classes)
            self.TCNs.append(tcn)
        self.last_TCN =  TCN(num_layers, num_f_maps, num_classes + dim, num_classes)

    def forward(self, x , mask):
        out = self.first_TCN(x,mask)   
        out = torch.cat((x, out), dim=1)
        #out_wrapped = out.unsqueeze(0)
        for stage in self.TCNs:
            out = stage(out,mask)
            out = F.softmax(out, dim=1)   
            out = torch.cat((x, out), dim=1)
            out = stage( out* mask[:, 0:1, :], mask)
            out = torch.cat((x, out), dim=1)
            #out_wrapped = torch.cat((out_wrapped, out.unsqueeze(0)), dim=0)
        out = self.last_TCN(out,mask)   
        #out_wrapped = torch.cat((out_wrapped, out.unsqueeze(0)), dim=0)
        return out
    

# For question 4, supporting down and upsampling  
class SampledTCN(torch.nn.Module):
    def __init__(self, sampling_factor, num_layers = 10, num_f_maps=64, dim = 2048, num_classes = 48 ):
        super(SampledTCN, self).__init__()
        # we downsamble using convolutions 
        self.downsamble_conv = nn.Conv1d(dim, dim//sampling_factor, 1)
        self.conv_1x1 = nn.Conv1d(dim//sampling_factor, num_f_maps, 1)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DilatedResidualLayer(i+1, num_f_maps, num_f_maps)
            self.layers.append(layer)
        self.upsample_conv = nn.Conv1d(num_f_maps, dim, 1)

    def forward(self, x , mask):
        out = self.downsamble_conv(x)
        out = self.conv_1x1(out)
        for layer in self.layers:
            out = layer(out,mask)
        out = self.upsample_conv(out) * mask[:, 0:1, :]
        return out
    
    
    
class ParallelTCNs(torch.nn.Module):
        def __init__(self, num_layers = 10, num_f_maps=64, dim = 2048, num_classes = 48):
            super(ParallelTCNs, self).__init__()
            self.TCN1 = SampledTCN(1, num_layers, num_f_maps, dim, num_classes)
            # scale reduced with factor 4 2048 -> 512
            self.TCN2 = SampledTCN(4, num_layers, num_f_maps, dim, num_classes)
             # scale reduced with factor 8 2048 -> 256
            self.TCN3 = SampledTCN(8, num_layers, num_f_maps, dim, num_classes)
            
            self.prediction_conv1 = nn.Conv1d(dim, num_classes, 1)
            self.prediction_conv2 = nn.Conv1d(dim, num_classes, 1)
            self.prediction_conv3 = nn.Conv1d(dim, num_classes, 1)

            self.prediction_conv_average = nn.Conv1d(dim, num_classes, 1)
            
            
        def forward(self, x, mask):
            out1 = self.TCN1(x,mask)
            out2 = self.TCN2(x,mask)
            out3 = self.TCN3(x,mask)
            
            average_out = torch.mean(torch.stack([out1,out2,out3]) , dim = 0)
            out1 = self.prediction_conv1(out1) * mask[:, 0:1, :]
            out2 = self.prediction_conv2(out2) * mask[:, 0:1, :]
            out3 = self.prediction_conv3(out3) * mask[:, 0:1, :]
            average_out = self.prediction_conv_average(average_out) * mask[:, 0:1, :]
            return out1 , out2 , out3 , average_out
            
            

            



        