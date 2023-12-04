import torch
from math import floor


class SweepNet(torch.nn.Module):
    
    def __init__(self, H, W, channels, channel_select=0):
        super(SweepNet, self).__init__()
        
        conv1_kernel = (2,2)
        conv2_kernel = (2,2)
        conv3_kernel = (2,2)
        
        conv1_stride = (1, 1)
        conv2_stride = (1, 1)
        conv3_stride = (1, 1)
        
        conv1_channels = 32
        conv2_channels = 32
        conv3_channels = 32
        
        pool1_kernel = 2
        pool2_kernel = 2
        pool3_kernel = 2
        
        pool1_stride = 1
        pool2_stride = 1
        pool3_stride = 1
        
        self.channel_select = channel_select
        self.channels = channels        
        self.conv1 = torch.nn.Conv2d(channels, conv1_channels, kernel_size=conv1_kernel, stride=conv1_stride, padding=0)
        self.pool1 = torch.nn.MaxPool2d(pool1_kernel, pool1_stride)
        self.relu1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(conv1_channels, conv2_channels, kernel_size=conv2_kernel, stride=conv2_stride, padding=0)
        self.pool2 = torch.nn.MaxPool2d(pool2_kernel, pool2_stride)
        self.relu2 = torch.nn.ReLU()
        
        self.conv3 = torch.nn.Conv2d(conv2_channels, conv3_channels, kernel_size=conv3_kernel, stride=conv3_stride, padding=0)
        self.pool3 = torch.nn.MaxPool2d(pool3_kernel, pool3_stride)
        self.relu3 = torch.nn.ReLU()
        
        # compute output size to FC
        out_conv1 = self.compute_out_shape((H, W), conv1_kernel, conv1_stride)
        out_pool1 = self.compute_out_shape(out_conv1, (pool1_kernel, pool1_kernel), (pool1_stride, pool1_stride))
        out_conv2 = self.compute_out_shape(out_pool1, conv2_kernel, conv2_stride)
        out_pool2 = self.compute_out_shape(out_conv2, (pool2_kernel, pool2_kernel), (pool2_stride, pool2_stride))
        out_conv3 = self.compute_out_shape(out_pool2, conv3_kernel, conv3_stride)
        out_pool3 = self.compute_out_shape(out_conv3, (pool3_kernel, pool3_kernel), (pool3_stride, pool3_stride))
        
        # init SE block layers
        r = 16
        self.se_pool = torch.nn.AvgPool2d(out_pool3)
        self.se_conv1 = torch.nn.Conv2d(in_channels=conv3_channels, out_channels=conv3_channels//r, kernel_size=1, stride=1)
        self.se_relu1 = torch.nn.ReLU()
        self.se_conv2 = torch.nn.Conv2d(in_channels=conv3_channels//r, out_channels=conv3_channels, kernel_size=1, stride=1)
        self.se_sig = torch.nn.Sigmoid()
        
        # init output layers
        self.fc1 = torch.nn.Linear(conv3_channels, 32)
        self.out_relu = torch.nn.ReLU()
        self.out_pool = torch.nn.AvgPool2d(out_pool3)
        self.fc2 = torch.nn.Linear(32, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def compute_out_shape(self, I, k, s):
        out_H = floor(((I[0] - (k[0] - 1) - 1) / s[0]) + 1)
        out_W = floor(((I[1] - (k[1] - 1) - 1) / s[1]) + 1)
        return (out_H, out_W)
    
        # execute SE block
    def SE_block(self, x_0):
        x = self.se_pool(x_0)      
        x = self.se_conv1(x)
        x = self.se_relu1(x)
        x = self.se_conv2(x)
        x = self.se_sig(x)
        x = torch.multiply(x_0, x)
        return x  
    
    def forward(self, x):
        if self.channels == 1:
            x = x[:,2,:,:].unsqueeze(1) # get one b/w channel, keep channels dim
        elif self.channels == 2:
            x_snp = x[:,2,:,:].unsqueeze(1)
            x = torch.cat((x[:,self.channel_select,:,:].unsqueeze(1), x_snp), dim=1)
        else:
            x = x[:,0:3,:,:]
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)

        x = self.SE_block(x)
        x = torch.movedim(x, 1, -1)
        x = self.fc1(x)
        x = self.out_relu(x)
        x = torch.movedim(x, -1, 1)
        x = self.out_pool(x)
        x = x.flatten(1)
        x = self.fc2(x)
        return x