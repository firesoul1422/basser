
from torch import nn
import torch
import torch.nn.functional as F


class CMTStem(nn.Module):
  def __init__(self, in_channels, output_channels):
    super(CMTStem, self).__init__()
    self.layer1 = self.conv_layer(in_channels, output_channels, 2)
    self.layer2 = self.conv_layer(output_channels, output_channels, 1)
    self.layer3 = self.conv_layer(output_channels, output_channels, 1)
    
    

  def conv_layer(self, in_channels, filters, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, filters, kernel_size=3, stride=stride, padding=1),
        nn.GELU(),
        nn.BatchNorm2d(filters))

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    
    return x 
  


class LightweightMHSA(nn.Module):
  def __init__(self,h, w, c, num_patches, input_channels, num_heads, reduction_ratio):
    super(LightweightMHSA, self).__init__()
    self.num_patches = num_patches
    
    self.dim = int((w * c * h) / self.num_patches)
    self.reduced_dim = int(((w / reduction_ratio) * (h / reduction_ratio) * c ) / self.num_patches)
    self.num_heads = num_heads
    self.head_dim = self.dim // num_heads
    assert self.head_dim * self.num_heads == self.dim, f"Dimension mismatch {self.dim}"

    self.reduced_head_dim = self.reduced_dim // num_heads
    self.dwconv = nn.Conv2d(input_channels, input_channels, kernel_size=reduction_ratio, stride=reduction_ratio, padding=0, groups=input_channels)
    
    self.value = nn.Linear(self.reduced_dim, self.reduced_dim)
    self.key = nn.Linear(self.reduced_dim, self.reduced_dim)
    self.query = nn.Linear(self.dim, self.dim)
    
    self.relative_position_bias = nn.Parameter(torch.zeros(self.num_heads, self.num_patches, self.num_patches))
    nn.init.trunc_normal_(self.relative_position_bias, std=0.02)


  def forward(self, x, dconv = True):
    batch, c, w, h = x.shape
    x_reduced = self.dwconv(x)
    query = self.query(x.reshape(batch, self.num_patches, -1)).reshape(batch, self.num_heads, self.num_patches, self.head_dim)
    key = self.key(x_reduced.reshape(batch, self.num_patches, -1)).reshape(batch, self.num_heads, self.num_patches, self.reduced_head_dim)
    value = self.value(x_reduced.reshape(batch, self.num_patches, -1)).reshape(batch, self.num_heads, self.num_patches, self.reduced_head_dim)
    
    key = F.interpolate(key, size=([query.shape[-2], query.shape[-1]]))
    value = F.interpolate(value, size=([query.shape[-2], query.shape[-1]]))
    
    attention_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.dim ** 0.5)
    attention_weights = F.softmax(attention_weights + self.relative_position_bias.unsqueeze(0), dim=-1)

    attention_values = torch.matmul(attention_weights, value)

    return attention_values.reshape(batch, c, w, h)




class InvertedResidual(nn.Module):
  def __init__(self, in_channels, out_channels, filters):
    super(InvertedResidual, self).__init__()

    self.conv1_1_1 = self.resblock(in_channels, filters, 1, 0, 1)
    self.conv = self.resblock(filters, filters, 3, 1, filters)

    self.conv1_1_2 = nn.Conv2d(filters, out_channels, kernel_size=1, stride=1, padding=0)
    self.batch_norm = nn.BatchNorm2d(out_channels)


  def resblock(self, in_channels, filters, kernal, padding, group):
    return nn.Sequential(
        nn.Conv2d(in_channels, filters, kernel_size=kernal, stride=1, padding=padding, groups=group),
        nn.GELU(),
        nn.BatchNorm2d(filters))

  def forward(self, x):
    
    
    x = self.conv1_1_1(x)
    
    skip_con = x

    x = self.conv(x)
    
    x = x + skip_con
    
    x = self.conv1_1_2(x)

    
    x = self.batch_norm(x)
    
    return x
  

class LocalPerceptionUnit(nn.Module):
  def __init__(self, in_channels, output_channels):
    super(LocalPerceptionUnit, self).__init__()
    self.in_channels = in_channels
    self.output_channels = output_channels
    self.conv1_1 = nn.Conv2d(self.in_channels, self.output_channels, 3, 1, 1)
    self.conv = nn.Conv2d(self.output_channels, self.output_channels, 3, 1, 1, groups=self.output_channels)
  def forward(self, x):
    x = self.conv1_1(x)
    x_output = self.conv(x)
    output = x + x_output
    return output
  

class CMTBlock(nn.Module):
    def __init__(self, h, w, c, num_patches, in_channels, output_channels, num_heads, reduction_ratio):
        super(CMTBlock, self).__init__()
        self.local_perception_unit = LocalPerceptionUnit(in_channels=in_channels, output_channels=output_channels)
        self.attention = LightweightMHSA(h, w, c, num_patches, output_channels, num_heads, reduction_ratio)
        self.inverted_residual = InvertedResidual(output_channels, output_channels, output_channels * 2)
        self.layer_norm1 = nn.LayerNorm([in_channels, w, h])
        self.layer_norm2 = nn.LayerNorm([in_channels, w, h])

    def training_pass(self, x, p):
        if torch.rand(1).item() > p:
            x = self.local_perception_unit(x)
            skip_con = x
            x = self.layer_norm1(x)
            x = self.attention(x)
            x = x + skip_con

            skip_con = x
            x = self.layer_norm2(x)
            x = self.inverted_residual(x)
            x = x + skip_con
            
        return x  

    def evaluation_pass(self, x):
        x = self.local_perception_unit(x)
        skip_con = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = x + skip_con

        skip_con = x
        x = self.layer_norm2(x)
        x = self.inverted_residual(x)
        x = x + skip_con
        
        return x

    def forward(self, x, p=0.1):
        if self.training:
            return self.training_pass(x, p)
        else:
            return self.evaluation_pass(x)
        

class CMTNetwork(nn.Module):
  def __init__(self, h, w, c, num_patches, in_channels, filters, drop_prob, layer_drop_prob):
    super(CMTNetwork, self).__init__()
    self.layer_drop_prob = layer_drop_prob
    
    
    
    self.stem = CMTStem(in_channels, 16)
    self.gated_stem = nn.Conv2d(in_channels, 16 , kernel_size=3, stride=2, padding=1)
    self.batch_norm_stem = nn.BatchNorm2d(16)
    
    
    w, h = w // 2, h // 2
    self.Stage_1 = nn.ModuleList([CMTBlock(h , w , 46, 46 , 46, 46, 8, 8) for _ in range(1)])
    w, h = w // 2, h // 2
    w, h = w // 2, h // 2

    w, h = w // 2, h // 2
    self.Stage_4 = nn.ModuleList([CMTBlock(h , w , 368, 368 , 368, 368, 8, 1) for _ in range(2)])


    self.conv1 = nn.Conv2d(16, 46, kernel_size=2, stride=2, padding=0)
    self.gated_cnn1 = nn.Conv2d(16, 46 , kernel_size=2, stride=2, padding=0)
    
    self.conv2 = nn.Conv2d(46, 92, kernel_size=2, stride=2, padding=0)
    self.gated_cnn2 = nn.Conv2d(46, 92 , kernel_size=2, stride=2, padding=0)
    
    self.conv3 = nn.Conv2d(92, 184, kernel_size=2, stride=2, padding=0)
    self.gated_cnn3 = nn.Conv2d(92, 184 , kernel_size=2, stride=2, padding=0)
    
    self.conv4 = nn.Conv2d(184, 368, kernel_size=2, stride=2, padding=0)
    self.gated_cnn4 = nn.Conv2d(184, 368 , kernel_size=2, stride=2, padding=0)
    
    
    self.dropout1 = nn.Dropout2d(drop_prob)
    self.dropout2 = nn.Dropout2d(drop_prob)
    self.dropout3 = nn.Dropout2d(drop_prob)
    self.dropout4 = nn.Dropout2d(drop_prob)
    
    
    
    
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.conv1_1 = nn.Conv2d(368, 736 , kernel_size=1, stride=1, padding=0)
    self.gated_cnn5 = nn.Conv2d(368, 736 , kernel_size=1, stride=2, padding=0)
    
    
    self.softmax = nn.Softmax()
    self.sigmoid = nn.Sigmoid()
    
    self.fc1 = nn.Linear(1 * 1 * 736 , 368)
    
    self.fc1_dropout = nn.Dropout1d(0.1)
    
    self.batch_norm = nn.BatchNorm1d(368)
    self.relu = nn.ReLU()
    
    
    
    self.clasifir = nn.Linear(736 , 4)
    
  def forward(self, x):
    gate_x = self.sigmoid(self.gated_stem(x))
    x = self.stem(x)
    x = x * gate_x

      
    x = self.conv1(x)

      
    skip_conn = x
    for layer in self.Stage_1:
      x = layer(x, self.layer_drop_prob)

    x = x + skip_conn
      
    x = self.conv2(x)
    x = self.dropout1(x)
    
    x = self.conv3(x)
    x = self.dropout2(x)
    

    
    x = self.conv4(x)
    x = self.dropout3(x)
    
    skip_conn = x
    for layer in self.Stage_4:
      x = layer(x, self.layer_drop_prob)

    x = x + skip_conn
    x = self.dropout4(x)
    
    x = self.avg_pool(x)
    
    x = self.conv1_1(x)

    x = torch.flatten(x, 1)
    
    
    x = self.clasifir(x)
    return x
