import torch
import torch.nn as nn


#切小方块操作
class PatchEmbedding(nn.Module):
    def __init__(self,in_channels,patch_size,embed_dim,num_patches,dropout):
        super(PatchEmbedding,self).__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size),
            nn.Flatten(2)
        )

        self.cls_token = nn.Parameter(torch.randn(size=(1,1,embed_dim)),requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(size=(1,num_patches+1,embed_dim)),requires_grad = True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)

        x = self.patcher(x).permute(0,2,1)
        x = torch.cat([cls_token,x],dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        return x
#vit结构流程
class Vit(nn.Module):
    def __init__(self,in_channels,patch_size,embed_dim,num_patches,dropout,
                 num_heads,activation,num_encoders,num_classes):
        super(Vit,self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels,patch_size,embed_dim,num_patches,dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,nhead=num_heads,dropout=dropout,
                                                    activation=activation,
                                                    batch_first=True,norm_first=True)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer,num_layers=num_classes)
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim,out_features=num_classes)
        )

    def forward(self,x):
        x = self.patch_embedding(x)

        x = self.encoder_layers(x)
        #x[:, 0, :] x.shape为(1,768) 经过MLP -> (1,10)
        x = self.MLP(x[:, 0, :])
        return x
        

if __name__ == "__main__":
    IMG_SIZE = 224
    IN_CHANNELS = 3
    PATCH_SIZE = 16
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 196
    EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS  # 768
    DROPOUT = 0.001

    NUM_HEADS = 8
    ACTIVATION = "gelu"
    NUM_ENCODERS = 4
    NUM_CLASSES = 10
    HIDDEN_LAYER = 768

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = PatchEmbedding(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT)
    model = Vit(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT, NUM_HEADS, ACTIVATION, NUM_ENCODERS,
                NUM_CLASSES).to(device)
    x = torch.randn(size=(1, 3, 224, 224)).to(device)
    prediction = model(x)
    print(prediction.shape)

    
"""
nn.TransformerEncoderLayer 和 nn.TransformerEncoder 是 PyTorch 中用于构建 Transformer 模型的两个关键组件。
nn.TransformerEncoderLayer
nn.TransformerEncoderLayer 表示一个单独的 Transformer 编码器层。它由以下几个部分组成：

多头自注意力机制（Multi-Head Self-Attention）：用于捕捉输入序列中的上下文信息。
前馈神经网络（Feedforward Neural Network）：用于在每个位置对编码器的输出进行非线性转换。
残差连接（Residual Connection）和层归一化（Layer Normalization）：用于稳定训练过程。
在创建 nn.TransformerEncoderLayer 时，你可以指定以下参数：

d_model：输入和输出的特征维度。
nhead：多头自注意力机制中注意力头的数量。
dim_feedforward：前馈神经网络中隐藏层的大小。
dropout：用于残差连接和注意力机制的 dropout 概率。
activation：前馈神经网络中使用的激活函数，默认为 ReLU。
nn.TransformerEncoder
nn.TransformerEncoder 是一个包含多个 nn.TransformerEncoderLayer 的编码器。它将输入序列传递给多个编码器层，并逐层处理。每个编码器层的输出作为下一层的输入，直到所有层都处理完毕。

在创建 nn.TransformerEncoder 时，你可以指定以下参数：

encoder_layer：用于构建编码器的 nn.TransformerEncoderLayer。
num_layers：编码器中编码器层的数量。
norm_first：表示是否应该先对输入进行归一化，默认为 False。
batch_first：表示输入张量的第一个维度是否为 batch 的大小，默认为 False。
这些组件一起构成了 Transformer 模型的编码器部分，它用于将输入序列转换为隐藏表示。Transformer 模型的解码器部分也类似，但包含了额外的注意力机制，用于生成输出序列。
"""