import os
import torch
import torch.nn as nn
from torch import optim
import timeit
from tqdm import tqdm
from dataloder import get_loaders
from model import Vit

#hyper parameter
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50

BATCH_SIZE = 16
TRAIN_DF_DIR = "./dataset/train.csv"
TEST_DF_DIR = "./dataset/test.csv"
SUBMISSION_DF_DIR = "./dataset/sample_submission.csv"

# Model Parameters
IN_CHANNELS = 1
IMG_SIZE = 28
PATCH_SIZE = 4
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
DROPOUT = 0.001

NUM_HEADS = 8
ACTIVATION = "gelu"
NUM_ENCODERS = 768
NUM_CLASSES = 10

LEARNING_RATE = 1e-4
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9,0.999)

train_dataloader,val_dataloader,test_dataloader = get_loaders(TRAIN_DF_DIR,TEST_DF_DIR,SUBMISSION_DF_DIR,BATCH_SIZE)
model = Vit(IN_CHANNELS,PATCH_SIZE,EMBED_DIM,NUM_PATCHES,DROPOUT,
            NUM_HEADS,ACTIVATION,NUM_ENCODERS,NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
"""
model.parameters()：这是传递给优化器的参数列表，表示要优化的模型的参数。model.parameters() 返回一个迭代器，包含了模型中所有需要优化的参数。
betas=ADAM_BETAS：betas 是 Adam 优化器的一个参数，它控制优化器中用于计算一阶矩和二阶矩的指数衰减率。ADAM_BETAS 可能是一个元组，包含了两个参数，通常是 (beta1, beta2)。这两个参数的默认值通常为 (0.9, 0.999)，它们在算法中起到平滑梯度和平方梯度的作用。
lr=LEARNING_RATE：lr 是学习率的意思，它是优化器的一个参数，用于控制每次参数更新的步长。LEARNING_RATE 是学习率的具体值，是一个标量（scalar），通常在训练模型时需要根据具体问题进行调整。
weight_decay=ADAM_WEIGHT_DECAY：weight_decay 是 Adam 优化器的一个参数，用于对模型参数进行 L2 正则化。ADAM_WEIGHT_DECAY 是 L2 正则化的权重衰减因子，通常是一个小的正数，用于防止模型过拟合。
"""
optimizer = optim.Adam(model.parameters(),betas=ADAM_BETAS,lr=LEARNING_RATE,weight_decay=ADAM_WEIGHT_DECAY)

start_time = timeit.default_timer()

for epoch in tqdm(range(EPOCHS),position=0,leave=True):
    model.train()
    train_labels = []
    train_preds = []
    train_running_loss = 0

    for idx,image_label in enumerate(tqdm(train_dataloader,position=0,leave=True,desc='Training...', dynamic_ncols=True)):
        img = image_label["image"].float().to(device)
        label = image_label["label"].type(torch.uint8).to(device)
        y_pred = model(img)
        y_pred_label = torch.argmax(y_pred,dim=1)

        #将train_labels和train_preds方到cup上并extend到对应列表中
        train_labels.extend(label.cpu().detach())
        train_preds.extend(y_pred_label.cpu().detach())

        loss = criterion(y_pred,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
    #train_running_loss单轮训练总损失   train_loss：多轮训练平均损失
    train_loss = train_running_loss / (idx + 1)

    #保存训练对象模型
    model_name = f"model_epoch_{epoch}.pth"
    save_dir = "models"  # 保存路径
    # 确保保存路径存在，如果不存在则创建
    os.makedirs(save_dir, exist_ok=True)
    # 保存模型参数
    torch.save(model.state_dict(), os.path.join(save_dir, model_name))

    model.eval()
    val_labels = []
    val_preds = []
    val_running_loss = 0
    with torch.no_grad():
        for idx, image_label in enumerate(tqdm(val_dataloader,position=0,leave=True,desc='Validating...', dynamic_ncols=True)):
            img = image_label["image"].float().to(device)
            label = image_label["label"].type(torch.uint8).to(device)
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred,dim=1)

            val_labels.extend(label.cpu().detach())
            val_preds.extend(y_pred_label.cpu().detach())

            loss = criterion(y_pred,label)
            val_running_loss += loss.item()
    val_loss = val_running_loss / (idx + 1)

    
    print("-" * 30)

    print(f"Train Loss Epoch:{epoch + 1} : {train_loss:.4f}")
    print(f"Val Loss Epoch:{epoch + 1} : {val_loss:.4f}")
    print(f"Train Accuracy Epoch {epoch + 1} : {sum(1 for x , y in zip(train_preds,train_labels) if x==y) /len(train_labels): .4f}")
    print(f"Val Accuracy Epoch {epoch + 1} : {sum(1 for x , y in zip(val_preds,val_labels) if x==y) /len(val_labels): .4f}")
    print("-" * 30)
stop_time = timeit.default_timer()

print(f"Train time : {stop_time - start_time:.4f}s")






