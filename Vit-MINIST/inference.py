import torch
from dataloder import get_loaders
from model import Vit
from tqdm import tqdm

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
TRAIN_DF_DIR = "./dataset/train.csv"
TEST_DF_DIR = "./dataset/test.csv"
SUBMISSION_DF_DIR = "./dataset/sample_submission.csv"

# 模型参数
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

# 加载模型
model = Vit(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT,
            NUM_HEADS, ACTIVATION, NUM_ENCODERS, NUM_CLASSES).to(device)

# 加载模型参数
model.load_state_dict(torch.load("models/model_epoch_5.pth", map_location=device))  # 指定路径

# 设置模型为评估模式
model.eval()

# 加载测试数据
_, val_dataloader, _ = get_loaders(TRAIN_DF_DIR, TEST_DF_DIR, SUBMISSION_DF_DIR, BATCH_SIZE)

# 进行推理
test_preds = []
test_labels = []
with torch.no_grad():
    for idx,image_label in enumerate(tqdm(val_dataloader,position=0,leave=True,desc='Training...', dynamic_ncols=True)):
        img = image_label["image"].float().to(device)
        label = image_label["label"].float().to(device)
        y_pred = model(img)
        y_pred_label = torch.argmax(y_pred, dim=1)

        test_preds.extend(y_pred_label.cpu().detach().numpy())
        test_labels.extend(label.cpu().detach())



# 输出预测结果
print(f"Inference Accuracy : {sum(1 for x , y in zip(test_preds,test_labels) if x==y) /len(test_labels): .4f}")

