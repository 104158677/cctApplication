from __future__ import print_function
import glob
import os
import random
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from cct import CCT
from torch import nn, einsum


print(f"Torch: {torch.__version__}")


# Training settings
batch_size = 64
epochs = 10
lr = 3e-5
gamma = 0.7
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cuda'

os.makedirs('data3', exist_ok=True)

train_dir = 'data3/train'
test_dir = 'data3/test'

DATA_DIR_TRAIN = 'data3//train'
DATA_DIR_TEST = 'data3//test'

# def extract_class(Datasets):
#   for vals in os.listdir(Datasets):
#     print(vals)
#
# extract_class(DATA_DIR_TRAIN)

# for root, dirs, files in os.walk(train_dir):
#     if not files:
#         continue
#     prefix = os.path.basename(root)
#     for f in files:
#         fname = os.path.join(root, "{}.{}".format(prefix, f))
#         os.rename(os.path.join(root, f), fname)
#         shutil.move(fname,"data2\\train")

# for root, dirs, files in os.walk(test_dir):
#     if not files:
#         continue
#     prefix = os.path.basename(root)
#     for f in files:
#         fname = os.path.join(root, "{}.{}".format(prefix, f))
#         os.rename(os.path.join(root, f), fname)
#         shutil.move(fname,"data2\\test")

#
# random_idx = np.random.randint(1, len(train_dataset), size=9)
# fig, axes = plt.subplots(3, 3, figsize=(16, 12))
#
# for idx, ax in enumerate(axes.ravel()):
#     img = Image.open(train_list[idx])
#     ax.set_title(labels[idx])
#     ax.imshow(img)

train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

print(os.listdir(train_dir)[:5])
print(len(train_list))

# random_idx = np.random.randint(1,8144,size=10)
#
# fig = plt.figure()
# i=1
# for idx in random_idx:
#     ax = fig.add_subplot(2,5,i)
#     img = Image.open(train_list[idx])
#     plt.imshow(img)
#     i+=1
#
# plt.axis('off')
# plt.show()

print(train_list[0].split('\\')[-1].split('.')[0])
print(test_list[0].split('\\')[-1].split('.')[0])

print(len(train_list), len(test_list))

labels = [path.split('\\')[-1].split('.')[0] for path in train_list]
print(type(labels))
train_classes = list(set(labels))
test_classes = train_classes
print(train_classes)
print(len(train_classes))

def link_idx(input):
    idx = train_classes.index(input)
    return idx

train_list, valid_list = train_test_split(train_list,
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)

print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")


# for i in range(8144):
#     img = Image.open(train_list[i])
#     print(img.shape)

train_transforms =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
     transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label = img_path.split("\\")[-1].split(".")[0]
        label = train_classes.index(label)
        return img_transformed, label


# for i in range(len(train_list)):
#     img_path = train_list[i]
#     print(img_path)
#     img = Image.open(img_path)
#     img_transformed = train_transforms(img)
#     print(img_transformed.shape)
#
# for i in range(len(valid_list)):
#     img_path = valid_list[i]
#     print(img_path)
#     img = Image.open(img_path)
#     img_transformed = val_transforms(img)
#     print(img_transformed.shape)
#
# for i in range(len(test_list)):
#     img_path = test_list[i]
#     print(img_path)
#     img = Image.open(img_path)
#     img_transformed = test_transforms(img)
#     print(img_transformed.shape)

train_data = CatsDogsDataset(train_list, transform=train_transforms)
valid_data = CatsDogsDataset(valid_list, transform=test_transforms)
test_data = CatsDogsDataset(test_list, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

print(len(train_data), len(train_loader))

print(len(valid_data), len(valid_loader))

# for i in range(len(train_data)):
#     if train_data[i][0].shape == torch.Size([1, 224, 224]):
#         train_data.pop(i)
#
# for i in range(len(valid_data)):
#     if valid_data[i][0].shape == torch.Size([1, 224, 224]):
#         valid_data.pop(i)
#
# for i in range(len(test_data)):
#     if test_data[i][0].shape == torch.Size([1, 224, 224]):
#         test_data.pop(i)

    # print((train_data[i][0]).shape)
    # print(train_data[i][1])
    # print(train_list[i])

cct = CCT(
    img_size = (224, 224),
    embedding_dim = 384,
    n_conv_layers = 2,
    kernel_size = 7,
    stride = 2,
    padding = 3,
    pooling_kernel_size = 3,
    pooling_stride = 2,
    pooling_padding = 1,
    num_layers = 14,
    num_heads = 6,
    mlp_ratio = 3.,
    num_classes = 1000,
    positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
)

model = cct.to(device)
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        # print(label)
        # print(type(label))
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
    print("label: ")
    print(label)
    print("output: ")
    print(output.argmax(dim=1))
