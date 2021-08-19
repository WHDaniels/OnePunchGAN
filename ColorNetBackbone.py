"""
(avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=<some number>, out_features=1000, bias=True)

  goes to

  (1): Sequential(
    (0): AdaptiveConcatPool2d(
      (ap): AdaptiveAvgPool2d(output_size=1)
      (mp): AdaptiveMaxPool2d(output_size=1)
    )
    (1): Flatten()
    (2): InstanceNorm1d(<some number>, ...)
    (3): Dropout(p=0.25)
    (4): Linear(in_features=<some number>, out_features=512, bias=True)
    (5): ReLU(inplace)
    (6): InstanceNorm1d(512, ...)
    (7): Dropout(p=0.5)
    (8): Linear(in_features=512, out_features=<#tags>, bias=True)
  )


  Use GroupNorm for ColorNet
"""

import copy
import os
import pickle
import random
import time

import torch
import torchvision
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as ts
from matplotlib import pyplot as plt


class BackboneDataset(Dataset):
    def __init__(self, path, batch_size, dim, mode='train'):
        self.data_path = path
        self.batch_size = batch_size
        self.resized_dim = dim
        self.mode = mode

        self.image_file_list = list()
        self.label_list = list()

        # for root, dirs, files in os.walk(self.data_path):
        # for name in files:
        # if name is not text file
        # if not name.endswith('.txt'):
        # self.image_file_list.append(os.path.join(root, name))

        if os.path.exists(".\\objects"):
            with open(".\\objects\\filename_tag_dict.p", 'rb') as file:
                self.filename_tag_dict = pickle.load(file)
            with open(".\\objects\\tag_freq_dict.p", 'rb') as file:
                self.tag_freq_dict = pickle.load(file)
            with open(".\\objects\\tag_set.p", 'rb') as file:
                self.tag_set = pickle.load(file)
        else:
            self.filename_tag_dict, self.tag_freq_dict, self.tag_set = self.get_dictionaries_and_sets()

        self.image_list = list(self.filename_tag_dict.keys())

    def __getitem__(self, i):
        # transform_seed = (random.randint(0, 2 ** 32))

        # get image path
        self.image_path = os.path.join(self.data_path, self.image_list[i])

        # get image and label
        image, label = self.get_image_and_label(self.filename_tag_dict)

        # get transform and one hot encoding
        image = self.get_transforms(image)

        # image = ts.Compose([ts.ToPILImage()])(image)
        # image.show()
        # exit(1)

        return {'image': image,
                'label': label}

    def __len__(self):
        return int(len(self.image_list) / 20)

    def get_dictionaries_and_sets(self):
        # code here should filter tags and filenames to get the best training labels
        # prob will modify how we grab images from directory above, as we will want to filter some out and not grab all
        filename_tag_dict, tag_freq_dict, tag_set = self.create_dicts_and_sets()
        # remove every image that is not in png or jpg format
        # for k, v in filename_tag_dict.items():
        # if not k.endswith('.png') and not k.endswith('.jpg'):
        # del filename_tag_dict[k]

        lowest_tag_threshold = 300
        tag_freq_copy = copy.deepcopy(tag_freq_dict)
        remove_list = ['monochrome', 'greyscale', 'rating:safe']

        for k, v in tag_freq_copy.items():
            # if the tag is not common enough
            if v < lowest_tag_threshold:
                del tag_freq_dict[k]
                tag_set.remove(k)
            if k in remove_list:
                del tag_freq_dict[k]
                tag_set.remove(k)

        for k, v in filename_tag_dict.items():
            filename_tag_dict[k] = [tag for tag in tag_set if tag in v and tag not in remove_list]

        os.mkdir(".\\objects")
        with open(".\\objects\\filename_tag_dict.p", 'wb') as file:
            pickle.dump(filename_tag_dict, file)
        with open(".\\objects\\tag_freq_dict.p", 'wb') as file:
            pickle.dump(tag_freq_dict, file)
        with open(".\\objects\\tag_set.p", 'wb') as file:
            pickle.dump(tag_set, file)
        return filename_tag_dict, tag_freq_dict, tag_set

    def create_dicts_and_sets(self):
        filename_tag_dict = {}
        tag_freq_dict = {}
        tag_set = set()

        for n, filename in enumerate(os.listdir(self.data_path)):
            print("Files iterated through:", n)
            if filename.endswith('.txt') and (filename[:-4].endswith('.png') or filename[:-4].endswith('.jpg')):
                if filename[-7:-4] == 'gif':
                    print(filename)
                    break
                with open(self.data_path + "\\" + filename, 'r') as file:
                    # get dict of tag frequency and get dict of filename and respective tag list
                    tag_list = []
                    for line in file.readlines():

                        line = line.strip("\n")
                        tag_set.add(line)

                        if line in tag_freq_dict.keys():
                            tag_freq_dict[line] = tag_freq_dict.get(line) + 1
                        else:
                            tag_freq_dict[line] = 1
                        tag_list.append(line)

                    filename_tag_dict[filename[:-4]] = tag_list

        # sort tag frequency list by frequency
        tag_freq_dict = {k: v for k, v in sorted(tag_freq_dict.items(), key=lambda item: item[1])}

        return filename_tag_dict, tag_freq_dict, tag_set

    def get_image_and_label(self, filename_tag_dict):

        # return image
        image = Image.open(self.image_path).convert('L')

        # return label
        # with open(self.label_path, 'r') as file:
        image_key = self.image_path.split('\\')[-1]
        tag_list = filename_tag_dict[image_key]

        encoding = self.get_OHE_from_tag_list(tag_list)

        return image, encoding

    def get_OHE_from_tag_list(self, tag_list):
        # gets the one hot encoding from a given list of tags
        # self.tag_set used to make OHE
        tag_set = list(self.tag_set)
        # print("tag", tag_set)
        labels = torch.Tensor([tag_set.index(x) for x in tag_list])
        # print("first labels shape", labels.shape)
        # print("first labels", labels)
        labels = labels.unsqueeze(0)
        # print("second labels shape", labels.shape)
        # print("second labels", labels)
        encoding = torch.zeros(labels.size(0), len(tag_set)).scatter_(1, labels.data.cpu().long(), 1.)
        encoding = torch.squeeze(encoding)
        # print("encoding shape", encoding.shape)
        # print("encoding", encoding)
        # exit(1)

        # should return an encoding like so [[0., 1., 0., 1., 1., 0., 1., ... etc]]
        return encoding

    def get_transforms(self, image):
        # return transformed_image
        compose_list = []
        # if image h:w ratio is higher than say 2:1 (?)
        # pad image by the side that is least by some random amount (up to the length of the longer side)
        if self.mode == 'train':
            compose_list += pad_image(image)

            compose_list += [
                ts.RandomHorizontalFlip(p=0.5),
                ts.RandomRotation(25, fill=255),
                ts.RandomPerspective(distortion_scale=0.667, p=0.667, fill=255),
                ts.RandomAdjustSharpness(sharpness_factor=2),
                ts.RandomAutocontrast(),
                ts.RandomEqualize()
            ]

            if random.random() > 0.5:
                filter_size = random.choice([3, 5, 7])
                compose_list += [ts.GaussianBlur(kernel_size=(filter_size, filter_size))]

        compose_list += [
            ts.Resize((self.resized_dim, self.resized_dim), interpolation=ts.InterpolationMode.BICUBIC),
            ts.ToTensor(),
            ts.Normalize((0.5,), (0.5,))
        ]

        """
        compose_list_val += [

            ts.RandomHorizontalFlip(p=0.5),
            ts.Grayscale(num_output_channels=1),
            ts.Resize((self.resized_dim, self.resized_dim), interpolation=ts.InterpolationMode.BICUBIC),
            ts.ToTensor(),
            ts.Normalize((0.5,), (0.5,))

        ]
        """
        return ts.Compose(compose_list)(image)


def pad_image(image):
    compose_list = list()
    # compose_list += ts.Grayscale(num_output_channels=1)

    w, h = image.size

    if w / h > 2:
        # pad height
        pad_amount = int((w - h) * random.uniform(0.5, 1) // 2)
        compose_list += [ts.Pad(padding=(0, pad_amount, 0, pad_amount), fill=255)]
    if w / h < 0.5:
        # pad width
        pad_amount = int((h - w) * random.uniform(0.5, 1) // 2)
        compose_list += [ts.Pad(padding=(pad_amount, 0, pad_amount, 0), fill=255)]
    return compose_list


# noinspection PyTypeChecker
def get_new_model():
    # cite: https://github.com/zhaoyuzhi/PyTorch-Special-Pre-trained-Models
    base_model = torchvision.models.resnet50(pretrained=False)

    # reconfigure resnet model for single channel input
    # base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # base_state_dict = base_model.state_dict()
    # conv1_weight = base_state_dict['conv1.weight']
    # base_state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
    # base_model.load_state_dict(base_state_dict)

    base_model = nn.Sequential(*(list(base_model.children())[:-2]))
    print(base_model)

    # model.conv1.weight = nn.Parameter(model.conv1.weight.sum(dim=1).unsqueeze(1))

    model = nn.Sequential(

        base_model,
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.AdaptiveMaxPool2d(output_size=1),
        nn.Flatten(),
        nn.BatchNorm1d(4096),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=4096, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=512, out_features=6000, bias=True)

    )

    print(model)

    # leave requires grad = True
    return model


# from pytorch's fine-tune tutorial
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, is_inception=False):
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for n, batch in enumerate(dataloaders[phase]):
                print(n)
                # print(batch)
                inputs = batch['image'].to('cuda:1')
                labels = batch['label'].to('cuda:1')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        # print(inputs.shape)
                        outputs = model(inputs)
                        # print(outputs.shape)
                        # print(outputs)
                        # print(outputs)
                        # print(labels.shape)
                        # print(labels.shape)
                        # print(labels)
                        # exit(1)
                        loss = criterion(outputs, labels)
                        # print(loss)
                        # print(inputs.size(0))

                    _, preds = torch.max(outputs, 0)
                    # print(preds)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # print(preds.shape, labels.data.shape)
                running_corrects += torch.sum(preds == labels.data)
                # print(preds == labels.data)
                # print("loss", running_loss / (n + 1))
                # print("corrects", running_corrects)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "./models/1_channel_resnet_backbone.pth")
            if phase == 'train':
                train_acc_history.append(epoch_acc.cpu() * .01)
                train_loss_history.append(epoch_loss)
            if phase == 'val':
                val_acc_history.append(epoch_acc.cpu() * .01)
                val_loss_history.append(epoch_loss)

        scheduler.step(epoch_loss)

        # plot metrics
        plt.figure(figsize=[8, 6])
        plt.plot(train_acc_history, 'r', linewidth=2, label='train_acc')
        plt.plot(val_acc_history, 'm', linewidth=2, label='val_acc')
        plt.plot(train_loss_history, 'b', linewidth=2, label='train_loss')
        plt.plot(val_loss_history, 'c', linewidth=2, label='val_loss')
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Accuracy and Loss', fontsize=16)
        plt.legend()
        plt.savefig(f"figures/epoch{epoch}.png")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == '__main__':
    data_path = 'D:\\greyscale_monochrome'
    batch_size, resize_dim = 32, 256
    epochs = 20 * 5
    # decay_epochs = 5
    model = get_new_model().to('cuda:1')

    # model.load_state_dict(torch.load("C:\\Users\\mercm\\Desktop\\pretrained\\resnet_backbone_128_dim.pth"))

    datasets = {'train': BackboneDataset(data_path, batch_size, resize_dim),
                'val': BackboneDataset(data_path, batch_size, resize_dim, mode='val')}
    dataloaders = {'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, pin_memory=True,
                                       num_workers=2),
                   'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=True, pin_memory=True,
                                     num_workers=2)}

    # dataloader = DataLoader(BackboneDataset, batch_size, shuffle=True, pin_memory=True, drop_last=True)
    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

    # def lambda_rule(epoch):
        # return 1.0 - max(0, epoch - epochs) / float(decay_epochs + 1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    train_model(model, dataloaders, loss, optimizer, scheduler, num_epochs=epochs)
