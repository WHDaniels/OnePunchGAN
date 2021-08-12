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


  Use GroupNorm for ResNet + ColorNet
"""

import copy
import os
import random
import time

import torch
import torchvision
from PIL.Image import Image
from torch.utils.data import Dataset


class BackboneDataset(Dataset):
    def __init__(self):
        self.data_path = ''
        self.batch_size = 1

        self.image_list = list()
        self.label_list = list()

        for root, dirs, files in os.walk(self.data_path):
            for name in files:
                if True:
                    # if name is not text file
                    self.image_list.append(os.path.join(root, name))
                if True:
                    # if it is
                    self.label_list.append(os.path.join(root, name))

        pass

    def __getitem__(self, i):
        transform_seed = (random.randint(0, 2 ** 32))

        # get image path
        self.image_path = os.path.join(self.data_path, self.image_list[i])
        # get label path
        self.label_path = os.path.join(self.data_path, self.label_list[i])

        # get image and label
        # image, label = self.get_image_and_label()

        # get transform and one hot encoding
        self.get_transforms()

        # return { 'image': image
        #          'label': label }
        pass

    def __len__(self):
        return len(self.image_list) // self.batch_size

    def get_image_and_label(self):

        # return image
        image = Image.open(self.image_path).convert('L')

        # return label
        with open(self.label_path, 'r') as file:

                

        pass

    def get_transforms(self):
        # return transformed_image and label_encoding
        pass


def get_new_model():
    model = torchvision.models.resnet50(pretrained=True)

    # drop last 2 layers and replace with above
    # leave requires grad = True
    return model


def make_data():
    # make dataset
    # should get picture along with labels one-hot encoded

    # return dataloader
    pass


# from pytorch's fine-tune tutorial
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

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
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to('cuda:1')
                labels = labels.to('cuda:1')

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
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
