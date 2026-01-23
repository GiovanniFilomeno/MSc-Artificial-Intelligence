# This implementation is heavily inspired by the following GitHub repository:
# https://github.com/prathamhc/CIFAR10-Image-Classification-CNN-PyTorch/

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Dataset with indexing
class CIFAR10WithID(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target, idx


# Load CIFAR-10 dataset
def load_data(batch_size=20, valid_size=0.2, num_workers=0, withid=False):
    # used to convert images to tensors and normalize them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load and process train and test dataset
    train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

    # wrap with id Dataset
    if withid:
        train_data = CIFAR10WithID(train_data)
        test_data = CIFAR10WithID(test_data)

    # shuffle then split training data into training and validation sets
    # we shuffle before the split in case the data is orderd in some way
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # create samplers
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # create dataloaders for batching
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

# Helper function to un-normalize and display an image
def im_show(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# Define the CNN model
class Net(nn.Module):
    def __init__(self, save_act=False):
        super(Net, self).__init__()
        self.save_act = save_act
        self.activations = {}

        # convolutional Layer 
        self.conv_layer = nn.Sequential(
            # (1.) Convolution Layer: 3 input channels (RGB), 32 output channels, 3x3 kernel, padding 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),     # 'conv0'
            nn.BatchNorm2d(32),                                                      # 'conv1'
            nn.ReLU(inplace=True),                                                   # 'conv2'

            # (2.) Convolution Layer: 32 input channels, 64 output channels, 3x3 kernel
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),    # 'conv3'
            nn.ReLU(inplace=True),                                                   # 'conv4'
            # max Pooling to reduce dimensionality (downsampling)
            nn.MaxPool2d(kernel_size=2, stride=2),                                   # 'conv5'

            # (3.) Convolution Layer: 64 input channels, 128 output channels, 3x3 kernel
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),   # 'conv6'
            nn.BatchNorm2d(128),                                                     # 'conv7'
            nn.ReLU(inplace=True),                                                   # 'conv8'

            # (4.) Convolution Layer: 128 input channels, 128 output channels, 3x3 kernel
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),  # 'conv9'
            nn.ReLU(inplace=True),                                                   # 'conv10'
            # max Pooling to reduce dimensionality again
            nn.MaxPool2d(kernel_size=2, stride=2),                                   # 'conv11'
            # dropout for regularization to prevent overfitting
            nn.Dropout2d(p=0.05),                                                    # 'conv12'

            # (5.) Convolution Layer: 128 input channels, 256 output channels, 3x3 kernel
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # 'conv13'
            nn.BatchNorm2d(256),                                                     # 'conv14'
            nn.ReLU(inplace=True),                                                   # 'conv15'

            # (6.) Convolution Layer: 256 input channels, 256 output channels, 3x3 kernel
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),  # 'conv16'
            nn.ReLU(inplace=True),                                                   # 'conv17'
            # max Pooling to further reduce dimensionality
            nn.MaxPool2d(kernel_size=2, stride=2),                                   # 'conv18'
        )
        
        # fully Connected Layer
        self.fc_layer = nn.Sequential(
            # dropout regularization
            nn.Dropout(p=0.1),                                                       # 'fc0'
            # (1.) fully connected layer: flatten input to a vector, 4096 features to 1024
            nn.Linear(4096, 1024),                                                   # 'fc1'
            nn.ReLU(inplace=True),                                                   # 'fc2'

            # (2.) fully connected layer: 1024 features to 512
            nn.Linear(1024, 512),                                                    # 'fc3'
            nn.ReLU(inplace=True),                                                   # 'fc4'
            # dropout for regularization
            nn.Dropout(p=0.1),                                                       # 'fc5'
            # output layer: 512 features to 10 classes (CIFAR-10 has 10 classes)
            nn.Linear(512, 10)                                                       # 'fc6'
        )


        # register hooks for capturing activations (if enabled)
        self._register_hooks()

    # Registers forward hooks for each layer in the convolutional and fully connected layers
    # They save the activations of each layer during the forward pass
    def _register_hooks(self):
        # Saves the curent activation if save_act is enabled
        def save_activation(name):
            def hook(model, input, output):
                if self.save_act: self.activations[name] = output.detach()  # store activation, detach from graph
            return hook

        # Register hooks for convolutional layers
        for name, layer in self.conv_layer.named_children():
            layer.register_forward_hook(save_activation('conv'+name))

        # Register hooks for fully connected layers
        for name, layer in self.fc_layer.named_children():
            layer.register_forward_hook(save_activation('fc'+name))

    def forward_from_layer(self, x, start_layer=0):
        # Process up to the specified layer
        x = nn.Sequential(*list(self.conv_layer.children())[start_layer:])(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layer(x)  # Pass through the fully connected layers
        return x
            
    # Defines the forward pass through the network.
    def forward(self, x):
        # pass through the convolutional layers
        x = self.conv_layer(x)
        # flatten the output of convolutional layers to a 1D vector
        x = x.view(x.size(0), -1)
        # pass through the fully connected layers (classifier)
        x = self.fc_layer(x)
        
        return x

# Train the model
def train_model(model, train_loader, valid_loader, device, n_epochs=30, lr=0.01, instanceflow=False, plot=False):

    # we use cross-entropy loss and stochastic gradient descent
    model = model.to(device) # move the model to cpu/gpu
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # to store InstanceFlow data if enabled
    if instanceflow:
        train_ground_truth = {}
        train_predictions = []
        val_ground_truth = {}
        val_predictions = []

    valid_loss_min = np.inf
    train_losses = []
    valid_losses = []
    saved_epochs = []

    for epoch in range(n_epochs):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        # to store InstanceFlow data if enabled
        if instanceflow:
            train_pred = {}
            valid_pred = {}

        # train the model
        model.train()
        for batch in train_loader:
            if instanceflow: 
                data, target, sid = batch
                # move data to cpu/gpu
                data, target, sid = data.to(device), target.to(device), sid.to(device)
            else:
                data, target = batch
                # move data to cpu/gpu
                data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

            # save truth and prdiction data for InstanceFlow if enabled
            if instanceflow:
                _, preds = torch.max(output, 1)
                for idx, pred in zip(sid.cpu().numpy(), preds.cpu().numpy()):
                    train_pred[int(idx)] = int(pred)
                if not epoch: 
                    for idx, targ in zip(sid.cpu().numpy(), target.cpu().numpy()):
                        train_ground_truth[int(idx)] = int(targ)

        # validate the model
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                if instanceflow: 
                    data, target, sid = batch
                    # move data to cpu/gpu
                    data, target, sid = data.to(device), target.to(device), sid.to(device)
                else:
                    data, target = batch
                    # move data to cpu/gpu
                    data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item() * data.size(0)

                # save truth and prdiction data for InstanceFlow if enabled
                if instanceflow:
                    _, preds = torch.max(output, 1)
                    for idx, pred in zip(sid.cpu().numpy(), preds.cpu().numpy()):
                        valid_pred[int(idx)] = int(pred)
                    if not epoch: 
                        for idx, targ in zip(sid.cpu().numpy(), target.cpu().numpy()):
                            val_ground_truth[int(idx)] = int(targ)

        # sort InstanceFlow data if enabled
        if instanceflow:             
            train_predictions.append(dict(sorted(train_pred.items())))
            val_predictions.append(dict(sorted(valid_pred.items())))
            if not epoch:
                train_ground_truth = dict(sorted(train_ground_truth.items()))
                val_ground_truth = dict(sorted(val_ground_truth.items()))
                

        # calculate average losses
        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print training/validation statistics
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...')
            torch.save(model.state_dict(), 'model_cifar.pt')
            valid_loss_min = valid_loss
            saved_epochs.append(epoch)
    if plot:
        # plot training and validation losses and when we saved the model
        plt.figure(figsize=(5, 3))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(valid_losses, label='Validation Loss')
        for epoch in saved_epochs:
            plt.axvline(x=epoch, color='r', linestyle='--', alpha=0.7, label=f'Saved Epoch {epoch}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses Over Epochs')
        plt.legend()
        plt.grid()
        plt.show()

    if instanceflow: 
        return train_losses, train_ground_truth, train_predictions, val_ground_truth, val_predictions
    return train_losses

# Load the model
def load_saved_model(model_class=Net, filepath='model_cifar.pt', device=None):
    # initialize the model
    model = model_class()
    # load the model's state_dict
    model.load_state_dict(torch.load(filepath, map_location=device))

    print(f"Model loaded successfully from {filepath}")
    return model


# Test the model
def test_model(model, test_loader, device, save_act=False):
    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    # track test loss and activations 
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    activations = []

    # we use cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # enable or disable activation saving and move the model to cpu/gpu
    model.save_act = save_act
    model = model.to(device)
    
    model.eval()
    # iterate over test data
    with torch.no_grad():
        for data, target in test_loader:
            # move data to cpu/gpu
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update test loss
            test_loss += loss.item() * data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            # calculate test accuracy for each object class
            for i in range(len(correct)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

            # if activations are saved, collect them for this batch
            if save_act:
                batch_activations = {name: act.cpu().numpy() for name, act in model.activations.items()}
                activations.append(batch_activations)

    # average test loss
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.6f}\n')
    for i in range(10):
        if class_total[i] > 0:
            print(f'Test Accuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:.2f}%')
    overall_acc = 100. * np.sum(class_correct) / np.sum(class_total)
    print(f'Test Accuracy (Overall): {overall_acc:.2f}%')

    if save_act: return activations
