import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from data_generate_fg import CNNDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from sklearn import metrics
from pytorchtools import EarlyStopping

# some pre-parameters
batch_size = 8
n_epochs = 120
# early stopping patience; how long to wait after last time validation loss improved.
patience = 16
# reduce lr patience
lr = 0.015
l_patience = 8

savep1_path = './pred/pred_sinle_check_fg4.npy'
savep2_path = './pred/true_single_check_fg4.npy'
save_early_path = './model/checkpoint_fg4.pth.tar'
save_train_loss_path = './loss/train_loss_fg4.npy'
save_valid_loss_path = './loss/pred_loss3_fg4.npy'

lenghtr = 80
lenghva = 96
lenghte = 96
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


# some class & function
# transform function
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, para = sample[0],  sample[1]
        image = np.expand_dims(image,axis=0)
        return [torch.from_numpy(image).float(),
                torch.from_numpy(para).float()]

# data loader
def create_datasets(bts):
    trainset = CNNDataset(length=lenghtr, prefix = 'Idlt-',
                                           root_dir='/scratch/zxs/',
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
    validateset = CNNDataset(length=lenghva, prefix = 'Idlv-',
                                           root_dir='/scratch/zxs/',
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))
    testset = CNNDataset(length=lenghte, prefix = 'Idlp-',
                                           root_dir='/scratch/zxs/',
                                           transform=transforms.Compose([
                                               ToTensor()
                                           ]))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validateset, batch_size=8,
                                          shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False)

    return train_loader, valid_loader, test_loader

# network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 32, 5, stride=2)
        self.batch31 = nn.BatchNorm3d(32, affine=False)
        self.conv2 = nn.Conv3d(32, 64, 5, stride=1)
        self.batch32 = nn.BatchNorm3d(64, affine=False)
        self.conv3 = nn.Conv3d(64, 128, 5, stride=1)
        self.batch33 = nn.BatchNorm3d(128, affine=False)

        self.batch11 = nn.BatchNorm1d(64, affine=False)
        self.batch12 = nn.BatchNorm1d(16, affine=False)
        self.batch13 = nn.BatchNorm1d(4, affine=False)

        self.fc1 = nn.Linear(83968, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)
        self.fc4 = nn.Linear(4, 2)

    def forward(self, input):
        x = self.pool(F.relu(self.batch31(self.conv1(input))))
        x = F.pad(x,(1,1,1,1,1,1),"constant", 0)

        x = self.pool(F.relu(self.batch32(self.conv2(x))))
        x = F.pad(x,(1,1,1,1,1,1),"constant", 0)
        x = F.dropout3d(x,p=0.4, training=True, inplace=False)

        x = self.pool(F.relu(self.batch33(self.conv3(x))))
        x = F.pad(x,(1,1,1,1,1,1),"constant", 0)
        x = F.dropout3d(x,p=0.4, training=True, inplace=False)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.batch11(self.fc1(x)))
        x = F.dropout(x,p=0.4, training=True, inplace=False)
        x = F.relu(self.batch12(self.fc2(x)))
        x = F.relu(self.batch13(self.fc3(x)))
        x = self.fc4(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# specify loss function & optimizer & device
#criterion = nn.MSELoss()
criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=l_patience, factor=0.2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #net = nn.DataParallel(net,device_ids=[0,1,2,3,4,5,6,7])
    #net = nn.DataParallel(net,device_ids=[0])
    net = nn.DataParallel(net)

net.to(device)


# model
def train_model(net, patience, n_epochs, save_early_path):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):  # loop over the dataset multiple times
        net.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, para = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, para)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            ######################
            # validate the model #
            ######################
        net.eval()
        with torch.no_grad():
            for val in valid_loader:
                inputsv, parav = val[0].to(device), val[1].to(device)
                outputsv = net(inputsv)
                loss = criterion(outputsv, parav)
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
            
        scheduler.step(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg, flush=True)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        state = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss
                }
        early_stopping(valid_loss, state, save_early_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    checkpoint = torch.load(save_early_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return  net, avg_train_losses, avg_valid_losses


train_loader, valid_loader, test_loader = create_datasets(batch_size)

start_time = time.time()
net, train_loss, valid_loss = train_model(net, patience, n_epochs, save_early_path)
end_time = time.time()
print('Done in {0:.6f} s'.format(end_time - start_time))

train_loss = np.array(train_loss)
valid_loss = np.array(valid_loss)
np.save(save_train_loss_path, train_loss)
np.save(save_valid_loss_path, valid_loss)

net.to(device)
net.eval()

true = np.zeros((lenghte,2))
pred = np.zeros((lenghte,2))
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images).to("cpu", torch.double).numpy()
        pred[i,0] = outputs[0][0]
        pred[i,1] = outputs[0][1]
        true[i,0] = labels[0][0].to("cpu", torch.double).numpy()
        true[i,1] = labels[0][1].to("cpu", torch.double).numpy()
np.save(savep1_path,pred)
np.save(savep2_path,true)

#coefficient of determination
coefficient1 = metrics.r2_score(true[:,0],pred[:,0])
print ("coefficient of determination of Tvir(on test data): %0.5f" % coefficient1)
coefficient2 = metrics.r2_score(true[:,1],pred[:,1])
print ("coefficient of determination of Zeta(on test data): %0.5f" % coefficient2)
