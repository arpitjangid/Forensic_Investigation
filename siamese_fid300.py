#Reference code: https://github.com/adambielski/siamese-triplet/blob/master/Experiments_MNIST.ipynb

import numpy as np
import torch
import os
from datasets import FID300, TripletFID
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim

from torch.autograd import Variable
from trainer import fit
from networks import * 
from losses import TripletLoss
import argparse

cuda = torch.cuda.is_available()

def extract_embeddings(dataloader, model, if_probe=False, ncc=False):
    with torch.no_grad():
        model.eval()
        if not ncc:
            embeddings = np.zeros((len(dataloader.dataset), 128))
        else:
            embeddings = np.zeros((len(dataloader.dataset), 256,56,28))
        if if_probe:
          labels = np.zeros(len(dataloader.dataset))
          k = 0
          for images, target in dataloader:
              if cuda:
                  images = images.cuda()
              embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy() 
              labels[k:k+len(images)] = target.numpy()
              k += len(images)
          return embeddings, labels
        else:
          k = 0
          for images in dataloader:
              if cuda:
                  images = images.cuda()
              embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy() 
              k += len(images)
          return embeddings

# def get_transforms(mean, std, transform_method=1):
#     if(transform_method == 1): # same transform for reference and probe images
#         transform_val = transforms.Compose([transforms.Resize((224,112)), transforms.ToTensor(), 
#                     transforms.Normalize(mean, std)]) # (224,112)
#         transform_train = transforms.Compose([transforms.Resize((224,112)), transforms.RandomHorizontalFlip(),
#                         transforms.ToTensor(), transforms.Normalize(mean, std)])
def get_transforms(mean, std, transform_method=1, transform_size=(224, 112)):
    if(transform_method == 1):
        transform_val = transforms.Compose([transforms.Resize(transform_size), transforms.ToTensor(), 
                    transforms.Normalize(mean, std)])
        transform_train = transforms.Compose([transforms.Resize(transform_size), 
                        transforms.ToTensor(), transforms.Normalize(mean, std)]) # transforms.RandomHorizontalFlip(),
        return transform_val, transform_train, transform_train
    else:
        transform_val = transforms.Compose([transforms.Resize(x), transforms.ToTensor(), transforms.Normalize(mean, std)])
        transform_train_ref = transforms.Compose([transforms.Resize((256,128)), transforms.RandomCrop((224,112)), 
                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
        transform_train_probe = transforms.Compose([transforms.Resize((256,128)), transforms.RandomRotation((-20, 20)), transforms.RandomCrop((224, 112)), 
                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    
        return transform_val, transform_train_ref, transform_train_probe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--divided', dest='divided', action='store_true')
    parser.add_argument('--ncc', dest='ncc', action='store_true')
    args = parser.parse_args()

    if args.divided:
        data_path = "../data/FID-300/divided"
    else:
        data_path = "../data/FID-300/"
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform_method = 1
    if args.divided:
        transform_size = (224, 224)
    else:
        transform_size = (224, 112)
    transform_val, transform_train_ref, transform_train_probe = get_transforms(mean, std,
        transform_method=transform_method, transform_size=transform_size)
   
    train_dataset = FID300(data_path, train = True, transform = [transform_train_ref, transform_train_probe], with_aug= False)
    val_dataset = FID300(data_path, train = False, transform = transform_val)

    triplet_train_dataset = TripletFID(train_dataset) # Returns pairs of images and target same/different
    triplet_val_dataset = TripletFID(val_dataset) #same as val
    batch_size = 16 #128

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_val_loader = torch.utils.data.DataLoader(triplet_val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    
    margin = 0.3
    # layer_id = 6
    # embedding_net = EmbeddingNet_ResNet18(layer_id)
    layer_id = 7 #5
    network_name='resnet50'
    # network_name='vgg19'
    
    embedding_net = EmbeddingNet(network_name=network_name, layer_id=layer_id, ncc=args.ncc)
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin, ncc=args.ncc)
    lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # optimizer = optim.Adam([
    #             {'params': model.embedding_net.net_base.parameters()},
    #             {'params': model.embedding_net.fc.parameters(), 'lr': lr*10}
    #         ], lr=lr)
    # lr = 1e-5
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    scheduler = lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=-1)
    n_epochs = 100
    log_interval = 100
    checkpoint_path = "../checkpoints_resnet50/"
    # checkpoint_path = "../checkpoints_vgg19/"
    if(not os.path.exists(checkpoint_path)):
        os.makedirs(checkpoint_path)
    save_freq = 10
    plot_name = "../loss_curves/resnet50.png"
    # plot_name = "../loss_curves/loss_vgg19.png"

    ### if(resume_tranining):
    # checkpoint = torch.load(checkpoint_path+"epoch_20.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    #checkpoint = torch.load(checkpoint_path+"epoch_20.pt")
    #model.load_state_dict(checkpoint['model_state_dict'])

    fit(triplet_train_loader, triplet_val_loader, model, loss_fn, optimizer, 
                    scheduler, n_epochs, cuda, log_interval, checkpoint_path, save_freq, plot_name)

