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

cuda = torch.cuda.is_available()

def extract_embeddings(dataloader, model, if_probe=False):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 128))
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

def get_transforms(transform_method=1):
    if(transform_method == 1):
        transform_val = transforms.Compose([transforms.Resize((224,112)), transforms.ToTensor(), 
                    transforms.Normalize(mean, std)])
        transform_train = transforms.Compose([transforms.Resize((224,112)), transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)])
        return transform_val, transform_train, transform_train
    else:
        transform_val = transforms.Compose([transforms.Resize((224,112)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        transform_train_ref = transforms.Compose([transforms.Resize((256,128)), transforms.RandomCrop((224,112)), 
                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
        transform_train_probe = transforms.Compose([transforms.Resize((256,128)), transforms.RandomRotation((-20, 20)), transforms.RandomCrop((224,112)), 
                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    
        return transform_val, transform_train_ref, transform_train_probe

if __name__ == "__main__":
    data_path = "../data/FID-300/"
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    transform_method = 1
    transform_val, transform_train_ref, transform_train_probe = get_transforms(transform_method=transform_method)
   
    
    #  transforms.RandomRotation(degrees=(-20,20)),
    # train_dataset = FID300(data_path, train = True, transform = transform_train, with_aug= False)
    # val_dataset = FID300(data_path, train = False, transform = transform_val)
    ## add different transforms for reference and croo images
    train_dataset = FID300(data_path, train = True, transform = [transform_train_ref, transform_train_probe], with_aug= False)
    val_dataset = FID300(data_path, train = False, transform = transform_val)

    triplet_train_dataset = TripletFID(train_dataset) # Returns pairs of images and target same/different
    triplet_val_dataset = TripletFID(val_dataset) #same as val
    batch_size = 16 #128

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=4, shuffle=True, **kwargs)
    triplet_val_loader = torch.utils.data.DataLoader(triplet_val_dataset, batch_size=4, shuffle=False, **kwargs)

    
    margin = 0.3 #1.
    # layer_id = 6
    # embedding_net = EmbeddingNet_ResNet18(layer_id)
    layer_id = 5
    network_name='resnet50'
    # network_name='vgg19'
    embedding_net = EmbeddingNet(network_name=network_name, layer_id=5)
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.Adam([
    #             {'params': model.embedding_net.net_base.parameters()},
    #             {'params': model.embedding_net.fc.parameters(), 'lr': lr*10}
    #         ], lr=lr)
    # optimizer = optim.SGD([
    #             {'params': model.embedding_net.net_base.parameters()},
    #             {'params': model.embedding_net.fc.parameters(), 'lr': lr*10}
    #         ], lr=lr, momentum=0.9)
    
    scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
    n_epochs = 100
    log_interval = 100
    checkpoint_path = "../checkpoints_resnet50_labelfix/"
    # checkpoint_path = "../checkpoints_vgg19/"
    if(not os.path.exists(checkpoint_path)):
        os.makedirs(checkpoint_path)
    save_freq = 10
    plot_name = "../loss_curves/resnet50_transform.png"
    # plot_name = "../loss_curves/loss_vgg19.png"
    # if(resume_tranining):
    #     checkpoint = torch.load(checkpoint_path+)
    #     model.load_state_dict(checkpoint['model_state_dict'])

    fit(triplet_train_loader, triplet_val_loader, model, loss_fn, optimizer, 
                    scheduler, n_epochs, cuda, log_interval, checkpoint_path, save_freq, plot_name)

