import numpy as np
from datasets import FID300, TripletFID
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys

# sys.path.append("../data/")
data_path = "../data/FID-300/"
trainset = FID300(data_path,transform=transforms.Compose([transforms.Resize((256,128)),
                                                                      transforms.ToTensor()
                                                                      ]))
triplet_dataset = TripletFID(trainset)
trainset_loader = DataLoader(triplet_dataset, batch_size=4, shuffle=True, num_workers=1)
# print(len(trainset))
# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(trainset_loader)
# images, refs, labels = dataiter.next()
(anchor, pos, neg),_ =  dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# plt.show()
# imshow(torchvision.utils.make_grid(refs))
# plt.show()
# print labels
# print(' '.join('%5s' % labels[j] for j in range(4)))

# show images
imshow(torchvision.utils.make_grid(anchor))
plt.show()
imshow(torchvision.utils.make_grid(pos))
plt.show()
imshow(torchvision.utils.make_grid(neg))
plt.show()