import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EmbeddingNet_ResNet18(nn.Module):
    def __init__(self, layer_id=6):
        super(EmbeddingNet_ResNet18, self).__init__()
        # resnet_base = models.resnet18(pretrained=True)
        # resnet_base.fc = nn.Linear(, 128)
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:layer_id] 
        self.resnet_base = nn.Sequential(*modules)
        if(layer_id == 6):
            self.fc = nn.Linear(128*28*14, 128)
            nn.init.xavier_uniform(self.fc.weight)
        elif(layer_id == 7):
            self.fc = nn.Linear(256*14*7, 128)
            nn.init.xavier_uniform(self.fc.weight)

        # modules = list(resnet18.children())[:8] 
        # self.resnet_base = nn.Sequential(*modules)
        # self.fc = nn.Linear(512*7*4, 128)
        
        # self.fc = nn.Sequential(nn.Linear(128*28*14, 1024),
        #                         nn.ReLU(),
        #                         nn.Linear(1024, 128))
        
        # modules = list(resnet18.children())[:7] 
        # self.resnet_base = nn.Sequential(*modules)
        # self.fc = nn.Linear(256*14*7, 128)
        
        # self.fc = nn.Sequential(nn.Linear(256*14*7, 1024),
        #                         nn.ReLU(),
        #                         nn.Linear(1024, 128))
        # for input size 224x112, feature map is 128x28x14 = 50176 

        for param in self.resnet_base.parameters():
            param.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True
        # for param in self.resnet_base.fc.parameters():
        #     param.requires_grad = True
        

    def forward(self, x):
        # output = self.resnet_base(x)
        x = self.resnet_base(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EmbeddingNet(nn.Module):
    def __init__(self, network_name='resnet50', layer_id=5, ncc=False):
        super(EmbeddingNet, self).__init__()
        self.ncc = ncc
        # self.resnet_base = models.resnet50(pretrained=True)
        # self.resnet_base.fc = nn.Linear(512, 128)
        if(network_name == "resnet50"):
            # print("layer_id = {}".format(layer_id))
            model = models.resnet50(pretrained=True)
            modules = list(model.children())[:layer_id] 
            self.net_base = nn.Sequential(*modules)
            # print("layer_id = {}".format(layer_id))
            if(layer_id == 5):
                self.fc = nn.Linear(256*56*28, 128)
            elif(layer_id == 6):
                self.fc = nn.Linear(512*28*14, 128)
            elif(layer_id == 7):
                self.fc = nn.Linear(1024*14*7, 128)
                # self.fc = nn.Sequential(nn.Linear(256*56*28, 1024),
                #                     nn.ReLU(),
                #                     nn.Linear(1024, 128))
            elif(layer_id == 6):
                # pass
                self.fc = nn.Linear(512*28*14, 128)
                nn.init.xavier_uniform(self.fc.weight)
            elif(layer_id == 7):
                # pass
                self.fc = nn.Linear(1024*14*7, 128)
                nn.init.xavier_uniform(self.fc.weight)
            else:
                print("layer_id not supported")
                exit()
            # self.fc = nn.Sequential(nn.Linear(256*28*14, 1024),
            #                         nn.ReLU(),
            #                         nn.Linear(1024, 128))
        elif(network_name == "vgg19"):
            model = models.vgg19(pretrained=True)
            modules = list(model.children())[:-1] 
            self.net_base = nn.Sequential(*modules)
            # self.net_base = nn.ModuleList(list(model.features[:18]))
            # self.net_base = nn.Sequential(*modules)
            # self.fc = nn.Linear(256*56*28, 128)
            self.fc = nn.Linear(512*7*7, 128)
            # self.fc = nn.Sequential(nn.Linear(128*28*14, 1024),
            #                        nn.ReLU(),
            #                        nn.Linear(1024, 128))

        for param in self.net_base.parameters():
            param.requires_grad = True #False
        for param in self.fc.parameters():
            param.requires_grad = True
        for param in self.net_base[-1][-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        # output = self.resnet_base(x)
        x = self.net_base(x)
        if not self.ncc:
            x = x.view(x.size(0), -1)
            output = self.fc(x)
        else:
            output = x
        return output

    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

# class TripletNet_MPCNN(nn.Module):
#     def __init__(self, embedding_net):
#         super(TripletNet, self).__init__()
#         self.embedding_net = embedding_net

#     def get_part_weights(self, x1):
#         pass
#     def forward(self, x1, x2, x3):
#         W = self.get_part_weights(x1) # shape?

#         output1 = self.embedding_net_MPCNN(x1, W)
#         output2 = self.embedding_net_MPCNN(x2, W)
#         output3 = self.embedding_net_MPCNN(x3, W)
#         return output1, output2, output3

#     def get_embedding(self, x):
#         return self.embedding_net(x)
