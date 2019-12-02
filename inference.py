import numpy as np
import torch
from datasets import FID300, TripletFID
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from networks import *
from siamese_fid300 import extract_embeddings
import os

cuda = torch.cuda.is_available()
# mean, std = 0.1307, 0.3081
#data_path = r"/content/drive/My Drive/701_project/FID-300/"


def get_feature_vecs(checkpoint_path, network_name, layer_id):
    checkpoint = torch.load(checkpoint_path)
    if(network_name == "resnet18"):
        model = TripletNet(EmbeddingNet_ResNet18(layer_id))
    else:
        model = TripletNet(EmbeddingNet(network_name, layer_id))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if cuda:
        model.cuda()
    batch_size = 32
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    transform = transforms.Compose([transforms.Resize((224,112)), transforms.ToTensor(), 
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    train_probe_dataset = FID300(data_path, get_probe=True, train=True, transform=transform)
    val_probe_dataset = FID300(data_path, get_probe=True, train=False, transform=transform)
    ref_dataset = FID300(data_path, get_probe=False, transform=transform)

    train_probe_loader = torch.utils.data.DataLoader(train_probe_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    val_probe_loader = torch.utils.data.DataLoader(val_probe_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    ref_loader = torch.utils.data.DataLoader(ref_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    train_embeddings_probe  = extract_embeddings(train_probe_loader, model, if_probe=True)
    val_embeddings_probe  = extract_embeddings(val_probe_loader, model, if_probe=True)
    ref_embeddings = extract_embeddings(ref_loader, model, if_probe= False)
    return ref_embeddings, val_embeddings_probe, train_embeddings_probe

def find_scores(ref_vec_list, test_vec_list, label_table):
  
    test_vec_list = (test_vec_list-test_vec_list.mean(1)[:, None])/test_vec_list.std(1)[:, None]
    ref_vec_list = (ref_vec_list-ref_vec_list.mean(1)[:, None])/ref_vec_list.std(1)[:, None]
    score = cosine_similarity(test_vec_list, ref_vec_list)
    score_sort = (-score).argsort(1)

    pos_array = []
    for a,b,c in zip(score_sort, score.argmax(1), label_table):
        pos_array.append(np.where(a==c)[0][0])
    pos_array = np.array(pos_array)

    t = 5
    thresh = t*len(ref_vec_list)/100 #pos_array.shape[0]/100
    # thresh = t
    acc = 100*np.sum((pos_array<thresh))/pos_array.shape[0]
    print("top {}%: {}".format(t, acc))

    t = 10
    thresh = t*len(ref_vec_list)/100  #pos_array.shape[0]/100
    # thresh = t
    acc = 100*np.sum((pos_array<thresh))/pos_array.shape[0]
    print("top {}%: {}".format(t, acc))
    return score, score_sort


if __name__ == "__main__":
    data_path = "../data/FID-300/"

    # network_name = "resnet50"
    network_name = "resnet18"
    epoch = 80
    layer_id = 6
    # checkpoint_path = "../checkpoints_full_resnet50_layerid5/"
    # checkpoint_path = "../checkpoints_full_resnet50_aug/"
    checkpoint_path = "../checkpoints_full_resnet18_fixed/"
    checkpoint_file = os.path.join(checkpoint_path, "epoch_{}.pt".format(epoch))
    # checkpoint_file = "../checkpoints_fulltrain_layer7/resnet_18_epoch_{}.pt".format(epoch)
    # checkpoint_file = "../checkpoints/resnet_18_epoch_{}.pt".format(epoch)
    
    reference_embeddings, val_embeddings, train_embeddings = get_feature_vecs(checkpoint_file,
                                                            network_name, layer_id)
    val_embeddings, val_labels = val_embeddings
    train_embeddings, train_labels = train_embeddings

    score, ranked_matches = find_scores(reference_embeddings, train_embeddings, train_labels)
    print("training data scores: ", score)
    
    score, ranked_matches = find_scores(reference_embeddings, val_embeddings, val_labels)
    print("validation data scores: ", score)

    print("validation data, ranked matches:", ranked_matches)