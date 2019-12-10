import numpy as np
import torch
from datasets import FID300, TripletFID
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from networks import *
from siamese_fid300 import extract_embeddings
import os
import argparse
from scipy.io import loadmat

cuda = torch.cuda.is_available()
# mean, std = 0.1307, 0.3081
#data_path = r"/content/drive/My Drive/701_project/FID-300/"


def get_feature_vecs(data_path, checkpoint_path, network_name, layer_id, transform_size=(224,112)):
    checkpoint = torch.load(checkpoint_path)
    if(network_name == "resnet18"):
        model = TripletNet(EmbeddingNet_ResNet18(layer_id))
    else:
        model = TripletNet(EmbeddingNet(network_name, layer_id))
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    if cuda:
        model.cuda()
    batch_size = 32
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    transform = transforms.Compose([transforms.Resize(transform_size), transforms.ToTensor(), 
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    # train_probe_dataset = FID300(data_path, get_probe=True, train=True, transform=transform)
    # val_probe_dataset = FID300(data_path, get_probe=True, train=False, transform=transform)
    # ref_dataset = FID300(data_path, get_probe=False, transform=transform)

    train_probe_dataset = FID300(data_path, get_probe=True, train=True, transform=[transform, transform])
    val_probe_dataset = FID300(data_path, get_probe=True, train=False, transform=transform)
    ref_dataset = FID300(data_path, get_probe=False, train=False, transform=transform)

    train_probe_loader = torch.utils.data.DataLoader(train_probe_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    val_probe_loader = torch.utils.data.DataLoader(val_probe_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    ref_loader = torch.utils.data.DataLoader(ref_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    train_embeddings_probe  = extract_embeddings(train_probe_loader, model, if_probe=True)
    val_embeddings_probe  = extract_embeddings(val_probe_loader, model, if_probe=True)
    ref_embeddings = extract_embeddings(ref_loader, model, if_probe= False)
    return ref_embeddings, val_embeddings_probe, train_embeddings_probe

def get_scoresort_divided(l2_dist_vec):
    data_path = "../data/FID-300/"
    divided_data_path = os.path.join(data_path, "divided")

    label_file = os.path.join(data_path,'label_table_train.csv')
    label_map = np.loadtxt(label_file,delimiter=',',dtype='int')
    label_table = label_map[:, 1]
    
    label_file = os.path.join(divided_data_path,'label_table_train.csv')
    div_label_map = np.loadtxt(label_file,delimiter=',',dtype='int')

    div_lhs = div_label_map[:, 0]
    lhs = label_map[:, 0]

    scores = np.zeros((len(label_map), 1175))

    # DIstance from upper/Lower shoe features
    du = l2_dist_vec.reshape(-1, int(l2_dist_vec.shape[1]/2), 2)[:, :, 0]
    dl = l2_dist_vec.reshape(-1, int(l2_dist_vec.shape[1]/2), 2)[:, :, 1]

    count = 0
    for i, l in enumerate(lhs):
        if 2*l not in div_lhs:
            scores[i] = du[count]
            count+=1
        else:
            scores[i] = np.maximum(du[count], dl[count+1])
    score_sort = scores.argsort(1)
    return score_sort, label_table


def find_scores(ref_vec_list, test_vec_list, label_table, divided=False):
    l2_dist_vec = [] # list of l2_distance for each test image
    for test_vec in test_vec_list:
        dist_from_ref = np.linalg.norm(ref_vec_list-test_vec, axis=1)**2 # as in loss, using sq distance
        l2_dist_vec.append(dist_from_ref)
    l2_dist_vec = np.stack(l2_dist_vec)
    if divided:
        score_sort, label_table = get_scoresort_divided(l2_dist_vec)
    else:
        # label_table -= 1 # commented as now handled in dataset.py
        score_sort = l2_dist_vec.argsort(1)
        # print("range label_table:", min(label_table), max(label_table))
        # print("range score_sort:", min(score_sort.ravel()), max(score_sort.ravel()))

    # print("score_sort.shape = {}".format(score_sort.shape))
    # print("label_table.shape = {}".format(label_table.shape))
    # print("label_table = {}".format(label_table))
    pos_array = []
    for a, c in zip(score_sort, label_table):
        pos_array.append(np.where(a==c)[0][0])
        # print("match_id ,c", np.where(a==c)[0][0], c)
    pos_array = np.array(pos_array)

    t = 5
    thresh = t*len(ref_vec_list)/100 #pos_array.shape[0]/100
    acc = 100*np.sum((pos_array<thresh))/pos_array.shape[0]
    print("top {}%: {}".format(t, acc))

    retreival_5_inds = np.where(pos_array < thresh)[0]
    score_5 = score_sort[retreival_5_inds][:,:int(thresh)]
    gts_correct = label_table[retreival_5_inds]
    score_5 = np.insert(score_5, 0, gts_correct, axis=1)# now first element is gt 

    t = 10
    thresh = t*len(ref_vec_list)/100  #pos_array.shape[0]/100
    acc = 100*np.sum((pos_array<thresh))/pos_array.shape[0]
    print("top {}%: {}".format(t, acc))

    retreival_10_inds = np.where(pos_array < thresh)[0]
    # print("score_sort ", score_sort.shape)
    score_10 = score_sort[retreival_10_inds][:,:int(thresh)]
    gts_correct = label_table[retreival_5_inds][1]
    score_10 = np.insert(score_10, 0, gts_correct, axis=1)# now first element is gt 

    # print("score_10 ", score_10.shape)
    return retreival_5_inds, retreival_10_inds, score_sort

def add_gt_info(scores, label_map, retreival_inds, thresh):
    score_thresh = scores[retreival_inds][:,:int(thresh)]
    ids = label_map[retreival_inds, 0]
    gts = label_map[retreival_inds, 1]
    score_thresh = np.insert(score_thresh, 0, gts, axis=1)# now first element is gt 
    score_thresh = np.insert(score_thresh, 0, ids, axis=1)
    return score_thresh


def save_results(data_path, label_fname, scores, reference_embeddings, retreival_5_inds, retreival_10_inds, network_name, epoch, split):
    label_file = os.path.join(data_path,label_fname)
    label_map = np.loadtxt(label_file,delimiter=',',dtype='int')
    thresh = 5*len(reference_embeddings)/100
    ranked_matches_5 = add_gt_info(scores, label_map, retreival_5_inds, thresh)
    thresh = 10*len(reference_embeddings)/100
    ranked_matches_10 = add_gt_info(scores, label_map, retreival_10_inds, thresh)
    np.savetxt('../results/{}_inf_5_{}_{}.txt'.format(split, network_name,epoch), ranked_matches_5.astype(int), fmt='%i', delimiter=',')
    np.savetxt('../results/{}_inf_10_{}_{}.txt'.format(split, network_name,epoch), ranked_matches_10.astype(int), fmt='%i', delimiter=',')
    pass

if __name__ == "__main__":
    # data_path = "../data/FID-300/"

    # network_name = "resnet50"
    # layer_id = 5
    # checkpoint_path = "../checkpoints_resnet50_labelfix/" 
    # # checkpoint_path = "../checkpoints_resnet50/"
    # # network_name = "resnet18"
    # # layer_id = 6
    # # checkpoint_path = "../checkpoints_resnet18/"
    # epoch = 50

    parser = argparse.ArgumentParser()
    parser.add_argument('--divided', dest='divided', action='store_true')
    args = parser.parse_args()
    if args.divided:
        data_path = "../data/FID-300/divided"
        transform_size=(224, 224)
    else:
        data_path = "../data/FID-300/"
        transform_size=(224, 112)

    network_name = "resnet50"
    layer_id = 5
    checkpoint_path = "../checkpoints_resnet50/"
    # network_name = "resnet18"
    # layer_id = 6
    # checkpoint_path = "../checkpoints_resnet18/"
    epoch = 20
    
    checkpoint_file = os.path.join(checkpoint_path, "epoch_{}.pt".format(epoch))
    
    reference_embeddings, val_embeddings, train_embeddings = get_feature_vecs(data_path, checkpoint_file,
                                                            network_name, layer_id, transform_size=transform_size)
    val_embeddings, val_labels = val_embeddings # dataloader is handling index 
    train_embeddings, train_labels = train_embeddings

    # print("train_labels = {}".format(train_labels))

    print("training data results:")
    retreival_5_inds, retreival_10_inds, scores = find_scores(reference_embeddings,
        train_embeddings, train_labels, divided=args.divided)
    
    # label_fname = 'label_table_train.csv'
    # save_results(data_path, label_fname, scores, reference_embeddings, retreival_5_inds, retreival_10_inds, network_name, epoch, 'train')
    
    print("validation data results:")
    retreival_5_inds, retreival_10_inds, scores = find_scores(reference_embeddings,
        val_embeddings, val_labels, divided=args.divided)
    
    # label_fname = 'label_table_val.csv'
    # save_results(data_path, label_fname, scores, reference_embeddings, retreival_5_inds, retreival_10_inds, network_name, epoch, 'val')
