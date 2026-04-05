import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time
from dataset import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix, average_precision_score
from DiffProt import *
from utils import *
from sklearn.model_selection import train_test_split
import random
import logging

import pickle
import os



def train(model, g, args):
    features = g.ndata['feature']
    labels = g.ndata['label']
    index = list(range(len(labels)))
    if dataset_name == 'amazon':
        index = list(range(3305, len(index)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1

    features = features.to(args.device)
    labels = labels.to(args.device)
    g = g.to(args.device)

    num_classes = int(labels.max().item()) + 1

    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
    best_auc = 0.
    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()

    data_indices = idx_train

    for e in range(args.epoch):
        model.train()
        logits, emb, min_distances = model(features, g)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]).to(args.device))
        
        if args.homo == 1:
            #cluster loss
            prototypes_of_correct_class = torch.t(model.prototype_class_identity[:, labels[train_mask]].bool()).to(args.device)
            min_distances_train = min_distances[train_mask]
            cluster_cost = torch.mean(torch.min(min_distances_train[prototypes_of_correct_class].reshape(-1, args.num_prototypes_per_class), dim=1)[0])

            #seperation loss
            separation_cost = -torch.mean(torch.min(min_distances_train[~prototypes_of_correct_class].reshape(-1, (num_classes-1)*args.num_prototypes_per_class), dim=1)[0])

            #diversity loss
            ld = 0
            for k in range(num_classes):
                p = model.prototypes[k*args.num_prototypes_per_class: (k+1)*args.num_prototypes_per_class]
                p = F.normalize(p, p=2, dim=1)
                matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(args.device) - 0.3
                matrix2 = torch.zeros(matrix1.shape).to(args.device)
                ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))

            with torch.no_grad():
                proto_idx = torch.argmin(min_distances, dim=1)
                proto_assignments = model.prototype_labels[proto_idx]

            recon_node, lim_node = model.train_diffusion_step(g, emb.detach(), proto_assignments)

            proto_loss, struct_loss = model.prototype_loss(emb.detach())

            loss += ((args.clt_weight*cluster_cost) + 
                    (args.sep_weight*separation_cost) + 
                    (args.ld_weight * ld) +
                    (args.lambda_1 * (recon_node + lim_node)) +
                    (args.lambda_2 * (proto_loss + args.lambda_con * struct_loss))
                    )
        else:
            prototypes_of_correct_class = torch.t(model.prototype_class_identity[:, labels[train_mask]].bool()).to(args.device)
            min_distances_train = min_distances[train_mask]
            num_correct_p = model.num_rels * model.num_prototypes_per_class
            num_incorrect_p = model.num_prototypes - num_correct_p

            # Cluster cost
            cluster_cost = torch.mean(torch.min(min_distances_train[prototypes_of_correct_class].reshape(-1, num_correct_p), dim=1)[0])

            # Separation cost
            if num_incorrect_p > 0:
                separation_cost = -torch.mean(torch.min(min_distances_train[~prototypes_of_correct_class].reshape(-1, num_incorrect_p), dim=1)[0])
            else:
                separation_cost = torch.tensor(0.0).to(args.device)


            ld = 0
            for i in range(model.num_rels):
                for k in range(model.output_dim):
                    start_idx = k * model.num_prototypes_per_class
                    end_idx = (k + 1) * model.num_prototypes_per_class
                    
                    rel_offset = i * model.num_prototypes_per_rel
                    global_start = rel_offset + start_idx
                    global_end = rel_offset + end_idx
                    p = model.prototypes[global_start:global_end]
                    
                    p = F.normalize(p, p=2, dim=1)
                    matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(args.device) - 0.3
                    matrix2 = torch.zeros(matrix1.shape).to(args.device)
                    ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))


            with torch.no_grad():
                proto_idx = torch.argmin(min_distances, dim=1)
                proto_assignments = model.prototype_labels[proto_idx]

            recon_node, lim_node = model.train_diffusion_step(g, emb.detach(), proto_assignments)

            proto_loss, struct_loss = model.prototype_loss(emb.detach())

            loss += ((args.clt_weight*cluster_cost) + 
                    (args.sep_weight*separation_cost) + 
                    (args.ld_weight * ld) +
                    (args.lambda_1 * (recon_node + lim_node)) +
                    (args.lambda_2 * (proto_loss + args.lambda_con * struct_loss))
                    )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        vauc = roc_auc_score(labels.cpu()[val_mask], probs.cpu()[val_mask][:, 1].detach().numpy())
        preds = numpy.zeros_like(labels.cpu())
        preds[probs.cpu()[:, 1] > thres] = 1
        trec = recall_score(labels.cpu()[test_mask], preds[test_mask])
        tpre = precision_score(labels.cpu()[test_mask], preds[test_mask])
        tmf1 = f1_score(labels.cpu()[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(labels.cpu()[test_mask], probs.cpu()[test_mask][:, 1].detach().numpy())
        tap = average_precision_score(labels.cpu()[test_mask], probs.cpu()[test_mask][:, 1].detach().numpy())

        if best_f1 < f1:
            best_f1 = f1
            best_auc = vauc
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            final_tap = tap
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, AUC: {:.4f}'.format(e, loss, f1, vauc))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} AP {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100, final_tap*100))
    
    logging.info('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} AP {:.2f}'.format(final_trec*100,
                                                                     final_tpre*100, final_tmf1*100, final_tauc*100, final_tap*100))
    
    if args.homo==1:
        model.eval()
        with torch.no_grad():
            logits, _, distances = model(features, g)
            probs = torch.softmax(logits, dim=1) # (N, num_classes)
            y_pred = logits.argmax(dim=1)        # (N,)
        
        num_nodes_total = features[test_mask].shape[0]
        k_nodes = int(math.ceil(num_nodes_total * args.sparsity))
        
        print(f"\n[Sparsity Target: {args.sparsity*100:.2f}% | k_nodes={k_nodes}]")
        
        epoch_fid_plus = []
        epoch_fid_minus = []
        
        for target_class in range(model.output_dim):
            target_eval_mask = test_mask & (y_pred.cpu() == target_class)
            if target_eval_mask.sum() == 0:
                continue
            Hnodes = extract_key_nodes_for_class(model, distances, target_class, k=k_nodes)
            
            if Hnodes.numel() == 0:
                continue
            src, dst = g.edges()
            mask_src_in = torch.isin(src, Hnodes)
            mask_dst_in = torch.isin(dst, Hnodes)
            Hedges_mask = mask_src_in & mask_dst_in
            Hedges_eids = Hedges_mask.nonzero(as_tuple=True)[0] 
            Rem_edges_eids = (~Hedges_mask).nonzero(as_tuple=True)[0]

            if Hedges_eids.numel() > 0:
                g_expl = dgl.edge_subgraph(g, Hedges_eids, relabel_nodes=False)
            else:
                g_expl = dgl.graph(([], []), num_nodes=g.num_nodes()).to(g.device)
            
            if Rem_edges_eids.numel() > 0:
                g_rem = dgl.edge_subgraph(g, Rem_edges_eids, relabel_nodes=False)
            else:
                g_rem = dgl.graph(([], []), num_nodes=g.num_nodes()).to(g.device)

            features_expl = features.clone()
            mask_not_hnodes = torch.ones(g.num_nodes(), dtype=torch.bool, device=features.device)
            mask_not_hnodes[Hnodes] = False
            features_expl[mask_not_hnodes] = 0.0 
            features_rem = features.clone()
            features_rem[Hnodes] = 0.0

            with torch.no_grad():
                logits_expl, _, _ = model(features_expl, g_expl)
                logits_rem, _, _ = model(features_rem, g_rem)

            probs_expl = torch.softmax(logits_expl, dim=1)
            fid_p = probs_expl[target_eval_mask, target_class].mean().item()
            epoch_fid_plus.append(fid_p)
            
            probs_rem = torch.softmax(logits_rem, dim=1)
            fid_m = probs_rem[target_eval_mask, target_class].mean().item()
            epoch_fid_minus.append(fid_m)
        
        if len(epoch_fid_plus) > 0:
            avg_fid_plus = np.mean(epoch_fid_plus)
            avg_fid_minus = np.mean(epoch_fid_minus)
            combined_score = avg_fid_plus - avg_fid_minus
            print(f"Average fid+: {avg_fid_plus:.4f}")
            print(f"Average fid-: {avg_fid_minus:.4f}")
            print(f"fid+ - fid-: {combined_score:.4f}")
        
        else:
            print(f"  Warning: No valid samples found for sparsity {node_sparsity}")

    return model, final_tmf1, final_tauc


# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    probs = probs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="Dataset for this model (amazon/yelp/tfinance/spoofing)")
    args = parser.parse_args()
    
    dataset_name = args.dataset

    cfg_path =f'./configs/{dataset_name}.json'
    cfg_dir = os.path.dirname(cfg_path) if os.path.dirname(cfg_path) != '' else cfg_path
    per_ds_path = os.path.join(cfg_dir, f"{dataset_name}.json")
    ds_cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, 'r') as f:
            ds_cfg = json.load(f)
    else:
        print(f"Config file not found: {cfg_path}")
        exit(1)

    for k, v in ds_cfg.items():
        setattr(args, k, v)

    print(args)

    homo = args.homo
    order = args.order
    h_feats = args.h_feats
    graph = Dataset(dataset_name, homo).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = args.num_classes

    logging.basicConfig(
        filename=f'log/{args.log_name}_{args.dataset}.log',
        filemode='a', 
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info(args)


    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # set the running device
    if int(args.device_id) >= 0 and torch.cuda.is_available():
        args.device = torch.device("cuda".format(args.device_id))
        torch.cuda.set_device(int(args.device_id))
        print('using gpu:{} to train the model'.format(args.device_id))
    else:
        args.device = torch.device("cpu")
        print('using cpu to train the model')

    if homo:
        model = DiffProt(in_feats, args)
    else:
        model = DiffProt_here(in_feats, args)

    model = model.to(args.device)
    train(model, graph, args)