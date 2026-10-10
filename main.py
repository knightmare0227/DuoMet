import os
import torch
import torch.nn as nn
import config
import pandas as pd
import numpy as np
import copy
from unified_model import UnifiedDuoMet
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score
from util import read_xml, set_random_seed, getGipKernel
import csv
import random
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_metabolite_data_for_training(metabolite_file, embedding_file, mmi_file, device):
    import pandas as pd
    import pickle as pkl
    import dgl
    
    metabolite_df = pd.read_csv(metabolite_file)
    bigg_ids = metabolite_df['bigg_id'].tolist()
    num_metabolites = len(bigg_ids)
    
    with open(embedding_file, 'rb') as f:
        embedding_dict = pkl.load(f)
    
    embeddings_list = []
    
    for bigg_id in bigg_ids:
        if bigg_id in embedding_dict:
            embeddings_list.append(embedding_dict[bigg_id])
        else:
            modified_bigg_id = bigg_id.split('_')[0] + '_c'
            if modified_bigg_id in embedding_dict:
                embeddings_list.append(embedding_dict[modified_bigg_id])
            else:
                embedding = torch.zeros(2304, dtype=torch.float32)
                embeddings_list.append(embedding)
    
    metabolite_features = torch.stack(embeddings_list).to(device)
    
    mmi_df = pd.read_csv(mmi_file)
    
    node_mapping = {bigg_id: idx for idx, bigg_id in enumerate(bigg_ids)}
    
    substrate_ids = [node_mapping.get(row['substrate_id'], 0) for _, row in mmi_df.iterrows()]
    product_ids = [node_mapping.get(row['product_id'], 0) for _, row in mmi_df.iterrows()]
    
    g = dgl.graph((substrate_ids, product_ids), num_nodes=num_metabolites)
    
    from dgl import LapPE
    transform = LapPE(k=3, feat_name='eig')
    g = transform(g)
    g = g.to(device)
    
    return {
        'metabolite_features': metabolite_features,
        'graph': g,
        'eig_vec': g.ndata['eig'],
        'node_mapping': node_mapping
    }

def test_pre(biochemical_features, g, eig_vec, incidence_matrix, model):
    model.eval()
    with torch.no_grad():
        y_pred = model.predict(biochemical_features, g, eig_vec, incidence_matrix)
    return torch.squeeze(y_pred)

def train(args, batch, metabolite_data, train_incidence_pos, incidence_train, incidence_valid, y_train, y_valid):
    print('---------------- calculating similarity ------------------------------')
    
    biochemical_features = metabolite_data['metabolite_features']
    g = metabolite_data['graph']
    eig_vec = metabolite_data['eig_vec']
    
    node_num = biochemical_features.shape[0]
    X_similarity_t = None
    if args.use_similarity:
        X_similarity_t = getGipKernel(biochemical_features.cpu(), False, args.g_lambda).to(device)
    
    model = UnifiedDuoMet(
        biochemical_in_dim=2304,
        biochemical_out_dim=512,
        dgnn_hid_dim=256,
        num_dgnn=2,
        input_num=node_num,
        hyper_in_dim=256,
        conv_dim=args.conv_dim,
        head=args.head,
        p=args.p,
        L=args.L,
        use_attention=True,
        similarity=X_similarity_t
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crossentropyloss = nn.CrossEntropyLoss()
    
    max_valid_f1 = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print('---------------- start training ------------------------------')
    for epoch in tqdm(range(args.epoch)):
        model.train()
        epoch_size = incidence_train.shape[1] // batch
        
        for e in range(epoch_size):
            optimizer.zero_grad()
            
            batch_indices = slice(e * batch, (e + 1) * batch)
            batch_incidence = incidence_train[:, batch_indices]
            batch_labels = y_train[batch_indices]
            
            y_pred = model(biochemical_features, g, eig_vec, batch_incidence)
            
            loss = crossentropyloss(y_pred, batch_labels)
            
            loss.backward()
            optimizer.step()
        
        valid_score = test_pre(biochemical_features, g, eig_vec, incidence_valid, model)
        true_valid_score = valid_score.cpu().numpy()[:, 1]
        b_score = [int(s >= 0.5) for s in true_valid_score]
        
        auc_score = roc_auc_score(y_valid.cpu().numpy(), true_valid_score)
        pr = precision_score(y_valid.cpu().numpy(), b_score)
        re = recall_score(y_valid.cpu().numpy(), b_score)
        f1 = f1_score(y_valid.cpu().numpy(), b_score)
        aupr = average_precision_score(y_valid.cpu().numpy(), true_valid_score)
        
        if max_valid_f1 < f1:
            max_valid_f1 = f1
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f'\nvalid, epoch:{epoch}, f1:{f1:.4f}, pr:{pr:.4f}, recall:{re:.4f}, auc:{auc_score:.4f}, aupr:{aupr:.4f}')
    
    model.load_state_dict(best_model_wts)
    return model

def predict(model, metabolite_data, incidence_test, y_test):
    print('---------------- start testing ------------------------------')
    
    biochemical_features = metabolite_data['metabolite_features']
    g = metabolite_data['graph']
    eig_vec = metabolite_data['eig_vec']
    
    y_pred = test_pre(biochemical_features, g, eig_vec, incidence_test, model)
    score_t = torch.squeeze(y_pred)
    true_test_score = score_t.cpu().numpy()[:, 1]
    b_score = [int(s >= 0.5) for s in true_test_score]
    
    auc_score = roc_auc_score(y_test.cpu().numpy(), true_test_score)
    pr = precision_score(y_test.cpu().numpy(), b_score)
    re = recall_score(y_test.cpu().numpy(), b_score)
    f1 = f1_score(y_test.cpu().numpy(), b_score)
    aupr = average_precision_score(y_test.cpu().numpy(), true_test_score)
    
    print(f'f1:{f1:.4f}, pr:{pr:.4f}, recall:{re:.4f}, auc:{auc_score:.4f}, aupr:{aupr:.4f}')
    return f1, pr, re, auc_score, aupr


if __name__ == '__main__':
    args = config.parse()
    os.makedirs('./result', exist_ok=True)
    set_random_seed(args.seed)
    xml_files = glob.glob(os.path.join('BiGG Models', '*.xml'))

    for xml_file in xml_files:
        model_name = os.path.splitext(os.path.basename(xml_file))[0]
        print(f'\n{"="*60}')
        print(f'Processing model: {model_name}')
        print(f'{"="*60}')
        
        with open(xml_file, 'r', encoding='utf-8') as file:
            xml_content = file.read()
        
        read_xml(xml_content)

        with open('metabolite.csv', 'r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file)
            data1 = [row for row in reader]

        with open('incidence_matrix_pos.csv', 'r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file)
            data2 = [row for row in reader]

        name = ['']
        for i in range(len(data1)-1):
            name.append(data1[i+1][0])

        reaction = {}
        for j in range(len(data2)-1):
            if int(data2[j+1][0]) not in reaction:
                reaction[int(data2[j+1][0])] = [int(data2[j+1][1])]
            else:
                reaction[int(data2[j+1][0])].append(int(data2[j+1][1]))

        matrix = []
        for k, v in reaction.items():
            matrix.append(['p_' + str(k)] + [0]*(len(data1)-1))
            matrix.append(['n_' + str(k)] + [0]*(len(data1)-1))
            for j in range(len(v)):
                matrix[-2][v[j]+1] = 1
            count = 0
            while (count < len(v)):
                random_int = random.randint(1, len(data1)-1)
                if matrix[-2][random_int] == 0 and matrix[-1][random_int] == 0:
                    matrix[-1][random_int] = 1
                    count += 1

        random.shuffle(matrix)
        n = len(matrix)
        matrix1 = matrix[0:round(n*0.6)]
        matrix2 = matrix[round(n*0.6):round(n*0.8)]
        matrix3 = matrix[round(n*0.8):]

        matrix1.insert(0, name)
        matrix2.insert(0, name)
        matrix3.insert(0, name)

        matrix1 = list(zip(*matrix1))
        matrix2 = list(zip(*matrix2))
        matrix3 = list(zip(*matrix3))

        os.makedirs('./data/' + model_name, exist_ok=True)

        with open('./data/' + model_name + '/train_df_ub_G.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for row in matrix1:
                writer.writerow(row)

        with open('./data/' + model_name + '/test_df_ub_G.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for row in matrix2:
                writer.writerow(row)

        with open('./data/' + model_name + '/valid_df_ub_G.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for row in matrix3:
                writer.writerow(row)

        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        print('---------------- loading metabolite data ------------------------------')
        metabolite_data = load_metabolite_data_for_training(
            metabolite_file='metabolite.csv',
            embedding_file='metabolite_emb.pkl',
            mmi_file='mmi.csv',
            device=device
        )

        result = [['iteration','f1','pr','recall','auc','aupr']]
        n = int(n/4.28)
        for i in range(args.iteration):
            print(f'---------------- iteration {i} ------------------------------')

            train_df = pd.read_csv(f'./data/{model_name}/train_df_ub_G.csv', index_col=0)
            test_df = pd.read_csv(f'./data/{model_name}/test_df_ub_G.csv', index_col=0)
            valid_df = pd.read_csv(f'./data/{model_name}/valid_df_ub_G.csv', index_col=0)
            train_pos_df = train_df.loc[:, lambda d: d.columns.str.contains('p')]

            y_train = torch.tensor(['p' in c for c in train_df.columns], dtype=torch.long).view(-1).to(device)
            y_test = torch.tensor(['p' in c for c in test_df.columns], dtype=torch.float).view(-1).to(device)
            y_valid = torch.tensor(['p' in c for c in valid_df.columns], dtype=torch.float).view(-1).to(device)

            incidence_train = torch.tensor(train_df.to_numpy(), dtype=torch.float).to(device)
            incidence_test = torch.tensor(test_df.to_numpy(), dtype=torch.float).to(device)
            incidence_valid = torch.tensor(valid_df.to_numpy(), dtype=torch.float).to(device)
            
            model = train(args, n, metabolite_data, None, incidence_train, incidence_valid, y_train, y_valid)
            
            f1, pr, re, auc_score, aupr = predict(model, metabolite_data, incidence_test, y_test)
            result.append([i+1, f1, pr, re, auc_score, aupr])
            
            torch.save({'model': model.state_dict()}, f'{args.output}{model_name}_unified_model_{i}.pth')

        with open('./result/' + model_name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for row in result:
                writer.writerow(row)
