import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, mean_squared_error
from data_split import split_data
import os


class multimodal_MLP(nn.Module):
    def __init__(self, input_size, downstream_task, modality):
        super(multimodal_MLP, self).__init__()
        self.modality = modality
        self.downstream_task = downstream_task
        self.low_dim_re = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

        self.mid_dim_re = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

        self.enlarge_dim_re = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )

        self.trans_high_re = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

        self.low_dim_cl = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.mid_dim_cl = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.enlarge_dim_cl = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.trans_high_cl = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.transformer = nn.Transformer(
            d_model=24,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256
        )

    def forward(self, features):
        if self.downstream_task == 're':  # regression task, correspond to outcome prediction
            # middle dimensional MLP for Photo (ResNet152) included features, so was classification.
            if 'P' in self.modality and self.modality != 'MTPD':
                out = self.mid_dim_re(features)
            # enlarge dimension when Meta+Text(BERT), so was classification
            elif self.modality == 'MT':
                out = self.enlarge_dim_re(features)
            # lower dimensional MLP for Meta or Text ONLY, so was classification.
            elif self.modality == 'M' or self.modality == 'T':
                out = self.low_dim_re(features)
            elif self.modality == 'MTD':
                static_data = features[:, :, :-24]  # extract static data
                temporal_data = features[:, :, -24:]  # extract temporal data for transformer input
                temporal_output = self.transformer(temporal_data, temporal_data)
                concatenated_features = torch.cat((static_data, temporal_output), dim=2)
                out = self.enlarge_dim_re(concatenated_features)
            elif self.modality == 'MTPD':
                static_data = features[:, :, :-24]  # extract static data
                temporal_data = features[:, :, -24:]  # extract temporal data for transformer input
                temporal_output = self.transformer(temporal_data, temporal_data)
                concatenated_features = torch.cat((static_data, temporal_output), dim=2)
                out = self.trans_high_re(concatenated_features)
            else:  # unexpected modality inputs
                print('Non Multimodal Experiment!')
                exit()
        elif self.downstream_task == 'cl':  # classification task, correspond to success prediction
            if 'P' in self.modality and self.modality != 'MTPD':
                out = self.mid_dim_cl(features)
            elif self.modality == 'MT':
                out = self.enlarge_dim_cl(features)
            elif self.modality == 'M' or self.modality == 'T':
                out = self.low_dim_cl(features)
            elif self.modality == 'MTD':
                static_data = features[:, :, :-24]  # extract static data
                temporal_data = features[:, :, -24:]  # extract temporal data for transformer input
                temporal_output = self.transformer(temporal_data, temporal_data)
                concatenated_features = torch.cat((static_data, temporal_output), dim=2)
                out = self.enlarge_dim_cl(concatenated_features)
            elif self.modality == 'MTPD':
                static_data = features[:, :, :-24]
                temporal_data = features[:, :, -24:]
                temporal_output = self.transformer(temporal_data, temporal_data)
                concatenated_features = torch.cat((static_data, temporal_output), dim=2)
                out = self.trans_high_cl(concatenated_features)
            else:
                print('Non Multimodal Experiment!')
                exit()
        else:
            print('Downstream Task Error.')
            exit()

        return out


class Multimodal_Dataset(Dataset):
    def __init__(self, data, downstream_task, modality):
        self.feature_normal = data['feature_meta_1']
        self.feature_bert = data['text_embedding']  # Text feature
        self.feature_res152 = data['photo_embedding']  # Photo feature
        self.feature_temporal = data['temporal_inputs']  # Temporal feature
        self.target_re = data['target_re']  # regression target
        self.target_cl = data['target_cl']  # classification target
        self.downstream_task = downstream_task
        self.modality = modality

    def __len__(self):
        return len(self.feature_normal)

    def __getitem__(self, index):
        if self.modality == 'M':
            features = self.feature_normal[index]
        elif self.modality == 'T':
            features = self.feature_bert[index]
        elif self.modality == 'P':
            features = self.feature_res152[index]
        elif self.modality == 'MT':
            features = torch.cat((self.feature_normal[index], self.feature_bert[index]), dim=1)
        elif self.modality == 'MP':
            features = torch.cat((self.feature_normal[index], self.feature_res152[index]), dim=1)
        elif self.modality == 'MTP':
            features = torch.cat((self.feature_normal[index], self.feature_bert[index], self.feature_res152[index]),
                                 dim=1)
        elif self.modality == 'MTD':
            features = torch.cat((self.feature_normal[index], self.feature_bert[index], self.feature_temporal[index]),
                                 dim=1)
        elif self.modality == 'MTPD':
            features = torch.cat((self.feature_normal[index], self.feature_bert[index], self.feature_res152[index],
                                  self.feature_temporal[index]), dim=1)
        else:
            print('Non Multimodal Experiment!')
            exit()

        if self.downstream_task == 're':
            target = self.target_re[index]
        elif self.downstream_task == 'cl':
            target = self.target_cl[index]
        else:
            print('Downstream Task Error.')
            exit()

        return features, target


# 定义训练函数
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, learning_rate, n_epochs_stop=10):
    min_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device).to(torch.float32)
            labels = labels.to(device).to(torch.float32).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(features).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device).to(torch.float32)
                labels = labels.to(device).to(torch.float32).unsqueeze(1)
                outputs = model(features).squeeze(1)
                val_loss += criterion(outputs, labels)

        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            epochs_no_improve = 0
            # Return best model
            best_model = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                print('min val loss:', min_val_loss, 'lr:', learning_rate, 'early_stopping_epoch', n_epochs_stop)
                break

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

    model.load_state_dict(best_model)  # load best model parameters
    return model


# 定义测试函数
def test(model, test_loader, downstream_task, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device).to(torch.float32)
            labels = labels.to(device).to(torch.float32).unsqueeze(1)

            outputs = model(features).squeeze(1)

            predictions.extend(outputs.tolist())
            true_labels.extend(labels.tolist())

    if downstream_task == 're':
        results = sqrt(mean_squared_error(true_labels, predictions))
        print(f"Test RMSE: {results:.4f}")

    elif downstream_task == 'cl':
        results = roc_auc_score(true_labels, predictions)
        print('auc', results)

    else:
        print('Downstream Task Error.')
        exit()

    return results


def multimodal_prediction(early_stopping, modality, downstream_task, device):
    # training parameters setting
    learning_rate = 0.0001
    num_epochs = 5000
    batch_size = 1024

    # set random_state with parameter, random split if vacancy
    train_data, val_data, test_data = split_data(random_state=24)

    # generate corresponding input size through dict
    input_size_dict = dict({'M': 21, 'T': 768, 'P': 2048, 'MT': 21 + 768, 'MP': 21 + 2048, 'MTP': 21 + 768 + 2048,
                            'MTD': 21 + 768 + 24, 'MTPD': 21 + 768 + 2048 + 24})
    input_size = input_size_dict[modality]

    # load dataset
    train_dataset = Multimodal_Dataset(train_data, downstream_task, modality)
    val_dataset = Multimodal_Dataset(val_data, downstream_task, modality)
    test_dataset = Multimodal_Dataset(test_data, downstream_task, modality)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initiate model
    model = multimodal_MLP(input_size, downstream_task=downstream_task, modality=modality).to(device).to(torch.float32)

    # define criterion by type of task
    if downstream_task == 're':  # regression
        criterion = nn.MSELoss()
    elif downstream_task == 'cl':  # classification
        criterion = nn.BCELoss()
    else:
        print('Only support regression and classification.')
        exit()

    # choose Adam as optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # model training
    best_model = train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, learning_rate,
                       n_epochs_stop=early_stopping)

    # model testing
    results = test(best_model, test_loader, downstream_task, device)
    # 斌 start; log
    if not os.path.exists("../Logs"):
        os.makedirs("../Logs")
    f = open("../Logs/multimodal_log.txt", "a")
    f.write("\n######################Multimodal__{0} ######################".format(modality))
    f.write("AUC(cl):   {:.4f}".format(results))
    f.close()
    # end
