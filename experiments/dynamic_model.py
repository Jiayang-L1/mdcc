import torch
import torch.nn as nn
import torch.optim as optim
from math import sqrt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, mean_squared_error
from data_split import split_data
import os


class dynamic_MLP(nn.Module):
    def __init__(self, input_size, downstream_task, modality, device):
        super(dynamic_MLP, self).__init__()
        self.modality = modality
        self.downstream_task = downstream_task
        self.device = device
        self.gru_don = nn.GRU(24, hidden_size=256, batch_first=True)
        self.gru_com = nn.GRU(768, hidden_size=256, batch_first=True)
        self.transformer = nn.Transformer(
            d_model=24,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
        )
        self.low_dim_re = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU()
        )

        self.enlarge_dim_re = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.Dropout(0.3),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )

        self.rnn_re = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

        self.low_dim_cl = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.enlarge_dim_cl = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.Dropout(0.3),
            nn.Linear(2048, 256),
            nn.Sigmoid(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        self.rnn_cl = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        if self.downstream_task == 're':  # regression task, correspond to outcome prediction
            if self.modality == 'M':
                out = self.low_dim_re(features)
            elif self.modality == 'G':
                features = features.squeeze()
                gru_hidden, _ = self.gru_don(features)
                gru_output = gru_hidden.squeeze()
                out = self.low_dim_re(gru_output).unsqueeze(-1)
            elif self.modality == 'Tr':
                temporal_output = self.transformer(features, features)
                out = self.low_dim_re(temporal_output)
            elif self.modality == 'S':
                out = self.low_dim_re(features)
            elif self.modality == 'MG':
                static_data = features[:, :, :-24].squeeze()
                temporal_data = features[:, :, -24:].squeeze()
                gru_hidden, _ = self.gru_don(temporal_data)
                temporal_output = gru_hidden
                concatenated_features = torch.cat((static_data, temporal_output), dim=1)
                out = self.low_dim_re(concatenated_features).unsqueeze(1)
            elif self.modality == 'MTr' or self.modality == 'MTrU':
                static_data = features[:, :, :-24]
                temporal_data = features[:, :, -24:]
                temporal_output = self.transformer(temporal_data, temporal_data)
                concatenated_features = torch.cat((static_data, temporal_output), dim=2)
                out = self.low_dim_re(concatenated_features)
            elif self.modality == 'MS':
                out = self.enlarge_dim_re(features)
            elif self.modality == 'MTrC':
                static_data = features[0][:, :, :-24].to(self.device).to(torch.float32)
                temporal_data = features[0][:, :, -24:].to(self.device).to(torch.float32)
                comments = features[1].to(self.device).to(torch.float32)
                temporal_output = self.transformer(temporal_data, temporal_data)
                gru_out_c, _ = self.gru_com(comments)
                comment_output = gru_out_c[:, -1, :].unsqueeze(1)
                concatenated_features = torch.cat((static_data, temporal_output, comment_output), dim=2)
                out = self.enlarge_dim_re(concatenated_features)
            else:
                print('Non Multimodal Experiment!')
                exit()
        elif self.downstream_task == 'cl':  # classification task, correspond to success prediction
            if self.modality == 'M':
                out = self.low_dim_cl(features)
            elif self.modality == 'G':
                features = features.squeeze()
                gru_hidden, _ = self.gru_don(features)
                gru_output = gru_hidden.squeeze()
                out = self.low_dim_cl(gru_output).unsqueeze(-1)
            elif self.modality == 'Tr':
                temporal_output = self.transformer(features, features)
                out = self.low_dim_cl(temporal_output)
            elif self.modality == 'S':
                out = self.low_dim_cl(features)
            elif self.modality == 'MG':
                static_data = features[:, :, :-24].squeeze()
                temporal_data = features[:, :, -24:].squeeze()
                gru_hidden, _ = self.gru_don(temporal_data)
                temporal_output = gru_hidden
                concatenated_features = torch.cat((static_data, temporal_output), dim=1)
                # out = self.low_dim_cl(concatenated_features).unsqueeze(1)
                out = self.rnn_cl(concatenated_features).unsqueeze(1)
            elif self.modality == 'MTr' or self.modality == 'MTrU':
                static_data = features[:, :, :-24]
                temporal_data = features[:, :, -24:]
                temporal_output = self.transformer(temporal_data, temporal_data)
                concatenated_features = torch.cat((static_data, temporal_output), dim=2)
                out = self.low_dim_cl(concatenated_features)
            elif self.modality == 'MS':
                out = self.enlarge_dim_cl(features)
            elif self.modality == 'MTrC':
                static_data = features[0][:, :, :-24].to(self.device).to(torch.float32)
                temporal_data = features[0][:, :, -24:].to(self.device).to(torch.float32)
                comments = features[1].to(self.device).to(torch.float32)
                temporal_output = self.transformer(temporal_data, temporal_data)
                gru_out_c, _ = self.gru_com(comments)
                comment_output = gru_out_c[:, -1, :].unsqueeze(1)
                concatenated_features = torch.cat((static_data, temporal_output, comment_output), dim=2)
                out = self.enlarge_dim_cl(concatenated_features)
            else:
                print('Non Multimodal Experiment!')
                exit()

        return out


class Dynamic_Dataset(Dataset):
    def __init__(self, data, downstream_task, modality):
        self.feature_normal_1 = data['feature_meta_1']  # metadata without comment & update
        self.feature_normal_2 = data['feature_meta_2']  # metadata with comment & update
        self.feature_bert = data['text_embedding']  # Text feature
        self.feature_res152 = data['photo_embedding']  # Photo feature
        self.feature_temporal_aggregation = data['temporal_inputs']  # temporal data for GRU or Transformer
        self.feature_temporal_infectious = data['point_process_inputs']  # temporal data for Self-exciting point process
        self.feature_comment = data['comment_bert']  # comment feature within the first day
        self.feature_update = data['update_bert']  # update feature within the first day
        self.target_re = data['target_re']  # regression target
        self.target_cl = data['target_cl']  # classification target
        self.downstream_task = downstream_task
        self.modality = modality

    def __len__(self):
        return len(self.feature_normal_1)

    def __getitem__(self, index):
        if self.modality == 'M':  # 21 dimension
            features = self.feature_normal_1[index]
        elif self.modality == 'G' or self.modality == 'Tr':
            features = self.feature_temporal_aggregation[index]
        elif self.modality == 'S':  # 1441*1 dimension
            features = self.feature_temporal_infectious[index]
        elif self.modality == 'MG' or self.modality == 'MTr':
            features = torch.cat((self.feature_normal_1[index], self.feature_temporal_aggregation[index]), dim=1)
        elif self.modality == 'MS':
            features = torch.cat((self.feature_normal_1[index], self.feature_temporal_infectious[index]), dim=1)

        # 23+24+(24*768) dimension, compared with normal 1 feature, comment and update count within the first day
        # are added. comment are converted into embedding respectively and aggregated through hours,
        # resulting in a 24*768 dimension
        elif self.modality == 'MTrC':
            section_1 = torch.cat((self.feature_normal_2[index], self.feature_temporal_aggregation[index]), dim=1)
            section_2 = self.feature_comment[index]
            features = [section_1, section_2]
        # 23+24+(1*768), updates are converted into embedding respectively and aggregated together.
        elif self.modality == 'MTrU':
            features = torch.cat((self.feature_normal_2[index], self.feature_update[index],
                                  self.feature_temporal_aggregation[index]), dim=1)
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


def train(model, train_loader, val_loader, combination, criterion, optimizer, num_epochs, device, learning_rate,
          n_epochs_stop=10):
    min_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for features, labels in train_loader:
            if combination != 'MTrC':
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
                if combination != 'MTrC':
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


def test(model, test_loader, downstream_task, device, combination):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            if combination != 'MTrC':
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


def dynamic_prediction(early_stopping, combination, downstream_task, device):
    # training parameters setting
    num_epochs = 5000
    batch_size = 1024
    learning_rate = 0.0001

    # set random_state with parameter, random split if vacancy
    train_data, val_data, test_data = split_data(random_state=24)

    # generate MLP input size from dict
    input_size_dict = dict({'M': 21, 'G': 256, 'S': 1441, 'Tr': 24, 'MG': 21 + 256, 'MS': 21 + 1441, 'MTr': 21 + 24,
                            'MTrC': 23 + 24 + 256, 'MTrU': 23 + 24 + 768})
    input_size = input_size_dict[combination]

    train_dataset = Dynamic_Dataset(train_data, downstream_task, combination)
    val_dataset = Dynamic_Dataset(val_data, downstream_task, combination)
    test_dataset = Dynamic_Dataset(test_data, downstream_task, combination)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = dynamic_MLP(input_size, downstream_task=downstream_task, modality=combination, device=device).to(device).to(
        torch.float32)

    # define criterion by type of task
    if downstream_task == 're':  # regression
        criterion = nn.MSELoss()
    elif downstream_task == 'cl':  # classification
        criterion = nn.BCELoss()
    else:
        print('Only support regression and classification.')
        exit()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # model training
    best_model = train(model, train_loader, val_loader, combination, criterion, optimizer, num_epochs, device,
                       learning_rate, n_epochs_stop=early_stopping)

    # model testing
    results = test(best_model, test_loader, downstream_task, device, combination)

    # 斌 start; log
    if not os.path.exists("../Logs"):
        os.makedirs("../Logs")
    f = open("../Logs/dynamic_log.txt", "a")
    f.write("\n######################Dynamic__{0} ######################".format(combination))
    f.write("AUC(cl):   {:.4f}".format(results))
    f.close()
    # end
