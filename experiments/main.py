import torch
from dynamic_model import dynamic_prediction
from multimodal_model import multimodal_prediction


def model_init(task, feature_type, prediction_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    early_stopping = 50     # Users can adjust the number of early stopping epochs by their preference.

    # enlarge number of early stop epochs when using Transformer as training speed is relatively low.
    if 'Tr' in feature_type:
        early_stopping = 150

    if task == 'multimodal':
        multimodal_prediction(early_stopping=early_stopping, modality=feature_type, downstream_task=prediction_type,
                              device=device)
    elif task == 'dynamic':
        dynamic_prediction(early_stopping=early_stopping, combination=feature_type, downstream_task=prediction_type,
                           device=device)


if __name__ == '__main__':
    task = 'multimodal'             # multimodal or dynamic

    '''
    feat_types: Abbreviations of feature involved
    Multimodal:
        M: Meta 
        T: Text
        P: Photo
        MT: M + T
        MP: M + P
        MTP: M + T + P
        MTD: MT + Donations (Transformer)
        MTPD: M + T + P + D
    Dynamic:
        M: Meta 
        G: GRU
        S: Self-exciting point process
        Tr: Transformer 
        MG: M + G 
        MS: M + S 
        MTr: M + Tr 
        MTrC: MTr + Comments 
        MTrU: MTr + Updates
    '''
    feat_types = 'M'

    prediction_type = 're'  # re for regression or cl for classification

    model_init(task, feat_types, prediction_type)
