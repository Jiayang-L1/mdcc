# MDCC: A Multimodal Dynamic Dataset for Donation-based Crowdfunding Campaigns

![](https://img.shields.io/badge/DOI-10.5281/zenodo.8075200-blue)

This repo provides a new multimodal dynamic dataset for donation-based crowdfunding campaigns on [GoFundMe](https://gofundme.com/). Our dataset contains project descriptions, campaign photos, project metadata, and dynamic data including donations, updates and comments. 

    Xovee Xu, Jiayang Li, and Fan Zhou
    MDCC: A Multimodal Dynamic Dataset for Donation-based Crowdfunding Campaigns
    Under Review for ACM CIKM, 2023

# Usage

## Dataset Download

The dataset can be obtained via Zenodo
(doi: [10.5281/zenodo.8075200](10.5281/zenodo.8075200)).

There are three files: raw data, experimental data, and Photos. 

- **raw data**: 14,961 rows, 14 columns, corresponding to 14,961 campaigns and all features extracted (except images).
- **experimental Data**: used to reproduce the results reporeted in the paper. 
- **Photos**: photos with human faces are blurred (with prefix `blurred_`). For cover photo, its file name has an extension of `homepage`, e.g., `celi-need-help_homepage.jpg`. For photos in main body if any, its file name has an extension of `description_n`, e.g., `save-kate_description_2.jpg`. 


```python
import pandas as pd

raw_data = pd.read_pickle('./raw data.pickle')
exp_data = pd.read_pickle('./experimental data.pickle')

print(raw_data.columns)
```

An example of blurred image shown below: 
<div style="text-align: center;">
<img src="./sample image.jpg" alt="blurred image" style="max-width: 600px; max-height: 400px;" />
</div>

## Experiment Running

To reproduce our results, run `main.py` in experiments. Use `feat_types` parameter to select different types of features when running the model. More details are shown in the codes. 

**Note 1**: Five crowdfunding campaigns missing their cover photos, and 20 campaigns have donations prior to the `launch_date`. 

**Note 2**: To preserve privacy, all photos with human faces are blurred. You can obtain the original photos with the original campaign homepage (https::www.gofundme.com/f/{campaign_id}). You can also contact us to request the original photos (used only for academic purposes). Photo embeddings are generated by the original photos. 

## Cite

If you find our work useful, please consider cite us:

```bibtex
@inproceedings{xu2023mdcc, 
  author = {Xovee Xu and Jiayang Li and Fan Zhou}, 
  title = {{MDCC}: A Multimodal Dynamic Dataset for Donation-based Crowdfunding Campaigns}, 
  booktitle = {Under Review}, 
  year = {2023}, 
}
```