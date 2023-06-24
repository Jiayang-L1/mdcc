Followings are instructions to reproduce the experimental results reported in Section 3. 

# Dependencies

Install dependencies for running the codes:

```shell
pip install -r requirements.txt
```

# Usage

Main function: `model_init(task, feature_type, prediction_type)`

`task`: Specify whether you want to run a multimodal or dynamic model. Choose `multimodal` for a multimodal model or `dynamic` for a dynamic model.

`feature_type`: Determines the type of features to be used according to the `task` parameters. 

`prediction_type`: Specify whether the task is a regression or classification. Use `re` for regression task or `cl` for classification task.

Here is an example:

```python
model_init(task='dynamic', feature_type='MTr', prediction_type='re')
```

More details please refer to the code parameters or code comments. 