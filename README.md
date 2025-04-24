# tsel-image-classification

# 1. Setup Environemnt
This step assume that Conda is already installed.

## 1.1. Create Conda Environment
```
conda create -n tsel-image-cls python=3.9
```

or 
```
conda env create -f environment.yml
```

## 1.2. Activate
```
conda activate tsel-image-cls
```

## 1.3. Install Dependencies
```
conda install jupyter
python3 -m ipykernel install --user --name tsel-image-cls --display-name "tsel-image-cls"
```

```
pip3 install -r requirements.txt
```


## 1.4. Export 
```
conda env export > environment.yml
```


## 1.5. Update
```
conda env update -f environment.yml --prune
```

# 2. Setup Kaggle

## 2.1. Download Kaggle API Token
```
mkdir -p ~/.kaggle
cp ./kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 2.2. Download Datasets
To download image, we can download manually using Web Browser (like chrome) and copy the download link.

### Training Datasets
```
wget -O images.zip "https://storage.googleapis.com/kaggle-data-sets/55098/107188/upload/images.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250421%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250421T080749Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=7f67dd39bc10c2b65cc3ab8e91fd668e5a43977fe973d881841d713386ea4c31e1eff3a2bb6f03e6cf173a9ff22f639418bb7aff00f8578d11d5c495dcb100558749b67487a70260e2ad7abf3b1eb57db4f19c192a51c232ff07dc37cec4496b4ec1411bd5941582c2a8425c8b5615ed5e231b15843b6b0a084ad9b6dab4f0b9945fbf13c5872ccb7a91e8c507e859c0dd740ddc7f4ac1d5f5f904a080699759942f6f15f4052d11b94c44857f7387dad9e1234e9a0a48d3cf23433ff7d55d36b06c1235161eb3954e5bc2bf89f1df790026227081e6c715904549031024e0a7bbdb0a692c32c04c6759ef8f813b4056e21926c04a6b86ed07f85ae4f5152c15"
```

### Validation Datasets
```
wget -O validation.zip "https://storage.googleapis.com/kaggle-data-sets/55098/107188/upload/validation.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250421%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250421T081636Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=2757f01571bfb33f595b85a85fab181837db01f6adcade639e3f5101cef7f5ffeb2ca7136d88291ac98c24409e3abc8ccd6a36b2c1af63f76f1685e53478c52ff495d790f19f2c336ffca62562006ba7e454954904a1c49202453614a7dfe9e806f3123b12e6625727a8e287ddecb1f7545d5f32d66d2836ca8c1bc4a3a0b6bf0fc55320c3eaba8bca32e024d76bb2ca21afe8b5ac5a8263c610381e8b9b9fe6bb066624b8c9d7e34dece1fecda49711beecaecb765fc6dc1ef57babdeb649e950a296e78b60d6f3a9804580cc741ec135236623f5b404ea17e22ef82877bc61d5e773fb27cabdb2bf8a32201efce782306332905b0b55cf9751cf228697e24a"
```

### Test Datasets
```
wget -O test.zip "https://storage.googleapis.com/kaggle-data-sets/55098/107188/upload/test.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250421%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250421T081728Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=68cdc83adea753a4eeea2c9d5d185e5dd568ad559fb4baaf77c1b570385d3a4269c95a70bfa7661a39745f15657472f050659448bf20d2325fc605c522f64c705fb265cb47f5ab21b57779cfdb139d85f201e65cc291d3316b8d98e5623d4f942a766d4f5eec0cf36dcaedb50a71eef419353974a7912ecba70f105d8fb9050c221011d418efc45e901b7c33e6e15c3d21bda2f964ea768dffa4224dd6f61d66f4f216d35a86e85794f6f897e12ead4142bbdc80833a0a69fef80b4204276fa9ea236ff3a3e824be438ce27cf2a3de9a82ab16f04103a7cdd44418b6c84e99badd6d741f7772406b83f19fd25c941ee8fa9f2f477cf8bf537f88c22986bad65e"
```

## 2.3. Unzip Datasets
```
mkdir datasets/
unzip validation.zip -d datasets/
unzip test.zip -d datasets/
unzip images.zip -d datasets/
```

## 2.4. Rename Datasets

### Rename Train Datasets
```
mv "datasets/images" "datasets/train"
mv "datasets/train/architecure" "datasets/train/architecture"
mv "datasets/train/art and culture" "datasets/train/art and culture"
mv "datasets/train/food and d rinks" "datasets/train/food and drinks"
mv "datasets/train/travel and  adventure" "datasets/train/travel and adventure"
```

### Rename Validation Datasets
```
mv "datasets/validation/architecture" "datasets/validation/architecture"
mv "datasets/validation/art and culture" "datasets/validation/art and culture"
mv "datasets/validation/food" "datasets/validation/food and drinks"
mv "datasets/validation/travel and adventure" "datasets/validation/travel and adventure"
```

### Delete ZIP Files
```
rm images.zip
rm test.zip
rm validation.zip
```

# Experiment Tracking with [MLFlow](https://github.com/aimhubio/aim)

## Install
```
pip3 install mlflow
```

## Run
```
mlflow server --host 127.0.0.1 --port 5000
```

# 3. Methodology
In this image clasification problems, we will use `TensorFlow` as a Deep Learning frame work to build a classifier that leverage `pre-trained` models and `fine tune` it to get better results.

##  3.1 Data Understanding and EDA
First we need to download, extract and prepare the dataset by following `step number 2` above then we can execute the notebook in `notebooks/01_eda.ipynb`

## 3.2 Model Training (Classifier)
Next, we will train a classifier model on the notebook located in `notebooks/02_training.ipynb`.

In summary, we wil use a MLFlow to track our experiments and log the model with it's hyper parameters so we can reproduce later. 

In the notebook, there is `parameters` to store all hyper-params values. We also use several base feature extractor models form `tf.keras.applications` that are trained on `ImageNet` such as `resnet_50`, `efficientnet`, `mobilenet` etc.

To achieve high accuracy, there are some strategies we can tune on `data`, `model` and `hyper-parameters`.

### 3.2.1 Data Strategy
To avoid overfitting on training dataset, we augment the dataset using several technique that include `Geometric` and `Color` transformation.

`Geometric` transformation augment the data by changeing the object location but keep the signal or color information. While `Color` transformation change the signal information so they can adapt to various condition that caused by lightning or camera quality.

We will try to use different `aug` values that can affect how model performance. To quick understand the impact on `training` and `validation` accuracy, we will use `MobileNet` as a base model.

Here are the results:

- Model with lower augmentation can achieve higher `accuracy` on `training` but lower on `validation` dataset.
    ![Train v.s Validation Accuracy](images/augmentaitons/accuracy-val_accuracy.png)
    *Train v.s Validation Accuracy*


- Even though with similar value of `validation accuracy` in the first step above, model has higher `validation loss` on low augmentation. This is beacuse the model overfit on traing dataset and can not generalize well.

    

    ![Train v.s Validation Loss](images/augmentaitons/loss-val_loss.png)
    *Train v.s Validation Loss*

- Data with high `augmentation` will have a better/lower `validation loss` means that the model generalize data very well. But too much augmentation can make the infromation difficult to understand (e.g. too dark or too bright) and did not improve so much.
    ![Augmenation magnitudes](images/augmentaitons/parallel.png)
    *Augmenation magnitudes*

In summary, with data augmentation we can achieve higher `accuracy` and lower `loss`. So, to achieve better model performance we will use augmentation with around `0.15` up to `0.3` because it have similar `validation loss` with higher augmentatio. There will be some impact on higher augmenation that can make information loss or missundersood, like the image become a mosaic pattern.

![Augmenation magnitudes](images/augmentaitons/table.png)
*Augmenation comparison*

![Augmenation magnitudes](images/augmentaitons/mosaic.png)
*Mosaic pattern due to high augmentation*


### 3.2.2 Model Strategy
#### 3.2.2.1 Transfer Learning
By using transfer learning from pre-trained model, we can achieve higher model performance and training duration compared to train the model from scratch.

For example, on this image below, we achieve the best `validation accuracy` only in just first 3 step/epochs.
![Validation Accuracy](images/augmentaitons/val_accuracy.png)
*Validation Accuracy*


#### 3.2.2.2 Base Model
We try to compare several base models availabe on `tf.keras.applications` and we found that `efficientnet` has a better accuracy.
![Pre-trained models](images/models/base-models.png)
*Pre-trained models*


#### 3.2.2.3 Callbacks
There are several callbacks we use during training:
- `Checkpoint`: to save the best model based on metrics we monitor on each epoch completed.
    ![Checkpoint](images/callbacks/checkpoint.png)

- `Early Stopping`: avoide to long training while the model did not have any improvement.
    ![Eary Stopping](images/callbacks/early-stopping.png)

- `Learning Rate Reducer`: in case the model stuck in local minima, we reduce the learning rate so they can learn carefully.
    ![Learning Rate](images/callbacks/learning-rate.png)


### 3.2.3 Hyper Parameters
We try several hyper-parameters tunning, tracking and comparing with `MLFlow`.
```json
parameters = {
    'model': BASE_MODEL_NAMES[1],
    'width': IMAGE_WIDTH,
    'height': IMAGE_HEIGHT,
    'image_size': IMAGE_SIZE,
    'class_names': CLASS_NAMES,
    'learning_rate': 4e-05,
    'batch_size': 128,
    'dropout': 0.15,
    'dense': 128,
    'epochs': 24,
    'aug': 0.05,
}
```

![ML FLow](images/mlflow.png)
*Experiment Tracking*

## 3.3 Model Fine-tuning
Fine tuning model can be followed on `notebooks/03_finetune.ipynb`. It continues to train not only the classifier, but also the feature extractor from the best model we get so far.

### 3.3.1 Best Classifier Model
The classifier layers we add on top of base model some times have performance that depends on how well the feature extractor have. 

We use `efficientnet` as a base model and achieve around `93.9%` accuracy.
![Best Model](images/transfer-learning/best-model-table.png)

Interestingly, even though we prove that higher augmentation can get higher accuracy, but in this model too much augmentatio can suffer the performance on training and validation data because the information is drifted away.

But with no or small augmentation can result higher accuracy on training, but lower on validation.
![Best Model](images/transfer-learning/best-model-parallel.png)

### 3.3.2 Learning rate
We reduce learning rate previously `4e-05` when performing transfer learning, to `3e-06` when we fine tune it.

### 3.3.2 Validation Accuracy Callback
We apply full fine tuning so the model can adapt with current dataset.
Because we would like to achive higher accuracy on validation data, we add additional `tf.keras.callbacks.ModelCheckpoint` callback to monitor `val_accuracy` and save the best model when improved.

```python
tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_acc, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max'
)
```

# 4. Result
After we fine tune the model improved to `94.8%` wich has a slightly +1% improvement on validation data.
![Best Model](images/fine-tuning/ft-table.png)

Here are the model result on test dataset
![Best Model](images/fine-tuning/test-scoring.png)

