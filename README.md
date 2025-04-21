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

# Experiment Tracking with [MLFlow](https://github.com/aimhubio/aim)

## Install
```
pip3 install mlflow
```

## Run
```
mlflow server --host 127.0.0.1 --port 8080
```

