# Captioning
The goal of this project is to generate smiles for ligands using pocket information.
Pocket is encoded using Euclidean Neural Networks. Then features are given to Lstm to generate smiles. This is similar to the image captioning task. There are two implemeted options: decoder without and with attention.
# Models
Model without attention. Every LSTM gets a previous hidden state and embedded caption.
![](images/model_without_attention.png)

Model with attention. Every LSTM gets a previous hidden state and embedded caption with attention-weighted-feature vector. On the illustration white parts of the image mean weighted features whee the Decoder should pay attention to generate the next word
![](images/model_attention_grey.png)
## Usage 

#### Install additional libraries

#### 1. Install lie-learn
```bash
   pip install lie_learn
```

#### 2. Install se3cnn
```bash
   git clone https://github.com/mariogeiger/se3cnn.git
   python setup.py install
```
#### Installation

#### 1. Clone the repositories
```bash
git clone https://gitlab.ethz.ch/rethink/geneuclidean/tree/master/geneuclidean
cd geneuclidean/binding
```

#### 2. Download the dataset

```bash 
   bash getDataset.sh 
```

#### 3. Preprocess dataset

```bash
   python preprocessing_all.py
```
#### 4. Training

```bash
python train.py configuration/config.json  
```

Configuration file is in configs/ folder. You may specify the path to the inputs and results, parameters of encoder and decoder, parameters of training (batch size, number of epoches). 

#### 5. Sampling

```bash
python sampling.py configuration/config.json  
```


#### 5. Results

Results (plots of train/test loss and scatter plots of predicted/target pkd) should be saved in the folder /results




<br>


