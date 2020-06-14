# Captioning
The goal of this project is to generate smiles for ligands using pocket information.
Pocket is encoded using Euclidean Neural Networks. Then features are given to Lstm to generate smiles. 


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

#### 5. Sampling

```bash
python sampling.py configuration/config.json  
```


#### 5. Results

Results (plots of train/test loss and scatter plots of predicted/target pkd) should be saved in the folder /results




<br>


