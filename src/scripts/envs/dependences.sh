conda env create -f e3nn_sampling.yml

# pip install torch-scatter==latest+cu101 torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html

# pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html

# pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html

# pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html

pip install torch-summary

pip install torch-geometric

mkdir pachages && cd pachages

pip install lie-learn

git clone https://github.com/mariogeiger/se3cnn.git

cd se3cnn 

python setup.py install

cd ../..


