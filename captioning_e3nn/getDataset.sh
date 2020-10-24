
if ! [ -d data ]
then 
   mkdir data
fi

cd data

echo "Retrieving PDBbind refined dataset . . . "
wget http://www.pdbbind.org.cn/download/pdbbind_v2019_refined.tar.gz
echo "Extracting refined files . . . "
tar -xvf pdbbind_v2019_refined.tar.gz -d refined-det

rm pdbbind_v2019_refined.tar.gz

echo "Retrieving PDBbind CASF dataset . . . "
wget http://www.pdbbind-cn.org/download/CASF-2016.tar.gz


echo "Extracting casf files . . . "
tar -xvf CASF-2016.tar.gz -d CASF-2016

rm CASF-2016.tar.gz