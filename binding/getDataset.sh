echo "Retrieving PDBbind refined dataset . . . "
wget http://www.pdbbind.org.cn/download/pdbbind_v2019_refined.tar.gz

if ! [ -d data ]
then 
   mkdir data
fi

cd data
echo "Extracting refined files . . . "
unzip pdbbind_v2019_refined.tar.gz -d refined-set

rm pdbbind_v2019_refined.tar.gz

echo "Retrieving PDBbind CASF dataset . . . "
wget http://www.pdbbind-cn.org/download/CASF-2016.tar.gz


echo "Extracting casf files . . . "
unzip CASF-2016.tar.gz -d CASF-2016

rm CASF-2016.tar.gz