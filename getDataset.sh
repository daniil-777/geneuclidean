
if ! [ -d data ]
then 
   mkdir data
fi


if ! [ -d data/refined-set ]
then 
   mkdir data/refined-set
fi

if ! [ -d data/CASF-2016]
then 
   mkdir data/CASF-2016
fi

echo "Retrieving PDBbind refined dataset . . . "
wget http://www.pdbbind.org.cn/download/pdbbind_v2019_refined.tar.gz
echo "Extracting refined files . . . "

tar -C data/refined-set -zxvf pdbbind_v2019_refined.tar.gz


rm pdbbind_v2019_refined.tar.gz

echo "Retrieving PDBbind CASF dataset . . . "
wget http://www.pdbbind-cn.org/download/CASF-2016.tar.gz


echo "Extracting casf files . . . "
tar -C data/CASF-2016 -zxvf pdbbind_v2019_refined.tar.gz

rm CASF-2016.tar.gz