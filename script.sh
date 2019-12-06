mkdir ../data
cd ../data
wget https://fid.dmi.unibas.ch/FID-300.zip
unzip FID-300.zip
cd -
python data_split.py
