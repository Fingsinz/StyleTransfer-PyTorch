# 国外地址，如果太卡可以替换为飞桨的数据集
# https://aistudio.baidu.com/datasetdetail/222219 下面的 WikiArt2.zip
wget https://ia801206.us.archive.org/11/items/WikiArt_dataset/WikiArt_000.tar

if [ ! -d "../datasets" ]; then
    mkdir ../datasets
fi

mv WikiArt_000.tar ../datasets/

cd ../datasets

tar -xvf WikiArt_000.tar

rm WikiArt_000.tar