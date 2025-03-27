# 国外地址，如果太卡可以替换为飞桨的数据集
# https://aistudio.baidu.com/datasetdetail/103218 下面的 test2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

if [ ! -d "../datasets" ]; then
    mkdir ../datasets
fi

mv test2017.zip ../datasets/

cd ../datasets

unzip test2017.zip

mv test2017/ coco2017/

rm test2017.zip