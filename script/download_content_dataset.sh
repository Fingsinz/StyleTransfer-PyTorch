# 飞桨数据集地址：https://aistudio.baidu.com/datasetdetail/103218
# 浦源数据集地址：https://openxlab.org.cn/datasets/OpenDataLab/COCO_2017/

# 飞桨数据集直接获取链接，然后wget下载即可

# 使用浦源数据集下载脚本如下

pip install openxlab
pip install -U openxlab

openxlab login

openxlab dataset download --dataset-repo OpenDataLab/COCO_2017 --source-path /raw/Images/test2017.zip --target-path .

unzip test2017.zip