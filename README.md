# icvseg
1. config 配置
    拷贝 config/human/mt_human_mtdata_changebg.py 到任意config 路径
    修改 txt_f root_dir 用于训练
2. 生成数据集txt文件
    修改 tools/create_traintxt.py

3. 训练

python train_remo.py --cfg config/xxx.py  --gpus 0,1,2,3
# icvseg
