echo "start..."
echo "generate dialogue..."
cd ./data
python3 yelp_generate_dialogue.py # 处理slot信息
echo "generate belief tracker data"
cd ./belief_tracker_data
python3 belief_tracker_data_generate.py # 获取词表
python3 belief_tracker_data_generate2.py # 转换数据
echo "generate FM data"
cd ../FM_data
python3 FM_data_generate_2.py # 获取userid，itemid，rating，utterance数据
echo "generate high score data"
cd ..
python3 generate_high_rating.py # 选择评分大于4.9的数据用于推荐部分
echo "generate agent data"
cd ./RL_data
python3 create_RL_data.py # 划分训练集，验证集，测试集
python3 create_RL_pretrain_data.py

