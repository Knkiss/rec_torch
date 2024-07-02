cd code || exit
echo "开始运行rec torch脚本"

# 关于SSM BPR Loss融合的实验
python main.py --model PCL --dataset lastfm --no_print --pcl_combine
python main.py --model PCL --dataset lastfm --no_print
python main.py --model PCL --dataset lastfm_kg --no_print --pcl_combine
python main.py --model PCL --dataset lastfm_kg --no_print
python main.py --model PCL --dataset movielens1m_kg --no_print --pcl_combine
python main.py --model PCL --dataset movielens1m_kg --no_print
python main.py --model PCL --dataset amazonbook --no_print --pcl_combine
python main.py --model PCL --dataset amazonbook --no_print