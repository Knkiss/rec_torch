cd code || exit
echo "开始运行rec torch脚本"

# 关于SSM BPR Loss融合的实验
#echo "lastfm PCL_combine PCL LightGCN"
#python main.py --model PCL --dataset lastfm --no_print --pcl_combine
#python main.py --model PCL --dataset lastfm --no_print
#python main.py --model LightGCN --dataset lastfm --no_print
#echo "lastfm_kg PCL_combine PCL LightGCN"
#python main.py --model PCL --dataset lastfm_kg --no_print --pcl_combine
#python main.py --model PCL --dataset lastfm_kg --no_print
#python main.py --model LightGCN --dataset lastfm_kg --no_print
#echo "movielens1m_kg PCL_combine PCL LightGCN"
#python main.py --model PCL --dataset movielens1m_kg --no_print --pcl_combine
#python main.py --model PCL --dataset movielens1m_kg --no_print
#python main.py --model LightGCN --dataset movielens1m_kg --no_print
#echo "amazonbook PCL_combine PCL LightGCN"
#python main.py --model PCL --dataset amazonbook --no_print --pcl_combine
#python main.py --model PCL --dataset amazonbook --no_print
#python main.py --model LightGCN --dataset amazonbook --no_print

# 关于cluster内跨视图对比学习的实验
#python main.py --no_print --model WORK2 --dataset lastfm_kg --hyper3 10 --hyper2 0.1
#python main.py --no_print --model WORK2 --dataset lastfm_kg --hyper3 10 --hyper2 0.01
#python main.py --no_print --model WORK2 --dataset lastfm_kg --hyper3 10 --hyper2 1
#python main.py --no_print --model WORK2 --dataset lastfm_kg --hyper3 11 --hyper2 0.1
#python main.py --no_print --model WORK2 --dataset lastfm_kg --hyper3 11 --hyper2 0.01
#python main.py --no_print --model WORK2 --dataset lastfm_kg --hyper3 11 --hyper2 1