# RecTorch 使用说明

# 概述
## 1. 框架
本系统框架包括五大模块：数据集输入与预处理、参数设置与模型初始化、模型训练与结果统计、结果测试与可视化分析，以及其他功能。
## 2. 模块描述
### 数据集输入与预处理模块
1. data_split.py：提供通用和知识数据集处理功能。包括从数据集构建用户交互字典、训练集和测试集分割、知识数据集分割以及生成以物品为首的数据文件。用户可以通过修改data_split主函数中的数据集参数，调用对应的函数进行数据处理。
2. dataLoader.py：提供通用和知识数据集读取功能。支持读取和分割额外的知识数据，提供灵活的数据处理方法，方便用户进行知识图谱相关的推荐任务。
### 参数设置与模型初始化模块
1. world.py：管理和初始化全局参数，使用argparse库解析命令行参数，支持设置根路径、模型类型、数据集、TopK、训练批次、学习率等常用参数，选择早停策略和绘图选项，并打印参数信息，帮助用户快速配置和选择模型。
2. model.py：管理和初始化知识模型，定义和初始化知识图谱模型的结构和参数，支持复杂模型的创建和训练，为用户提供灵活的模型配置和扩展。
### 模型训练与结果统计模块
1. main.py：主训练脚本，负责调用模型进行训练和测试，包括初始化模型、准备每个epoch的方法、计算embedding和损失的方法等。
2. losses.py：定义推荐任务中的常用损失函数，如BPR、MAE、InfoNCE、TransE、KD等，提供灵活的损失函数选择和扩展。 
3. decorator.py：使用装饰器函数统计模型训练和测试的时间和性能，帮助开发人员进行性能分析和代码优化。
4. ablation.py：进行消融实验或参数分析实验，分析不同模块或参数对模型性能的影响，提供详细的实验结果和分析。 
### 结果测试与可视化分析模块
1. metrics.py：定义常用的推荐系统评价指标，如Recall、NDCG、Hit、MAP、Precision，支持用户自定义和扩展评价指标。
2. experiment.py：提供可视化分析方法，通过matplotlib库绘制柱状图、折线图等多种图表，展示模型性能对比和结果分析。 
### 其他功能模块
1. utils.py：提供各种辅助工具函数，包括早停策略、邮件发送实验结果、计算余弦相似度、构建稀疏图矩阵、dropout操作、随机丢弃图的边、将稀疏矩阵转换为系数张量、k-means聚类、根据用户-物品交互数据创建邻接矩阵等。

# 使用说明
## 1. 数据集输入与预处理模块
### 数据集输入
dataLoader.py文件位于目录code/train/下，其中提供了对于通用数据集以及扩展知识数据集的绝大多数常见方法，包括对于通用数据集的一般属性和方法：训练集用户交互网络UserItemNet；构建稀疏图getSparseGraph()；构建测试数据__build_test()；构建用户正样本对getUserPosItems()。对于知识数据集的扩展属性和方法：物品实体邻接表kg_dict；从数据集读取知识数据generate_kg_data()；构建知识图谱稀疏图get_kg_graph()；获取物品实体邻接表get_kg_dict()；采样物品实体邻接表get_kg_dict_random()。数据集输入文件在model.py文件中被调用以初始化数据集并用于之后进行的训练阶段和测试阶段。
```
class KGDataset(Dataset):
    def __init__(self, kg_path=None):
        if kg_path is None:
            kg_path = join(world.PATH_DATA, world.dataset, "kg.txt")
        kg_data = pd.read_csv(kg_path, sep=' ', names=['h', 'r', 't'], engine='python')
        self.kg_data = kg_data.drop_duplicates()
        self.kg_dict, self.heads = self.generate_kg_data()
        self.item_net_path = join(world.PATH_DATA, world.dataset)
        self.length = len(self.kg_dict)

    def extra_functions(self):
        pass
```
用户可以通过在通用数据集类UIDataset和扩展知识数据集类KGDataset中添加额外方法为数据集提供更多自定义功能，如上所示。

### 数据集预处理
data_split.py文件处于目录data/下，其中提供了诸多常用的方法：
- 从数据集读用户交互字典get_dict_from_data()；
- 写入用户交互字典到文件中write_dict_to_file()；
- 训练集、测试集分割train_test_split()；
- 知识图谱分割kg_split()；
- 转换用户交互数据为以物品为首的数据文件ui_to_iu()；
- 重新排列知识图谱实体ID序号kg_resort()。

使用时更改在data_split.py主函数中调用对应的函数即可完成数据处理。
## 2. 参数设置与模型初始化模块
### 运行时参数设置
world.py文件位于目录code/下，使用 Python 中的argparse模块解析命令行参数，允许用户通过命令行脚本形式设置全局变量并运行实验。支持的命令行参数设置包括：根路径、模型、数据集、TopK、训练批次、学习率等推荐任务深度学习常用参数；同时world文件中还提供对于实验早停策略的选择参数、结果统计的频率参数；多种模型的关键超参数设置等。所有参数均会在模型初始化前被打印以区分本次实验的实验环境。用户可以通过手动修改文件中内容或命令行参数设置修改运行时参数。
### 模型初始化
模型初始化由抽象模型类abstract_model.py和__init_.py共同构成model包，位置目录code/model/下，根据抽象模型类本系统提供了16个深度学习推荐模型，均位于/code/model/的子目录中，典型模型包括MF、LightGCN、CGCL、KGRec、PCL等。模型选择由全局参数world设置并于__init__中初始化对应名称的模型，模型的通用参数由抽象模型类构建，其主要包括的参数如下所示。

不同深度学习模型包括各自的特定参数，这类信息在继承类中实现，如code/model/general/SGL.py中定义了self.graph_1和self.graph_2的两个子图，并于每个epoch开始阶段初始化，自定义新模型的具体方法在下一小节中介绍。
```
class AbstractRecModel(nn.Module):
    def __init__(self):
        super().__init__()
        if world.sys_root_model:
            return
        world.sys_root_model = True
        self.ui_dataset = dataloader.UIDataset()
        self.num_users = self.ui_dataset.n_users
        self.num_items = self.ui_dataset.m_items
        self.Graph = self.ui_dataset.getSparseGraph()

        self.embedding_dim = world.hyper_embedding_dim
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
```
### 自定义新模型
abstractModel.py中有一些抽象方法和属性定义，提供了模型的基本结构。主要包括初始化函数、每个epoch开始前的初始化方法、计算embedding的方法、计算损失的方法和计算用户评分的方法。这些方法是抽象方法，需要在实际模型中进行具体实现，示例如下所示。
```
class LightGCN(model.AbstractRecModel):
    def __init__(self):
        super().__init__()

    def calculate_embedding(self):
        return self.forward(self.embedding_user.weight, self.embedding_item.weight, self.Graph)

    def calculate_loss(self, users, pos, neg):
        loss = {}
        # 自定义损失函数
        return loss

    def forward(self, all_users, all_items, graph, dropout=True, drop_prob=0.2, n_layers=3, output_one_layer=False):
        # 自定义前向计算
        return all_users, all_items
```
以LightGCN算法为例，在calculate_embedding方法中，它继承了之前提到的AbstractRecModel类。LightGCN类中实现了计算embedding和计算损失的具体方法。它通过调用forward方法来计算用户和物品的embedding。在calculate_loss方法中，计算了BPR损失和正则化损失，并返回总的损失。在forward方法中，根据输入参数计算了多层的embedding信息，其中实现了多层GCN的结构。通过对图进行多次稀疏矩阵乘法操作，得到不同层次的节点embedding信息。在dropout操作中，根据训练模式和设置的dropout概率来对图进行节点的部分删除。仿照以上函数设计，用户可以在任意位置新建模型并在__init__.py中导入对应model从而实现自己的模型。
## 3. 模型训练与结果统计模块
### 模型训练过程
模型训练与结果统计的切换过程由main.py（code/）进行，它负责指导其他计算模块的执行顺序。

在模型训练过程中，由于上文中定义的抽象模型类，因此模型的训练与测试过程可以采用黑盒设计，在每个epoch中会计算模型的损失并使用定义好参数的优化器进行反向梯度传播。损失函数设计定义在目录code/train/下的losses.py中，其中提供BPR、SSM、InfoNCE、L2正则化、KD、MAE等多种损失函数，用户也可根据自己的需要自行添加并于自定义模型中进行使用。

位于目录code/train/下的decorator文件对于模型的研究工作具有相当作用，其中提供对于模型关键函数的时间统计装饰器和CPU占用率统计装饰器，用户可以通过@注解形式对自定义模型的方法进行考量以用于优化模型设计，装饰器的使用示例如下所示。
```
from train.decorator import count_cpu_once

class Lightgcn(model.AbstractRecModel):
    def __init__(self):
        super().__init__()

    @count_cpu_once
    def calculate_loss(self, users, pos, neg):
        loss = {}
        # loss计算
        return loss

    def calculate_embedding(self):
        return self.embedding_user.weight, self.embedding_item.weight
```
ablation文件提供了对于深度学习模型参数搜索与消融实验的自动化实验方法，用户可以通过在ablation文件中手动修改ablation_list和dataset_list以进行笛卡尔积形式的多数据下参数搜索，每次均将执行独立的main进程以得到稳定的实验结果。
### 结果统计与存储
main方法同时管理着运行结果的存储，其中包括：保存运行时的中间时刻checkpoint防止遭遇中断后运行结果丢失；保存模型输出的推荐列表，并通过临时的性能测试判断最佳推荐列表；保存模型对于数据集中用户和物品节点的高维特征向量，可以用于预训练模型的研究或下游任务的应用。用户通过world参数中对于上述三种功能的开启后，可以在新生成的文件夹output中查看到保存的内容。
## 4. 结果测试与可视化分析模块
### 结果测试
metrics.py文件位于code/train/目录下，提供了对于推荐任务评估的五种常见指标，包括：召回率Recall、准确率Precision、折损排序指标NDCG、平均倒数排名MRR和平均距离MAD，用户可以在Metrics枚举类中添加指标分类并构建自己的评价指标计算函数。

根据上文中提到的main文件保存的推荐列表，code/experiment/中的RQ0_calculate文件可以用于测试该推荐列表与原始数据集中测试集的相关评价指标，生成计算结果并保存到.npy文件中输出到output文件夹中，用于后续的可视化分析任务。结果测试不仅包括全数据集的结果统计，还包括根据用户稀疏度或物品稀疏度分组的计算方式。
### 可视化分析
experiment位于code/experiment下，提供了对测试结果的多种纵向和横向比较方法和可视化方法，包括：RQ1整体性能比较、RQ2消融实验比较、RQ3物品侧长尾分布实验比较、RQ4用户侧稀疏度分布实验比较、RQ5超参数实验比较。具体的函数说明如下：

- RQ0_calculate_all函数是用来计算数据集和模型的一系列结果，并根据需求进行调整计算和输出。
- RQ0_datasets_statistics函数是用来对数据集进行统计分析，包括用户、物品、交互信息、稀疏度等指标的统计。
- RQ1_compare_all函数是用来对不同数据集、模型的性能进行独立Top-N评估，并将结果绘制成折线图；同时还会输出数据表格展示模型在不同数据集下的性能对比。
- RQ3_compare_longTail函数是用来比较不同物品长尾分布情况对模型性能的影响，并根据不同的可视化形式（如Recall、数量、类别）进行绘图比较。
- RQ4_compare_sparsity函数是用来比较不同用户群体稀疏度对模型性能的影响，并根据不同的可视化形式（如指标、数量、类别）进行绘图比较。
- RQ2_ablation.py使用matplotlib库绘制了关于不同指标（Recall和NDCG）的柱状图，用以展示不同模型在指标上的表现。代码首先指定了一些绘图的参数和颜色列表，然后根据数据将不同模型在相应的指标上的表现绘制成了柱状图。在绘图的过程中，还设置了不同的y轴刻度和位置，以及一些美化的绘图效果，如字体大小、坐标轴标签、标题等。最后，代码根据数据集名字存储了绘制结果，并将其保存成eps或png格式的图片。
## 5. 其他功能模块
utils.py文件位于code/train/目录下，其中实现了一些在深度学习模型训练中会用到的常用工具函数：设置随机种子、计算两个张量之间的余弦相似度、构建稀疏图矩阵、稀疏矩阵dropout操作、随即丢弃图的边、将稀疏矩阵转换为系数张量、k-means算法对数据聚类、根据用户-物品交互数据创建邻接矩阵等，关键的方法说明如下：
### 随机种子的设置
在深度学习中，随机种子（random seed）是用来控制随机数生成的初值。设置随机种子可以使得在相同随机种子的情况下，每次运行得到的随机结果是一样的，这有助于代码的可重复性以及实验结果的可控性，随机种子的设置应当在训练模型之前进行。同时，由于pytorch库的随机数机制，batch采样的结果也受到模型初始化的影响，因此在数据集batch采样过程前重新设置随机种子，从而得到可靠的实验比较结果。

*希望所有使用者能通过本系统快速的构建实验，并预祝取得较好的实验结果*