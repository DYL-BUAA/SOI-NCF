import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def preprocess_dataset(df: pd.DataFrame, test_size=0.2, shuffle=False, random_state=42, n_features: int = None):
    """
    进行数据预处理
    :param df: dataset
    :param test_size: 测试集在数据集中占的比例
    :param shuffle: 是否打乱顺序
    :param random_state: 随机数状态实例
    :param n_features: 获取和标签相关性最高的n_features个特征
    :return: X_train, X_test, y_train, y_test
    """
    """
    在进行机器学习之前，对数据集进行适当的预处理是非常重要的。以下是一些常见的数据预处理步骤，这些步骤可以帮助提高模型的性能和准确性：

        1、数据清洗：
        处理缺失值：决定是填充它们、删除它们，还是用其他方式处理。
        移除重复数据：检查是否有重复的记录，并进行删除或合并。

        2、数据探索（Exploratory Data Analysis, EDA）：
        通过统计摘要和可视化来了解数据的分布和结构。

        3、特征选择（Feature Selection）：
        确定哪些特征对模型最重要，移除无关或冗余的特征。

        4、特征工程（Feature Engineering）：
        创建新特征或转换现有特征，以提高模型的性能。

        5、数据转换：
        对于分类变量，使用编码技术如独热编码（One-Hot Encoding）。
        对于数值变量，进行标准化（Standardization）或归一化（Normalization）。

        6、数据集划分：
        将数据集划分为训练集、验证集和测试集。

        7、类别不平衡处理：
        如果目标变量的类别分布不均匀，可能需要使用过采样或欠采样技术。

        8、异常值检测和处理：
        识别异常值并决定是移除它们、修正它们还是保留。

        9、数据类型转换：
        确保数据的类型适合模型的需求，例如将分类变量转换为整数或字符串。

        10、数据集的保存和加载：
        将清洗和预处理后的数据集保存为文件，以便于模型训练时加载。
    """
    target_column = df.columns[-1]
    # 1. 处理缺失值
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # 2. 移除重复数据
    df_unique = df_imputed.drop_duplicates()

    # 3. 特征选择（示例：移除具有高度相关性的特征）
    # # # 依据：线性相关
    # correlation_matrix = df_unique.corr()
    # # # # 移除自相关性，即对角线上的1
    # correlation_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
    # # # # 找出相关性高于阈值的特征对
    # high_correlation = correlation_matrix.apply(lambda x: x > 0.95)
    # to_drop = high_correlation[high_correlation.any(axis=1)].index
    # # # # 移除相关性高的特征列，但不包括目标列
    # to_drop = [column for column in to_drop if column != target_column]
    # # # # 应用特征选择
    # df_reduced = df_unique.drop(columns=to_drop)

    # 3. 特征选择 - 基于互信息
    # # 根据分类或回归任务选择适当的函数
    # # # . 划分特征和目标变量
    X = df_unique.drop(columns=[target_column])
    y = df_unique[target_column]
    if y.dtype == 'object' or (isinstance(y, pd.Series) and len(y.unique()) > 10):
        mi_score = mutual_info_classif(X, y)
    else:
        mi_score = mutual_info_regression(X, y)
    # # # . 获取互信息分数最高的n_features个特征
    if n_features:
        selected_features = X.columns[np.argsort(mi_score)[-n_features:]]
    else:
        selected_features = X.columns[np.argsort(mi_score)]

    # # # . 使用选择的特征来更新X
    X_selected = X[selected_features]
    df_reduced = X_selected

    # 4. 特征工程（根据需要添加新特征或转换现有特征）
    # 例如：df_reduced['new_feature'] = df_reduced['existing_feature'] ** 2

    # 5. 数据转换
    # 仅对特征进行标准化，不包括目标列
    features = df_reduced.columns
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_reduced[features]), columns=features)

    # 6. 数据集划分
    # 确保目标列没有被标准化
    # y = df_reduced[target_column]
    # X = pd.concat([df_scaled, y], axis=1)  # 将目标列添加回DataFrame
    X = df_scaled
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    return X_train, X_test, y_train, y_test

# 使用示例
# 假设 df 是您的 DataFrame，target_column 是目标列的名称
# X_train, X_test, y_train, y_test = preprocess_dataset(df, 'target_column')

