import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_data(datapath: str = './data/ObesityDataSet_raw_and_data_sinthetic.csv'):
    df = pd.read_csv(datapath)

    continue_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    discrete_features = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']
    discrete_features_size = {'Gender': 2, 'CALC': 4, 'FAVC': 2, 'SCC': 2, 'SMOKE': 2,
                              'family_history_with_overweight': 2, 'CAEC': 4, 'MTRANS': 5}

    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    # encode discrete str to number, eg. male&female to 0&1
    label_encoder = LabelEncoder()
    for col in discrete_features:
        X[col] = label_encoder.fit(X[col]).transform(X[col])
    y = label_encoder.fit(y).fit_transform(y)

    # 将 X, y 转为 numpy 数组
    X = X.values
    y = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


class MyDecisionTreeClassifier:
    def __init__(
            self,
            max_depth: int = 10,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            threshold: float = 0.1
    ):
        """
        :param max_depth: 决策树最大深度
        :param min_samples_split: 内部节点再划分所需最小样本数
        :param min_samples_leaf: 叶子节点最少样本数
        :param threshold: 特征值唯一率阈值，用于判断特征是连续还是离散
        """
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        assert 0 < threshold <= 0.5
        self.threshold = threshold

    class MetaTreeNode:
        def __init__(self, feature_idx=None, splits=None, threshold=None, value=None):
            self.feature_idx = feature_idx
            self.splits = splits  # 分割字典，离散特征：键为特征属性值；连续特征：键为'left'和'right'，值为子树节点
            self.threshold = threshold  # 连续特征的划分点，离散特征为 None
            self.value = value  # 叶节点的取值，也就是 _y 中的某个值，仅当节点是叶节点时使用

    def __feature_type(self, _X):
        """
        :param _X: 特征列
        :return: 'discrete' or 'continuous'
        """
        unique_values = np.unique(_X)
        unique_ratio = len(unique_values) / len(_X)
        if len(_X) < int(1 / self.threshold):
            threshold = 0.5
        else:
            threshold = self.threshold
        if unique_ratio > threshold:
            return 'continuous'
        else:
            return 'discrete'

    @staticmethod
    def __calc_entropy(_y):
        """
        :param _y: 标签列
        :return: entropy
        """
        _, counts = np.unique(_y, return_counts=True)
        probs = counts / len(_y)
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def __calc_information_gain(self, _X, _y, _type):
        """
        :param _X: 特征列
        :param _y: 标签列
        :return:
        """
        total_entropy = self.__calc_entropy(_y)
        feature_gain = 0
        feature_threshold = None

        if _type == 'discrete':
            unique_values, counts = np.unique(_X, return_counts=True)
            weighted_entropy = 0

            for value, count in zip(unique_values, counts):
                subset_entropy = self.__calc_entropy(_y[_X == value])
                weighted_entropy += (count / len(_X)) * subset_entropy

            feature_gain = total_entropy - weighted_entropy
        else:
            # 连续变量，找到最佳划分点
            sorted_indices = np.argsort(_X)
            sorted_values = _X[sorted_indices]
            sorted_y = _y[sorted_indices]

            for i in range(1, len(sorted_values)):
                if sorted_values[i] != sorted_values[i - 1]:
                    threshold = (sorted_values[i] + sorted_values[i - 1]) / 2
                    subset_left, subset_right = sorted_y[:i], sorted_y[i:]
                    weighted_left, weighted_right = len(subset_left) / len(sorted_y), len(subset_right) / len(sorted_y)
                    subset_entropy = weighted_left * self.__calc_entropy(
                        subset_left) + weighted_right * self.__calc_entropy(subset_right)
                    gain = total_entropy - subset_entropy
                    if gain >= feature_gain:
                        # 这里不能只用大于，因为可能出现 gain 为 0 的情况
                        feature_gain = gain
                        feature_threshold = threshold

        return feature_gain, feature_threshold

    def __best_split(self, _D, _y):
        num_features = _D.shape[1]
        best_feature_idx = None
        best_splits = None
        best_gain = 0
        best_threshold = None

        for feature_idx in range(num_features):
            # 如果该特征是连续特征，那么最终只会有一个最优划分点，也就是两棵子树
            # 如果该特征是离散特征，那么要按特征的值来分割
            feature_values = _D[:, feature_idx]
            feature_type = self.__feature_type(feature_values)

            gain, threshold = self.__calc_information_gain(feature_values, _y, feature_type)

            splits = {}  # 键为特征取值，值为子树节点下标
            if feature_type == 'discrete':
                # 如果该特征是离散特征，那么按特征的值来分割
                unique_values = np.unique(feature_values)
                for value in unique_values:
                    indices = _D[:, feature_idx] == value
                    splits[value] = indices
            elif feature_type == 'continuous':
                # 如果该特征是连续特征，此时可以确定最优划分点，按最优划分点分割
                assert threshold is not None
                left_indices = _D[:, feature_idx] < threshold
                right_indices = _D[:, feature_idx] >= threshold
                splits['left'] = left_indices
                splits['right'] = right_indices
            else:
                raise ValueError('Invalid feature type')

            if gain >= best_gain:
                best_gain = gain
                best_feature_idx = feature_idx
                best_splits = splits
                best_threshold = threshold
        return best_feature_idx, best_splits, best_threshold

    def __gen_tree(self, _X, _y, _depth):
        # 计算 X 中样本最多的类别
        X_max_samples_feature_idx = np.bincount(_y).argmax()
        # 超过最大深度或样本数不足以再分割，返回叶子节点
        if _depth >= self.max_depth or len(_X) <= self.min_samples_split:
            return self.MetaTreeNode(value=X_max_samples_feature_idx)

        # X 中所有样本属于同一类别，返回叶子节点
        if len(np.unique(_y)) == 1:
            return self.MetaTreeNode(value=_y[0])

        # A 为空或样本在 A 上取值相同，返回叶子节点
        if len(np.unique(_X, axis=0)) == 1:
            return self.MetaTreeNode(value=X_max_samples_feature_idx)

        # 从 A 中选择最优划分特征
        best_feature_idx, best_splits, best_threshold = self.__best_split(_X, _y)

        # 建空结点，用于划分子树
        node = self.MetaTreeNode(feature_idx=best_feature_idx, splits={}, threshold=best_threshold)
        # 遍历每个 splits，建树
        for value, indices in best_splits.items():
            # 若子树节点样本数不大于 min_samples_leaf，返回叶子节点，将该节点的类别值设为 X 中样本最多的类别
            if len(indices) <= self.min_samples_leaf:
                node.splits[value] = self.MetaTreeNode(value=X_max_samples_feature_idx)
            else:
                node.splits[value] = self.__gen_tree(_X[indices], _y[indices], _depth + 1)

        return node

    def __predict(self, _x, node):
        """
        :param _x: 一条样本
        :param node: 当前节点
        :return: 预测值
        """
        if node.feature_idx is None:
            # 到达叶子节点
            return node.value
        else:
            feature_value = _x[node.feature_idx]
            if node.threshold is not None:
                # 连续特征
                if feature_value < node.threshold:
                    return self.__predict(_x, node.splits['left'])
                else:
                    return self.__predict(_x, node.splits['right'])
            else:
                # 离散特征
                if feature_value in node.splits.keys():
                    return self.__predict(_x, node.splits[feature_value])
                else:
                    # 未知的特征值，返回 X 中样本最多的类别
                    return node.value

    def predict(self, X):
        return [self.__predict(x, self.tree) for x in X]

    def fit(self, X, y):
        self.tree = self.__gen_tree(X, y, 0)

    def score(self, X, y):
        y_pred = self.predict(X)
        # print(y_pred[:10], y[:10])
        return np.sum(y_pred == y) / len(y)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))  # 0.9408983451536643

    my_clf = MyDecisionTreeClassifier(max_depth=10)
    my_clf.fit(X_train, y_train)
    print(my_clf.score(X_test, y_test))  # 0.9361702127659575
