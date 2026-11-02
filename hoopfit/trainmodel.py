#导入外部package做机器学习训练因为python自己实现不了
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression









import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# 加载数据 用pandas读取csv文件
data = pd.read_csv('data.csv')
# data = pd.read_excel('addMoreData1.xlsx', sheet_name='Sheet1')

# 特征和目标
features = ['GP', 'Pts', 'OReb', 'DReb', 'Ast', 'Stl', 'Blk']
targets = ['F', 'G', 'C']

X = data[features]
y_forward = data[targets[0]].astype(int)
y_guard = data[targets[1]].astype(int)
y_center = data[targets[2]].astype(int)

# 分割数据集
X_train, X_test, y_train_forward, y_test_forward = train_test_split(X, y_forward, test_size=0.2, random_state=12)
_, _, y_train_guard, y_test_guard = train_test_split(X, y_guard, test_size=0.2, random_state=12)
_, _, y_train_center, y_test_center = train_test_split(X, y_center, test_size=0.2, random_state=12)

# 标准化 对metric做处理以更适合机器学习训练 让各个metric平均值=0 方差=1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 模型和参数网格 写gridsearch时要输入所有参数组合 参考文献/资料根据经验选择
param_grid = {
    "Logistic Regression": {
        "C": [0.1, 1, 10],
        "solver": ['lbfgs', 'liblinear']
    },
    "Random Forest": {
        "n_estimators": [50],
        "max_depth": [5]
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5],
        "weights": ['uniform', 'distance']
    },
    "Decision Tree": {
        "max_depth": [5],
        "criterion": ['gini']
    },
    "XGBoost": {
        "n_estimators": [50],
        "learning_rate": [0.01]
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ['linear', 'rbf']
    }
}


# 网格搜索和模型选择
def train_and_select_best(X_train, X_test, y_train, y_test):
    best_model = None
    best_acc = 0
    best_params = {}
    
    for model_name, param in param_grid.items():
        print(f"Training {model_name} with GridSearchCV...")

        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "K-Nearest Neighbors":
            model = KNeighborsClassifier()
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        elif model_name == "SVM":
            model = SVC(probability=True)

        grid_search = GridSearchCV(model, param, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best = grid_search.best_estimator_
        test_acc = accuracy_score(y_test, best.predict(X_test))

            # 训练集预测和指标计算 Training set prediction and metrics calculation
        y_pred_train = grid_search.predict(X_train)
        y_proba_train = grid_search.predict_proba(X_train)[:, 1]
        auc_train = roc_auc_score(y_train, y_proba_train)
        acc_train = accuracy_score(y_train, y_pred_train)

        # 测试集预测和指标计算 Testing set prediction and metrics calculation
        y_pred_test = grid_search.predict(X_test)
        y_proba_test = grid_search.predict_proba(X_test)[:, 1]
        auc_test = roc_auc_score(y_test, y_proba_test)
        acc_test = accuracy_score(y_test, y_pred_test)

        print(f" Best Params for {model_name}: {grid_search.best_params_}")
        print(f" Train Accuracy: {auc_train:.2f}\n")
        print(f" Test Accuracy: {auc_test:.2f}\n")

        if auc_test > best_acc:
            best_acc = auc_test
            best_model = best
            best_params = grid_search.best_params_

    return best_model, best_params


# 分别训练三个位置的模型
best_forward, params_forward = train_and_select_best(X_train, X_test, y_train_forward, y_test_forward)
print("Forward Finished!")
best_guard, params_guard = train_and_select_best(X_train, X_test, y_train_guard, y_test_guard)
print("Guard Finished!")
best_center, params_center = train_and_select_best(X_train, X_test, y_train_center, y_test_center)
print("Center Finished!")

print("\n Optimal parameters：")
print("Forward:", params_forward)
print("Guard:", params_guard)
print("Center:", params_center)


# # GUI 界面
# def create_gui():
#     root = tk.Tk()
#     root.title("Player Position Suitability & Rating")
#     root.configure(bg='#d4edda')

#     labels = ['GP', 'Pts', 'OReb', 'DReb', 'Ast', 'Stl', 'Blk']
#     entries = {}

#     for i, label in enumerate(labels):
#         tk.Label(root, text=label, bg='#d4edda', fg='green', font=('Helvetica', 12)).grid(row=i, column=0, sticky='w', padx=10, pady=5)
#         entry = tk.Entry(root, font=('Helvetica', 12))
#         entry.grid(row=i, column=1, padx=10, pady=5)
#         entries[label] = entry

#     # 下拉选择框
#     position_var = tk.StringVar(value='Forward')
#     tk.Label(root, text="Select Position", bg='#d4edda', fg='green', font=('Helvetica', 12)).grid(row=len(labels), column=0, sticky='w', padx=10, pady=5)
#     position_menu = ttk.Combobox(root, textvariable=position_var, values=['Forward', 'Guard', 'Center'], font=('Helvetica', 12))
#     position_menu.grid(row=len(labels), column=1, padx=10, pady=5)

#     # 预测功能
#     def predict():
#         try:
#             input_data = [float(entries[label].get()) for label in labels]
#             input_data = scaler.transform([input_data])
#             position = position_var.get()

#             if position == 'Forward':
#                 model = best_forward
#             elif position == 'Guard':
#                 model = best_guard
#             else:
#                 model = best_center

#             prediction = model.predict(input_data)
#             probability = model.predict_proba(input_data)[0][1]

#             result = f"{position} Suitability: {'Yes' if prediction[0] == 1 else 'No'}\nScore: {probability:.2f}"
#             messagebox.showinfo("Prediction Result", result)

#         except Exception as e:
#             messagebox.showerror("Error", f"Invalid input: {e}")

#     # 按钮
#     predict_button = tk.Button(root, text="Predict", command=predict, font=('Helvetica', 12, 'bold'), bg='green', fg='white')
#     predict_button.grid(row=len(labels) + 1, column=0, columnspan=2, pady=15)

#     root.mainloop()


# # 防止打包时报错
# if __name__ == "__main__":
#     create_gui()
