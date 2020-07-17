import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from boruta import BorutaPy


def readFile(file):
    df = pd.read_csv(file)
    return df


def readFile2(file):
    df = pd.read_csv(file, delimiter=";")
    return df


def lasso(X, y):
    reg = LassoCV()
    reg.fit(X, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(X, y))
    coef = pd.Series(reg.coef_, index=X.columns)
    imp_coef = coef.sort_values()
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind="barh")
    plt.title("Modelo Lasso para selección de variables")
    plt.show()


def boruta(X, y, features):
    X_train, X_test, y_train, y_test = split(X, y)
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    boruta_feature = BorutaPy(rf, n_estimators="auto", random_state=1, max_iter=50, perc=90)
    boruta_feature.fit(X_train, y_train)
    final_features = list()
    indexes = np.where(boruta_feature.support_ == True)
    for x in np.nditer(indexes):
        final_features.append(features[x])
    print(final_features)


def correlacion(df):
    print(df.corr(method='pearson'))
    corr = df.corr()
    # plt.figure(figsize=(30, 20))
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                linewidths=0.5,
                linecolor="black",
                annot=True)
    plt.show()

    # corr_target = abs(corr["G3"])
    # relevant_features = corr_target[corr_target > 0.15]
    # print(relevant_features)


def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test):
    sc = StandardScaler()
    sc.fit(X_train)
    sc.fit(X_test)
    X_train = sc.transform(X_train)
    #X_test = sc.transform(X_test)
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.fit_transform(X_test)
    return X_train, X_test


def smote(X_train, y_train):
    os = SMOTE(random_state=1)
    X_train, y_train = os.fit_resample(X_train, y_train)
    return X_train, y_train


def ROS(X_train, y_train):
    ros = RandomOverSampler(random_state=1)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    return X_train, y_train


def grid_search(X, y, model, modelo):
    if modelo == "nn":
        params = {"activation": ["relu", "logistic", "tanh"],
                  "hidden_layer_sizes": [10, 20, 50], "batch_size": [4, 8, 10]}
    elif modelo == "regression":
        params = {}
    elif modelo == "rf":
        params = {"criterion": ["mse", "mae"], 'max_depth': [None, 3, 5, 7],
                  'n_estimators': [100, 1000, 5000]}
    elif modelo == "svr":
        params = {"kernel": ["linear", "rbf", "poly", "sigmoid"], "C": [1, 5, 10]}
    grid = GridSearchCV(estimator=model, param_grid=params, cv=5)
    grid.fit(X, y)
    return grid


def nn(X, y):
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, y_train = ROS(X_train, y_train)
    X_train, y_train = smote(X_train, y_train)
    X_train, X_test = normalize(X_train, X_test)
    # mlp = MLPRegressor(hidden_layer_sizes=(50,), solver='lbfgs',
    #                    activation='logistic', max_iter=10000,
    #                    alpha=0.01, momentum=0.5, batch_size=4)
    mlp = MLPRegressor(hidden_layer_sizes=(20,), solver='lbfgs',
                       activation='relu', max_iter=10000,
                       alpha=0.01, momentum=0.5, batch_size=10)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    print("------Redes Neuronales-------")
    print(mlp.score(X_test, y_test))

    # print(accuracy_score(y_test, y_pred))

    resultados = cross_val_score(mlp, X, y, cv=5)
    print(resultados)
    print("Accuracy (cross-validation): %0.2f (+/- %0.2f)" % (resultados.mean(), resultados.std() * 2))

    # grid = grid_search(X, y, mlp, "nn")
    # print(grid.best_params_)
    # print(pd.DataFrame(grid.cv_results_))


def regression(X, y):
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, y_train = ROS(X_train, y_train)
    X_train, y_train = smote(X_train, y_train)
    X_train, X_test = normalize(X_train, X_test)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("------Regresión Linear------")
    print(lr.score(X_test, y_test))

    rmse = mean_squared_error(y_test, y_pred)
    print(rmse)


def random_forest(X, y):
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, y_train = ROS(X_train, y_train)
    X_train, y_train = smote(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=5000, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("------Random Forest------")
    print(rf.score(X_test, y_test))

    # print(accuracy_score(y_test, y_pred))

    # resultados = cross_val_score(rf, X, y, cv=5)
    # print(resultados)
    # print("Accuracy (cross-validation): %0.2f (+/- %0.2f)" % (resultados.mean(), resultados.std() * 2))

    # grid = grid_search(X, y, rf, "rf")
    # print(grid.best_params_)
    # print(pd.DataFrame(grid.cv_results_))


def svr(X, y):
    X_train, X_test, y_train, y_test = split(X, y)
    X_train, y_train = ROS(X_train, y_train)
    X_train, y_train = smote(X_train, y_train)
    X_train, X_test = normalize(X_train, X_test)
    # vm = SVR(kernel="linear")
    vm = SVR(kernel="rbf", C=10)
    vm.fit(X_train, y_train)
    y_pred = vm.predict(X_test)
    print("------SVR------")
    print(vm.score(X_test, y_test))

    # print(accuracy_score(y_test, y_pred))

    resultados = cross_val_score(vm, X, y, cv=5)
    print(resultados)
    print("Accuracy (cross-validation): %0.2f (+/- %0.2f)" % (resultados.mean(), resultados.std() * 2))

    # grid = grid_search(X, y, vm, "svr")
    # print(grid.best_params_)
    # print(pd.DataFrame(grid.cv_results_))


def main():
    pd.options.display.width = 0
    f = "D:/Google Drive/Universidad/Noveno/Seminario1/Experimentacion/DatasetLimpio.csv"
    df = readFile(f)
    X1 = df.iloc[:, :-1].values
    y1 = df.iloc[:, -1].values
    X2 = df.iloc[:, :-1]
    y2 = df.iloc[:, -1]

    # X = df[['health', 'absences', 'G1', 'G2']].values  # Comun
    X = df[['age', 'studytime', 'Walc', 'famrel', 'health', 'absences', 'G1', 'G2']]  # Todos
    # X = df[['studytime', 'health', 'absences', 'G1', 'G2']].values  # Boruta
    # X = df[['age', 'Walc', 'famrel', 'health', 'absences', 'G1', 'G2']]  # Lasso
    y = df['G3'].values

    # features = [i for i in df.columns if i not in ['G3']]
    # print(features)
    # boruta(X1, y1, features)
    # lasso(X2, y2)
    nn(X, y)
    regression(X, y)
    random_forest(X, y)
    svr(X, y)

    # ds = df[['age', 'studytime', 'Walc', 'famrel', 'health', 'absences', 'G1', 'G2', 'G3']]
    # correlacion(ds)

    # f2 = "D:/Google Drive/Universidad/Noveno/Seminario1/Pruebas/student-mat.csv"
    # df2 = readFile2(f2)
    # X2 = df2.iloc[:, :-1].values
    # y2 = df2.iloc[:, -1].values


if __name__ == '__main__':
    main()
