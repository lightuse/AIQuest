from .supervised_learning import function

import numpy as np
np.random.seed(seed=32)

# 表示オプションの変更
import pandas as pd
pd.options.display.max_columns = 100
pd.set_option('display.max_rows', 500)

evaluation_list = {'AUC':'roc_auc',
                   'F1':'f1',
                   'Recall':'recall',
                   'Precision':'precision',
                   'Accuracy':'accuracy'}
options_algorithm = ['lightgbm', 'knn', 'ols', 'ridge', 'tree', 'rf', 'gbr1', 'gbr2', 'xgboost', 'catboost']

feature_importances_algorithm_list = ['tree', 'rf', 'gbr1', 'gbr2', 'lightgbm', 'xgboost', 'catboost']

xgboost_params = {
    'objective' : 'reg:squarederror',
    #'tree_method' : 'gpu_hist',
    'random_state' : 1,
    'eval_metric': 'rmse'
}
catboost_params = {
    #'task_type' : 'GPU',
    'random_state' : 1,
    'eval_metric': 'RMSE',
    'num_boost_round' : 10000
}

def setup_algorithm(pipelines = {}):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import RFE
    pipelines = {
        'lightgbm':
            Pipeline([('pca', PCA(random_state=1)),
                    #('est', lgb.LGBMClassifier(random_state=1, objective='binary', metric='binary_logloss', device='gpu'))]),
                    ('est', lgb.LGBMRegressor(random_state=1, objective='regression', metric='rmse'))]),
	    'catboost':
	        Pipeline([('pca', PCA(random_state=1)),
	                  ('est', CatBoostRegressor(**catboost_params))]),
	    'xgboost':
	        Pipeline([('pca', PCA(random_state=1)),
	                  ('est', xgb.XGBRegressor(**xgboost_params))]),
	    'knn': Pipeline([('reduct', PCA(random_state=1)),
	                      ('est', KNeighborsRegressor())]), 
	    'ols': Pipeline([('scl', StandardScaler()),
	                      ('est', LinearRegression())]),
	    'ridge':Pipeline([('scl', StandardScaler()),
	                      ('est', Ridge(random_state=0))]),
	    'tree': Pipeline([('reduct', PCA(random_state=1)),
	                    ('est', DecisionTreeRegressor(random_state=1))]),
	    'rf': Pipeline([#('reduct', PCA(random_state=1, n_components=n_components)),
                        ('reduct', PCA(random_state=1)),
	                    ('est', RandomForestRegressor(max_depth=5, n_estimators=10, random_state=0))]),
	    'gbr1': Pipeline([('reduct', PCA(random_state=1)),
	                      ('est', GradientBoostingRegressor(random_state=0))]),
	    'gbr2': Pipeline([('reduct', PCA(random_state=1)),
	                      ('est', GradientBoostingRegressor(n_estimators=250, random_state=0))]),
    }
    return pipelines

tuning_prarameter_list = []
# パラメータグリッドの設定
tuning_prarameter = {
    'lightgbm':{
        'est__learning_rate': [0.1,0.05,0.01],
        'est__n_estimators':[1000,2000],
        'est__num_leaves':[31,15,7,3],
        'est__max_depth':[4,8,16]
    },
    'tree':{
        "est__min_samples_split": [10, 20, 40],
        "est__max_depth": [2, 6, 8],
        "est__min_samples_leaf": [20, 40, 100],
        "est__max_leaf_nodes": [5, 20, 100],
    },
    'rf':{
        'est__n_estimators':[5,10,20,50,100],
        'est__max_depth':[1,2,3,4,5],
    }
}

def scoring(output_data_dir, algorithm_name, X):
    from joblib import load
    print('scoring')
    clf = load(output_data_dir + 'model/' + algorithm_name + '_regressor.joblib')
    print('scoring_clf')
    return clf.predict(X)
'''
from sklearn.model_selection import cross_val_score
def cross_validatior(scorer, output_data_dir, n_features_to_select, pipelines, input_evaluation, X, y):
    from sklearn.model_selection import KFold
    skf = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)
    for model, target in enumerate(targets, 1):
        for trn_idx, test_idx in skf.split(X, y):
            for pipe_name, est in pipelines.items():
                oof[test_idx] += scoring(output_data_dir, pipe_name, X.iloc[test_idx]) / skf.n_split
                reds += clf.predict(test[features]) / skf.n_splits
'''
def cross_validatior(kf, scorer, output_data_dir, n_features_to_select, pipelines, input_evaluation, X_train, y_train):
    from sklearn.model_selection import cross_val_score

    str_all_print = 'n_features_to_select:' + str(n_features_to_select) + '\n'
    print('評価指標:' + input_evaluation.value)
    str_print = ''
    for pipe_name, est in pipelines.items():
        cv_results = cross_val_score(est,
                                    X_train, y_train,
                                    cv=kf,
                                    scoring=scorer)  
        str_print = '----------' + '\n' + 'algorithm:' + str(pipe_name) + '\n' + 'cv_results:' + str(cv_results) + '\n' + 'avg +- std_dev ' + str(cv_results.mean()) + '+-' + str(cv_results.std()) + '\n'
        print(str_print)
        str_all_print += str_print
    import datetime
    with open(output_data_dir + 'cv_results' + '_' + str(n_features_to_select) + "_" +  datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.txt', mode='w') as f:
        f.write(str_all_print)

def decide_evaluation(input_evaluation):
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import mean_absolute_error
	from sklearn.metrics import r2_score
	from sklearn.metrics import log_loss
	function_evaluation = mean_squared_error
	if input_evaluation.value == 'LOG_LOSS':
	    function_evaluation = log_loss
	if input_evaluation.value == 'RMSE':
	    function_evaluation = root_mean_squared_error
	elif input_evaluation.value == 'MAE':
	    function_evaluation = mean_absolute_error
	elif input_evaluation.value == 'R2':
	    function_evaluation = r2_score
	elif input_evaluation.value == 'RMSLE':
	    function_evaluation = root_mean_squared_log_error
	return function_evaluation

def evaluation(output_data_dir, scores, pipelines, X_train, y_train, phase, evaluation_function_list,
                input_evaluation):
    print(input_evaluation.value)
    for pipe_name, _ in pipelines.items():
        scores[(pipe_name, phase)] = evaluation_function_list[input_evaluation.value](y_train[pipe_name],
                                        scoring(output_data_dir, pipe_name, X_train[pipe_name]))

def exec_evaluation(scores, is_holdout, output_data_dir, pipelines, function_evaluation, input_evaluation,
                     X_train, Y_train, X_valid, Y_valid):
    if is_holdout:
        evaluation(output_data_dir, scores, pipelines, X_train, Y_train, 'train', function_evaluation, input_evaluation)
        evaluation(output_data_dir, scores, pipelines, X_valid, Y_valid, 'valid', function_evaluation, input_evaluation)
    else:
        evaluation(output_data_dir, scores, pipelines, X_train, Y_train, 'train', function_evaluation, input_evaluation)

def display_evaluation(is_holdout, output_data_dir, pipelines, function_evaluation, input_evaluation,
                        X_train, y_train, X_valid, y_valid):
    import pandas as pd
    scores = {}
    exec_evaluation(scores, is_holdout, output_data_dir, pipelines, function_evaluation, input_evaluation,
                    X_train, y_train, X_valid, y_valid)
    # sort score
    #sorted_score = sorted(scores.items(), key=lambda x:-x[1])
    ascending = True
    if input_evaluation.value == 'R2':
        ascending = False
    print('評価指標:' + input_evaluation.value)
    if is_holdout:
        display(pd.Series(scores).unstack().sort_values(by=['valid'], ascending=[ascending]))
    else:
        display(pd.Series(scores).unstack().sort_values(by=['train'], ascending=[ascending]))

# train
def train_model(output_data_dir, pipelines, X_train, X_valid, y_train, y_valid,
                evaluation, is_holdout, is_optuna=False, is_sklearn=True, categorical_feature = [], y_column='', ):
    if is_optuna:
        import optuna.integration.lightgbm as lgb
    elif not is_sklearn:
        import lightgbm as lgb
    from joblib import dump
    from sklearn.model_selection import GridSearchCV
    for pipe_name, pipeline in pipelines.items():
        print(pipe_name)
        if pipe_name in tuning_prarameter_list:
            gs = GridSearchCV(estimator=pipeline,
                        param_grid=tuning_prarameter[pipe_name],
                        scoring=evaluation_list[evaluation],
                        cv=3,
                        return_train_score=False)
            gs.fit(X_train, y_train)
            dump(gs, output_data_dir + 'model/' + pipe_name + '_regressor.joblib')
            gs.fit(X_valid, y_valid)
            # 探索した結果のベストスコアとパラメータの取得
            print(pipe_name + ' Best Score:', gs.best_score_)
            print(pipe_name + ' Best Params', gs.best_params_)
        else:
            if pipe_name in 'lightgbm' and ((is_optuna) or (not is_sklearn)):
                lgb_train = lgb.Dataset(X_train[pipe_name], y_train[pipe_name],
                            categorical_feature = categorical_feature
                            )
                lgb_eval = lgb.Dataset(X_valid[pipe_name], y_valid[pipe_name], reference=lgb_train)
                params = {
                    # 二値分類問題
                    'objective': 'regression',
                    # 損失関数は二値のlogloss
                    #'metric': 'auc',
                    'metric': 'rmse',
                    # 最大イテレーション回数指定
                    'num_iterations' : 1000,
                    # early_stopping 回数指定
                    'early_stopping_rounds' : 100,
                    #'metric': 'binary_logloss',
                    #'verbosity': -1,
                    #'boosting_type': 'gbdt',
                }
                #params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'feature_pre_filter': False, 'lambda_l1': 8.033158760223655, 'lambda_l2': 1.1530347880300857e-07, 'num_leaves': 4, 'feature_fraction': 0.5, 'bagging_fraction': 0.9430649230190336, 'bagging_freq': 1, 'min_child_samples': 20}
                #params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'feature_pre_filter': False, 'lambda_l1': 7.448574459066696e-08, 'lambda_l2': 1.3771631987966848e-05, 'num_leaves': 3, 'feature_fraction': 0.516, 'bagging_fraction': 0.8471270267389193, 'bagging_freq': 5, 'min_child_samples': 20}
                best = lgb.train(params,
                            lgb_train,
                            valid_sets=[lgb_train, lgb_eval],
                            verbose_eval=0,
                            categorical_feature = categorical_feature
                            )
                dump(best, output_data_dir + 'model/' + pipe_name + '_regressor.joblib') 
            else:
                print('normal')
                if pipe_name in 'lightgbm':
                    clf = pipeline.fit(X_train[pipe_name], y_train[pipe_name],
                                        #categorical_feature = categorical_feature
                                        )
                else:
                    #X_train[pipe_name] = function.one_hot_encoding(X_train[pipe_name], categorical_feature)
                    clf = pipeline.fit(X_train[pipe_name], y_train[pipe_name])
                dump(clf, output_data_dir + 'model/' + pipe_name + '_regressor.joblib')
                if is_holdout:
                    clf = pipeline.fit(X_valid[pipe_name], y_valid[pipe_name])
    return X_train, X_valid, y_train, y_valid