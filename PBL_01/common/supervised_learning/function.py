import numpy as np
np.random.seed(seed=32)

# 表示オプションの変更
import pandas as pd
pd.options.display.max_columns = 100
pd.set_option('display.max_rows', 500)

# imputation
def imputation(output_data_dir, model_columns_file_name, X_ohe):
    from sklearn.impute import SimpleImputer
    for column, is_empty in X_ohe.isnull().all(axis=0).iteritems():
        if is_empty:
            print(column)
            X_ohe = X_ohe.drop(column, axis=1)
    X_ohe = X_ohe.reset_index(drop=True)
    imp = SimpleImputer(strategy='mean')
    #imp.fit_transform(X_ohe)
    imp.fit(X_ohe)
    from joblib import dump
    dump(imp, output_data_dir + 'imputer.joblib')
    X_ohe_columns = X_ohe.columns.values
    X_ohe = pd.DataFrame(imp.transform(X_ohe), columns=X_ohe_columns)
    pd.DataFrame(X_ohe_columns).to_csv(model_columns_file_name, index=False)
    pd.DataFrame(X_ohe_columns).to_csv(output_data_dir + 'model_columns' + "_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.csv', index=False)
    return X_ohe, X_ohe_columns

# one-hot encoding
def one_hot_encoding(X, ohe_columns=[]):
    X_ohe = pd.get_dummies(X, dummy_na=True, columns=ohe_columns)
    return X_ohe

# feature selection
def feature_selection(output_data_dir, n_features_to_select,
                      X, y, X_ohe_columns, algorithm_name, estimator):
    from sklearn.feature_selection import RFE
    from joblib import dump
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=.05)
    #selector.fit_transform(X, y)
    selector.fit(X, y)
    dump(selector, output_data_dir + algorithm_name + '_selector.joblib')
    X_fin = X.loc[:, X_ohe_columns[selector.support_]]
    return X_fin

def root_mean_squared_log_error(y_true, y_pred):
    from sklearn.metrics import mean_squared_log_error
    #y_true = np.log1p(y_true)
    #y_pred = np.log1p(y_pred)
    y_pred[y_pred<0] = 0
    y_true[y_true<0] = 0
    #return np.sqrt(np.mean((((y_true)-(y_pred))**2)))
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
    #return np.sqrt(np.mean(((np.log1p(y_true+1)-np.log1p(y_pred+1))**2)))
    #return np.sqrt(np.mean(((np.log(y_true+1))**2)-np.log(y_pred+1)))
    #return np.sqrt(mean_squared_log_error(np.exp(y_true), np.exp(y_pred)))

def root_mean_squared_error(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import mean_squared_error
    #y_true = np.log1p(y_true)
    #y_pred = np.log1p(y_pred)
    #y_pred[y_pred<0] = 0
    #y_true[y_true<0] = 0
    #y_pred = np.expm1(y_pred)
    #y_true = np.expm1(y_true)
    #return np.sqrt(np.mean((((y_true)-(y_pred))**2)))
    return np.sqrt(mean_squared_error(y_true, y_pred))

# holdout
def holdout(X_ohe, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_ohe,
                                                y,
                                                test_size=0.2,
                                                random_state=1)
    return X_train, X_test, y_train, y_test

def evaluation(pipelines, output_data_dir, scores, X_train, y_train, phase, function_evaluation):
    for pipe_name, _ in pipelines.items():
        scores[(pipe_name, phase)] = function_evaluation(y_train, (scoring(output_data_dir, pipe_name, X_train)))

def execute_feature_selection(output_data_dir, pipelines, feature_selection_rf_list, n_features_to_select,
                              X_ohe, y, X_ohe_columns, is_holdout, is_optuna, is_feature_selection=True, y_column=''):
    X_train = {}
    y_train = {}
    X_valid = {}
    y_valid = {}
    for pipe_name, pipeline in pipelines.items():
        print(pipe_name)
        if is_feature_selection:
            print('feature_selection')
            if pipe_name in feature_selection_rf_list:
                X_featured = feature_selection(output_data_dir, n_features_to_select,
                                               X_ohe, y, X_ohe_columns, pipe_name,
                                               pipelines['rf'].named_steps['est'])
            else:
                X_featured = feature_selection(output_data_dir, n_features_to_select,
                                               X_ohe, y, X_ohe_columns, pipe_name,
                                               pipeline.named_steps['est'])
            #X_featured.to_csv(output_data_dir + pipe_name + "_" + str(n_features_to_select)
            #                  + "_X_featured.csv", index=False, header=True)
        else:
            X_featured = X_ohe
        #if is_holdout | (pipe_name in 'lightgbm' and is_optuna):
        if is_holdout:
            X_train[pipe_name], X_valid[pipe_name], y_train[pipe_name], y_valid[pipe_name] = holdout(X_featured, y)
        else:
            X_train[pipe_name], X_valid[pipe_name], y_train[pipe_name], y_valid[pipe_name] = X_featured, X_featured, y, y
    return X_train, X_valid, y_train, y_valid

def get_input(x):
    return x

def choice(options):
    from ipywidgets import interact,interactive,fixed,interact_manual
    from IPython.display import display
    import ipywidgets as widgets
    input = get_input(widgets.RadioButtons(options=options))
    display(input)
    return input

def cross_validatior(scorer, output_data_dir, n_features_to_select, pipelines, input_evaluation, X_ohe, y):
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    str_all_print = '評価指標:' + input_evaluation.value + '\n'
    str_all_print += 'n_features_to_select:' + str(n_features_to_select) + '\n'
    print('評価指標:' + input_evaluation.value)
    str_print = ''
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for pipe_name, est in pipelines.items():
        cv_results = -cross_val_score(est, X_ohe, y, cv=kf, scoring=scorer)
        str_print = '----------' + '\n' + 'algorithm:' + str(pipe_name) + '\n' + 'cv_results:' + str(cv_results) + '\n' + 'avg +- std_dev ' + str(cv_results.mean()) + '+-' + str(cv_results.std()) + '\n'
        #print('----------')
        #print('algorithm:', pipe_name)
        #print('cv_results:', cv_results)
        #print('avg +- std_dev', cv_results.mean(),'+-', cv_results.std())
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

# preprocessing
def preprocessing(output_data_dir, model_columns_file_name, score_columns_file_name, 
                algorithm_name, X_ohe, X_ohe_s,
                is_imputation=True, is_feature_selection=True):
    #from pandas.io.common import EmptyDataError
    import pandas
    from joblib import load
    try:
        model_columns = pd.read_csv(model_columns_file_name)
        X_ohe_columns = model_columns.values.flatten()
        cols_model = set(X_ohe_columns)
    except pd.errors.EmptyDataError:
        X_ohe_columns = X_ohe.columns
        cols_model = set(X_ohe.columns.values)
    except FileNotFoundError:
        X_ohe_columns = X_ohe.columns
        cols_model = set(X_ohe.columns.values) 
    #except pandas.io.common.EmptyDataError:
    #    X_ohe_columns = X_ohe.columns
    #    cols_model = set(X_ohe.columns.values)
    '''
    model_columns = pd.read_csv(model_columns_file_name)
    X_ohe_columns = model_columns.values.flatten()
    cols_model = set(X_ohe_columns)
    '''

    X_ohe_s_columns = X_ohe_s.columns.values
    pd.DataFrame(X_ohe_s_columns).to_csv(score_columns_file_name, index=False)

    cols_score = set(X_ohe_s.columns.values)
    diff1 = cols_model - cols_score
    print('モデルのみに存在する項目: %s' % diff1)
    diff2 = cols_score - cols_model
    print('スコアのみに存在する項目: %s' % diff2)
    df_cols_m = pd.DataFrame(None, columns=X_ohe_columns, dtype=float)
    X_ohe_s2 = pd.concat([df_cols_m, X_ohe_s])

    pd.DataFrame(X_ohe_s2).to_csv(output_data_dir + "X_ohe_s2.csv", index=False)

    set_Xm = set(X_ohe.columns.values)
    set_Xs = set(X_ohe_s.columns.values)
    X_ohe_s3 = X_ohe_s2.drop(list(set_Xs - set_Xm), axis=1)
    #pd.DataFrame(X_ohe_columns).to_csv(model_columns_file_name, index=False)

    pd.DataFrame(X_ohe_s3).to_csv(output_data_dir + "X_ohe_s3.csv", index=False)

    X_ohe_s3.loc[:,list(set_Xm - set_Xs)] = X_ohe_s3.loc[:,list(set_Xm - set_Xs)].fillna(0, axis=1)
    X_ohe_s3 = X_ohe_s3.reindex(X_ohe.columns.values, axis=1)
    X_ohe_s4 = X_ohe_s3
    if is_imputation:
        imp = load(output_data_dir + 'imputer.joblib')
        X_ohe_s4 = pd.DataFrame(imp.transform(X_ohe_s3), columns=X_ohe_columns)
    X_fin_s = X_ohe_s4
    if is_feature_selection:
        selector = load(output_data_dir + algorithm_name + '_selector.joblib')
        X_fin_s = X_ohe_s4.loc[:, X_ohe_columns[selector.support_]]
    return X_fin_s

# preprocessing
def preprocessing2(output_data_dir, model_columns_file_name, algorithm_name, X_ohe, X_ohe_s):
    import pandas as pd
    from joblib import load
    model_columns = pd.read_csv(model_columns_file_name)
    X_ohe_columns = model_columns.values.flatten()
    cols_model = set(X_ohe_columns)
    cols_score = set(X_ohe_s.columns.values)
    diff1 = cols_model - cols_score
    print('モデルのみに存在する項目: %s' % diff1)
    diff2 = cols_score - cols_model
    print('スコアのみに存在する項目: %s' % diff2)
    df_cols_m = pd.DataFrame(None, columns=X_ohe_columns, dtype=float)
    X_ohe_s2 = pd.concat([df_cols_m, X_ohe_s])
    set_Xm = set(X_ohe.columns.values)
    set_Xs = set(X_ohe_s.columns.values)
    X_ohe_s3 = X_ohe_s2.drop(list(set_Xs-set_Xm), axis=1)
    X_ohe_s3.loc[:,list(set_Xm-set_Xs)] = X_ohe_s3.loc[:,list(set_Xm-set_Xs)].fillna(0, axis=1)
    X_ohe_s3 = X_ohe_s3.reindex(X_ohe.columns.values, axis=1)
    imp = load(output_data_dir + 'imputer.joblib')
    X_fin_s = pd.DataFrame(imp.transform(X_ohe_s3), columns=X_ohe_columns)
    return X_fin_s

import datetime
def output_file(output_data_dir, n_features_to_select, target_label, df, id_label,
                y, model_name, extension, header=True):
    #y = [i if i > 0.01 else 0 for i in y]
    publish_file_name = output_data_dir + "submission_" + model_name + "_" + str(n_features_to_select) + "_" +  datetime.datetime.now().strftime('%Y%m%d%H%M%S') + "." + extension
    file_name = output_data_dir + "submission_" + model_name + "." + extension
    separator = ','
    if extension == 'tsv':
        separator = '\t'
    if id_label != '':
        df_output = pd.concat([df[id_label], pd.DataFrame(y, columns=target_label)], axis=1)
        df_output.to_csv(publish_file_name, index=False, sep=separator, header=header)
        df_output.to_csv(file_name, index=False, sep=separator, header=header)
        #concated = pd.concat([df, pd.DataFrame(y, columns=[target_label])], axis=1)
        #pd.DataFrame(concated[id_label, target_label]).to_csv(file_name, index=False, sep=separator, header=header)
        #concated = pd.concat([df, pd.DataFrame(y, columns=[target_label])], axis=1)
        #pd.DataFrame(concated[id_label, target_label]).to_csv(file_name, index=False, sep=separator, header=header)
        #concated[id_label, target_label].to_csv(file_name, index=False, sep=separator, header=header)
    else:
        df_output = pd.concat([df, pd.DataFrame(y, columns=target_label)], axis=1)
        df_output.to_csv(publish_file_name, index=False, sep=separator, header=header)
        df_output.to_csv(file_name, index=False, sep=separator, header=header)

def convert_to_count_encode(X:pd.core.series.Series, column:str):
    import collections
    counter = collections.Counter(X[column].values)
    count_dict = dict(counter.most_common())
    encoded = X[column].map(lambda x: count_dict[x]).values
    return encoded

def convert_to_label_encode(X:pd.core.series.Series, column:str):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    encoded = le.fit_transform(X[column].astype(str).values)
    return encoded

def convert_to_label_count_encode(X:pd.core.series.Series, column:str):
    import collections
    counter = collections.Counter(X[column].values)
    count_dict = dict(counter.most_common())
    label_count_dict = {key:i for i, key in enumerate(count_dict.keys(), start=1)}
    encoded = X[column].map(lambda x: label_count_dict[x]).values
    return encoded

def target_encode(X:pd.core.series.Series, y:pd.core.series.Series,
                   df_s:pd.core.series.Series, X_s:pd.core.series.Series,
                   ohe_columns=[]):
    for column in ohe_columns:
        convert_to_target_encode(X, y, X_s, column, "target_encoded_" + column)
    return X, X_s

def convert_to_target_encode(X, y, X_s, input_column_name, output_column_name):
    from sklearn.model_selection import KFold
    # nan埋め処理
    X[input_column_name] = X[input_column_name].fillna('-1')
    X_s[input_column_name] = X_s[input_column_name].fillna('-1')

    kf = KFold(n_splits=5, shuffle=True, random_state=71)
    #=========================================================#
    c = input_column_name
    # 学習データ全体で各カテゴリにおけるyの平均を計算
    data_tmp = pd.DataFrame({c: X[c], 'target':y})
    target_mean = data_tmp.groupby(c)['target'].mean()
    #テストデータのカテゴリを置換
    X_s[output_column_name] = X_s[c].map(target_mean)
    
    # 変換後の値を格納する配列を準備
    tmp = np.repeat(np.nan, X.shape[0])

    for i, (train_index, test_index) in enumerate(kf.split(X)): # NFOLDS回まわる
        #学習データについて、各カテゴリにおける目的変数の平均を計算
        target_mean = data_tmp.iloc[train_index].groupby(c)['target'].mean()
        #バリデーションデータについて、変換後の値を一時配列に格納
        tmp[test_index] = X[c].iloc[test_index].map(target_mean) 

    #変換後のデータで元の変数を置換
    X[output_column_name] = tmp

#def convert_to_target_encode(X:pd.core.series.Series, column:str, target:str):
#    target_dict = X[[column, target]].groupby([column])[target].mean().to_dict()
#    encoded = X[column].map(lambda x: target_dict[x]).values
#    return encoded

def convert_to_frequency_encode(X:pd.core.series.Series, column:str):
    encoded = X.groupby(column).size() / len(X)
    encoded = X[column].map(encoded)
    return encoded

def convert_to_multi_label_binarized(X:pd.core.series.Series, columns=[]):
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    mlb.fit([set(X[columns[0]].unique())])
    #weapon = X.fillna('none')
    #pd.DataFrame(weapon).to_csv(out_put_data_dir + "weapon.csv", index=False)
    X_binarized = mlb.transform(X[columns].values)
    #return pd.DataFrame(weapon_binarized, columns=mlb.classes_).drop('none', axis=1)
    return pd.DataFrame(X_binarized, columns=mlb.classes_)



