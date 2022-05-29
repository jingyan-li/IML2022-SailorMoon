import sklearn.linear_model
from sklearn.metrics import mean_squared_error
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F


def extract_data(identifier="pretrain"):
    feature_filename = "data/" + identifier + "_features.csv/" + identifier + "_features.csv"
    label_filename = "data/" + identifier + "_labels.csv/" + identifier + "_labels.csv"
    feature = pd.read_csv(feature_filename).drop(columns=['Id','smiles'])
    if identifier != "test":
        label = pd.read_csv(label_filename).drop(columns=(['Id']))
    else:
        label = None
    return feature, label


def pretrained(x,y):
    ## several different models and automatically choose one

    # data split
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3,
                                                                                random_state=666)

    ## neurual network model (assume one epoch)
    model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(500,100,20))
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    score = mean_squared_error(y_test,y_pred,squared=False)
    print("Pretrained scoring:"+score)
    return model


### todo: pretrained model
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #self.conv1 =nn.Conv1d(1,)

    def forward(self, x):



def encoder(x,y):

    # data splitting
    #x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3,
#                                                                                random_state=666)

    #estimator =
    #encoder_model = sklearn.model_selection.GridSearchCV(estimator )
    encoder_model = sklearn.linear_model.RidgeCV(alphas = [0.05, 0.1, 1],store_cv_values=True)
    encoder_model.fit(x,y)
    #y_pred = encoder_model.predict(x_test)
    #score = mean_squared_error(y_test, y_pred, squared=False)
    #print("encoder scoring:"+score)
    return  encoder_model


if __name__ == '__main__':
    print("start task 4")
    ### data extraction
    pretrain_x, pretrain_y = extract_data("pretrain")
    print("data extracted")
    ### pretrain
    pretrain_model = pretrained(pretrain_x,pretrain_y)
    print("pretrained done")

    ### get encoder data
    x, y_encoder = extract_data("train")
    x_encoder = pretrain_model.predict(x)

    ### train x_encoder, y_encoder
    encoder_model = encoder(x_encoder,y_encoder)
    print("encoder done")
    ### test
    x_test, _ = extract_data("test")

    y_pre = pretrain_model.predict(x_test)
    y_pre_enc = encoder_model.predict(y_pre)
    print("predicted")
    ### save

    # need id
    id = pd.read_csv("test_features.csv/test_features.csv",usecols=['Id'])

    pred = pd.DataFrame(id,y_pre_enc)
    pred.to_csv("result.csv")
    print("task done")

    #print(mean_squared_error(y_true,y_pre_enc,squared=False))
