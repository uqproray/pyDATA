# -*- coding: utf-8 -*-

import pandas as pd

base = pd.read_csv('dadus.csv')


## pegando todas as linhas e da coluna 0 até 13
predictors = base.iloc[:, 0:13].values

classe = base.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()

## transformando em atributo discreto/numerico. alguns algoritmos precisam dessa action.
## o atributo 'qllq' por exemplo será identificado pelo numero 5
##labels = labelencoder_previsores.fit_transform(previsores[:,1])

predictors[:,1] = labelencoder_previsores.fit_transform(predictors[:,1])
predictors[:,2] = labelencoder_previsores.fit_transform(predictors[:,2])
predictors[:,4] = labelencoder_previsores.fit_transform(predictors[:,4])
predictors[:,5] = labelencoder_previsores.fit_transform(predictors[:,5])
predictors[:,6] = labelencoder_previsores.fit_transform(predictors[:,6])
predictors[:,7] = labelencoder_previsores.fit_transform(predictors[:,7])
predictors[:,8] = labelencoder_previsores.fit_transform(predictors[:,8])
predictors[:,12] = labelencoder_previsores.fit_transform(predictors[:,12])
