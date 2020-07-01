# -*- coding: utf-8 -*-

import pandas as pd

base = pd.read_csv('dadus.csv')


## pegando todas as linhas e da coluna 0 até 13
previsores = base.iloc[:, 0:13].values

classe = base.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()

## transformando em atributo discreto/numerico. alguns algoritmos precisam dessa action.
## o atributo 'qllq' por exemplo será identificado pelo numero 5
##labels = labelencoder_previsores.fit_transform(previsores[:,1])


previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])