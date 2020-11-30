""" Tabulate the results """

import tabulate
import numpy as np
import pandas as pd

headers= ['model','precision','recall','accuracy','f1']
# Read result file
df=pd.read_json('results.json',lines=True)
df=df.drop(['model','time'],axis=1)

table=list(df.values)
table.sort(key=lambda r:r[-1], reverse=True)
print(tabulate.tabulate(table,headers=headers))
