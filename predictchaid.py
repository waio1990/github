# Predict CHAID generated tree
from IPython.display import display
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
import pyodbc
import os
import pandas.io.sql as psql
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import savefig
from jupyterthemes import jtplot
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
#from cholesky import estandar
from sklearn.model_selection import train_test_split
import pweave
from pweave import convert
from IPython.display import Markdown, display
from scipy.special import boxcox, inv_boxcox
import matplotlib.style as style
from CHAID import Tree


## FUNCION AUXILIAR

## FIN FUNCION

os.chdir(r'D:\Notebook')
plt.rc("font", size=14)
sns.set()
style.use('ggplot')
mpl.rcParams["figure.figsize"] = (15, 6)

# Conexion SQL
cnxn = pyodbc.connect(
    "Driver={SQL Server};Server=s2k_micro-rsg1" +
    ";Database=Ambiental;trusted_connection=yes;")
cursor = cnxn.cursor()
sql = "SELECT * FROM [S2K_DATAMART04].[dbo].[BANAL_EXPERIAN_SH_CONSOLIDADO]"
df = psql.read_sql_query(sql, cnxn)
cnxn.close()


# limpieza general

# Eliminar Rut raros
df[['Ventas_IT', 'Costos_Fijos_IT', 'Costos_Variables_IT', 'RUT']] = df[[
    'Ventas_IT', 'Costos_Fijos_IT', 'Costos_Variables_IT', 'RUT']].apply(pd.to_numeric)
df = df[df['RUT'] < 50000000]
df = df[df['RUT'] > 4000000]


# Eliminar duplicados
df.drop_duplicates(subset=['RUT'], keep=False)

# Normalizar frecuencia de ingresos
df['FRECUENCIA_INGRESO'] = df['FRECUENCIA_INGRESO'].str.strip()

# Mensualizar valores

df.loc[df['FRECUENCIA_INGRESO'] == 'A', 'Ventas_IT'] = df['Ventas_IT'] / 12
df.loc[df['FRECUENCIA_INGRESO'] == 'S', 'Ventas_IT'] = df['Ventas_IT'] / 6
df.loc[df['FRECUENCIA_INGRESO'] == 'T', 'Ventas_IT'] = df['Ventas_IT'] / 3
df.loc[df['FRECUENCIA_INGRESO'] == 'B', 'Ventas_IT'] = df['Ventas_IT'] / 2

df.loc[df['FRECUENCIA_INGRESO'] == 'A',
       'Costos_Fijos_IT'] = df['Costos_Fijos_IT'] / 12
df.loc[df['FRECUENCIA_INGRESO'] == 'S',
       'Costos_Fijos_IT'] = df['Costos_Fijos_IT'] / 6
df.loc[df['FRECUENCIA_INGRESO'] == 'T',
       'Costos_Fijos_IT'] = df['Costos_Fijos_IT'] / 3
df.loc[df['FRECUENCIA_INGRESO'] == 'B',
       'Costos_Fijos_IT'] = df['Costos_Fijos_IT'] / 2

df.loc[df['FRECUENCIA_INGRESO'] == 'A',
       'Costos_Variables_IT'] = df['Costos_Variables_IT'] / 12
df.loc[df['FRECUENCIA_INGRESO'] == 'S',
       'Costos_Variables_IT'] = df['Costos_Variables_IT'] / 6
df.loc[df['FRECUENCIA_INGRESO'] == 'T',
       'Costos_Variables_IT'] = df['Costos_Variables_IT'] / 3
df.loc[df['FRECUENCIA_INGRESO'] == 'B',
       'Costos_Variables_IT'] = df['Costos_Variables_IT'] / 2


# Valores numericos


df[['CANT_COMPRAS_MENSUALES', 'MONTO_PROM_COMPRA', 'OTRAS_DEUDAS', 'OTROS_GASTOS_ME', 'TOTAL_ACT_CIRC', 'TOTAL_ACT_FIJO']] = df[[
    'CANT_COMPRAS_MENSUALES', 'MONTO_PROM_COMPRA', 'OTRAS_DEUDAS', 'OTROS_GASTOS_ME', 'TOTAL_ACT_CIRC', 'TOTAL_ACT_FIJO']].apply(pd.to_numeric)

df[['Otros_Ingresos_IT', 'Capacidad_Pago_IT', 'Capacidad_Pago_Ajustada_IT', 'Resultado_Operacional_IT', 'Deudas_IT']] = df[[
    'Otros_Ingresos_IT', 'Capacidad_Pago_IT', 'Capacidad_Pago_Ajustada_IT', 'Resultado_Operacional_IT', 'Deudas_IT']].apply(pd.to_numeric)

# Discretizar

df['edad'] = pd.cut(df['edad'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 90])

# Guardar Base
df.to_csv('data.csv')

df['OFICINA'] = df['OFICINA'].str.strip()
independent_variable_columns = ['OFICINA']
dep_variable = 'Ventas_IT'
minsplit=len(df['Ventas_IT'])*0.03
prueba= Tree.from_pandas_df(df,
        dict(zip(independent_variable_columns, ['nominal'] * 1)), dep_variable, dep_variable_type='continuous',max_depth=5,min_parent_node_size=minsplit)
prueba
prueba.print_tree()
prueba2=prueba.tree_store

probando=predict(df,prueba)
prueba2
for i in prueba2:
    print(i.members['mean'])

rules = prueba.classification_rules()
rules
lenrules  = len(rules)
df.index = range(0,df.shape[0])
r1 = rules[2]
ruleset = list(r1.items())[1][1]

r=ruleset[0]
v = r.get('variable')
d = r.get('data')
type(r1)

def predict(df,tree):
    #we have a single ruleset
    rules = tree.classification_rules()
    df.index = range(0,df.shape[0])
    Response = np.repeat(0, df.shape[0])
    for i in range(0,len(df)-1):
        for j in range (0,len(rules)-1):
            r1 = rules[j]
            ruleset = list(r1.items())[1][1]
            r=ruleset[0]
            v = r.get('variable')
            d = r.get('data')
            #Check agains all rules
            if df[v].iloc[i] in d:
                Response[i]=r1['node']

    return Response

asdf=predict(df,prueba)
# Replace with numbers
prueba
def promedio(tree,nodes):
    Response = pd.DataFrame(columns=['Promedio'])
    nodos=tree.tree_store
    print(type(nodos))
    for i in range(0,len(nodes)-1):
        #check each node

        Response.loc[i,"Promedio"]=nodos[nodes[i]].members['mean']

    return Response

asdf2=promedio(prueba,asdf)
asdf2
