#!/usr/bin/env python
# coding: utf-8

# 
# ![](https://www.iteso.mx/documents/27014/202031/Logo-ITESO-MinimoV.png)
# 
# # Departamento de Matemáticas y Física
# ## Ciencia de datos e inteligencia de negocios
# ## Proyecto de aplicación: Manejo de Datos, Similitud y Clustering

# ### Integrantes
# - Flavio Cesar Palacios Salas
# - Andres Gonzales Luna Diaz del Castillo
# - Maximiliano Garcia Mora

# ### Introducción
# 
#     
# Actualmente, con los avances en las tecnologías de la información, la
# generación de datos de diversos tipos en un solo día tiene volúmenes muy
# altos y con tendencia creciente. Con el fin del aprovechamiento de la
# información valiosa que pueda estar oculta en los datos que se generan, se
# requieren de tener conocimientos básicos de manejo de información y de
# análisis exploratorio de datos.  
# 
# De forma general, a menos que la persona sea una experta en el fenómeno en
# el cual se están generando los datos, el ingeniero que se disponga al análisis
# de los datos generados debe de realizar un análisis exploratorio para rescatar
# las características básicas que poseen los datos. Además de realizar un
# agrupamiento de los mismos datos en base a una característica de interés.

# ### Objetivo:
# El objetivo de este proyecto de aplicación se puede separar en tres
# fases:  
# 1.- La limpieza y extracción de la información estadística básica que
# tienen los datos que se están analizando.  
# 2.- Realización de un agrupamiento (“Clustering”) de los datos en
# base a una característica de interés.  
# 3.- La obtención o formulación de conclusiones sobre el fenómeno del
# cual provienen los datos en base a los resultados de los análisis
# anteriores  

# ### Actividades
# >**1** Obtención de una base de datos que fuera generada por un
# fenómeno de interés (La orientación o tema de los datos será 
# especificada para cada equipo por el profesor).  
# **2** Aplicar el estudio de calidad de los datos para determinar el tipo
# de datos, categorías e información estadística básica.  
# **3** Realizar una limpieza de datos y obtener un análisis exploratorio
# de datos (EDA) donde se muestren gráficas y conclusiones acerca del
# análisis. Al menos obtener 5 insights.  
# **4** En base al estudio anterior, realizar un análisis de similitud entre
# variables y entre muestras disponibles en su base de datos.  
# **5** Crear agrupamientos o “clusters” basados en el algoritmo
# “hierarchical clustering” ó “Kmeans”, y presentar sus resultados (si
# los datos lo permiten).  
# **6** Basados en los análisis anteriores, formular conclusiones sobre la
# información importante que se haya logrado encontrar de los datos
# analizados.
# 

# ## Desarrollo del trabajo

# ### Importación de librerías

# In[1]:


import numpy as np
import pandas as pd
from CDIN import CDIN as cd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ### Bases de datos

# In[2]:


data2017 = pd.read_csv('../Data/incidentes-viales-c5-2017.csv')


# In[3]:


data2018 = pd.read_csv('../Data/incidentes-viales-c5-2018.csv')


# In[4]:


data2019 = pd.read_csv('../Data/incidentes-viales-c5-2019.csv')


# La base de datos como indica la descripción contiene accidentes registrados por el C4 un sistema mexicano que registra todos los incidentes de tráfico, en este caso en particular contienen datos entre 2017 y 2019 y fueron obtenidos a través de la siguiente liga: https://datos.cdmx.gob.mx/explore/dataset/incidentes-viales-c5/table/?disjunctive.incidente_c4
# Además, contiene las siguientes explicaciones para cada una de las variables:
# * folio: una identificación única para cada registro  
# * fecha_creacion: fecha de creación  
# * hora_creacion: hora de creación  
# * dia_semana: día de la semana en que ocurre el incidente  
# * codigo_cierre: clasificación interna. La columna puede contener los siguientes códigos.  
#     - R: Afirmativo, si el incidente es confirmado por el equipo de emergencias.
#     - N: Negativo, si el equipo de emergencias no confirma el incidente en el punto de ubicación.
#     - I: Informativo, en caso de que los equipos de atención quieran agregar información extra.
#     - F: Falso, si el informe inicial no coincide con los eventos reales.
#     - D: Duplicados, registros con código de cierre afirmativo, negativo o falso pero los operadores los identifican  
# 
# * fecha_cierre: fecha de cierre, la fecha en la que se resolvió el incidente  
# * año_cierre: año de cierre  
# * mes_cierre: mes de cierre  
# * hora_cierre: tiempo de cierre  
# * delegacion_inicio: entidad dentro de la Ciudad de México donde se registró el incidente  
# * incidente_c4: una breve explicación sobre el incidente.  
# * latitud: latitud del accidente  
# * longitud: longitud del accidente  
# * clasconf_alarma: código que identifica la gravedad de la situación  
# * tipo_entrada: cómo se informó el incidente  
# * delegacion_cierre: entidad dentro de la Ciudad de México donde se cerró el incidente  
# * geopoint: columnas de latitud y longitud combinadas  
# * mes: mes en el que se informó el incidente  

# ### Data Quality Report

# In[5]:


dqr2017 = cd.dqr(data2017)
dqr2017


# In[6]:


dqr2018 = cd.dqr(data2018)
dqr2018


# In[7]:


dqr2019 = cd.dqr(data2019)
dqr2019


# In[8]:


dqr2017.index.to_list() == dqr2018.index.to_list() == dqr2019.index.to_list()


# In[9]:


dqr2017['Present_values'][0]+dqr2018['Present_values'][0]+dqr2019['Present_values'][0]


# Se puede comprobar que los datos de todos los años contienen las mismas columnas y que se pueden usar indistintamente
# los datos o inclusive se pueden combinar las tablas, una de las ventajas de los datos es que en la mayoría de estos hay apenas una pequeña cantidad de datos perdidos, alrededor de 500 lo que facilitará el análisis además de que se cuentan con aproximadamente 680,000 datos y no se perderá una parte considerable de la información, en algunos casos alguna de las columnas pueden contener información que puede ser repetitiva o innecesaria debido a que otra de las columnas lo contiene previamente como por ejemplo la columna de geopoint pues contiene los mismos datos que latitud y longitud.

# In[5]:


# unir data sets

data = pd.concat([data2017,data2018, data2019])

#quitar datos faltantes (los datos faltantes eran muy pocos comparados con el tamaño del data frame, por lo que los eliminé)

data = data.dropna()
data.reset_index(inplace = True)


# ### EDA Exploratory Data Analysis

# In[6]:


#1.- Numero de accdientes por dia de la semana (dia_semana)

insight1 = (data['dia_semana'].value_counts())

insight1.plot(kind='bar', color = '#410071')

plt.xlabel("Día de la semana", labelpad=14)
plt.ylabel("Cantidad de accidentes", labelpad=14)
plt.title("Cantidad de accidentes por día de la semana", y=1.02)
plt.show()


# Aquí se puede observar, como los días de la semana que más accidentes tiene son el viernes y el sábado. Se puede observar como a medida que la semana avanza (comienza en domingo), la cantidad de accidentes aumenta.

# In[7]:


#2. - numero de accidentes por tipo de accidente (incidente_c4)

insight5 = (data['incidente_c4'].value_counts())

plt.figure(figsize=(15,12))
insight5.plot(kind='bar')

plt.xlabel("Tipo de incidente", labelpad=14)
plt.ylabel("Cantidad de accidentes", labelpad=14)
plt.title("Cantidad de accidentes por tipo de accidente", y=1.02)
plt.show()


# <div style="text-align: justify">
# Se puede observar que la gran mayoría de los accidentes son choques sin lesionados. Después sigue accidentes con choque con lesionados, pero la diferencia es bastante grande. Esto significa que la gran mayoría de los choques son choques leves. También hubo bastante atropellados. Fuera de eso los incidentes se vuelven menos comunes siguiendo un comportamiento logarítmico y apenas hay un par de incidentes para las ultimas categorías como en el caso de los sismos.

# In[8]:


#3.- numero de accidentes por delegacion (delegacion_inicio)

insight2 = (data['delegacion_inicio'].value_counts())



insight2.plot(kind='bar', color = '#FF5733')

plt.xlabel("Delegación", labelpad=14)
plt.ylabel("Cantidad de accidentes", labelpad=14)
plt.title("Cantidad de accidentes por delegación", y=1.02)
plt.show()


# Aqui podemos observar cuales son las delegaciones con más accidentes. La delegación de Iztapalapa es el lugar con más accidentes y supera a los demás significativamente, en contraste milpa alta es la delegación que tiene menos incidentes.

# In[9]:


#4.- numero de accidentes por tipo de entrada (tipo_entrada)

insight3 = (data['tipo_entrada'].value_counts())

insight3.plot(kind='bar', color = '#160D4B')

plt.xlabel("Tipo de entrada", labelpad=14)
plt.ylabel("Cantidad de accidentes", labelpad=14)
plt.title("Cantidad de accidentes por tipo de entrada", y=1.02)
plt.show()


# El tipo de entrada que se uso para reportar el accidente fue una llamada al 911 por mucho. 

# In[10]:


#5.- numero de accidentes por mes (mes_cierre)

insight4 = (data['mes_cierre'].value_counts())

insight4.plot(kind='bar')

plt.xlabel("Mes", labelpad=14)
plt.ylabel("Cantidad de accidentes", labelpad=14)
plt.title("Cantidad de accidentes por mes", y=1.02)
plt.show()


# <div style="text-align: justify">
# El mes en el que más accidentes hubo fue el de Octubre. Sorprendentemente diciembre fue el mes con menos accidentes. Debido a que la mayoría de los accidentes habían sido por accidentes automovilísticos el sentido común nos indicaba lo contrario porque con las posadas y las fiestas se generarían condiciones para que más gente chocara por ir bajo la influencia del alcohol.

# In[11]:


#6.- numero de accidentes por hora (hora_creacion)
data['Hora'] = data['hora_creacion'].str[:2]

data['Hora'] = data['Hora'].apply(lambda x: cd.remove_puntuation(x))

data['Hora'] = data['Hora'].apply(lambda x: cd.zeros(x,2))

i6 = (data['Hora'].value_counts())

i6 = i6.to_frame()

i6 = i6.sort_index()

i6.plot(kind='bar', color = '#900C3F')

plt.xlabel("Hora", labelpad=14)
plt.ylabel("Cantidad de accidentes", labelpad=14)
plt.title("Cantidad de accidentes por hora", y=1.02)
plt.show()


# <div style="text-align: justify">
# Podemos observar como la mayoría de los accidentes ocurrieron a las 19, 20, 21, 18 y 16 horas. Esto fue sorpresivo ya que se podía esperar que la mayoría de los accidentes ocurrieran a más altas horas de la noche cuando la gente sale de bares y antros. Pero no, ocurrieron a las horas en las que la gente va saliendo del trabajo y el tráfico aumenta. 

# In[12]:


#7.- numero de accidentes por hora  y tipos de accidentes principales (hora_creacion)
fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(20,10))
data[data['incidente_c4'] == 'accidente-choque con lesionados'].groupby(['Hora'])['folio'].count().plot(kind = 'bar',
                                                                                                        ax = ax1[0])
data[data['incidente_c4'] == 'accidente-choque sin lesionados'].groupby(['Hora'])['folio'].count().plot(kind = 'bar',
                                                                                              color= '#09C43F',ax = ax1[1])
data[data['incidente_c4'] == 'accidente-volcadura'].groupby(['Hora'])['folio'].count().plot(kind = 'bar', color= '#FFC300',
                                                                                            ax = ax2[0])
data[data['incidente_c4'] == 'lesionado-atropellado'].groupby(['Hora'])['folio'].count().plot(kind = 'bar',color ='#C70039',
                                                                                             ax = ax2[1])
ax1[0].title.set_text('accidente-choque con lesionados')
ax1[1].title.set_text('accidente-choque sin lesionados')
ax2[0].title.set_text('accidente-volcadura')
ax2[1].title.set_text('lesionado-atropellado')
plt.show()


# <div style="text-align: justify">
# Dada la conclusión anterior resultaba muy particular por lo que al comparar los principales tipos de accidentes se puede observar una mayor tendencia que corresponde a los pensamientos derivados del sentido común, si tomamos los accidentes choque sin lesionados como nuestro control y nos enfocamos en el número de incidentes proporcionales entonces se puede observar que durante las horas de la madrugada se incrementan los accidentes con lesionados y volcadura principalmente aunque en el primer caso sin ser tan drástico como en el del caso de volcaduras, los atropellados no se ven influenciados, posiblemente por la baja movilidad en esas horas y que hay un aporte por el alcohol que sea de relevancia.

# In[13]:


alarma = data['clas_con_f_alarma'].unique()


# In[14]:


fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(20,10))
data[data['clas_con_f_alarma'] == alarma[0]].groupby(['Hora'])['folio'].count().plot(kind = 'bar',color = '#0AD852',
                                                                                                        ax = ax1[0])
data[data['clas_con_f_alarma'] == alarma[1]].groupby(['Hora'])['folio'].count().plot(kind = 'bar',color= '#148779',
                                                                                                    ax = ax1[1])
data[data['clas_con_f_alarma'] == alarma[2]].groupby(['Hora'])['folio'].count().plot(kind = 'bar', color= '#391787',
                                                                                             ax = ax2[0])
data[data['clas_con_f_alarma'] == alarma[3]].groupby(['Hora'])['folio'].count().plot(kind = 'bar',color ='#0833B9',
                                                                                             ax = ax2[1])
ax1[0].title.set_text( alarma[0])
ax1[1].title.set_text( alarma[1])
ax2[0].title.set_text( alarma[2])
ax2[1].title.set_text( alarma[3])
plt.show()


# De igual manera quisimos checar si la hora influía en alguna manera en la clasificación de la alarma del incidente, en general todos se comportan de manera similar con una disminución del tipo de alarma hacia la madrugada pero con diferencias en sobre la media tarde, hay un pico de urgencias a las 8 de la mañana mientras que las falsas alarmas tienen el pico hacia la una de la tarde y los delitos a las 3 de la tarde, todas hacia el final de día es donde tienen los valores máximos.

# In[15]:


pd.DataFrame(data['clas_con_f_alarma'].value_counts())


# In[59]:


### Visualización de la incidencia espacial de incidentes
plt.figure(figsize = (15,15))
plt.scatter(data['longitud'][:600000],data['latitud'][:600000],alpha = 0.002, s = 3)
plt.xlabel('longitud')
plt.ylabel('latitud')
plt.show()


# ## Similitud entre los datos

# In[16]:


### Convertir la hora en dato númerico continuo
data['hora_creacion'] = data['hora_creacion'].apply(lambda x: cd.remove_puntuation(x))
data['hora_creacion'] = data['hora_creacion'].apply(lambda x: cd.zeros(x,6))
data['hora_creacion'] = data['hora_creacion'].apply(lambda x: int(x[:2])*60 + int(x[2:4])+ int(x[4:])/60)


# Si consideramos que la mayor parte de la información esta contenida dentro de los datos que tuvieron alguna de las clasificaciones del C4 como los accidentes o los lesionados, por ello vamos a trabajar con ese subset para poder hacer la medición de la similitud de los datos.

# In[17]:


incidentes = insight5[:5].index.to_list()


# In[18]:


new_data = pd.concat([data[data['incidente_c4'] == i] for i in incidentes])
new_data.reset_index(inplace = True)


# Vamos a considerar como información relevante para comparar nuestros datos el tipo incidente según el código del C4, la hora, la latitud, longitud y la delegación donde fue reportado el incidente, esto también considerando como en algunas otras variables que pudieran ser relevantes a partir del EDA se pudo observar que su distribución es aproximadamente uniforme indistintamente de la condición, debido a la cantidad de datos, el considerar más variables podría hacer que los cálculos se tarden o no se puedan hacer.

# ### Selección de variables

# Debido a que vamos a trabajar con datos mixtos (categóricos y numéricos) procedimos a crear variables dummies para los datos categóricos y finalmente estandarizar los datos para poder sacar la similitud entre las variables.

# In[19]:


### Variables de importancia
cols = ['longitud','latitud','hora_creacion','incidente_c4','delegacion_inicio','codigo_cierre']
incidentes_mixtos = new_data[cols]

#Variables categoricas
cols_cat = ['incidente_c4','delegacion_inicio','codigo_cierre']

#Variables continuas
cols_cont = ['longitud','latitud','hora_creacion']


# In[20]:


### Creación de variables dummies
incidentes_dummy = pd.get_dummies(incidentes_mixtos[cols_cat[0]], prefix = cols_cat[0])

for col in cols_cat[1:]:
    temp = pd.get_dummies(incidentes_mixtos[col], prefix = col)
    incidentes_dummy = incidentes_dummy.join(temp)

del temp

col_list_cat_dummy = incidentes_dummy.columns.to_list()

incidentes_mixtos_dummy = incidentes_mixtos [cols_cont].join(incidentes_dummy)


# ### Visualización de las variables dummy

# In[21]:


incidentes_mixtos_dummy.head()


# In[22]:


### Estandarización de los datos
incidentes_std = (incidentes_mixtos_dummy -incidentes_mixtos_dummy.mean(axis = 0))/incidentes_mixtos_dummy.std(axis = 0)
incidentes_std[col_list_cat_dummy] = incidentes_std[col_list_cat_dummy]*(1/np.sqrt(2))


# In[23]:


### Importar librerías para las metricas
import scipy.spatial.distance as sc # Metricas de distancia


# ### Datos de similitud

# Finalmente aplicamos a las medidas de similitud de distancia euclidiana y correlación de los datos obteniendo resultados similares en los primeros datos que eran más afines.

# In[26]:


### Sublista para poder manejar los datos
accidents_std100 = incidentes_std.loc[0:10000,:]

### Medida de similitud distancia euclideana
euc = sc.pdist(accidents_std100,'euclidean')
euc = sc.squareform(euc)


# In[27]:


temp = pd.DataFrame(euc)


# In[29]:


temp


# In[30]:


D1 = temp.iloc[:,0]
D1_sort = np.sort(D1)

D1_index = np.argsort(D1)


# In[31]:


D1_index[2]


# In[32]:


new_data.iloc[0,:]


# In[33]:


new_data.iloc[4146,:]


# Se puede observar que con este ejemplo estos dos incidentes tienen aproximadamente el mismo comportamiento, son accidentes de tipo choque sin lesionados, en la delegación milpa alta, con coordenadas similares y que fueron reportados aproximadamente a la misma hora aunque con día y hora de diferencia

# In[34]:


### Medida de similitud con correlacion
corr = sc.pdist(accidents_std100,'correlation')
corr = sc.squareform(corr)


# In[35]:


temp2 = pd.DataFrame(corr)
D2 = temp2.iloc[:,0]
D2_sort = np.sort(D2)

D2_index = np.argsort(D2)


# In[36]:


comp = pd.concat([D2_index,D1_index], axis = 1)
comp


# Con esta comparativa se puede observar cómo se obtienen resultados similares por lo menos dentro de los primeros valores que son los más relevantes por ser los más cercanos si se utilizan tanto la métrica euclidiana como la correlación.

# ### Clustering

# La mayor parte de los datos son variables categoricas por lo que hacer clustering con cualquiera de las tecnicas aprendidas dentro del curso como clustering jerarquico o Kmeans no podría realizarse ya que esos algoritmos están reservados exclusivamente para datos categoricos, dentro de las variables númericas con las que si podemos contar de las que más destacan son la longitud y latitud de donde se iniciarón las llamadas de emergencia por lo que serán las variables que utilizaremos para el clustering
# 
# Decidimos de utilizar el algoritmo de clustering de Kmeans debido a la gran cantidad de datos ya que hacer el algoritmo de clustering jerárquico al ser un método exhaustivo tardaría demasiado tiempo en computarse y posiblemente la memoria se desbordaría.
# 
# Para delimitar el número optimo de grupos en base a nuestros datos utilizamos el criterio del codo el cuál nos quedo de la siguiente forma:
# 

# In[37]:


### criterio del codo
data_clust = data[['latitud','longitud']]


data_std = (data_clust-data_clust.mean(axis=0))/data_clust.std(axis=0)


inercias = np.zeros(10)
for k in np.arange(10)+1:
    model = KMeans(n_clusters=k,init = 'random')
    model = model.fit(data_std)
    inercias[k-1] = model.inertia_
    
plt.plot(np.arange(1,11),inercias)
plt.title('KMeans')
plt.xlabel('# Grupos')
plt.ylabel('Inercia global')
plt.show()


# Como se puede ver en la inercia global cuando se alcanzan 3 grupos la tendencia cambia y por tanto decidimos seleccionar ese valor como nuestro número de grupos para hacer el clustering.

# In[38]:


# clustering
model = KMeans(n_clusters = 3, init = 'k-means++')
model = model.fit(data_std)
grupo = model.predict(data_std)
centroides = model.cluster_centers_


# Después de realizar el clustering utilizando como inicialización K-means ++ para evitar que los centroides inicien en un punto alejado y los resultados no sean los más adecuados se obtuvo el siguiente resultado.

# In[40]:


#visualización de los resultados
plt.scatter(data_std['latitud'],data_std['longitud'],c = grupo)
plt.plot(centroides[:300000,1],centroides[:300000,0],'rx', ms =5)
plt.title('Resultado de cluster K-means')
plt.ylabel('latitud std')
plt.xlabel('longitud std')
plt.show()


# ### Conclusiones

# Este es el análisis de los grupos que nos otorgó, el problema es que los datos que se tomaron que son de latitud y longitud, son los únicos que se pueden tomar en cuenta para este tipo de análisis y no es tan relevante esta información de los grupos que se forman ya que no se toman en cuenta todas las variables posibles.
