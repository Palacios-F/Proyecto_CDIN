# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 09:22:28 2021

@author: flavi
"""
import pandas as pd
import string

class CDIN:
    
    #Función data quality report
    def dqr(data):
        #%% lista de variables de la base de datos
        columns = pd.DataFrame(list(data.columns.values),columns = ['Nombres'], index = list(data.columns.values))
    
        #%% Lista de tipos de datos
        data_types = pd.DataFrame(data.dtypes, columns = ['Data_Types'])
    
        #%% lista de datos perdidos
        missing_values = pd.DataFrame(data.isnull().sum(), columns = ['Missing_values'])
    
        #%% Lista de valores presentes
        present_values = pd.DataFrame(data.count(), columns = ['Present_values'])
    
        #%% Lista de valores unicos para cada variable
        unique_values =  pd.DataFrame(columns = ['Unique_values'])
        for col in list(data.columns.values):
            unique_values.loc[col] = [data[col].nunique()]
        
        #%% Lista de valores mínimos
        min_values = pd.DataFrame(columns = ['Min'])
        for col in list(data.columns.values):
            try:
                min_values.loc[col] = [data[col].min()]
            except:
                pass
    
        #%% Lista de valores máximos
        max_values = pd.DataFrame(columns = ['Max'])
        for col in list(data.columns.values):
            try:
                max_values.loc[col] = [data[col].max()]
            except:
                pass

    
        #%% Columna categorica que sea booleana que cuando sea true represente que la variables es categorica y false represente
        # cuando sea numerica
        return columns.join(data_types).join(missing_values).join(present_values).join(unique_values).join(min_values).join(max_values)
    
    #%% Funcinoes para limpieza de datos
    def remove_puntuation(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.punctuation)
        except:
            pass
        return x
    
    #%%Remover digitos
    def remove_digits(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.digits)
        except:
            pass
        return x
    
    #%%Remover espacios en blanco
    def remove_whitespace(x):
        try:
            x = ''.join(x.split(),)
        except:
            pass
        return x
    
    #%%Reemplazar texto
    def replace_text(x, to_replace, replacement):
        try:
            x = x.replace(to_replace, replacement)
        except:
            pass
        return x
    
    #%%Texto en mayusculas
    def uppercase_text(x):
        try:
            x=x.upper()
        except:
            pass
        return x
    
    #%%Texto en mínusculas
    def lowercase_text(x):
        try:
            x=x.lower()
        except:
            pass
        return x
    
    #%%Remover caracteres
    def remove_char(x):
        try:
            x = ''.join(ch for ch in x if ch not in string.ascii_letters)
        except:
            pass
        return x
    
    #%%Aumentar ceros
    def zeros(x):
        if len(x) < 9: 
            x = (9 - len(x))*'0' + x
        return x
    