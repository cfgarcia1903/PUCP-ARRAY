import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from timeit import default_timer
from datetime import timedelta
from tqdm import tqdm
import os
import pickle
import re

## FUNCTIONS
def formato_tiempo(segundos):
    delta_tiempo = timedelta(seconds=segundos)
    # Construye la cadena de tiempo
    tiempo_formateado = f"({delta_tiempo.days})D ({delta_tiempo.seconds//3600})H ({(delta_tiempo.seconds//60)%60})M ({(delta_tiempo.seconds%60)})S"
    return tiempo_formateado
def imprimir_barra_de_carga(tiempo,iteracion, total, longitud=50):
    porcentaje = int(iteracion / total * 100)
    carga = int(iteracion / total * longitud)
    tiempo_promedio=tiempo/iteracion
    tiempo_faltante=(tiempo_promedio*(total-iteracion))
    barra_de_carga = f"[{'■' * carga}{' ' * (longitud - carga)}] {porcentaje}%            REMAINING TIME: {formato_tiempo(segundos=tiempo_faltante)}"
    print(barra_de_carga, end='\r', flush=True)
def txt_to_df(path,xlims=None,ylims=None,inclined=True):
    # Lists to save the data
    ids = []
    x_values = []
    y_values = []
    t_values = []
    px_values = []
    py_values = []
    pz_values = []
    ek_values = []
    w_values = []
    lev_values = []

    # accessing the .txt
    with open(path, 'r') as archivo:
        for linea in archivo:
            try:
                # Divide la línea en partes usando el espacio como separador
                partes = linea.split()

                # Extrae los valores que contienen 'x=', 'y=', 't=', etc.
                id_valor = int(partes[1])
                x_valor = float(partes[2].split('=')[1])/(100)   #en kilometros
                y_valor = float(partes[3].split('=')[1])/(100)   #en kilometros          
                t_valor = float(partes[4].split('=')[1])
                px_valor = float(partes[5].split('=')[1])
                py_valor = float(partes[6].split('=')[1])
                pz_valor = float(partes[7].split('=')[1])
                if inclined==True:
                    x_valor,y_valor= (-y_valor),x_valor
                    px_valor,py_valor= (-py_valor),px_valor
                    pz_valor=-pz_valor
                    #Now Y means upwards the inclined plane and X means to the right 
                ek_valor = float(partes[8].split('=')[1])
                w_valor = float(partes[9].split('=')[1])
                lev_valor = int(partes[10].split('=')[1])

                #if (det_X_inf<=x_valor<=det_X_sup) and (det_Y_inf<=y_valor<=det_Y_sup):
                    # Agrega los valores a las listas
                ids.append(id_valor)
                x_values.append(x_valor)
                y_values.append(y_valor)
                t_values.append(t_valor)
                px_values.append(px_valor)
                py_values.append(py_valor)
                pz_values.append(pz_valor)
                ek_values.append(ek_valor)
                w_values.append(w_valor)
                lev_values.append(lev_valor)
            except:
                pass

    # Crea un DataFrame de Pandas
    data = {
        'id': ids,
        'x': x_values,
        'y': y_values,
        't': t_values,
        'px': px_values,
        'py': py_values,
        'pz': pz_values,
        'ek': ek_values,
        'w': w_values,
        'lev': lev_values,
        'detector': np.nan
    }

    all_data = pd.DataFrame(data).astype({'detector':object})
    if xlims != None:
        all_data = all_data[(all_data['x']>= xlims[0]) & (all_data['x']<= xlims[1])].reset_index(drop=True)
    if ylims != None:
        all_data = all_data[(all_data['y']>= ylims[0]) & (all_data['y']<= ylims[1])].reset_index(drop=True)
    
    return all_data

    '''
    given a detector position and a tolerance (radius), assign_to_detector(det_position,df,tol) filters the particles that fall
    inside that given detector and updates the dataframe of particles, assigning the
    detector position to the 'detector' column of those entries that fall inside the detector
    
    it also deletes the entries that are in the neighbourhood of the detector but do not fall inside the detector
    
    the parameters are:
    det_position:              a tuple that contains the position (x,y) of the detector
    df:                        the DataFrame of all entries
    tol:                       a tolerance for particle detection (radius of the detector)
    pf_tol=(pf_tolx,pf_toly):  [IGNORE] a tolerance for a preliminary filtering of particles in a rectangular neighbourhood of the detector 
                               (dimensions: 2*pf_tolx by 2*pf_toly) centered at the detector.
                               it is necesary that tol<=pf_tol(both components). large values will cause problems if the rectangular
                               neighbourhood is too big and overlaps with the bounds of other detectors 
    
    the function returns the updated DataFrame
                          
    '''
    if pf_tol==(None,None):
        pf_tol=(1.01*tol,1.01*tol)
    
    det_x,det_y=det_position
    pf_tolx,pf_toly=pf_tol
    possible_particles_index=df.index[(df['x']<=det_x+pf_tolx) & (df['x']>=det_x-pf_tolx) & (df['y']<=det_y+pf_toly) & (df['y']>=det_y-pf_toly)].tolist()
    for index in possible_particles_index:
        x,y=df.loc[index,'x'],df.loc[index,'y']
        if (x-det_x)**2 + (y-det_y)**2 <= tol**2:
            df.at[index,'detector']= det_position
        else:
            df.drop(index, inplace=True, axis=0)
    return df    

### FUNCIÓN DE FIORELLA
def get_shower_info(nombre_archivo):
    valores_encontrados = {}

    palabras_clave = ["PRMPAR = ", "PRME = ", "THETAP = ", "PHIP = "]

    # Abre el archivo y lo lee línea por línea
    with open(nombre_archivo, 'r') as archivo:
        for linea in archivo:
            for palabra in palabras_clave:
                # Busca la palabra clave seguida de un número
                if palabra in linea:
                    # Encuentra el número al costado de la palabra clave
                    indice_palabra = linea.index(palabra)
                    inicio_numero = indice_palabra + len(palabra)
                    
                    # Busca el número después de la palabra clave
                    numero = ""
                    for char in linea[inicio_numero:]:
                        if char.isdigit() or char == '.':
                            numero += char
                        else:
                            break
                    
                    if numero:
                        valores_encontrados[palabra] = float(numero) if '.' in numero else int(numero)

    for palabra, valor in valores_encontrados.items():
        print(f"{palabra[:-3]} {valor}")
# Ejemplo de uso:
nombre_archivo = "datos.txt"
# Función (⁠｢･⁠ω⁠･⁠)⁠｢
get_shower_info(nombre_archivo)

### LISTA DE PARTÍCULAS DE JP


### FUNCIÓN DE JD
def assign_to_detector2(det_position,df,d_side=1):
    '''
    given a detector position and a tolerance (radius), assign_to_detector(det_position,df,tol) filters the particles that fall
    inside that given detector and updates the dataframe of particles, assigning the
    detector position to the 'detector' column of those entries that fall inside the detector
    
    it also deletes the entries that are in the neighbourhood of the detector but do not fall inside the detector
    
    the parameters are:
    det_position:              a tuple that contains the position (x,y) of the detector
    df:                        the DataFrame of all entries
    tol:                       a tolerance for particle detection (radius of the detector)
    pf_tol=(pf_tolx,pf_toly):  [IGNORE] a tolerance for a preliminary filtering of particles in a rectangular neighbourhood of the detector 
                               (dimensions: 2*pf_tolx by 2*pf_toly) centered at the detector.
                               it is necesary that tol<=pf_tol(both components). large values will cause problems if the rectangular
                               neighbourhood is too big and overlaps with the bounds of other detectors 
    
    the function returns the updated DataFrame
                          
    '''
        
    det_x,det_y=det_position
    possible_particles_index=df.index[(df['x']<=det_x+d_side/2.0) & (df['x']>=det_x-d_side/2.0) & (df['y']<=det_y+d_side/2.0) & (df['y']>=det_y-d_side/2.0)].tolist()
    for index in possible_particles_index:
        df.at[index,'detector']= det_position
    return df    

###

def list_directories(path):
    directories = []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path) and not name[0]=='.':
            directories.append(name)
    return directories
def list_dats(path):
    directories = []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            directories.append(name)
    return directories



## PROCESS_DATA FUNCTION
def process_data(txt_path):
    ## TXT to Dataframe
    all_particles_df=txt_to_df(txt_path,xlims=xlims,ylims=ylims,inclined=False)  ## edit limits
    ## Call Fiorella's Function

    ## Declare JP's list

    ## Call JD's Function


    return None ## return the three dataframes, and the shower parameters
    

## PARAMETERS
# Paths
# txt_path='8-12/DAT000008-inclined (2).txt'


data_directory = r'C:\Users\cg_h2\Documents\data_tambo\DATA'
energy_directories = list_directories(data_directory)
dat_list= ## list dats function (to be programmed)

exceptions=[]
count=0
for dat in dat_list:
    print(f'{dat} is being processed.')
    try:
        dat_path= os.path.join(data_directory, dat)
        ## call process_data function

        ## append dataframes and shower info

        print(f'\n{dat} successful.')
    except Exception as e :
        print(f'\n{dat} Failed.')
        exceptions.append((dat,e))

    
    count+=1
    left=len(dat_list)-count
    print(f'{left} dats remaining')
print('Data has been processed')
print(f'The following exceptions have been encountered: \n{exceptions}')




if len(primaries_list)==len(det_energies_list) and len(det_energies_list)== len(det_totals_list):  ##check if the amount of processed data is correct
    save_flag = input('Save data?[y/n]: ')
    if not save_flag=='n':
        pickle_path=r'C:\Users\cg_h2\Documents\data_tambo'
        ## declare pickle file paths. pimaries_path=os.path.join(pickle_path, 'primaries.pickle')
        
        with open(pimaries_path, 'wb') as file:              ## edit 
            pickle.dump(primaries_list, file)
            
    else:
        print('Not saved')
else:
    print('List lengths don\'t match, please debug')


