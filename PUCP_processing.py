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
                        valores_encontrados[palabra[:-3]] = float(numero) if '.' in numero else int(numero)

    #for palabra, valor in valores_encontrados.items():
    #    print(f"{palabra[:-3]} {valor}")
    return valores_encontrados

### FUNCIÓN DE JD
def filter_geometry(all_particles_df, allowed_particles, length_triangle, a, b):
    """
    Apply geometric and charged particle ID filters to the dataframe of all particles using vectorized operations.
    Adjusted to include only particle ID and time in the output for the provided data structure.
    
    Parameters:
    all_particles_df (pd.DataFrame): Dataframe with all particle data, including 'id', 'x', 'y', and 't' columns.
    charged_particles (list): List of charged particle IDs to filter.
    length_triangle (float): Length of the triangle side.
    a (float): The width of the detector.
    b (float): The height of the detector.
    
    Returns:
    pd.DataFrame: Filtered dataframe with only the ID and time of particles that hit the detectors.
    """
    # Crear una máscara booleana para los IDs de partículas cargadas
    mask_charged = all_particles_df['id'].isin(allowed_particles)

    # Calcular las máscaras booleanas para cada detector
    mask_d1 = ((all_particles_df['x'] >= (length_triangle / np.sqrt(3) - b / 2)) & 
               (all_particles_df['x'] <= (length_triangle / np.sqrt(3) + b / 2)) & 
               (all_particles_df['y'] >= -a / 2) & 
               (all_particles_df['y'] <= a / 2))
    
    mask_d2 = ((all_particles_df['x'] >= (-length_triangle / (2 * np.sqrt(3)) - b / 2)) & 
               (all_particles_df['x'] <= (-length_triangle / (2 * np.sqrt(3)) + b / 2)) & 
               (all_particles_df['y'] >= (length_triangle / 2 - a / 2)) & 
               (all_particles_df['y'] <= (length_triangle / 2 + a / 2)))
    
    mask_d3 = ((all_particles_df['x'] >= (-length_triangle / (2 * np.sqrt(3)) - b / 2)) & 
               (all_particles_df['x'] <= (-length_triangle / (2 * np.sqrt(3)) + b / 2)) & 
               (all_particles_df['y'] >= (-length_triangle / 2 - a / 2)) & 
               (all_particles_df['y'] <= (-length_triangle / 2 + a / 2)))

    # Aplicar la máscara de IDs cargados y las máscaras de detectores
    all_particles_df['Detector'] = 0
    all_particles_df.loc[mask_charged & mask_d1, 'Detector'] = 1
    all_particles_df.loc[mask_charged & mask_d2, 'Detector'] = 2
    all_particles_df.loc[mask_charged & mask_d3, 'Detector'] = 3

    # Filtrar las filas donde 'Detector' es diferente de 0 y seleccionar solo las columnas 'id' y 't'
    filtered_df = all_particles_df[all_particles_df['Detector'] != 0][['id', 't', 'Detector']].reset_index(drop=True)
    df_detector_1 = filtered_df[filtered_df['Detector'] == 1].reset_index(drop=True)
    df_detector_2 = filtered_df[filtered_df['Detector'] == 2].reset_index(drop=True)
    df_detector_3 = filtered_df[filtered_df['Detector'] == 3].reset_index(drop=True)

    return df_detector_1, df_detector_2, df_detector_3

###
def list_directories(path):
    directories = []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path) and not name[0]=='.':
            directories.append(name)
    return directories
def list_dats(path):
    dat_list = []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if full_path[-5:]=='.txt':
            dat_list.append(name)
    return dat_list



## PROCESS_DATA FUNCTION
def process_data(txt_path,length_triangle,a,b):
    ## TXT to Dataframe
    all_particles_df=txt_to_df(txt_path,xlims=xlims,ylims=ylims,inclined=False)  ## edit limits
    ## Call Fiorella's Function
    shower_info=get_shower_info(txt_path)
    ## Declare JP's list
    allowed_particles=
    ## Call JD's Function
    df_det1,df_det2,df_det3=filter_geometry(all_particles_df,allowed_particles,length_triangle,a,b)

    return shower_info,df_det1,df_det2,df_det3 ## return the three dataframes, and the shower parameters
    



if __name__ == "__main__":
    ## DETECTOR PARAMETERS
    a=
    b=
    length_triangle= 
    ## DATA PARAMETERS
    data_directory = r'C:\Users\cg_h2\Documents\data_tambo\DATA'

    ## 
    dat_list= list_dats(data_directory)
    exceptions=[]
    count=0
    all_showers_data=[]
    for dat in dat_list:
        print(f'{dat} is being processed.')
        try:
            dat_path= os.path.join(data_directory, dat)
            ## call process_data function
            shower_info,df_det1,df_det2,df_det3=process_data(dat_path,length_triangle,a,b)
            ## append dataframes and shower info
            shower_summary=[shower_info,df_det1,df_det2,df_det3]
            all_showers_data.append(shower_summary)
            print(f'\n{dat} successful.')
        except Exception as e :
            print(f'\n{dat} Failed.')
            exceptions.append((dat,e))

        count+=1
        left=len(dat_list)-count
        print(f'{left} dats remaining')
    print('Data has been processed')
    print(f'The following exceptions have been encountered: \n{exceptions}')

        #### PENDING
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


