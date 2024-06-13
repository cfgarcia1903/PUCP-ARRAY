import numpy as np 
import pandas as pd
import os
import pickle


## FUNCTIONS
def txt_to_df(path,xlims=None,ylims=None,inclined=True):
    # Lists to save the data
    ids = []
    x_values   = []
    y_values   = []
    t_values   = []
    px_values  = []
    py_values  = []
    pz_values  = []
    ek_values  = []
    w_values   = []
    lev_values = []

    # accessing the .txt
    with open(path, 'r') as archivo:
        for linea in archivo:
            try:
                # Divide la línea en partes usando el espacio como separador
                partes = linea.split()

                # Extrae los valores que contienen 'x=', 'y=', 't=', etc.
                id_valor = int(partes[1])
                x_valor = float(partes[2].split('=')[1])/(100)   #en metros
                y_valor = float(partes[3].split('=')[1])/(100)   #en metros          
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

def list_dats(path):
    dat_list = []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if full_path[-4:]=='.txt':
            dat_list.append(name)
    return dat_list

def list_directories(parent_directory):
    directories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    return directories

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
    mask_d0 = ((all_particles_df['x'] >=  (- b / 2)) & 
               (all_particles_df['x'] <=  (+ b / 2)) & 
               (all_particles_df['y'] >=  (- a / 2)) & 
               (all_particles_df['y'] <=  (+ a / 2)))

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
    all_particles_df['Detector'] = np.nan
    all_particles_df.loc[mask_charged & mask_d0, 'Detector'] = 0
    all_particles_df.loc[mask_charged & mask_d1, 'Detector'] = 1
    all_particles_df.loc[mask_charged & mask_d2, 'Detector'] = 2
    all_particles_df.loc[mask_charged & mask_d3, 'Detector'] = 3

    # Filtrar las filas donde 'Detector' es diferente de 0 y seleccionar solo las columnas 'id' y 't'
    filtered_df = all_particles_df[all_particles_df['Detector'] != np.nan][['id', 't', 'Detector']].reset_index(drop=True)
    df_detector_0 = filtered_df[filtered_df['Detector'] == 0].reset_index(drop=True)
    df_detector_1 = filtered_df[filtered_df['Detector'] == 1].reset_index(drop=True)
    df_detector_2 = filtered_df[filtered_df['Detector'] == 2].reset_index(drop=True)
    df_detector_3 = filtered_df[filtered_df['Detector'] == 3].reset_index(drop=True)

    return df_detector_0, df_detector_1, df_detector_2, df_detector_3

## PROCESS_DATA FUNCTION
def process_data(txt_path,length_triangle,a,b,allowed_particles):
    ## TXT to Dataframe
    all_particles_df=txt_to_df(txt_path,inclined=False)  ## edit limits
    ## Call Fiorella's Function
    shower_info=get_shower_info(txt_path)
    ## Call JD's Function
    df_det0,df_det1,df_det2,df_det3=filter_geometry(all_particles_df,allowed_particles,length_triangle,a,b)
    return shower_info,df_det1,df_det2,df_det3,df_det0 ## return the three dataframes, and the shower parameters
    

if __name__ == "__main__":
    ## USER
    print('------------ USER SELECTION ------------\n')
    users=['francisco','fiorella','bruno']
    print(f"1| Francisco ")
    print(f"2| Fiorella ")
    print(f"3| Bruno\n ")
    user_id=int(input("Enter user ID: "))
    user=users[user_id-1]
    print('')
    ## DETECTOR PARAMETERS
    print('----------- ARRAY PARAMETERS -----------\n')
    a=0.05*16
    b=1.85
    length_triangle= float(input('Enter the separation of the detectors (m): '))
    allowed_particles=(1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 19, 21, 23, 24, 27, 29, 31, 32, 52, 53, 54, 55, 57, 58, 59, 61, 63, 64, 117, 118, 120, 121, 124, 125, 127, 128, 131, 132, 137, 138, 140, 141, 143, 149, 150, 152, 153, 155, 161, 162, 171, 172, 177, 178, 182, 183, 185, 186, 188, 189, 191, 192, 194, 195)
    print("")
    ## DATA FILES PARAMETERS
    print('------------ DATA SELECTION ------------\n')
    #parent_directory = r'C:\Users\cg_h2\Documents\pucp_array\data'
    parent_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')
    #print found directories
    directories= list_directories(parent_directory)
    print('Avaliable directories:\n')
    for n,d in zip(list(range(1,len(directories)+1)),directories):
        print(f"{n}| {d} ")
    print("")
    sim_dir_ids=input('Enter the IDs of the directories to process (separated by comas \',\' ): ')
    sim_dir_ids=sim_dir_ids.split(',')
    sim_dir_ids=[int(dir_id) for dir_id in sim_dir_ids]
    sim_dirs=[directories[id-1] for id in sim_dir_ids]
    print("")
    print('Selected directories: ')
    print(sim_dirs)
    print("")
    confirm_dirs=input('confirm_dirs [y/n]')
    if confirm_dirs=='y':
        pass
    else:
        sim_dirs=[]
    print("")
    for sim_dir in sim_dirs:
        print(f'{sim_dir} directory is being processed')
        data_directory=os.path.join(parent_directory,sim_dir)
        ## DATA PROCESSING
        dat_list= list_dats(data_directory)
        exceptions=[]
        count=0
        all_showers_data=[]
        for dat in dat_list:
            print(f'{dat} is being processed.')
            try:
                dat_path= os.path.join(data_directory, dat)
                ## call process_data function
                shower_info,df_det1,df_det2,df_det3,det_0=process_data(dat_path,length_triangle,a,b,allowed_particles)
                ## append dataframes and shower info
                shower_summary=(shower_info,df_det1,df_det2,df_det3,det_0)
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

        ## DATA SAVING
        if not exceptions:
            save_flag = 'y'
        else:
            save_flag = input('Save data?[y/n]: ')

        if not save_flag=='n':
            pickle_rel_dir=user+'/pickles'
            pickle_dir_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),pickle_rel_dir)
            ## declare pickle file paths. pimaries_path=os.path.join(pickle_path, 'primaries.pickle')
            pickle_name=sim_dir
            pickle_name=pickle_name+'.pickle'
            pickle_file_path=os.path.join(pickle_dir_path,pickle_name)
            with open(pickle_file_path, 'wb') as file:             
                pickle.dump(all_showers_data, file)
            print(f'pickle file saved as {pickle_file_path}')
        else:
            print('Not saved')
        
    input('Press Enter to continue')



