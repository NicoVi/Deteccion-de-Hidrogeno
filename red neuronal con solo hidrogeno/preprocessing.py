import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

###funcion que no se utilizan ###
def data_bug_fix(bad_table):
    bad_table[np.abs(bad_table)==np.inf] = 0
    bad_table[np.isnan(bad_table)]=0   
    return bad_table
###funcion que no se utilizan ###

def default_preprocessing(x_table):
    x_table_lg10 = data_bug_fix(np.log(x_table))
    out_table = x_table_lg10 - np.min(x_table_lg10, axis=1).reshape([-1,1])
    return out_table

def default_normalize_r(x_table):
    out_table=np.zeros((x_table.shape[0], x_table.shape[1]))
    for i in range(x_table.shape[0]):
        out_table[i,:]=(x_table[i,:]-min(x_table[i,:]))/(max(x_table[i,:])-min(x_table[i,:]))
    return out_table

def default_normalize_c(x_table):
    out_table=np.zeros((x_table.shape[0], x_table.shape[1]))
    for i in range(x_table.shape[1]):
        out_table[:,i]=x_table[:,i]/sum(x_table[:,i])
    return out_table

def get_poly_coeffs(input_vec, a_coef=9):
    from scipy import polyfit
    z1, resid, rank, singular, rcond = list(polyfit(np.arange(0,len(input_vec)),input_vec/np.sum(input_vec), a_coef, full=True))
    return(z1, resid)

def get_poly_coeffs_lg10(input_vec, a_coef=9):
    from scipy import polyfit
    z1, resid, rank, singular, rcond = list(polyfit(np.arange(0,len(input_vec)),input_vec, a_coef, full=True))
    return(z1, resid)

###funcion que no se utilizan ###
def poly_extracter(x_table, poly_coef=9):
    x_table_coefs = np.array(list(map(lambda x: get_poly_coeffs(x/np.sum(x), a_coef=poly_coef), x_table)))
    out_table = np.array([list(i[0]) for i in x_table_coefs])
    return out_table
###funcion que no se utilizan ###


def dataset_from_files(experiments_folder, output_folder, sensor='R1', dataset='train'):

#Comentarios de los rusos:
    
	# INPUTS
	#experiments_folder - path to folder with experiment dated folders, by default: './methane-propane/methane-propane-raw_data/' 
	    #following experiment folders at experiment_folder path should be exist:
	    # ['2018.12.12_20.14.18',
	    #  '2018.12.13_20.31.40',
	    #  '2018.12.14_20.53.45',  
	    #  '2018.12.19_21.35.25',
	    #  '2018.12.20_21.36.43',
	    #  '2018.12.21_21.46.34',
	    #  '2019.01.11_15.51.25',
	    #  '2019.01.12_19.32.29',
	    #  '2019.01.14_19.58.00',
	    #  '2019.01.16_11.08.22',
	    #  '2019.01.18_15.19.45',
	    #  '2019.01.25_12.26.23',
	    #  '2019.01.27_12.47.08',
	    #  '2019.01.30_11.26.44',
	    #  '2019.01.31_20.25.57',
	    #  '2019.02.01_11.08.18',
	    #  '2019.02.20_22.18.39']
    #output_folder - path to folder with precessed files, by default: './methane-propane/training_data'
	#sensor - sensor string index, maybe 3 possible values: 'S1', 'S2', 'S3'
	#dataset - should data be randomly splitted to training and validating subsets. If not, "test" value should be selected.
	
	# OUTPUTS
	# there are 4 files types will be created: X, Xt Y, log 
		# X - file containing sensor responses, no colnames and indexes. rows - samples, columns - features; 
        # Xt - file containing temperature observations, no colnames and indexes. rows - samples, columns - features;
        # Y - file containing one-hot encoded classes, corresponding to each sample at X\Xt file, no colnames and indexes. rows - samples, columns - ['air', 'methane', 'propane']; 
        # log - file containing information about each sample in X\Xt\Y files:
        	#experiment - name of experimnet folder (day and time info)
        	#gas - gas type: 'air', 'methane', 'propane';
        	#concentration - gas concentation, ppm;
        	#file - file name of observation series in experiment folder where sample was initially stored; 
        	#total obs - total number of samples of selected observation series
        	#curr obs - the number of selected sample during observation series 
 


    experiment_list = os.listdir(experiments_folder)[0::2]
    if dataset=='test': experiment_list = os.listdir(experiments_folder)[1::2]   #Divide los datos en test y validate:train, dependiendo del dataset que se pide en el main, en este caso se divide en pares e impares

    glob_cnt = 0 #para contar iteraciones
    log_arr_train = [] #se crea el archivo de los datos del experimento

    for exper_cnt in experiment_list:
        gas_list = np.sort(os.listdir(os.path.join(experiments_folder, exper_cnt))) #lista de gases de dataset\fechas\

        for gas_cnt in gas_list:
            conc_list = np.sort(os.listdir(os.path.join(experiments_folder, exper_cnt, gas_cnt))) #lista de concentraciones de gases de dataset\fechas\gases\

            for conc_cnt in conc_list:

                file_list = np.sort(os.listdir(os.path.join(experiments_folder, exper_cnt, gas_cnt, conc_cnt))) #lista de mediciones para sensores, hora y concentracion ej:16.03.2019_R1_400_num10, folder gases de dataset\fechas\gases\concentracion\
                file_list_sensor = [i for i in file_list  if i.split('_')[1]==sensor] #aqui se filtran solo los del sensor 

                for file_cnt in file_list_sensor:
                    raw_file = np.array(pd.read_csv(os.path.join(experiments_folder,exper_cnt,gas_cnt,conc_cnt,file_cnt), header=None)) #se lee cada medicion 
                    R_data=raw_file[1:,2:] #se filtra los ??ndices de los datos
                    
                    #raw_file_noind=raw_file[1:,2:]
                    #R_data = raw_file_noind[list(range(0,raw_file_noind.shape[0],1))]
                    #T_data = raw_file[list(range(0,raw_file.shape[0],2))] #Se extrae la temperatura en un vector

                    y_data = np.zeros([R_data.shape[0], 2]) #crea un array con la cantidad de samples n por el total de diferentes tipos de gases, n x total de gases, se llenan de 0
                    y_data[:,np.where(gas_cnt==gas_list)[0][0]]=conc_cnt # se llenan igual a la concentracion del gas en la fila que corresponda



                    if glob_cnt==0: #primera iteraci??n
                        out_xr = R_data.copy() #copia resistencia
                        #out_xt = T_data.copy() #copia temperatura
                        out_y = y_data.copy() #copia el vector de la concentracion
                        log_arr_train = np.hstack([np.repeat(np.array([exper_cnt, gas_cnt, conc_cnt, file_cnt, R_data.shape[0]]).reshape([1,-1]),R_data.shape[0], axis=0), np.arange(R_data.shape[0]).reshape(-1,1)]) #llena un array con el vector de [exp,gas, conc, medicion, total de minuto, minuto], cada vector cambia minuto, 0-57 minuto

                    else: #Siguientes iteraciones, va llenando las filas
                        out_xr = np.vstack([out_xr, R_data.copy()]) 
                        #out_xt = np.vstack([out_xt, T_data.copy()])
                        out_y = np.vstack([out_y,y_data.copy()])

                        log_arr_train = np.vstack([log_arr_train, np.hstack([np.repeat(np.array([exper_cnt, gas_cnt, conc_cnt, file_cnt,  R_data.shape[0]]).reshape([1,-1]),R_data.shape[0], axis=0), np.arange(R_data.shape[0]).reshape(-1,1)])])


                    glob_cnt+=1 #contador de iteraciones


    os.makedirs(output_folder, exist_ok=True)                

    #hace un dataset, hace los datos de train y de validate
    if dataset=='train':
        X_train, X_val, y_train, y_val = train_test_split(out_xr, out_y, test_size=0.1, random_state=42) #se crea el dataset con 20% de los datos para validate
        #Xt_train, Xt_val, yt_train, yt_val = train_test_split(out_xt, out_y, test_size=0.2, random_state=42)

        log_arr_trn, log_arr_val, y_train2, y_val2 = train_test_split(log_arr_train,  out_y, test_size=0.1, random_state=42) #se crea el log de los datos

        np.savetxt(os.path.join(output_folder,sensor+'_X_train.csv'), X_train, delimiter=',') #se salva en csv los datos de train de la resistencia
        #np.savetxt(os.path.join(output_folder,sensor+'_Xt_train.csv'), Xt_train, delimiter=',')
        np.savetxt(os.path.join(output_folder,sensor+'_X_val.csv'), X_val, delimiter=',') #se salva en csv los datos de validate de la resistencia
        #np.savetxt(os.path.join(output_folder,sensor+'_Xt_val.csv'), Xt_val, delimiter=',')
        np.savetxt(os.path.join(output_folder,sensor+'_Y_train.csv'), y_train, delimiter=',') #se salva en csv los datos de train de la concentracion del gas
        np.savetxt(os.path.join(output_folder,sensor+'_Y_val.csv'), y_val, delimiter=',') #se salva en csv los datos de validate de la concentracion del gas
        pd.DataFrame(log_arr_trn, columns=['experiment', 'gas','concentration', 'file', 'total obs', 'curr obs']).to_csv(os.path.join(output_folder,sensor+'_log_train.csv')) # se salva el log de los datos para los valores de train, se agregan indices tambien
        pd.DataFrame(log_arr_val, columns=['experiment', 'gas','concentration', 'file', 'total obs', 'curr obs']).to_csv(os.path.join(output_folder,sensor+'_log_val.csv')) # se salva el log de los datos para los valores de validate, se agregan indices tambien

    #hace un dataset con datos para prueba para la red
    if dataset == 'test':

        np.savetxt(os.path.join(output_folder,sensor+'_X_test.csv'), out_xr, delimiter=',') #se salva en csv los datos de test de la resistencia
        #np.savetxt(os.path.join(output_folder,sensor+'_Xt_test.csv'), out_xt, delimiter=',')
        np.savetxt(os.path.join(output_folder,sensor+'_Y_test.csv'), out_y, delimiter=',') #se salva en csv los datos de test de la concentracion del gas
        pd.DataFrame(log_arr_train, columns=['experiment', 'gas','concentration', 'file', 'total obs', 'curr obs']).to_csv(os.path.join(output_folder,sensor+'_log_test.csv')) # se salva el log de los datos para los valores de test, se agregan indices tambien               
         
    if dataset == 'real test': 
        #Prueba de se??al triangular #1
        fs=10*60 #numero de muestras por segundo por segundos en minuto=accesar por minuto 
        raw_file_1 = np.array(pd.read_csv(os.path.join(os.getcwd(),experiments_folder,'Prueba05-TRI-Aire-30-470-950-Fecha05032022-1045.csv'))) #se lee cada medicion              
        trim_file_1=np.concatenate((raw_file_1[fs*5:fs*38,3:],raw_file_1[fs*43:fs*70,3:],raw_file_1[fs*75:fs*111,3:],raw_file_1[fs*116:fs*140,3:]), axis=0) #se filtra los ??ndices de los datos
        reshape_file_1=np.reshape(trim_file_1, (trim_file_1.shape[0]//600, 600))           
        R_data_1=reshape_file_1[:,50:]
        
        resp_file_1 = np.concatenate((np.ones(raw_file_1[fs*5:fs*38,3:].shape[0],dtype=int)*0,np.ones(raw_file_1[fs*43:fs*70,3:].shape[0],dtype=int)*30,np.ones(raw_file_1[fs*75:fs*111,3:].shape[0],dtype=int)*470,np.ones(raw_file_1[fs*116:fs*140,3:].shape[0],dtype=int)*950), axis=0) #se crea las concentraciones correspondientes segun los datos
        y_data_1=np.vstack([np.zeros(resp_file_1.shape[0], dtype=int), resp_file_1]).transpose()
        
        #Prueba de se??al triangular #2
        raw_file_2 = np.array(pd.read_csv(os.path.join(os.getcwd(),experiments_folder,'Prueba06-TRI-Aire-100-400-800-1600-Fecha05032022-1306.csv'))) #se lee cada medicion              
        trim_file_2=np.concatenate((raw_file_2[fs*4:fs*7,3:],raw_file_2[fs*12:fs*38,3:],raw_file_2[fs*43:fs*69,3:],raw_file_2[fs*74:fs*102,3:],raw_file_2[fs*107:fs*135,3:]), axis=0) #se filtra los ??ndices de los datos
        reshape_file_2=np.reshape(trim_file_2, (trim_file_2.shape[0]//600, 600))           
        R_data_2=reshape_file_2[:,50:]
        
        resp_file_2 = np.concatenate((np.ones(raw_file_2[fs*4:fs*7,3:].shape[0],dtype=int)*0,np.ones(raw_file_2[fs*12:fs*38,3:].shape[0],dtype=int)*100,np.ones(raw_file_2[fs*43:fs*69,3:].shape[0],dtype=int)*400,np.ones(raw_file_2[fs*74:fs*102,3:].shape[0],dtype=int)*800,np.ones(raw_file_2[fs*107:fs*135,3:].shape[0],dtype=int)*1600), axis=0) #se crea las concentraciones correspondientes segun los datos
        y_data_2=np.vstack([np.zeros(resp_file_2.shape[0], dtype=int), resp_file_2]).transpose()

        #Se unen las dos pruebas en un vector de salida
        out_xr = np.vstack([R_data_1,R_data_2])
        out_y = np.vstack([y_data_1,y_data_2])
        
        os.makedirs(output_folder, exist_ok=True) 
        
        np.savetxt(os.path.join(output_folder,sensor+'_X_test.csv'), out_xr, delimiter=',') #se salva en csv los datos de test de la resistencia
        np.savetxt(os.path.join(output_folder,sensor+'_Y_test.csv'), out_y, delimiter=',') #se salva en csv los datos de test de la concentracion del gas
        
        