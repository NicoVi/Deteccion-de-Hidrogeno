import pandas as pd
import numpy as np
import os
def dataset_from_files(experiments_folder, output_folder):


    #Prueba de señal triangular #1
    fs=10*60 #numero de muestras por segundo por segundos en minuto=accesar por minuto 
    raw_file_1 = np.array(pd.read_csv(os.path.join(os.getcwd(),experiments_folder,'Prueba05-TRI-Aire-30-470-950-Fecha05032022-1045.csv'))) #se lee cada medicion              
    trim_file_1=np.concatenate((raw_file_1[fs*5:fs*38,3:],raw_file_1[fs*43:fs*70,3:],raw_file_1[fs*75:fs*111,3:],raw_file_1[fs*116:fs*140,3:]), axis=0) #se filtra los índices de los datos
    reshape_file_1=np.reshape(trim_file_1, (trim_file_1.shape[0]//600, 600))  
    
    resp_file_1 = np.concatenate((np.ones(raw_file_1[fs*5:fs*38,3:].shape[0],dtype=int)*0,np.ones(raw_file_1[fs*43:fs*70,3:].shape[0],dtype=int)*30,np.ones(raw_file_1[fs*75:fs*111,3:].shape[0],dtype=int)*470,np.ones(raw_file_1[fs*116:fs*140,3:].shape[0],dtype=int)*950), axis=0) #se crea las concentraciones correspondientes segun los datos
    resp_shape_file_1=np.reshape(resp_file_1, (resp_file_1.shape[0]//600, 600)) 
    y_data_1=np.vstack([np.zeros(resp_shape_file_1.shape[0], dtype=int), resp_shape_file_1[:,0]]).transpose()
    
    #Prueba de señal triangular #2
    raw_file_2 = np.array(pd.read_csv(os.path.join(os.getcwd(),experiments_folder,'Prueba06-TRI-Aire-100-400-800-1600-Fecha05032022-1306.csv'))) #se lee cada medicion              
    trim_file_2=np.concatenate((raw_file_2[fs*4:fs*7,3:],raw_file_2[fs*12:fs*38,3:],raw_file_2[fs*43:fs*69,3:],raw_file_2[fs*74:fs*102,3:],raw_file_2[fs*107:fs*135,3:]), axis=0) #se filtra los índices de los datos
    reshape_file_2=np.reshape(trim_file_2, (trim_file_2.shape[0]//600, 600))  
    
    resp_file_2 = np.concatenate((np.ones(raw_file_2[fs*4:fs*7,3:].shape[0],dtype=int)*0,np.ones(raw_file_2[fs*12:fs*38,3:].shape[0],dtype=int)*100,np.ones(raw_file_2[fs*43:fs*69,3:].shape[0],dtype=int)*400,np.ones(raw_file_2[fs*74:fs*102,3:].shape[0],dtype=int)*800,np.ones(raw_file_2[fs*107:fs*135,3:].shape[0],dtype=int)*1600), axis=0) #se crea las concentraciones correspondientes segun los datos
    resp_shape_file_2=np.reshape(resp_file_2, (resp_file_2.shape[0]//600, 600)) 
    y_data_2=np.vstack([np.zeros(resp_shape_file_2.shape[0], dtype=int), resp_shape_file_2[:,0]]).transpose()
    
    #Se unen las dos pruebas en un vector de salida
    R_data_1 = np.vstack([reshape_file_1,reshape_file_2])
    out_xr=np.sort(R_data_1,axis=1)
    out_xr_des=out_xr[:,50:]
    
    out_y = np.vstack([y_data_1,y_data_2])
    
    #Se consigue el orden del indíce que da el orden de los valores de una muestra de los datos de entrenamiento
    x_train_order = np.array(pd.read_csv(os.path.join(os.getcwd(),output_folder,'R1_X_train.csv'))) #se lee cada medicion              
    x_train_df = pd.DataFrame(np.arange(1, x_train_order.shape[1]+1, 1), index=x_train_order[2000,:], columns=['A'])
    x_train_index=x_train_df.sort_index(ascending=True)
    x_train_index=np.array(x_train_index)
    
    
    out_xr_order=np.zeros((out_xr_des.shape[0], out_xr_des.shape[1]))
    for i in range(out_xr_des.shape[0]):
        a,out_xr_order[i,:]=zip(*sorted(zip(x_train_index,out_xr_des[i,:])))
    #gx_data = np.linspace(-1, 1, out_xr.shape[1])
    #gy_data = stats.norm.pdf(gx_data, 0, 1)
    
    #out_gxr = [x for _,x in sorted(zip(gy_data,out_xr[0,:]))]
 
    
    np.savetxt(os.path.join(output_folder,'R1_X_test.csv'), out_xr_order, delimiter=',') #se salva en csv los datos de test de la resistencia
    np.savetxt(os.path.join(output_folder,'R1_Y_test.csv'), out_y, delimiter=',') #se salva en csv los datos de test de la concentracion del gas
    