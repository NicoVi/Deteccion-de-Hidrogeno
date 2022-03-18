import preprocessing as pre
import preprocessing_real_data as real

#en este código se realiza la creacion del dataset, se llama a preprocessing y se hacen los csv de los datos de train, validate y test
pre.dataset_from_files('./data', './preprocessed data',) #

pre.dataset_from_files('./data', './preprocessed data', 'R1','test')

#real test

pre.dataset_from_files('./data', './real data',)

real.dataset_from_files('.\\sensor data', '.\\real data')