from config import Parameters as prms, bc
from split_class import Splitting
from scaler_class import MyStandardScaler
from LogRegr_class import LogRegr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sys import exit, argv
import pickle, csv
from os.path import isfile, join

if __name__ == '__main__':
	assert len(argv) == 3,  f"Please pass {bc.WARN}executor, \
test dataset and weights{bc.ENDC} as arguments."
	path_test = join(prms.datasets_folder, argv[1])
	if not isfile(path_test):
		exit('Test dataset is absent. Please provide it \
in "datasets" folder')
	assert isfile(prms.columns_file), "Please be sure to have file with columns"
	assert isfile(prms.scaler_params_file), "Please be sure to have scaler parameters"
	assert isfile(argv[2]), "Please be sure to train model first"
	#get columns for predicting
	with open(prms.columns_file, 'rb') as file:
		columns = pickle.load(file)
	#get parameters for scaling
	sc = MyStandardScaler()	
	with open(prms.scaler_params_file, 'rb') as file:
		[sc._mean, sc._std] = pickle.load(file)
	#define model and get weights
	model = LogRegr(prms.learning_rate, prms.n_epochs)
	model.load_model(prms.model_output_file)
	#shoould be read from file
	#get dictionary of labels
	with open(prms.dict_file, 'rb') as file:
		y_dict, y_arg_max = pickle.load(file)
	#get_labels
	y_inv_dict = [None] * y_arg_max
	for u,v in y_dict.items():
		y_inv_dict[v] = u
	#read and preprocess validation data
	df = pd.read_csv(path_test)
	X = df[columns].values
	X_sc = sc.transform(X)
	X_sc_cl = sc.fillna_with_zero(X_sc)
	X_sc_cl_add_col = np.insert(X_sc_cl, 0, 1, axis=1)
	y_pred_probabilities = model.predict(X_sc_cl_add_col)
	y_pred = np.argmax(y_pred_probabilities, axis=1)
	y_test = [None] * y_pred.shape[0]
	for i in range(y_pred.shape[0]):
		y_test[i] = y_inv_dict[y_pred[i]]
	#save results
	f = open('houses.csv', 'wt')
	writer = csv.writer(f)
	writer.writerow(['Index','Hogwarts House'])
	for i in range(y_pred.shape[0]):
		row = [i, y_test[i]]
		writer.writerow(row)
	f.close()
