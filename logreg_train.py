from config import Parameters as prms, bc
from LogRegr_class import LogRegr
from split_class import Splitting
from scaler_class import MyStandardScaler
import pandas as pd
import numpy as np
from sys import exit, argv
import pickle
from used_functions import form_targets_dictionary, convert_targets_num
from os.path import isfile, join

if __name__ == '__main__':
	assert len(argv) == 2,  f"Please pass {bc.WARN}executor and \
test dataset{bc.ENDC} as arguments."
	path_train = join(prms.datasets_folder, argv[1])
	if not isfile(path_train):
		exit(f'{bc.FAIL}Train dataset is absent.{bc.ENDC} Please provide it \
in "datasets" folder')
	df = pd.read_csv(path_train)
	# columns = ['Hogwarts House','Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts']
	columns = ['Hogwarts House',
	'Herbology','Defense Against the Dark Arts',
	'Divination','Muggle Studies','Charms']
	#save feature column names
	with open(prms.columns_file, 'wb') as file:
		pickle.dump(columns[1:], file)

	df = df[columns]
	X = df.drop(['Hogwarts House'],axis=1).values
	y = df['Hogwarts House'].values

#Scaling and clearing nans
	sc = MyStandardScaler()
	sc.fit(X)
	X_sc = sc.transform(X)
	X_sc_clear = sc.fillna_with_zero(X_sc)

	with open(prms.scaler_params_file, 'wb') as file:
		pickle.dump([sc._mean, sc._std], file)

#map y to numbers, i.e. indices for argmax
	y_dict, y_arg_max = form_targets_dictionary(y)
	with open(prms.dict_file, 'wb') as file:
		pickle.dump([y_dict, y_arg_max], file)

#convert y to numbers according to dictionary
	y_conv = convert_targets_num(y, y_dict) #converted to numeric format
#insert bias column as ones
	X_sc_clear_add_col = np.insert(X_sc_clear, 0, 1, axis=1)
#Splitting
	spl = Splitting(y_conv.shape[0], prms.test_ratio, shuffle=False) 
	X_train, X_test, y_train, y_test = spl.split_dataset(X_sc_clear_add_col, y_conv)
#training
	model = LogRegr(prms.learning_rate, prms.n_epochs, 
	early_stopping=prms.early_stopping)
	model.train(X_train, y_train, X_test, y_test, y_arg_max)
	model.save_model(prms.model_output_file)
	model.plot_costs_accuracy()
