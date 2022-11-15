from dataclasses import dataclass

@dataclass
class Parameters:
	datasets_folder: str = "./datasets"
	is_header: bool = True
	test_ratio: float = 0.2
	learning_rate: float = 0.03
	n_epochs: int = 150
	early_stopping: bool = True
	random_seed: int = 21
	train_set: str = "dataset_train.csv"
	test_set: str = "dataset_test.csv"
	dict_file:str = "houses_dictionary.pkl"
	model_output_file: str = "model_params.pkl"
	scaler_params_file: str = "scaler_params.pkl"
	columns_file: str = "columns.pkl"

class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'