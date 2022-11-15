import numpy as np
import csv

def read_file(file_n, strings=False):
	"""
	if strings == True, parse NaNs as strings
	"""
	ds = [] #dataset
	#we have earlier checked that file is present
	with open(file_n) as f:
		reader = csv.reader(f)
		try:
			#locate header
			for line in reader:
				row_len = ft_count_all(line)
				feat_names = line
				break
			for i, line in enumerate(reader):
				if ft_count_all(line) != row_len:
					exit(f"Row length problem on line {i}.")
				row = [None] * row_len
				for k, el in enumerate(line):
					try:
						value = float(el)
					except:
						value = el if strings else np.nan
					row[k] = value				
				ds.append(row)
		except:
			exit(f"Reading problem: line {reader.line_num}")
	return feat_names, np.array(ds, dtype=object)

# def ft_mean_arr(arr):
# 	"""
# 	Be sure that arr doesn't have NaNs
# 	"""
# 	total = 0
# 	count = 0
# 	for el in arr:
# 		assert ~np.isnan(el), "ft_mean_arr: NaN encountered"
# 		total = total + el
# 		count += 1
# 	return total / count

# def ft_std_arr(arr, mean):
# 	"""
# 	Be sure that arr doesn't have NaNs
# 	"""
# 	total = 0
# 	count = 0
# 	for el in arr:
# 		assert ~np.isnan(el), "ft_mean_arr: NaN encountered"
# 		total = total + (el - mean) ** 2
# 		count += 1
# 	return (total / count) ** 0.5

def mean_(X):
	"""
	calculate mean of an array without considering NaNs
	"""
	total = 0
	count = 0
	for x in X:
		if np.isnan(x):
			continue
		total = total + x
		count += 1
	return total / count

def std_(X, mean):
	"""
	calculate std of an array without considering NaNs
	"""
	# mean = mean_(X)
	total = 0
	count = 0
	# print(mean)
	# input()
	for x in X:
		# print("      in std: x = ", x)
		if np.isnan(x):
			continue
		total = total + (x - mean) ** 2
		count += 1
	# print("in std_: total = ", total)
	return (total / count) ** 0.5

def ft_count_nonan_and_args(arr):
	"""
	collect numeric elements from arr and return them with their amount
	"""
	c = 0
	args = []
	for i, el in enumerate(arr):
		if np.isnan(el):
			continue
		c += 1
		args.append(i)
	return c, args

def ft_count_all(arr):
	"""
	count length of arr including NaNs
	"""
	c = 0
	for el in arr:
		c += 1
	return c

def ft_count_nans(arr):
	"""
	count only NaNs
	"""
	c = 0
	for el in arr:
		if np.isnan(el):
			c += 1
	return c

def ft_count_nonan(arr):
	"""
	count total length of arr
	"""
	c = 0
	for el in arr:
		if np.isnan(el):
			continue
		c += 1
	return c



def ft_count(X):
	try:
		X = X.astype('float')
		X = X[~np.isnan(X)]
		return len(X)
	except:
		return len(X)

def ft_sum_arr(arr):
	sum_ = 0
	for el in arr:
		assert isinstance(el, float), "ft_sum_arr: The value is not float"
		if np.isnan(el):
			continue
		sum_ += el
	return sum_

def ft_sq_deviation(arr, mean):
	total = 0
	for el in arr:
		assert isinstance(el, float), "ft_sq_deviation: The value is not float"
		if np.isnan(el):
			continue
		total = total + (el - mean) ** 2
	return total

def ft_min(X):
	min_value = X[0]
	for x in X:
		if np.isnan(x):
			continue 
		val = x
		if val < min_value:
			min_value = val
	return min_value

def ft_max(X):
	max_value = X[0]
	for x in X:
		if np.isnan(x):
			continue
		val = x
		if val > max_value:
			max_value = val
	return max_value

def ft_percentile(arr, percentile):
	arr = [x for x in arr if str(x) != 'nan']
	n = len(arr)
	p = n * percentile / 100
	if p.is_integer():
		return sorted(arr)[int(p)]
	else:
		return sorted(arr)[int(np.ceil(p)) - 1]


def form_targets_dictionary(y):	
	y_d = dict()
	count = 0
	for el in y:
		if y_d.get(el, None) is None:
			y_d[el] = count
			count += 1
	return y_d, count
	
def convert_targets_num(y, y_dict):
	y_converted = [None] * len(y)
	for i, el in enumerate(y):
		y_converted[i] = y_dict[el]
	return np.asarray(y_converted)
