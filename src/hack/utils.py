import pandas as pd


def numpy_to_csv(np_array, header, index, file_name):
	data_frame = pd.DataFrame(np_array, index=index, columns=header)
	print(data_frame)
	data_frame.to_csv(file_name, header=header, index=True)
