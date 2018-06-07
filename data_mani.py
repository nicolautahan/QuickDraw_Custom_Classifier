# ====================================	#
# Quick Draw Custom Classifier 			#
#	- Data Manipulation Module			#
#										#
# Nicolau Tahan 		07/06/2018		#
# ====================================	#

''' Nesse modulo estao as funcoes de importar dados assim como 
	as Input Functions para treinamento e evaluation
'''

import numpy as np
import tensorflow as tf


basic_path = 'numpy-data/full-numpy_bitmap-'
labels_list = ['airplane', 'apple', 'bicycle']	# Tipos de desenho

label_index = {
	'airplane'	: [0],
	'apple'		: [1],
	'bicycle'	: [2]
}

IMAGE_LIMIT = 300


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


# Classe definida para juntar a label a uma imagem
class LabeledImage():
	def __init__(self, img, label):
		self.img = img
		self.label = label

# Funcao pra carregar as imagens 28x28 de numpys arquivos
def load_data():

	data_obj_list = []
	for label in labels_list:
		data = np.load(basic_path + label + '.npy')
		i = 0
		print("Carregando " + label + ' data')
		printProgressBar(i, IMAGE_LIMIT, length= 20)
		for img in data:

			# Cria um objeto LabeledImage para unir a label e a imagem
			aux_obj = LabeledImage(img, label_index[label])
			data_obj_list.append(aux_obj)

			i = i + 1
			printProgressBar(i, IMAGE_LIMIT, length= 20)
			if i >= IMAGE_LIMIT:
				break

	np.random.shuffle(data_obj_list)

	return data_obj_list

# INPUT FUNCTIONS
"""	O esquema e que todos os metodos do estimador (train, evaluate, estimate)
	precisam de uma input function para interpretar os dados.

	Input functions sao funcoes que retornam ou uma tuple ou um objeto da 
	classe tf.data.Dataset que contem o set de (features, label). Ou seja
	um Tensor com as imagens e um Tensor com as labels respectivas.

	Nas funcoes abaixo uso um Dataset. Ele eh de dimensoes ((img:[28 28]), (label : [1]))
"""
def train_input_fn(obj_list, batch_size):
	features_dict = {'img' : []}
	labels_list = []

	for img_obj in obj_list:
		img_tensor = tf.convert_to_tensor(img_obj.img)
		img_tensor = tf.reshape(img_tensor, [28, 28])

		features_dict['img'].append(img_tensor)
		labels_list.append(img_obj.label)

	train_ds = tf.data.Dataset.from_tensor_slices((features_dict, labels_list))
	train_ds = train_ds.shuffle(100).repeat().batch(batch_size)

	return train_ds

def test_input_fn(obj_list, batch_size):
	features_dict = {'img' : []}
	labels_list = []

	for img_obj in obj_list:
		img_tensor = tf.convert_to_tensor(img_obj.img)
		img_tensor = tf.reshape(img_tensor, [28, 28])

		features_dict['img'].append(img_tensor)
		labels_list.append(img_obj.label)

	test_ds = tf.data.Dataset.from_tensor_slices((features_dict, labels_list))
	test_ds = test_ds.batch(batch_size)

	return test_ds