# ====================================	#
# Quick Draw Custom Classifier 			#
#	- Estimator Model Module			#
#										#
# Nicolau Tahan 		07/06/2018		#
# ====================================	#

''' Nesse modulo estao o modelo do Estimador e o processo 
	de treinamento e evaluation
'''

import numpy as np
import tensorflow as tf

import data_mani
from rnn_model import rnn_model
from cnn_model import cnn_model

BATCH_SIZE = 100
MAX_STEPS = 1000
LEARNING_RATE = 0.01
EPSLON = 0.001


def parse_data():
	data_obj_list = data_mani.load_data()

	# Os dados sao dividos da forma: 70% para treinamento
	#								 20% para validacao
	#								 10% para mostrar
	corte_a = int(0.7*len(data_obj_list))
	corte_b = corte_a + int(0.2*len(data_obj_list))

	train_obj_list = data_obj_list[0 : corte_a]
	test_obj_list = data_obj_list[corte_a : corte_b]
	predict_obj_list = data_obj_list[corte_b : len(data_obj_list)]

	return train_obj_list, test_obj_list, predict_obj_list

def run_prog(arg1, arg2, rnn, train_obj_list, test_obj_list, predict_obj_list):

	# FEATURE COLUMN
	""" Entao o esquema da feature column e que o estimador precisa saber como ler o Dataset que input funciton retorna
		entao precisa criar esse vetor. 

		A feature column e um vetor que, para cada feature do input, ele define a key do dicionario e as dimensoes (shape)
		desses tensores. No caso dos desenhos e apenas a imagem [28 28]. Mas poderia-se utilizar outros como media (shape [1])
		e qlq outro input com qualquer forma
	"""
	feature_column = [tf.feature_column.numeric_column(key = 'img', shape = [28, 28])]

	cnn_config = {'feature_column' : feature_column,
				 'kernels'		   : [[arg1, arg1], [arg1, arg1]],
				  'dense'		   : arg2,
				  'n_classes'  	   : 3,
				  'learning_rate'  : LEARNING_RATE
									}

	rnn_config = {'feature_column' : feature_column,
				  'hidden_units'   : [arg1, arg2],
				  'n_classes'  	   : 3,
				  'learning_rate'  : LEARNING_RATE
									}
	true_config = rnn_config if rnn else cnn_config

	auxsrt = "rnn_" if rnn else "cnn_"
	direr = "model/" + auxsrt + str(arg1) + "-" + str(arg2) + "--2" 

	meu_classificador = tf.estimator.Estimator(
									model_fn = cnn_model,
									params = cnn_config,
									model_dir = direr)

	meu_classificador.train(input_fn= lambda:data_mani.train_input_fn(train_obj_list, BATCH_SIZE), max_steps= MAX_STEPS)

	class_eval = meu_classificador.evaluate(input_fn= lambda:data_mani.test_input_fn(test_obj_list, BATCH_SIZE))

	predictions = meu_classificador.predict(input_fn= lambda:data_mani.test_input_fn(predict_obj_list, BATCH_SIZE))


	print('')
	print('')
	print('Tamanho dos Datasets:')
	print('  Treinamento\t=> ' + str(len(train_obj_list)))
	print('  Teste\t\t=> ' + str(len(test_obj_list)))

	for result ,expected in zip(predictions, predict_obj_list):
		aux_labels = ['Aviao', 'Maca', 'Bike']
		
		result_index = result['class_ids'][0]
		expected_index = expected.label[0]

		prop = result['probs'][result_index] * 100
		prop = str(prop)[0:4]

		info_string = 'Resultado = ' + aux_labels[result_index] + '\t Esperado = ' + aux_labels[expected_index] + '\t(' + str(prop) + ')'
		if aux_labels[result_index] == aux_labels[expected_index]:
			info_string = info_string + ' V'
		
		print(info_string)

def main(args):
	argss = [[9, 6, True],
			[5, 5, True],
			[15, 9, True],
			[3, 5, False],
			[5, 5, False],
			]

	train_obj_list, test_obj_list, predict_obj_list = parse_data()
	for (arg1, arg2, rnn) in argss:
		run_prog(arg1, arg2, rnn, train_obj_list, test_obj_list, predict_obj_list)

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()