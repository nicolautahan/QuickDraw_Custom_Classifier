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

BATCH_SIZE = 100
MAX_STEPS = 1000
LEARNING_RATE = 0.01
EPSLON = 0.001

# RNN Classifier Model
''' Comecando apenas um RNN com 2 hidden com 10 nodes
	config = {
		'feature_column' : Coloca a feature columns,
		'hidden_units'	 : [10, 10], no caso	,
		'n_classes'		 : 3, no caso,
		'learning_rate'	 : 0.1
	}
'''
def rnn_model(features, labels, params, mode):
	with tf.Session() as sess:
		# Input layer => recebe o dataset e a coluna de features
		#  Se tiver uma feature na feature_column com a key 'img' ele vai procurar
		#  essa key no dicionario
		neural_net = tf.feature_column.input_layer(features, params['feature_column'])

		# Cria uma layer densa(fully connected) com o tanto de nodes
		# especificado
		for i, units in enumerate(params['hidden_units']):
			neural_net = tf.layers.dense(neural_net, units= units, activation= tf.nn.relu, name = 'DenseLayer' + str(i))

		# Computa as saidas, um node para cada classe
		logits = tf.layers.dense(neural_net, units= params['n_classes'], activation= None, name = 'Logits')
		eps_tensor = tf.constant([EPSLON, EPSLON, EPSLON], name = 'Epsilon')

		logits = tf.add(logits, eps_tensor)

		# Agora eh a parte de computar as predicoes
		predictions_prob = tf.argmax(logits, 1, name = 'MaiorCerteza')
		if mode == tf.estimator.ModeKeys.PREDICT:
			predictions = {
				'class_ids' : predictions_prob[:, tf.newaxis],
				'probs'		: tf.nn.softmax(logits, name = 'Probabilidade'),
				'logits'	: logits
			}
			return tf.estimator.EstimatorSpec(mode, predictions= predictions)

		# Apatir de agora e definindo o treinamento e a evaluation

		# Perda
		loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits= logits)
		loss_summary = tf.summary.scalar('loss', loss)
		# Compute evaluation metrics.
		accuracy = tf.metrics.accuracy(labels=labels,
										predictions=predictions_prob,
										name='accuracy')
		metrics = {'accuracy': accuracy}
		acc_summary = tf.summary.scalar('accuracy', accuracy[1])

		if mode == tf.estimator.ModeKeys.EVAL:
			return tf.estimator.EstimatorSpec(
				mode, loss=loss, eval_metric_ops=metrics)

		
		# Create training op.
		assert mode == tf.estimator.ModeKeys.TRAIN

		optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
		train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# CNN Classifier Model
#	Com duas layers de conv e duas de pool
''' Params = {
		'feature_column' :
		'kernels'		 : Vetor com os kernels usados (2)
		'dense'			 : Numero de nodes na camada densa
		'n_classes'		 : Numero de classes
		'learning_rate'	 : 0.1
	}
'''
def cnn_model(features, labels, params, mode):
	# Input layer => recebe o dataset e a coluna de features
	#  Se tiver uma feature na feature_column com a key 'img' ele vai procurar
	#  essa key no dicionario
	#conv_net = tf.feature_column.input_layer(features, params['feature_column'])
	with tf.Session() as sess:
		conv_net = tf.reshape(features['img'], [-1, 28, 28, 1], name = 'InputReshape')

		# Primeira Convolutional Layer
		conv_net = tf.layers.conv2d(conv_net,
								filters = 1,
								kernel_size = params['kernels'][0],
								padding = 'SAME')
		conv_net = tf.layers.max_pooling2d(conv_net,
								 pool_size = [2, 2],
								 strides = 2)

		# Segunda Convolutional Layer
		conv_net = tf.layers.conv2d(conv_net,
								filters = 1,
								kernel_size = params['kernels'][1], 
								padding = 'SAME')
		conv_net = tf.layers.max_pooling2d(conv_net,
								 pool_size = [2, 2],
								 strides = 2)

		# Final Dense Layer
		conv_net = tf.reshape(conv_net, [-1, 7*7], name = 'reshapeForDense')
		conv_net = tf.layers.dense(conv_net, units = params['dense'], activation = tf.nn.relu)

		# Computa as saidas, um node para cada classe
		logits = tf.layers.dense(conv_net, units= params['n_classes'], activation= None)

		# Agora eh a parte de computar as predicoes
		predictions_prob = tf.argmax(logits, 1, name = 'Prediction')
		if mode == tf.estimator.ModeKeys.PREDICT:
			predictions = {
				'class_ids' : predictions_prob[:, tf.newaxis],
				'probs'		: tf.nn.softmax(logits),
				'logits'	: logits
			}
			return tf.estimator.EstimatorSpec(mode, predictions= predictions)

		# Apatir de agora e definindo o treinamento e a evaluation

		# Perda
		loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits= logits)

		# Compute evaluation metrics.
		accuracy = tf.metrics.accuracy(labels=labels,
										predictions=predictions_prob,
										name='acc_op')
		metrics = {'accuracy': accuracy}
		tf.summary.scalar('accuracy', accuracy[1])

		if mode == tf.estimator.ModeKeys.EVAL:
			return tf.estimator.EstimatorSpec(
				mode, loss=loss, eval_metric_ops=metrics)

		# Create training op.
		assert mode == tf.estimator.ModeKeys.TRAIN

		optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], epsilon= EPSLON)
		train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main(args):
	data_obj_list = data_mani.load_data()

	# Os dados sao dividos da forma: 70% para treinamento
	#								 20% para validacao
	#								 10% para mostrar
	corte_a = int(0.7*len(data_obj_list))
	corte_b = corte_a + int(0.2*len(data_obj_list))

	train_obj_list = data_obj_list[0 : corte_a]
	test_obj_list = data_obj_list[corte_a : corte_b]
	predict_obj_list = data_obj_list[corte_b : len(data_obj_list)]

	# FEATURE COLUMN
	""" Entao o esquema da feature column e que o estimador precisa saber como ler o Dataset que input funciton retorna
		entao precisa criar esse vetor. 

		A feature column e um vetor que, para cada feature do input, ele define a key do dicionario e as dimensoes (shape)
		desses tensores. No caso dos desenhos e apenas a imagem [28 28]. Mas poderia-se utilizar outros como media (shape [1])
		e qlq outro input com qualquer forma
	"""
	feature_column = [tf.feature_column.numeric_column(key = 'img', shape = [28, 28])]

	cnn_config = {'feature_column' : feature_column,
				 'kernels'		   : [[3, 3], [3, 3]],
				  'dense'		   : 5,
				  'n_classes'  	   : 3,
				  'learning_rate'  : LEARNING_RATE
									}

	rnn_config = {'feature_column' : feature_column,
				  'hidden_units'   : [5, 5],
				  'n_classes'  	   : 3,
				  'learning_rate'  : LEARNING_RATE
									}

	meu_classificador = tf.estimator.Estimator(
									model_fn = cnn_model,
									params = cnn_config,
									model_dir = 'model/cnn')

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

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()