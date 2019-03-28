
import numpy as np
import tensorflow as tf

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
		EPSLON = 0.001
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