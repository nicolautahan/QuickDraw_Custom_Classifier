import numpy as np
import tensorflow as tf

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
	EPSLON = 0.001
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