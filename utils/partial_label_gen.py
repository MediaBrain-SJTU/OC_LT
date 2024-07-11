import numpy as np
import torch
import ipdb

def generate_handcraft_partial_label(total_time,weight,train_labels,t_power=1.0):
	''' 
	for each class we design a hand craft partial label set
	which depends on diffusion time and weight 
	'''
	
	weight_norm = min(weight)/max(weight)
	if torch.min(train_labels) > 1:
		raise RuntimeError('testError')
	elif torch.min(train_labels) == 1:
		train_labels = train_labels - 1
	K = torch.max(train_labels) - torch.min(train_labels) + 1
	n = train_labels.shape[0]
	partialY = torch.zeros(total_time, n, K)
	partialY[:,torch.arange(n), train_labels] = 1.0
	transition_matrix = torch.zeros(total_time, K, K)
	for each_t in range(total_time):
		for each_class in range(K):
			temp_norm_cons = 0.0
			for transited_class in range(K):
				if transited_class >= each_class:
					continue
				else:
					unnormalized_prob = (weight[transited_class]/weight[each_class])* np.power(each_t/total_time,t_power)
					transition_matrix[each_t,each_class,transited_class] = unnormalized_prob*weight_norm
	transition_matrix[:,range(K),range(K)] = 1.0

	random_n = torch.from_numpy(np.random.uniform(0, 1, size=(total_time,n, K)))
	# for j in range(n):  # for each instance
	# 	for jj in range(K): # for each class 
	# 		if jj == train_labels[j]: # except true class
	# 			continue
	# 		for each_t in range(total_time):
	# 			if random_n[each_t, j, jj] < transition_matrix[each_t, train_labels[j], jj]:
	# 				partialY[each_t, j, jj] = 1.0
	for j in range(n):  # for each instance
		for jj in range(K): 
			if jj == train_labels[j] or jj>train_labels[j]: # except true class
				continue
			partialY[:, j, jj] = (random_n[:, j, jj] < transition_matrix[:,train_labels[j], jj]).float()
	partialY = np.transpose(partialY,(1,2,0))
	return partialY


def generate_handcraft_partial_label_full_matrix(total_time,weight,train_labels,t_power=1.0):
	''' 
	for each class we design a hand craft partial label set
	which depends on diffusion time and weight 
	'''
	weight_norm = min(weight)/max(weight)
	if torch.min(train_labels) > 1:
		raise RuntimeError('testError')
	elif torch.min(train_labels) == 1:
		train_labels = train_labels - 1
	K = torch.max(train_labels) - torch.min(train_labels) + 1
	n = train_labels.shape[0]
	partialY = torch.zeros(total_time, n, K)
	partialY[:,torch.arange(n), train_labels] = 1.0
	transition_matrix = torch.zeros(total_time, K, K)
	for each_t in range(total_time):
		for each_class in range(K):
			temp_norm_cons = 0.0
			for transited_class in range(K):
				if transited_class == each_class:
					continue
				else:
					imb_weight = (weight[transited_class]/weight[each_class]) if weight[transited_class] > weight[each_class] else (weight[each_class]/weight[transited_class])
					unnormalized_prob = imb_weight*np.power(each_t/total_time,t_power)*weight_norm
					transition_matrix[each_t,each_class,transited_class] = unnormalized_prob
	transition_matrix[:,range(K),range(K)] = 1.0
	random_n = torch.from_numpy(np.random.uniform(0, 1, size=(total_time,n, K)))
	# for j in range(n):  # for each instance
	# 	for jj in range(K): # for each class 
	# 		if jj == train_labels[j]: # except true class
	# 			continue
	# 		for each_t in range(total_time):
	# 			if random_n[each_t, j, jj] < transition_matrix[each_t, train_labels[j], jj]:
	# 				partialY[each_t, j, jj] = 1.0
	for j in range(n):  # for each instance
		for jj in range(K): 
			if jj == train_labels[j]: # except true class
				continue
			partialY[:, j, jj] = (random_n[:, j, jj] < transition_matrix[:,train_labels[j], jj]).float()
	partialY = np.transpose(partialY,(1,2,0))
	return partialY


if __name__ == '__main__':
	weight = [1.0,0.5,0.25,0.05,0.01]
	train_labels = torch.tensor([0,4,3,2,1])
	partialY = generate_handcraft_partial_label_full_matrix(1000,weight,train_labels)
	ipdb.set_trace()



