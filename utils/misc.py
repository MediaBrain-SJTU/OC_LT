import numpy as np
import torch
import ipdb
from .partial_label_gen import generate_handcraft_partial_label,generate_handcraft_partial_label_full_matrix

from torch.utils.data import Dataset


class gen_index_dataset(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, index):
        each_image = self.images[index]
        each_label = self.given_label_matrix[index]

        return each_image, each_label

def prepare_full_train_loader(dtset):
	train_size = len(dtset)
	full_train_loader =  torch.utils.data.DataLoader(dataset=dtset,
										batch_size=train_size,
										shuffle=False,
										drop_last=False,
										num_workers=0)
	return full_train_loader



def prepare_train_loaders_for_partial_labels(dtset, batch_size,weight,total_time=1000,t_norm=2.0,full_transition=False):
	full_train_loader = prepare_full_train_loader(dtset)
	for i, (data, labels) in enumerate(full_train_loader):
		K = torch.max(labels) + 1  # K is number of classes, 
	gen_label_func = generate_handcraft_partial_label_full_matrix if full_transition else generate_handcraft_partial_label
	partialY = gen_label_func(total_time,weight,labels,t_power=t_norm)
	partial_matrix_dataset = gen_index_dataset(data, partialY.float(),
	                                           partialY.float())
	partial_matrix_train_loader = torch.utils.data.DataLoader(
		dataset=partial_matrix_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=8)
	return partial_matrix_train_loader
