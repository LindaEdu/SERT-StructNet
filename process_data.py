# -*- coding: utf-8 -*-
# @Time : 2023/6/4 16:11
# @Author: LZ
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

'''
Training dataset
'''

class TrainDataset:
    def __init__(self, sequence_list, pssm_folder, label_list, train_max_length):

        self.sequences = sequence_list
        self.pssm_folder = pssm_folder
        self.labels = label_list
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
                            'W', 'Y']

        self.propertie_data = {"property"}

        self.amino_acid_encoding = {acid: i for i, acid in enumerate(self.amino_acids)}
        self.label_class = ['H', 'E', 'C']

        self.labels_encoding = {lab: i for i, lab in enumerate(self.label_class)}

        self.max_length = train_max_length
    def __getitem__(self, index):

        sequence = self.sequences[index]
        label = self.labels[index]
        encoded_sequence = self.one_hot_encode_sequence(sequence)
        encoded_properties = self.encode_propertie(sequence)
        encoded_label = self.one_hot_encode_label(label)

        pssm_file = f"{index + 1}.csv"
        pssm_path = os.path.join(self.pssm_folder, pssm_file)

        encoded_pssm = self.extract_pssm_data(pssm_path)

        encoded_fusion1 = torch.cat((encoded_pssm, encoded_sequence), dim=1)
        encoded_fusion = torch.cat((encoded_fusion1, encoded_properties), dim=1)

        return encoded_fusion, encoded_label

    def __len__(self):

        return len(self.sequences)

    def one_hot_encode_sequence(self, sequence):
        encoded_sequence = []

        for aa in sequence:
            if aa in self.amino_acid_encoding:
                encoded_aa = [0.0] * len(self.amino_acids)
                encoded_aa[self.amino_acid_encoding[aa]] = 1.0
                encoded_sequence.append(encoded_aa)

        padding_length = self.max_length - len(encoded_sequence)
        encoded_sequence += [[0.0] * len(self.amino_acids)] * padding_length

        return torch.tensor(encoded_sequence, dtype=torch.float32)

    def encode_propertie(self, sequence):
        encoded_properties = []

        for aa in sequence:
            if aa in self.propertie_data:
                encoded_properties.append(self.propertie_data[aa])
            else:
                encoded_properties.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        padding_length = self.max_length - len(encoded_properties)
        encoded_properties += [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * padding_length

        return torch.tensor(encoded_properties, dtype=torch.float32)

    def extract_pssm_data(self, pssm_file):
        pssm_data = []
        with open(pssm_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)


            for row in reader:
                selected_data = [float(value) for value in row[2:22]]
                pssm_data.append(selected_data)

        padding_length = self.max_length - len(pssm_data)
        pssm_data += [[0.0] * 20] * padding_length

        return torch.tensor(pssm_data, dtype=torch.float32)

    def one_hot_encode_label(self, label):
        encoded_label = []

        for bb in label:
            if bb in self.labels_encoding:
                encoded_bb = [0] * len(self.label_class)
                encoded_bb[self.labels_encoding[bb]] = 1
                encoded_label.append(encoded_bb)

        padding_length = self.max_length - len(encoded_label)
        encoded_label += [[0] * (len(self.label_class))] * padding_length


        return torch.tensor(encoded_label)


'''
Testing dataset

'''


class TestDataset:

    def __init__(self, sequence_list, pssm_folder, label_list, test_max_length):
        # 将传入的序列列表赋值给对象
        self.sequences = sequence_list
        self.pssm_folder = pssm_folder
        self.labels = label_list
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
                            'W', 'Y']

        self.propertie_data = {"property"}

        self.amino_acid_encoding = {acid: i for i, acid in enumerate(self.amino_acids)}
        self.label_class = ['H', 'E', 'C']
        self.labels_encoding = {lab: i for i, lab in enumerate(self.label_class)}

        self.max_length = test_max_length

    def __getitem__(self, index):

        sequence = self.sequences[index]
        label = self.labels[index]

        encoded_sequence = self.one_hot_encode_sequence(sequence)

        encoded_properties = self.encode_propertie(sequence)

        encoded_label = self.one_hot_encode_label(label)

        pssm_file = f"{index + 1}.csv"
        pssm_path = os.path.join(self.pssm_folder, pssm_file)

        encoded_pssm = self.extract_pssm_data(pssm_path)

        encoded_fusion1 = torch.cat((encoded_pssm, encoded_sequence), dim=1)
        encoded_fusion = torch.cat((encoded_fusion1, encoded_properties), dim=1)

        return encoded_fusion, encoded_label

    def __len__(self):

        return len(self.sequences)

    def one_hot_encode_sequence(self, sequence):
        encoded_sequence = []

        for aa in sequence:
            if aa in self.amino_acid_encoding:
                encoded_aa = [0.0] * len(self.amino_acids)
                encoded_aa[self.amino_acid_encoding[aa]] = 1.0
                encoded_sequence.append(encoded_aa)

        padding_length = self.max_length - len(encoded_sequence)
        encoded_sequence += [[0.0] * len(self.amino_acids)] * padding_length

        return torch.tensor(encoded_sequence, dtype=torch.float32)

    def encode_propertie(self, sequence):
        encoded_properties = []

        for aa in sequence:
            if aa in self.propertie_data:
                encoded_properties.append(self.propertie_data[aa])
            else:
                encoded_properties.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        padding_length = self.max_length - len(encoded_properties)
        encoded_properties += [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * padding_length

        return torch.tensor(encoded_properties, dtype=torch.float32)

    def extract_pssm_data(self, pssm_file):
        pssm_data = []
        with open(pssm_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                selected_data = [float(value) for value in row[2:22]]
                pssm_data.append(selected_data)

        padding_length = self.max_length - len(pssm_data)
        pssm_data += [[0.0] * 20] * padding_length

        return torch.tensor(pssm_data, dtype=torch.float32)

    def one_hot_encode_label(self, label):
        encoded_label = []

        for bb in label:
            if bb in self.labels_encoding:
                encoded_bb = [0] * len(self.label_class)
                encoded_bb[self.labels_encoding[bb]] = 1
                encoded_label.append(encoded_bb)

        # 填充到最大长度
        padding_length = self.max_length - len(encoded_label)
        encoded_label += [[0] * (len(self.label_class))] * padding_length

        return torch.tensor(encoded_label)