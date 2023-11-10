# -*- coding: utf-8 -*-            
# @Time : 2023/6/2 20:05
# @Author: LZ
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from process_data import TrainDataset
from process_data import TestDataset
import datetime
from model import SERT_StructNet

batch_size = 8
train_max_length = 100
test_max_length = 150
test_num_samples= 209
train_num_samples= 5600
test_drop_last = test_num_samples % batch_size != 0
train_drop_last = train_num_samples % batch_size != 0
Learning_rate = 0.0001

epoch = 100

print(datetime.datetime.now())

sequence_list = []
label_list = []
with open('dataset/train/GPU/train.csv', 'r') as sequence_file:
    reader = csv.reader(sequence_file)
    next(reader)
    for row in reader:
        sequence = row[1]
        label = row[2]
        sequence_list.append(sequence)
        label_list.append(label)


pssm_folder = 'dataset/train/GPU/pssm'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = TrainDataset(sequence_list, pssm_folder, label_list, train_max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=train_drop_last)  # 创建数据加载器，设置合适的批量大小和其他参数


sequence_list = []
label_list = []
with open('dataset/test/GPU-test/CB513(100-150)/cb513.csv', 'r') as sequence_file:
    reader = csv.reader(sequence_file)
    next(reader)
    for row in reader:
        sequence = row[0]
        label = row[1]
        sequence_list.append(sequence)
        label_list.append(label)

pssm_folder = 'dataset/test/GPU-test/CB513(100-150)/pssm_cb513'

test_dataset = TestDataset(sequence_list, pssm_folder, label_list, test_max_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=test_drop_last)  # 创建数据加载器，设置合适的批量大小和其他参数

model = SERT_StructNet()

model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

loss_fn = loss_fn.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate, weight_decay=1e-8)

def calculate_sov(S1, S2):
    if len(S1) != len(S2):
        raise ValueError("S1 and S2 must have the same length")

    NSov = len(S1)
    length_S1 = len(S1)
    length_S2 = len(S2)

    minov = 0
    maxov = 0
    S0 = 0

    for i in range(NSov):
        if S1[i] == S2[i]:
            minov += 1
            maxov += 1
            S0 += 1
        else:
            maxov += 1

    sigma = min(maxov - minov, minov, length_S1 // 2, length_S2 // 2)

    numerator = minov + sigma
    denominator = maxov

    a = numerator / denominator
    b = a * length_S1
    c = b * S0

    sov = c / NSov

    return sov

def train(model, train_dataloader, loss_fn, optimizer):

    train_loss_history = []
    train_acc_history = []

    best_loss = 9999

    for i in range(epoch):

        print("--------第 {} 轮训练开始--------".format(i+1))
        model.train()

        correct_predictions = 0
        train_loss_total = 0

        train_total_sov = 0.0
        train_total_batches = 0

        # 训练开始
        for batch_data in train_dataloader:

            encoded_fusion = batch_data[0].to(device)
            labels = batch_data[1].to(device)

            x = model(encoded_fusion)
            loss = 0.0
            batch_size, sequence_length, num_classes = x.size()

            S1 = ""
            S2 = ""

            for t in range(sequence_length):
                protein_pred_t = x[:, t, :]
                labels_t = labels[:, t, :]
                protein_pred_t = protein_pred_t.view(batch_size, num_classes)
                labels_t = labels_t.view(batch_size, num_classes)

                loss_t = loss_fn(protein_pred_t, torch.argmax(labels_t, dim=1))
                loss += loss_t

                predicted_labels = torch.argmax(protein_pred_t, dim=1)
                true_labels = torch.argmax(labels_t, dim=1)
                correct_predictions += torch.sum(predicted_labels == true_labels).item()

                S1 += "H" if true_labels[0].item() == 0 else ("E" if true_labels[0].item() == 1 else "C")
                S2 += "H" if predicted_labels[0].item() == 0 else ("E" if predicted_labels[0].item() == 1 else "C")

            sov = calculate_sov(S1, S2)
            train_total_sov += sov
            train_total_batches += 1

            total_predictions = sequence_length * batch_size

            loss /= total_predictions

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += loss

        print("epoch: {}".format(i+1))
        epoch_loss = train_loss_total.item() / len(train_dataloader)
        train_loss_history.append(epoch_loss)
        print("第 {} 轮的平均LOSS： {} ".format(i + 1, epoch_loss))

        epoch_accuracy = correct_predictions / (len(train_dataloader) * train_max_length * batch_size)
        train_acc_history.append(epoch_accuracy)
        print("第 {} 轮的平均ACC： {} ".format(i + 1, epoch_accuracy))

        average_sov = (train_total_sov / train_total_batches) / 100
        print("第 {} 轮的平均SOV： {} ".format(i + 1, average_sov))

        with open('./log/loss.txt','a+') as f:
            f.write(str(epoch_loss))
            f.write('\n')

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "./model_save/best_model.pkl")

        model.eval()

        with torch.no_grad():

            correct_predictions = 0
            test_loss_total = 0
            test_total_sov = 0.0
            test_total_batches = 0.0

            for batch_data in test_dataloader:
                encoded_fusion = batch_data[0].to(device)
                labels = batch_data[1].to(device)

                x = model(encoded_fusion)
                test_loss = 0.0
                batch_size, sequence_length, num_classes = x.size()
                S1 = ""
                S2 = ""

                for t in range(sequence_length):

                    protein_pred_t = x[:, t, :]
                    labels_t = labels[:, t, :]
                    protein_pred_t = protein_pred_t.view(batch_size, num_classes)
                    labels_t = labels_t.view(batch_size, num_classes)

                    loss_t = loss_fn(protein_pred_t, torch.argmax(labels_t, dim=1))
                    test_loss += loss_t

                    predicted_labels = torch.argmax(protein_pred_t, dim=1)
                    true_labels = torch.argmax(labels_t, dim=1)
                    correct_predictions += torch.sum(predicted_labels == true_labels).item()

                    S1 += "H" if true_labels[0].item() == 0 else ("E" if true_labels[0].item() == 1 else "C")
                    S2 += "H" if predicted_labels[0].item() == 0 else ("E" if predicted_labels[0].item() == 1 else "C")

                sov = calculate_sov(S1, S2)
                test_total_sov += sov
                test_total_batches += 1

                test_loss = test_loss / sequence_length
                test_loss_total += test_loss

            test_final_loss = test_loss_total / len(test_dataloader)
            print("第 {} 轮的测试LOSS： {} ".format(i + 1, test_final_loss))

            test_final_accuracy = correct_predictions / (len(test_dataloader) * batch_size * test_max_length)
            print("第 {} 轮的总测试ACC： {} ".format(i + 1, test_final_accuracy))
            test_average_sov = (test_total_sov / test_total_batches) / 150
            print("第 {} 轮的总测试SOV： {} ".format(i + 1, test_average_sov))


    print("--------------------------------------------------------------------------------------------------------")

if __name__ == '__main__':
    train(model, train_dataloader, loss_fn, optimizer)