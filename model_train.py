# -*- coding: utf-8 -*-
# @Time : 2023/3/16 14:42
# @Author : Jclian91
# @File : model_train.py
# @Place : Minghang, Shanghai
import torch
from torch.optim import RMSprop, Adam
from torch.nn import CrossEntropyLoss, Softmax
from torch.utils.data import DataLoader
from numpy import vstack, argmax
from sklearn.metrics import accuracy_score

from model import TextClassifier
from text_featuring import CSVDataset
from params import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, LEARNING_RATE, EPOCHS, TRAIN_FILE_PATH, TEST_FILE_PATH


# model train
class ModelTrainer(object):
    # evaluate the model
    @staticmethod
    def evaluate_model(test_dl, model):
        predictions, actuals = [], []
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yhat = model(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            # convert to class labels
            yhat = argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return acc

    # Model Training, evaluation and metrics calculation
    def train(self, model):
        # calculate split
        train, test = CSVDataset(TRAIN_FILE_PATH), CSVDataset(TEST_FILE_PATH)
        # prepare data loaders
        train_dl = DataLoader(train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_dl = DataLoader(test, batch_size=TEST_BATCH_SIZE)

        # Define optimizer
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        # Starts training phase
        for epoch in range(EPOCHS):
            # Starts batch training
            for x_batch, y_batch in train_dl:
                y_batch = y_batch.long()
                # Clean gradients
                optimizer.zero_grad()
                # Feed the model
                y_pred = model(x_batch)
                # Loss calculation
                loss = CrossEntropyLoss()(y_pred, y_batch)
                # Gradients calculation
                loss.backward()
                # Gradients update
                optimizer.step()

            # Evaluation
            test_accuracy = self.evaluate_model(test_dl, model)
            print("Epoch: %d, loss: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), test_accuracy))


if __name__ == '__main__':
    model = TextClassifier(nhead=10,             # number of heads in the multi-head-attention models
                           dim_feedforward=128,  # dimension of the feedforward network model in nn.TransformerEncoder
                           num_layers=1,
                           dropout=0.0,
                           classifier_dropout=0.0)
    # 统计参数量
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    ModelTrainer().train(model)
    torch.save(model, 'sougou_mini_cls.pth')
