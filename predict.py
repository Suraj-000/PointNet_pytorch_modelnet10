from pointnet import PointNet
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from dataset import ModelNet10Datset
from path import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



epochs=0
class PredictAndPlot:
    def __init__(self):
        path=Path("ModelNet10")
        valid_dataset = ModelNet10Datset(path, valid=True, folder='test')
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,drop_last=False,shuffle=True)
        self.classes = {i: cat for cat, i in valid_dataset.classes.items()}
        correct = total = 0
        pointnet = PointNet()
        pointnet.load_state_dict(torch.load("save_"+str(epochs)+".pth",map_location=torch.device('cpu')))
        pointnet.eval()
        self.predicted = []
        self.actual = []
        with torch.no_grad():
            for data in tqdm(valid_loader):
                inputs, labels = data['pointcloud'].float(), data['category']
                outputs,__,__ = pointnet(inputs.transpose(1,2))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                self.predicted += list(predicted.numpy())
                self.actual += list(labels.numpy())
        val_acc = 100. * correct / total
        print('Valid accuracy: %d %%' % val_acc)
        self.plot()

    def plot(self):
        cm = confusion_matrix(self.actual,self.predicted)
        sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=self.classes.values(),
            yticklabels=self.classes.values())
        plt.xlabel('Prediction',fontsize=13)
        plt.ylabel('Actual',fontsize=13)
        plt.title('Confusion Matrix',fontsize=17)
        plt.show()

# k=PredictAndPlot()
