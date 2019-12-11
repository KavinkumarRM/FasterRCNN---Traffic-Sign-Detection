
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score
import matplotlib.pyplot as plt

prediction = pd.read_csv('../prediction.csv')
annotation=pd.read_csv('../annotation_resampled.txt',sep=',',names=["path","UCX","UCY","LCX","LCY","Annotation"])
path='training_images/'+prediction['Image_path'][6]
true_predict=list(annotation['Annotation'][annotation['path']==path])
count=0

true_pred_list=[]
predicted_list=[]

for i in range(len(prediction)):
    predicted_ann=prediction['prediction'][i].split("'")
    try:
        if predicted_ann[1]:
            predicted_list.append(1)
    except:
        predicted_list.append(0)
    path='training_images/'+prediction['Image_path'][i]
    try:
        true_predict=list(annotation['Annotation'][annotation['path']==path])[0]
    except:
        true_predict=None
    if true_predict:
        true_pred_list.append(1)
    else:
        true_pred_list.append(0)
    try:
        if true_predict:
            index=predicted_ann.index(true_predict)
            count=count+1
        elif prediction['prediction'][i]=='[]':
            count=count+1
    except:
        index=0
        
print("precision",precision_score(true_pred_list,predicted_list))
print("recall ",recall_score(true_pred_list,predicted_list))
print("f1 score",f1_score(true_pred_list,predicted_list))
print("accuracy",count/len(prediction))

with open('../History.txt',encoding='utf-8') as text:
    data=text.read()

splited=data.split('\n')
history={}
k=0
for i in splited:
    if i.startswith('Mean') or i.startswith('Classifier') or i.startswith('Loss') or i.startswith('Elapsed'):
        j=i.split(':')
        if j[0] not in history.keys():
            history[j[0]]=[float(j[1].replace(" ",""))]
        else:
            history[j[0]].append(float(j[1].replace(" ","")))
    else:
        if i.startswith('Epoch'):
            k=k+1
            
graph=pd.DataFrame.from_dict(history)

graph['Mean number of bounding boxes from RPN overlapping ground truth boxes'].plot(kind='line')
plt.title('Mean number of bounding boxes from RPN overlapping ground truth boxes')

graph['Classifier accuracy for bounding boxes from RPN'].plot(kind='line')
plt.title('Classifier accuracy for bounding boxes from RPN')

graph['Loss RPN classifier'].plot(kind='line')
plt.title('Loss RPN classifier')

graph['Loss RPN regression'].plot(kind='line')
plt.title('Loss RPN regression')

graph['Loss Detector classifier'].plot(kind='line')
plt.title('Loss Detector classifier')

graph['Loss Detector regression'].plot(kind='line')
plt.title('Loss Detector regression')

graph['Elapsed time'].plot(kind='line')
plt.title('Elapsed time')

