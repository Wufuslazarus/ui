from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()

print (iris.feature_names)
print (iris.target_names)
print (iris.target)
print (iris.data)

X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 10)
print("Eğitim veri seti boyutu=",len(X_train))
print("Test veri seti boyutu=",len(X_test))

model = DecisionTreeClassifier() 
model.fit(X_train, Y_train)



Y_tahmin = model.predict(X_test)
hata_matrisi = confusion_matrix(Y_test, Y_tahmin)
print(hata_matrisi)

index = ['setosa','versicolor','virginica']
columns = ['setosa','versicolor','virginica']
hata_goster = pd.DataFrame(hata_matrisi,columns,index)
plt.figure(figsize=(10,6))
sns.heatmap(hata_goster, annot=True) 
plt.show()