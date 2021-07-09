# -*- coding: utf-8 -*-
"""+
Created on Tue Feb  5 20:58:55 2019

@author: saihe
"""
#importing packages
from tkinter import*
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier






#for the first gui window
me=Tk()
me.geometry("700x700")
me.title("Dataset accuracy finder")
melabel = Label(me,text="   dataset analyzer,upload file path  below and click upload button then you can check the below features",bg='White',font=("Times",20,'bold'))
melabel.pack(side=TOP)
me.config(background='navy')



textin=StringVar()
text1=StringVar()
c=""
a1=0.0
a2=0.0
a3=0.0
a4=0.0
k=0.3



# Importing the dataset
#c='C:/Users/saihe/Downloads/car2.csv'

#this function is for analysis part
def dataset():
  global b
  global c
  global k
  global l
  c=b.get()
  
  if(metext1.get()==''):
   k=0.3
  else:
    k=float(metext1.get())
	
  global f
  global z
  global a1
  global a2
  global a3
  global a4
  
  dataframe = pd.read_csv(c)
  X = dataframe.iloc[:,:-1 ]
  y = dataframe.iloc[:, -1]
#Encoding Categorical Data to Numerical Data

  for i in X.columns:
    if(X[i].dtype == 'object'):
        labelencoder_X = LabelEncoder()
        X[i] = labelencoder_X.fit_transform(X[i].astype(str))
  X.info()
  y.head()
#splitting the data set

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = k, random_state = 0)


#knn classification
#from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
#from sklearn.neighbors import KNeighborsClassifier
  classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
#from sklearn.model_selection import cross_val_score
  accuracies1=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
  print(accuracies1.mean())
  a1=accuracies1.mean()
  
#Decision Tree classification
#from sklearn.tree import DecisionTreeClassifier
  classifier = DecisionTreeClassifier()
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
#from sklearn.model_selection import cross_val_score
  accuracies2=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
  print(accuracies2.mean())
  a2=accuracies2.mean()


#naivebayes Classification
# Importing the dataset
  dataframe = pd.read_csv(c)
  X = dataframe.iloc[:,:-1 ]
  #X.iloc[0]=['vhigh','vhigh',2,4,'big','high']
  
  y = dataframe.iloc[:, -1]
#Encoding Categorical Data to Numerical Data
#from sklearn.preprocessing import LabelEncoder
  for i in X.columns:
    if(X[i].dtype == 'object'):
        labelencoder_X = LabelEncoder()
        X[i] = labelencoder_X.fit_transform(X[i].astype(str))
  #f=X.transform([['vhigh','med',2,4,'big','high']])
 # l=X.iloc[0]
  X.info()
  
  y.head()
#splitting the data set
#from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = k, random_state = 0)

#from sklearn.naive_bayes import BernoulliNB
  classifier = BernoulliNB()
  classifier.fit(X_train,y_train)
  y_pred=classifier.predict(X_test)
#from sklearn.model_selection import cross_val_score
  accuracies3=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
  print(accuracies3.mean())
  a3=accuracies3.mean()

#Random Forest classification
#from sklearn.ensemble import RandomForestClassifier
  classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
#from sklearn.model_selection import cross_val_score
  accuracies4=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
  print(accuracies4.mean())
  a4=accuracies4.mean()
  






# below functions are for showing results on the right  

def clrbut3():
 
 
  sentence="for the given dataset "+str(c)+" random forest accuracy is "+str(a4)+"with test size"+str(k)
  t.delete('0.0','end')
  t.insert(0.0,sentence)
  

def clrbut2():
  sentence="for the given dataset "+str(c)+"   decision tree accuracy is "+str(a3)+"with test size"+str(k)
  t.delete('0.0','end')
  t.insert(0.0,sentence)

 

def clrbut1():
  sentence="for the given dataset "+str(c)+"   naive bayes accuracy is "+str(a2)+"with test size"+str(k)
  t.delete('0.0','end')
  t.insert(0.0,sentence)

 

def clrbut0():
 
  sentence="for the given dataset "+str(c)+"   knn accuracy is "+str(a1)+"with test size"+str(k)
  t.delete('0.0','end')
  t.insert(0.0,sentence)





# below function is for finding best algorithm
 
def clrbut4():
 
 if((a1==a2)and(a1==a3)and(a1==a4)and(a1==0.0)):
  
  sentence="file is not uploaded"
  t.delete('0.0','end')
  t.insert(0.0,sentence)
 elif((a2>a1)and(a2>a3)and(a2>a4)):
  
  sentence="for the given dataset "+str(c)+" NAIVE BAYES classification algorithm has best accuracy among the 4 classification algorithms the accuracy by using this model is "+str(a2)
  t.delete('0.0','end')
  t.insert(0.0,sentence)
 elif((a1>a2)and(a1>a3)and(a1>a4)):
 # sentence="for the given dataset  knn classification algorithm has best accuracy among the 4 classification algorithms the accuracy by using this model is "+a1
  sentence="for the given dataset "+str(c)+" KNN classification algorithm has best accuracy among the 4 classification algorithms the accuracy by using this model is "+str(a1)
  t.delete('0.0','end')
  t.insert(0.0,sentence)
 elif((a3>a2)and(a3>a1)and(a3>a4)):
 # sentence="for the given dataset  decision classification algorithm has best accuracy among the 4 classification algorithms the accuracy by using this model is "+a3
  sentence="for the given dataset "+str(c)+" DECISION TREE classification algorithm has best accuracy among the 4 classification algorithms the accuracy by using this model is "+str(a3)
  t.delete('0.0','end')
  t.insert(0.0,sentence)
 elif((a4>a2)and(a4>a3)and(a4>a1)):
  

  sentence="for the given dataset "+str(c)+" random forest classification algorithm has best accuracy among the 4 classification algorithms the accuracy by using this model is "+str(a4)
  t.delete('0.0','end')
  t.insert(0.0,sentence)
  
  
 



 
  
  # below function is for comparision graph
 
def open_window():
    root = Tk()
    root.geometry('1200x700+200+100')
    root.title('This is my root window')
    root.state('zoomed')
    root.config(background='#fafafa')

    xar = [1,2,3,4]
    yar = [a1,a2,a3,a4]

    style.use('ggplot')
    fig = plt.figure(figsize=(14, 4.5), dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_ylim(0, 1)
    line, = ax1.plot(xar, yar, 'r', marker='o')
    #ser = serial.Serial('com3', 9600)
    

    def animate():
     line.set_data(xar, yar)
     ax1.set_xlim(0, 1)


    plotcanvas = FigureCanvasTkAgg(fig, root)
    plotcanvas.get_tk_widget().grid(column=1, row=1)
    ani = animation.FuncAnimation(fig, animate, interval=1000, blit=False)
    butequal=Button(root,padx=70,pady=14,bd=4,bg='white',command=root.destroy,text="BACK TO HOME PAGE  ",font=("Courier New",16,'bold'))
    butequal.place(x=15,y=650)
    lable3=Label(root,padx=5,pady=4,bd=4,text="    KNN       ",font=("Courier New",10,'bold'))
    lable3.place(x=180,y=410)
    lable3=Label(root,padx=5,pady=4,bd=4,text="Naive bayes   ",font=("Courier New",10,'bold'))
    lable3.place(x=500,y=410)
    lable3=Label(root,padx=5,pady=4,bd=4,text="Decision tree ",font=("Courier New",10,'bold'))
    lable3.place(x=850,y=410)
    lable3=Label(root,padx=5,pady=4,bd=4,text="Random Forest ",font=("Courier New",10,'bold'))
    lable3.place(x=1200,y=410)
	
    






# below function is front end of gui


def open_window2():
    me=Tk()
    me.geometry("1200x700+200+100")
    me.title("Dataset accuracy finder")
    me.state('zoomed')
    melabel = Label(me,text="prediction part",bg='White',font=("Times",20,'bold'))
    melabel.pack(side=TOP)
    me.config(background='navy')
    global b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,c1,c2,c3,c4,c5,c7,c6,c8,c9,c10,m,b11,b12,c11,c12,b13,c13
	
    
    
    lable2=Label(me,padx=5,pady=4,bd=4,text="below we give values for prediction",font=("Courier New",10,'bold'))
    lable2.place(x=0,y=0)

    lable3=Label(me,padx=5,pady=4,bd=4,text="1st value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=50)
    lable3=Label(me,padx=5,pady=4,bd=4,text="2nd value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=100)
    lable3=Label(me,padx=5,pady=4,bd=4,text="3rd value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=150)
    lable3=Label(me,padx=5,pady=4,bd=4,text="4th value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=200)
    lable3=Label(me,padx=5,pady=4,bd=4,text="5th value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=250)
    lable3=Label(me,padx=5,pady=4,bd=4,text="6th value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=300)
    lable3=Label(me,padx=5,pady=4,bd=4,text="7th value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=350)
    lable3=Label(me,padx=5,pady=4,bd=4,text="8th value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=400)
    lable3=Label(me,padx=5,pady=4,bd=4,text="9th value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=450)
    lable3=Label(me,padx=5,pady=4,bd=4,text="10th value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=500)
    lable3=Label(me,padx=5,pady=4,bd=4,text="11th value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=550)
    lable3=Label(me,padx=5,pady=4,bd=4,text="12th value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=600)
    lable3=Label(me,padx=5,pady=4,bd=4,text="13th value ",font=("Courier New",10,'bold'))
    lable3.place(x=0,y=650)


#metext.pack(side=LEFT)
    c1=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c1.place(x=100,y=50)
    c2=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c2.place(x=100,y=100)
    c3=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c3.place(x=100,y=150)
    c4=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c4.place(x=100,y=200)
    c5=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c5.place(x=100,y=250)
    c6=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c6.place(x=100,y=300)
    c7=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c7.place(x=100,y=350)
    c8=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c8.place(x=100,y=400)
    c9=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c9.place(x=100,y=450)
    c10=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c10.place(x=100,y=500)
    c11=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c11.place(x=100,y=550)
    c12=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c12.place(x=100,y=600)
    c13=Entry(me,width=30,bd=4,bg='white',font=("Courier New",12,'bold'))
    c13.place(x=100,y=650)	
    t1=Text(me,font=("Courier New",10,'bold'),width=110,bd=130,bg='cornflower blue')

    t1.pack(side=RIGHT)
    dataframe = pd.read_csv(c)
    f = dataframe
    
    sentence="dataset sample\n"+str(f)
    t1.delete('0.0','end')
    t1.insert(0.0,sentence)
   


    butequal=Button(me,padx=50,pady=14,bd=4,bg='white',command=me.destroy,text="BACK TO HOME PAGE  ",font=("Courier New",13,'bold'))
    butequal.place(x=0,y=720)
	
	
	
	# below function is for prediction
    def predict():
     dataframe = pd.read_csv(c)
     X = dataframe.iloc[:,:-1 ]
 
     
     if(len(X.columns)==2):
      b1=c1.get()
      b2=c2.get()
      m=[b1,b2]	  
     elif(len(X.columns)==3):
      b1=c1.get()
      b2=c2.get()
      b3=c3.get()
      m=[b1,b2,b3]
     elif(len(X.columns)==4):
      b1=c1.get()
      b2=c2.get()
      b3=c3.get()
      b4=c4.get()
      m=[b1,b2,b3,b4]	  
     elif(len(X.columns)==5):
      b1=c1.get()
      b2=c2.get()
      b3=c3.get()
      b4=c4.get()
      b5=c5.get()
      m=[b1,b2,b3,b4,b5]	  
     elif(len(X.columns)==6):
      b1=c1.get()
      b2=c2.get()
      b5=c5.get()
      b3=c3.get()
      b4=c4.get()
      b6=c6.get()
      m=[b1,b2,b3,b4,b5,b6]	  
     elif(len(X.columns)==7):
      b7=c7.get()
      b1=c1.get()
      b2=c2.get()
      b5=c5.get()
      b3=c3.get()
      b4=c4.get()
      b6=c6.get()
      m=[b1,b2,b3,b4,b5,b6,b7]      
     elif(len(X.columns)==8):
      b8=c8.get()
      b7=c7.get()
      b1=c1.get()
      b2=c2.get()
      b5=c5.get()
      b3=c3.get()
      b4=c4.get()
      b6=c6.get()
      m=[b1,b2,b3,b4,b5,b6,b7,b8]	  
     elif(len(X.columns)==9):
      b9=c9.get()
      b8=c8.get()
      b7=c7.get()
      b1=c1.get()
      b2=c2.get()
      b5=c5.get()
      b3=c3.get()
      b4=c4.get()
      b6=c6.get()
      m=[b1,b2,b3,b4,b5,b6,b7,b8,b9]
     elif(len(X.columns)==10):
      b10=c10.get()
      b9=c9.get()
      b8=c8.get()
      b7=c7.get()
      b1=c1.get()
      b2=c2.get()
      b5=c5.get()
      b3=c3.get()
      b4=c4.get()
      b6=c6.get()
      m=[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10]
     elif(len(X.columns)==11):
      b10=c10.get()
      b9=c9.get()
      b8=c8.get()
      b7=c7.get()
      b1=c1.get()
      b2=c2.get()
      b5=c5.get()
      b3=c3.get()
      b4=c4.get()
      b6=c6.get()
      b11=c11.get()
      m=[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11]
     elif(len(X.columns)==12):
      b10=c10.get()
      b9=c9.get()
      b8=c8.get()
      b7=c7.get()
      b1=c1.get()
      b2=c2.get()
      b5=c5.get()
      b3=c3.get()
      b4=c4.get()
      b6=c6.get()
      b12=c12.get()
      b11=c11.get()
      m=[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12]
     elif(len(X.columns)==13):
      b10=c10.get()
      b9=c9.get()
      b8=c8.get()
      b7=c7.get()
      b1=c1.get()
      b2=c2.get()
      b5=c5.get()
      b3=c3.get()
      b4=c4.get()
      b6=c6.get()
      b12=c12.get()
      b11=c11.get()
      b13=c13.get()	  
      m=[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13]	  
     X.iloc[0]=m
  
     y = dataframe.iloc[:, -1]
#Encoding Categorical Data to Numerical Data
#from sklearn.preprocessing import LabelEncoder
     for i in X.columns:
       if(X[i].dtype == 'object'):
        labelencoder_X = LabelEncoder()
        X[i] = labelencoder_X.fit_transform(X[i].astype(str))
  #f=X.transform([['vhigh','med',2,4,'big','high']])
     l=X.iloc[0]
     X.info()
  
     y.head()
#splitting the data set
#from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = k, random_state = 0)

     if((a1==a2)and(a1==a3)and(a1==a4)and(a1==0.0)):
  
      sentence="file is not uploaded"
      t1.delete('0.0','end')
      t1.insert(0.0,sentence)
     elif((a2>a1)and(a2>a3)and(a2>a4)):
      sc = StandardScaler()
      X_train = sc.fit_transform(X_train)
      X_test = sc.transform(X_test)
      classifier = DecisionTreeClassifier()
      classifier.fit(X_train, y_train)
      y_pred = classifier.predict(X_test)
      f=classifier.predict([l])

      sentence="the prediction is"+str(f)
      t1.delete('0.0','end')
      t1.insert(0.0,sentence)
     elif((a1>a2)and(a1>a3)and(a1>a4)):
 # sentence="for the given dataset  knn classification algorithm has best accuracy among the 4 classification algorithms the accuracy by using this model is "+a1
      sc = StandardScaler()
      X_train = sc.fit_transform(X_train)
      X_test = sc.transform(X_test)
#from sklearn.neighbors import KNeighborsClassifier
      classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
      classifier.fit(X_train, y_train)
      y_pred = classifier.predict(X_test)
      f=classifier.predict([l])

      sentence="the prediction is"+str(f)
      t1.delete('0.0','end')
      t1.insert(0.0,sentence)
     elif((a3>a2)and(a3>a1)and(a3>a4)):
      classifier = BernoulliNB()
      classifier.fit(X_train,y_train)
      y_pred=classifier.predict(X_test)

      f=classifier.predict([l])

      sentence="the prediction is"+str(f)
      t1.delete('0.0','end')
      t1.insert(0.0,sentence)
     elif((a4>a2)and(a4>a3)and(a4>a1)):
     
      classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
      classifier.fit(X_train, y_train)
      y_pred = classifier.predict(X_test)
      f=classifier.predict([l])

      sentence="the prediction is"+str(f)
      t1.delete('0.0','end')
      t1.insert(0.0,sentence)
    butequal=Button(me,padx=10,pady=10,bd=4,bg='white',command=predict,text="predict              ",font=("Courier New",10,'bold'))
    butequal.place(x=5,y=680)
 



	
	
	

    
# code below is for few design part


lable1=Label(me,padx=65,pady=5,bd=4,text="dataset file path",font=("Courier New",16,'bold'))
lable1.place(x=4,y=40)


b = Entry(me,font=("Courier New",20,'bold'),width=50,bd=5)
b.pack()
b.focus_set()

lable2=Label(me,padx=5,pady=4,bd=4,text="below we can change testing %(eg:0.5)",font=("Courier New",10,'bold'))
lable2.place(x=0,y=150)

lable3=Label(me,padx=5,pady=4,bd=4,text="testing   ",font=("Courier New",10,'bold'))
lable3.place(x=0,y=200)



#metext.pack(side=LEFT)
metext1=Entry(me,width=30,bd=4,bg='white',textvar=textin,font=("Courier New",12,'bold'))
metext1.place(x=100,y=200)

#sentence="conclusion after studying data:              for huge datasets      /n    for less datasets"
                                           #   numeric   random forest               random forest
											#  nominal                               random forest
											#  mixed     naive bayes                 random forest    "
#t.delete('0.0','end')
#t.insert(0.0,sentence)



t=Text(me,font=("Courier New",20,'bold'),width=60,bd=130,bg='cornflower blue')

t.pack(side=RIGHT)

sentence="best algorithms 								  data   :for huge datasets   for less datasets             numeric    random forest       randomforest                 nominal                        randomforest                 mixed      naive bayes         randomforest    "
                                           #   numeric   random forest               random forest
											#  nominal                               random forest
											#  mixed     naive bayes                 random forest    "
t.delete('0.0','end')
t.insert(0.0,sentence)




butequal=Button(me,padx=40,pady=10,bd=4,bg='white',command=open_window2,text="prediction           ",font=("Courier New",12,'bold'))
butequal.place(x=5,y=250)


butequal=Button(me,padx=40,pady=10,bd=4,bg='white',command=dataset,text="upload file          ",font=("Courier New",12,'bold'))
butequal.place(x=5,y=90)

butequal=Button(me,padx=40,pady=10,bd=4,bg='white',command=clrbut0,text="Knn algorithm        ",font=("Courier New",12,'bold'))
butequal.place(x=5,y=325)

butequal=Button(me,padx=40,pady=10,bd=4,bg='white',command=clrbut1,text="naivebayes           ",font=("Courier New",12,'bold'))
butequal.place(x=5,y=400)

butequal=Button(me,padx=40,pady=10,bd=4,bg='white',command=clrbut2,text="DecisionTree         ",font=("Courier New",12,'bold'))
butequal.place(x=5,y=475)

butequal=Button(me,padx=40,pady=10,bd=4,bg='white',command=clrbut3,text="RandomForest         ",font=("Courier New",12,'bold'))
butequal.place(x=5,y=550)


butequal=Button(me,padx=40,pady=10,bd=4,bg='white',command=clrbut4,text="performance analysis ",font=("Courier New",12,'bold'))
butequal.place(x=5,y=625)

butequal=Button(me,padx=40,pady=10,bd=4,bg='white',command=open_window,text="comparision graph    ",font=("Courier New",12,'bold'))
butequal.place(x=5,y=700)

me.mainloop()