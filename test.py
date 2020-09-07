from tkinter import *

root = Tk()



def LiRf():
	import pandas as pd
	import numpy as np
	import seaborn as sns
	from datetime import datetime
	from sklearn.linear_model import LinearRegression
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.ensemble import ExtraTreesRegressor
	from xgboost import XGBClassifier
	from sklearn.svm import LinearSVC
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import r2_score

	data1 =pd.read_excel("littoralis1516.xlsx")

	data1.head()

	data1.drop(["Date","Larves"],axis=1,inplace=True)

	data1.head()

	data1=data1.astype(float)

	data1["GDD"]=data1.Temp.astype(float)-10

	data1.head()

	x=data1.iloc[:,0].values

	y=data1.iloc[:,1:8].values

	x

	y

	xtrain,xtest,ytrain,ytest=train_test_split(y,x,test_size=0.2,random_state=0)

	regressor=LinearRegression()
	#regressor=RandomForestRegressor(n_estimators=10,random_state=0,max_depth=20)                                     #max depth=10
	#regressor=ExtraTreesRegressor(n_estimators=100,random_state=0,max_depth=10,min_samples_split=5)                  #max depth=5
	#regressor=XGBClassifier()
	#regressor=LinearSVC()
	#regressor = LogisticRegression()

	regressor.fit(xtrain,ytrain)

	y_pred=regressor.predict(xtest)

	data1_cmp=pd.DataFrame(list(zip(y_pred, ytest)))

	data1_cmp['Difference']=abs(data1_cmp[0]-data1_cmp[1])

	data1_cmp.rename(columns={0:"Predicted",1:"Actual"},inplace=True)

	data1_cmp.head()

	MAPE = data1_cmp['Difference'].mean()
	x000=float("{:.5f}".format(MAPE))
	print("MAPE: %.5f" % (MAPE))
	Error = np.mean(data1_cmp["Difference"])/np.mean(data1_cmp["Actual"])
	x11=Error*100
	x111=float("{:.2f}".format(x11))
	print("Error: %.2f%%" % (Error*100))

	Accuracy = accuracy_score((ytest*100).astype(int), (y_pred*100).astype(int))
	#Accuracy = r2_score(ytest,y_pred)

	print("Accuracy: %.2f%%" % (Accuracy * 100.0))
	x22=Accuracy*100
	x222=float("{:.2f}".format(x22))
	#plt.plot(data1_cmp.Actual, color="r")
	#plt.plot(data1_cmp.Predicted, color ="b")
	global Label11
	Label11=Label(root,text="MAPE=")
	global Label12
	Label12=Label(root, text=x000)

	global Label21
	Label21=Label(root,text="Error=")
	global Label22
	Label22=Label(root,text=x111)

	global Label31
	Label31=Label(root,text="Accuracy=")
	global Label32
	Label32=Label(root,text=x222)

	Label11.grid(row=10, column=5)
	Label12.grid(row=10, column=6)

	Label21.grid(row=11, column=5)
	Label22.grid(row=11, column=6)

	Label31.grid(row=12, column=5)
	Label32.grid(row=12, column=6)
	LiR['state']=DISABLED
def LoRf():

	import pandas as pd
	import numpy as np
	import seaborn as sns
	from datetime import datetime
	from sklearn.linear_model import LinearRegression
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.ensemble import ExtraTreesRegressor
	from xgboost import XGBClassifier
	from sklearn.svm import LinearSVC
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import r2_score

	data1 =pd.read_excel("littoralis1516.xlsx")

	data1.head()

	data1.drop(["Date","Larves"],axis=1,inplace=True)

	data1.head()

	data1=data1.astype(float)

	data1["GDD"]=data1.Temp.astype(float)-10

	data1.head()

	x=data1.iloc[:,0].values

	y=data1.iloc[:,1:8].values

	x

	y

	xtrain,xtest,ytrain,ytest=train_test_split(y,x,test_size=0.2,random_state=0)

	#regressor=LinearRegression()
	#regressor=RandomForestRegressor(n_estimators=10,random_state=0,max_depth=20)                                     #max depth=10
	#regressor=ExtraTreesRegressor(n_estimators=100,random_state=0,max_depth=10,min_samples_split=5)                  #max depth=5
	#regressor=XGBClassifier()
	#regressor=LinearSVC()
	regressor = LogisticRegression()

	regressor.fit(xtrain,ytrain)

	y_pred=regressor.predict(xtest)

	data1_cmp=pd.DataFrame(list(zip(y_pred, ytest)))

	data1_cmp['Difference']=abs(data1_cmp[0]-data1_cmp[1])

	data1_cmp.rename(columns={0:"Predicted",1:"Actual"},inplace=True)

	data1_cmp.head()

	MAPE = data1_cmp['Difference'].mean()
	x000=float("{:.5f}".format(MAPE))
	print("MAPE: %.5f" % (MAPE))
	Error = np.mean(data1_cmp["Difference"])/np.mean(data1_cmp["Actual"])
	x11=Error*100
	x111=float("{:.2f}".format(x11))
	print("Error: %.2f%%" % (Error*100))

	Accuracy = accuracy_score((ytest*100).astype(int), (y_pred*100).astype(int))
	#Accuracy = r2_score(ytest,y_pred)

	print("Accuracy: %.2f%%" % (Accuracy * 100.0))
	x22=Accuracy*100
	x222=float("{:.2f}".format(x22))
	#plt.plot(data1_cmp.Actual, color="r")
	#plt.plot(data1_cmp.Predicted, color ="b")
	global Label11
	Label11=Label(root,text="MAPE=")
	global Label12
	Label12=Label(root, text=x000)

	global Label21
	Label21=Label(root,text="Error=")
	global Label22
	Label22=Label(root,text=x111)

	global Label31
	Label31=Label(root,text="Accuracy=")
	global Label32
	Label32=Label(root,text=x222)

	Label11.grid(row=10, column=5)
	Label12.grid(row=10, column=6)

	Label21.grid(row=11, column=5)
	Label22.grid(row=11, column=6)

	Label31.grid(row=12, column=5)
	Label32.grid(row=12, column=6)
	LoR['state']=DISABLED
def SVMf():

	import pandas as pd
	import numpy as np
	import seaborn as sns
	from datetime import datetime
	from sklearn.linear_model import LinearRegression
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.ensemble import ExtraTreesRegressor
	from xgboost import XGBClassifier
	from sklearn.svm import LinearSVC
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import r2_score

	data1 =pd.read_excel("littoralis1516.xlsx")

	data1.head()

	data1.drop(["Date","Larves"],axis=1,inplace=True)

	data1.head()

	data1=data1.astype(float)

	data1["GDD"]=data1.Temp.astype(float)-10

	data1.head()

	x=data1.iloc[:,0].values

	y=data1.iloc[:,1:8].values

	x

	y

	xtrain,xtest,ytrain,ytest=train_test_split(y,x,test_size=0.2,random_state=0)

	#regressor=LinearRegression()
	#regressor=RandomForestRegressor(n_estimators=10,random_state=0,max_depth=20)                                     #max depth=10
	#regressor=ExtraTreesRegressor(n_estimators=100,random_state=0,max_depth=10,min_samples_split=5)                  #max depth=5
	#regressor=XGBClassifier()
	regressor=LinearSVC()
	#regressor = LogisticRegression()

	regressor.fit(xtrain,ytrain)

	y_pred=regressor.predict(xtest)

	data1_cmp=pd.DataFrame(list(zip(y_pred, ytest)))

	data1_cmp['Difference']=abs(data1_cmp[0]-data1_cmp[1])

	data1_cmp.rename(columns={0:"Predicted",1:"Actual"},inplace=True)

	data1_cmp.head()

	MAPE = data1_cmp['Difference'].mean()
	x000=float("{:.5f}".format(MAPE))
	print("MAPE: %.5f" % (MAPE))
	Error = np.mean(data1_cmp["Difference"])/np.mean(data1_cmp["Actual"])
	x11=Error*100
	x111=float("{:.2f}".format(x11))
	print("Error: %.2f%%" % (Error*100))

	Accuracy = accuracy_score((ytest*100).astype(int), (y_pred*100).astype(int))
	#Accuracy = r2_score(ytest,y_pred)

	print("Accuracy: %.2f%%" % (Accuracy * 100.0))
	x22=Accuracy*100
	x222=float("{:.2f}".format(x22))
	#plt.plot(data1_cmp.Actual, color="r")
	#plt.plot(data1_cmp.Predicted, color ="b")
	global Label11
	Label11=Label(root,text="MAPE=")
	global Label12
	Label12=Label(root, text=x000)

	global Label21
	Label21=Label(root,text="Error=")
	global Label22
	Label22=Label(root,text=x111)

	global Label31
	Label31=Label(root,text="Accuracy=")
	global Label32
	Label32=Label(root,text=x222)

	Label11.grid(row=10, column=5)
	Label12.grid(row=10, column=6)

	Label21.grid(row=11, column=5)
	Label22.grid(row=11, column=6)

	Label31.grid(row=12, column=5)
	Label32.grid(row=12, column=6)
	SVM['state']=DISABLED
def ETCf():
	import pandas as pd
	import numpy as np
	import seaborn as sns
	from datetime import datetime
	from sklearn.linear_model import LinearRegression
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.ensemble import ExtraTreesRegressor
	from xgboost import XGBClassifier
	from sklearn.svm import LinearSVC
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import r2_score

	data1 =pd.read_excel("littoralis1516.xlsx")

	data1.head()

	data1.drop(["Date","Larves"],axis=1,inplace=True)

	data1.head()

	data1=data1.astype(float)

	data1["GDD"]=data1.Temp.astype(float)-10

	data1.head()

	x=data1.iloc[:,0].values

	y=data1.iloc[:,1:8].values

	x

	y

	xtrain,xtest,ytrain,ytest=train_test_split(y,x,test_size=0.2,random_state=0)

	#regressor=LinearRegression()
	#regressor=RandomForestRegressor(n_estimators=10,random_state=0,max_depth=20)                                     #max depth=10
	regressor=ExtraTreesRegressor(n_estimators=100,random_state=0,max_depth=10,min_samples_split=5)                  #max depth=5
	#regressor=XGBClassifier()
	#regressor=LinearSVC()
	#regressor = LogisticRegression()

	regressor.fit(xtrain,ytrain)

	y_pred=regressor.predict(xtest)

	data1_cmp=pd.DataFrame(list(zip(y_pred, ytest)))

	data1_cmp['Difference']=abs(data1_cmp[0]-data1_cmp[1])

	data1_cmp.rename(columns={0:"Predicted",1:"Actual"},inplace=True)

	data1_cmp.head()

	MAPE = data1_cmp['Difference'].mean()
	x000=float("{:.5f}".format(MAPE))
	print("MAPE: %.5f" % (MAPE))
	Error = np.mean(data1_cmp["Difference"])/np.mean(data1_cmp["Actual"])
	x11=Error*100
	x111=float("{:.2f}".format(x11))
	print("Error: %.2f%%" % (Error*100))

	Accuracy = accuracy_score((ytest*100).astype(int), (y_pred*100).astype(int))
	#Accuracy = r2_score(ytest,y_pred)

	print("Accuracy: %.2f%%" % (Accuracy * 100.0))
	x22=Accuracy*100
	x222=float("{:.2f}".format(x22))
	#plt.plot(data1_cmp.Actual, color="r")
	#plt.plot(data1_cmp.Predicted, color ="b")
	global Label11
	Label11=Label(root,text="MAPE=")
	global Label12
	Label12=Label(root, text=x000)

	global Label21
	Label21=Label(root,text="Error=")
	global Label22
	Label22=Label(root,text=x111)

	global Label31
	Label31=Label(root,text="Accuracy=")
	global Label32
	Label32=Label(root,text=x222)

	Label11.grid(row=10, column=5)
	Label12.grid(row=10, column=6)

	Label21.grid(row=11, column=5)
	Label22.grid(row=11, column=6)

	Label31.grid(row=12, column=5)
	Label32.grid(row=12, column=6)
	ETC['state']=DISABLED

def RFf():
	import pandas as pd
	import numpy as np
	import seaborn as sns
	from datetime import datetime
	from sklearn.linear_model import LinearRegression
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.ensemble import ExtraTreesRegressor
	from xgboost import XGBClassifier
	from sklearn.svm import LinearSVC
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import r2_score

	data1 =pd.read_excel("littoralis1516.xlsx")

	data1.head()

	data1.drop(["Date","Larves"],axis=1,inplace=True)

	data1.head()

	data1=data1.astype(float)

	data1["GDD"]=data1.Temp.astype(float)-10

	data1.head()

	x=data1.iloc[:,0].values

	y=data1.iloc[:,1:8].values

	x

	y

	xtrain,xtest,ytrain,ytest=train_test_split(y,x,test_size=0.2,random_state=0)

	#regressor=LinearRegression()
	regressor=RandomForestRegressor(n_estimators=10,random_state=0,max_depth=20)                                     #max depth=10
	#regressor=ExtraTreesRegressor(n_estimators=100,random_state=0,max_depth=10,min_samples_split=5)                  #max depth=5
	#regressor=XGBClassifier()
	#regressor=LinearSVC()
	#regressor = LogisticRegression()

	regressor.fit(xtrain,ytrain)

	y_pred=regressor.predict(xtest)

	data1_cmp=pd.DataFrame(list(zip(y_pred, ytest)))

	data1_cmp['Difference']=abs(data1_cmp[0]-data1_cmp[1])

	data1_cmp.rename(columns={0:"Predicted",1:"Actual"},inplace=True)

	data1_cmp.head()

	MAPE = data1_cmp['Difference'].mean()
	x000=float("{:.5f}".format(MAPE))
	print("MAPE: %.5f" % (MAPE))
	Error = np.mean(data1_cmp["Difference"])/np.mean(data1_cmp["Actual"])
	x11=Error*100
	x111=float("{:.2f}".format(x11))
	print("Error: %.2f%%" % (Error*100))

	Accuracy = accuracy_score((ytest*100).astype(int), (y_pred*100).astype(int))
	#Accuracy = r2_score(ytest,y_pred)

	print("Accuracy: %.2f%%" % (Accuracy * 100.0))
	x22=Accuracy*100
	x222=float("{:.2f}".format(x22))
	#plt.plot(data1_cmp.Actual, color="r")
	#plt.plot(data1_cmp.Predicted, color ="b")
	global Label11
	Label11=Label(root,text="MAPE=")
	global Label12
	Label12=Label(root, text=x000)

	global Label21
	Label21=Label(root,text="Error=")
	global Label22
	Label22=Label(root,text=x111)

	global Label31
	Label31=Label(root,text="Accuracy=")
	global Label32
	Label32=Label(root,text=x222)

	Label11.grid(row=10, column=5)
	Label12.grid(row=10, column=6)

	Label21.grid(row=11, column=5)
	Label22.grid(row=11, column=6)

	Label31.grid(row=12, column=5)
	Label32.grid(row=12, column=6)
	RF['state']=DISABLED

##########################
def myDelete():
	Label11.grid_forget()
	Label12.grid_forget()
	Label21.grid_forget()
	Label22.grid_forget()
	Label31.grid_forget()
	Label32.grid_forget()
	LiR['state']=NORMAL
	LoR['state']=NORMAL
	SVM['state']=NORMAL
	ETC['state']=NORMAL
	RF['state']=NORMAL
	#print(LiR.winfo_exists())
root.title("Machine Learning Algorithm Simulator")
#e = Entry(root)
#e.pack()
#e.get
#creating label widget
myLabel = Label(root, text="Algorithm Simulator")
Label2 = Label(root, text="Please choose the required algorithm")
#myLabel2 = Label(root, text="wehoo")
#shoving it onto screen
myLabel.grid(row=1, column=1)
Label2.grid(row=2, column =1)
#myLabel2.grid(row=2, column=0)
#make it into a loop
#def myClick():
#	labell = Label(root, text="look it's a button")
#	labell.pack()

LiR = Button(root, text="Linear Regression", padx=25,pady=20, fg="blue", command=LiRf)
LoR = Button(root, text="Logestic Regression", padx=25, pady=20, fg="green", command=LoRf)
SVM = Button(root, text="Support Vector Machine", padx=25, pady=20, fg="red", command=SVMf)
ETC = Button(root, text= "Extra Trees Classifyer", padx=25, pady=20, fg="black", command=ETCf)
RF = Button(root, text = "Random Forests", padx=25, pady=20,fg="purple", command = RFf)
DeleteButton = Button(root, text="Delete Text", command=myDelete)
LiR.grid(row = 3, column=1 )
LoR.grid(row = 4,column=1)
SVM.grid(row=5, column=1)
ETC.grid(row=6,column=1)
RF.grid(row=7,column=1)
DeleteButton.grid(row=14,column=5)
root.mainloop()