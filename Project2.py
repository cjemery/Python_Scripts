import scipy
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pre
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

outputs=pd.read_csv('inputs.csv',names = ["column 1","column 2","column 3","column 4","column 5","column 6","column 7" ])
inputs=pd.read_csv('outputs.csv',names = ["column 1","column 2","column 3","column 4","column 5","column 6","column 7","column 8" ])
print "outputs",outputs.shape


outputs_tst=pd.read_csv('Outputs_tst.csv',names = ["column 1","column 2","column 3","column 4","column 5","column 6","column 7" ])
inputs_tst=pd.read_csv('Inputs_tst.csv',names = ["column 1","column 2","column 3","column 4","column 5","column 6","column 7","column 8" ])



x_tr=np.array(inputs,dtype='float')
y_tr=np.array(outputs,dtype='float')

x_te=np.array(inputs_tst,dtype='float')
y_te=np.array(outputs_tst,dtype='float')

print "training inputs shape:",x_tr.shape
print "training outputs shape:",y_tr.shape
print "testing inputs shape:",x_te.shape
print "testing outputs shape:",y_te.shape

def get_mse(): #find mse for best performing topology
	mlp = MLPRegressor(activation='tanh',hidden_layer_sizes=(100,),
							learning_rate='adaptive', max_iter=1000,
							solver='lbfgs',warm_start=True)
							
	print("MLP",mlp.fit(y_tr,x_tr))
	print "mlp training score:",mlp.score(y_tr,x_tr)
	print "mlp testing score:",mlp.score(y_te,x_te)
	y_test=mlp.predict(y_te)
	print "predict", y_test.shape
	print x_te.shape
	print "MSE with Activation function: tanh, Solver: lbfgs"
	print "mse=",mean_squared_error(y_test,x_te)

def get_mse_RF():	
	RF=RandomForestRegressor(n_estimators=100,warm_start=True)
	print "\nRandom Forest"
	print RF.fit(y_tr,x_tr)
	print "training:",RF.score(y_tr,x_tr)
	print "validation:",RF.score(y_te,x_te)
	y_test=RF.predict(y_te)
	print "MSE with n_estimators=100"
	print "mse=",mean_squared_error(y_test,x_te)	

#get_mse_RF()
#get_mse()
############## Training with outputs over inputs (inverse R^2) #########


### MLP Regressor ###
solver_list=['lbfgs','sgd','adam']

print"\nMLP"
mlpfile=open("mlp_data.txt",'a')
mlpfile.write("Outputs over Inputs\n")
mlpfile.write("\n\n Activation function: tanh\n")
for i in range(3):
	
	mlp = MLPRegressor(activation='tanh',hidden_layer_sizes=(100,),
						learning_rate='adaptive', max_iter=1000,
						solver=solver_list[i],warm_start=True)
						
	print("MLP",mlp.fit(y_tr,x_tr))
	print "mlp training score:",mlp.score(y_tr,x_tr)
	print "mlp testing score:",mlp.score(y_te,x_te)

	mlpfile.write("\n\nSolver: ")
	mlpfile.write(str(solver_list[i]))
	mlpfile.write("\ntraining:")
	mlpfile.write(str(mlp.score(y_tr,x_tr)))
	mlpfile.write("\ntesting:")
	mlpfile.write(str(mlp.score(y_te,x_te)))

mlpfile.write("\n\n Activation function: logistic\n")
for i in range(3):
	
	mlp = MLPRegressor(activation='logistic',hidden_layer_sizes=(100,),
						learning_rate='adaptive', max_iter=1000,
						solver=solver_list[i],warm_start=True)
						
	print("MLP",mlp.fit(y_tr,x_tr))
	print "mlp training score:",mlp.score(y_tr,x_tr)
	print "mlp testing score:",mlp.score(y_te,x_te)

	mlpfile.write("\n\nSolver: ")
	mlpfile.write(str(solver_list[i]))
	mlpfile.write("\ntraining:")
	mlpfile.write(str(mlp.score(y_tr,x_tr)))
	mlpfile.write("\ntesting:")
	mlpfile.write(str(mlp.score(y_te,x_te)))
	
mlpfile.write("\n\n Activation function: identity\n")
for i in range(3):
	
	mlp = MLPRegressor(activation='identity',hidden_layer_sizes=(100,),
						learning_rate='adaptive', max_iter=1000,
						solver=solver_list[i],warm_start=True)
						
	print("MLP",mlp.fit(y_tr,x_tr))
	print "mlp training score:",mlp.score(y_tr,x_tr)
	print "mlp testing score:",mlp.score(y_te,x_te)

	mlpfile.write("\n\nSolver: ")
	mlpfile.write(str(solver_list[i]))
	mlpfile.write("\ntraining:")
	mlpfile.write(str(mlp.score(y_tr,x_tr)))
	mlpfile.write("\ntesting:")
	mlpfile.write(str(mlp.score(y_te,x_te)))	
	
mlpfile.write("\n\n Activation function: relu\n")
for i in range(3):
	
	mlp = MLPRegressor(activation='relu',hidden_layer_sizes=(100,),
						learning_rate='adaptive', max_iter=1000,
						solver=solver_list[i],warm_start=True)
						
	print("MLP",mlp.fit(y_tr,x_tr))
	print "mlp training score:",mlp.score(y_tr,x_tr)
	print "mlp testing score:",mlp.score(y_te,x_te)

	mlpfile.write("\n\nSolver: ")
	mlpfile.write(str(solver_list[i]))
	mlpfile.write("\ntraining:")
	mlpfile.write(str(mlp.score(y_tr,x_tr)))
	mlpfile.write("\ntesting:")
	mlpfile.write(str(mlp.score(y_te,x_te)))	


### Random Forest Regressor ###
rffile=open("rf_data.txt",'a')
rffile.write("outputs over inputs\n")
for i in range(200,2200,200):
	print i
	RF=RandomForestRegressor(n_estimators=i,warm_start=True)
	print "\nRandom Forest"
	print RF.fit(y_tr,x_tr)
	print "training:",RF.score(y_tr,x_tr)
	print "validation:",RF.score(y_te,x_te)
	rffile.write("\n\n")
	rffile.write("n_estimators: ")
	rffile.write(str(i))
	rffile.write("\n")
	rffile.write("\ntraining:")
	rffile.write(str(RF.score(y_tr,x_tr)))
	rffile.write("\ntesting:")
	rffile.write(str(RF.score(y_te,x_te)))


	

############## Training with inputs over outputs ###################



print"\nMLP"
mlpfile=open("mlp_data.txt",'a')
mlpfile.write("Inputs over Outputs\n")
mlpfile.write("\n\n Activation function: tanh\n")
for i in range(3):
	
	mlp = MLPRegressor(activation='tanh',hidden_layer_sizes=(100,),
						learning_rate='adaptive', max_iter=1000,
						solver=solver_list[i],warm_start=True)
						
	print("MLP",mlp.fit(x_tr,y_tr))
	print "mlp training score:",mlp.score(x_tr,y_tr)
	print "mlp testing score:",mlp.score(x_te,y_te)

	mlpfile.write("\n\nSolver: ")
	mlpfile.write(str(solver_list[i]))
	mlpfile.write("\ntraining:")
	mlpfile.write(str(mlp.score(x_tr,y_tr)))
	mlpfile.write("\ntesting:")
	mlpfile.write(str(mlp.score(x_te,y_te)))

mlpfile.write("\n\n Activation function: logistic\n")
for i in range(3):
	
	mlp = MLPRegressor(activation='logistic',hidden_layer_sizes=(100,),
						learning_rate='adaptive', max_iter=1000,
						solver=solver_list[i],warm_start=True)
						
	print("MLP",mlp.fit(x_tr,y_tr))
	print "mlp training score:",mlp.score(x_tr,y_tr)
	print "mlp testing score:",mlp.score(x_te,y_te)

	mlpfile.write("\n\nSolver: ")
	mlpfile.write(str(solver_list[i]))
	mlpfile.write("\ntraining:")
	mlpfile.write(str(mlp.score(x_tr,y_tr)))
	mlpfile.write("\ntesting:")
	mlpfile.write(str(mlp.score(x_te,y_te)))
	
mlpfile.write("\n\n Activation function: identity\n")
for i in range(3):
	
	mlp = MLPRegressor(activation='identity',hidden_layer_sizes=(100,),
						learning_rate='adaptive', max_iter=1000,
						solver=solver_list[i],warm_start=True)
						
	print("MLP",mlp.fit(x_tr,y_tr))
	print "mlp training score:",mlp.score(x_tr,y_tr)
	print "mlp testing score:",mlp.score(x_te,y_te)

	mlpfile.write("\n\nSolver: ")
	mlpfile.write(str(solver_list[i]))
	mlpfile.write("\ntraining:")
	mlpfile.write(str(mlp.score(x_tr,y_tr)))
	mlpfile.write("\ntesting:")
	mlpfile.write(str(mlp.score(x_te,y_te)))	
	
mlpfile.write("\n\n Activation function: relu\n")
for i in range(3):
	
	mlp = MLPRegressor(activation='relu',hidden_layer_sizes=(100,),
						learning_rate='adaptive', max_iter=1000,
						solver=solver_list[i],warm_start=True)
						
	print("MLP",mlp.fit(x_tr,y_tr))
	print "mlp training score:",mlp.score(x_tr,y_tr)
	print "mlp testing score:",mlp.score(x_te,y_te)

	mlpfile.write("\n\nSolver: ")
	mlpfile.write(str(solver_list[i]))
	mlpfile.write("\ntraining:")
	mlpfile.write(str(mlp.score(x_tr,y_tr)))
	mlpfile.write("\ntesting:")
	mlpfile.write(str(mlp.score(x_te,y_te)))






### Random Forest ###

rffile=open("rf_data.txt",'a')
rffile.write("Inputs over outputs\n")
for i in range(200,2200,200):
	print i
	RF=RandomForestRegressor(n_estimators=i,warm_start=True)
	print "\nRandom Forest"
	print RF.fit(x_tr,y_tr)
	print "training:",RF.score(x_tr,y_tr)
	print "validation:",RF.score(x_te,y_te)
	rffile.write("\n\n")
	rffile.write("n_estimators: ")
	rffile.write(str(i))
	rffile.write("\n")
	rffile.write("\ntraining:")
	rffile.write(str(RF.score(x_tr,y_tr)))
	rffile.write("\ntesting:")
	rffile.write(str(RF.score(x_te,y_te)))

