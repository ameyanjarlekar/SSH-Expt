# import numpy as np 
# import math
# #import np.random
# sigma = 1.0
# alpha1 = 0.05
# alpha2 = 0.02
# sample = [-3,-2,-1,0,1,2,3]
# epoch1 = 20000
# k = 0.079789
# epoch2 = 50000
# hyp1 = 0.01
# #n1 = 2
# integral = 0
# def factorial(n):
# 	if type(n) != int or n < 0:
# 		return -10000000
# 	else:
# 		if (n==0):
# 			return 1
# 		else:
# 			return n*factorial(n-1)
# def poissons (a,lambd):
# 	i = 0
# 	b = np.zeros((len(a),2))
# 	while i< len(a):
# 		b[i][0] = a[i]
# 		b[i][1] = math.exp(-lambd)*((lambd)**a[i])/factorial(int(a[i]))
# 		i = i+1
# 	return b

# para = 10.0
# sampletrain = 50
# c = np.random.poisson(para, sampletrain)
# c = c.astype(float)
# #train = poisson(c,sigma)

# meana1 = np.mean(c)
# #print(mean1)
# meana2 = np.mean(np.power(c,2))
# #print(mean2)
# meana3 = np.mean(np.power(c,3))
# #print(mean3)
# vara1 = np.std(c)
# #print(var1)
# vara2 = np.std(np.power(c,2))
# #print(var2)
# vara3 = np.std(np.power(c,3))
# train2 = np.zeros((sampletrain,4))
# i = 0
# while i < sampletrain:
# 	train2[i][0] = 1.0
# 	train2[i][1] = (c[i])/vara1
# 	train2[i][2] = (c[i]**2)/vara2
# 	train2[i][3] = para
# 	i = i+1

# space = sampletrain
# N = 25
# integratept = 30
# i = 0
# integraterin =  np.zeros((integratept,1))
# integraterout =  np.zeros((integratept,1))
# while i <integratept:
# 	integraterin[i] = 3*i
# 	m = 0
# 	track = 0
# 	while m < sampletrain:
# 		track = track + math.exp(((c[m]-integraterin[i])**2)/((-2.0)*(sigma**2)))
# 		m = m+1
# 	integraterout[i] = track*k
# 	i = i+1

# w1 = np.random.normal(0.0,1.0,(4,6))
# w2 = np.random.normal(0.0,1.0,(6,1))
# train1 = np.zeros((integratept,4))
# i = 0
# j = 0

# # copy = np.zeros((26,1))
# # i = 0
# # while i <26:
# # 	copy[i] = integrater[i][0]
# # 	i = i+1

# mean1 = np.mean(integraterin)
# #print(mean1)
# mean2 = np.mean(np.power(integraterin,2))
# #print(mean2)
# mean3 = np.mean(np.power(integraterin,3))
# #print(mean3)
# var1 = np.std(integraterin)
# #print(var1)
# var2 = np.std(np.power(integraterin,2))
# #print(var2)
# var3 = np.std(np.power(integraterin,3))
# #print(var3)
# i = 0
# while i < integratept:
# 	train1[i][0] = 1.0
# 	train1[i][1] = (integraterin[i])/var1
# 	train1[i][2] = (integraterin[i]**2)/var2
# 	train1[i][3] = para
# 	i = i+1
# counter = 0
# # train1 = np.array([[0,0,0],[0,0,1],[0,1,0],[1,0,1],[1,1,1],[0,1,1],[1,1,0]])
# # integraterout = np.array([[0],[1],[1],[0],[1],[0],[0]])

# while counter < epoch1:
# 	stage1 = np.matmul(train1,w1)
# 	stage2 = 1.0/(1.0+np.exp(-stage1))
# 	outputa = np.matmul(stage2,w2)
# 	#print(outputa)

# 	# cost1 = 0

# 	# i = 0
# 	# while i < 7:
# 	# 	cost1 = cost1 + (outputa[i]-integraterout[i])**2
# 	# 	i = i+1
# 	#print(cost1)

# 	dw2 = 2.0*(np.matmul((outputa-integraterout).T,stage2)).T
# 	#print(np.shape(dw2))
# 	#print(dw2)
# 	dw1 = 2.0*(np.matmul(train1.T,(np.matmul((outputa-integraterout),w2.T)*stage2*(1.0-stage2))))
# 	#print(dw1)
# 	#print(np.shape(dw1))

# 	w1 = w1 - alpha1*dw1
# 	w2 = w2 - alpha1*dw2
# 	counter = counter +1
# 	#print(cost1)

# # test1 = np.array([[1,0,0],[1,1,0],[0,1,1],[1,1,1],[0,0,1],[1,1,0]])
# # stag1 = np.matmul(test1,w1)
# # stag2 = 1/(1+np.exp(-stag1))
# # outputy = np.matmul(stag2,w2)
# # print(outputy)
# #print(outputa)
# #print(integraterout)

# put = 0
# while put < epoch2:
# 	bit = np.random.choice(sample,p = [0.05,0.05,0.1,0.1,0.3,0.2,0.2])
# 	if ((bit + integraterin[0]) > 0 ):
# 		integraterin = integraterin + bit
# 	i = 0
# 	mean1 = np.mean(integraterin)
# 	#print(mean1)
# 	mean2 = np.mean(np.power(integraterin,2))
# 	#print(mean2)
# 	mean3 = np.mean(np.power(integraterin,3))
# 	#print(mean3)
# 	var1 = np.std(integraterin)
# 	#print(var1)
# 	var2 = np.std(np.power(integraterin,2))
# 	#print(var2)
# 	var3 = np.std(np.power(integraterin,3))
# 	while i < integratept:
# 		train1[i][0] = 1.0
# 		train1[i][1] = (integraterin[i])/var1
# 		train1[i][2] = (integraterin[i]**2)/var2
# 		train1[i][3] = para
# 		i = i+1


# 	# print(train2)
# 	# print('5')
# 	# print(w1)
# 	trip1 = np.matmul(train2,w1)
# 	# print('4')
# 	# print(trip1)
# 	trip2 = 1.0/(1.0+np.exp(-trip1))
# 	# print('2')
# 	# print(trip2)
# 	# print('3')
# 	# print(w2)
# 	# print('6')
# 	outputtrainmain = np.matmul(trip2,w2)
# 	#print(outputtrainmain)
# 	#print(outputtrainmain[1])

# 	tripp1 = np.matmul(train1,w1)
# 	tripp2 = 1.0/(1.0+np.exp(-tripp1))
# 	outputtrainintr = np.matmul(tripp2,w2)

# 	i = 0
# 	while i < integratept:
# 		integral = integral + 1.0*outputtrainintr[i]
# 		i = i+1
# 	#print(integral)
# 	#print(integral)
# 	#matr = np.multiply(((outputtrainmain/integral)-train[:][1]),outputtrainmain/(integral*integral))

# 	# DW2_l_a = 2*np.matmul(trip2.T,(outputtrainmain/(integral*integral)-train[:][1]/integral))
# 	# DW2_l_b = 2*50*np.matmul(tripp2.T,matr)
# 	# DW2_l = -DW2_l_a + (100.0/integral)*DW2_l_b

# 	# DW1_l_a = 2*np.matmul(train2.T,(np.matmul((outputtrainmain/(integral*integral)-train[:][1]/integral),w2.T)*trip2*(1.0-trip2)))
# 	# DW1_l_b = 2*50*np.matmul(train1.T,(np.matmul(matr,w2.T)*tripp2*(1.0-tripp2)))
# 	# DW1_l = -DW1_l_a + (100.0/integral)*DW1_l_b


# 	DW2_a = np.matmul(trip2.T,np.reciprocal(outputtrainmain))
# 	DW2_b = np.matmul(tripp2.T,np.ones((integratept,1)))
# 	#DW2 = -DW2_a + (10000.0/integral)*DW2_b  
# 	DW2 = -DW2_a + hyp1*1

# 	DW1_a = np.matmul(train2.T,(np.matmul(np.reciprocal(outputtrainmain),w2.T)*trip2*(1.0-trip2)))
# 	DW1_b = np.matmul(train1.T,(np.matmul(np.ones((integratept,1)),w2.T)*tripp2*(1.0-tripp2)))
# 	#DW1 = -DW1_a + (10000.0/integral)*DW1_b
# 	DW1 = -DW1_a + hyp1*1

# 	w1 = w1 - alpha2*DW1
# 	w2 = w2 - alpha2*DW2
# 	put = put +1

# 	loss = 0.0
# 	#if (put % 100 == 0 ):
# 	lossa = 0.0
# 	lossb =	 0.0
# 	i = 0
# 	while i < sampletrain:
# 		if (outputtrainmain[i] > 0):
# 			lossa = lossa - math.log(outputtrainmain[i])
# 		i = i+1
# 	i = 0
# 	while i < 15:
# 		lossb = lossb + 2.0*outputtrainintr[i]
# 		i = i+1
# 	i = 0
# 	if (lossb > 0):
# 		lossb = (50.0)*math.log(lossb)
# 	loss = lossa + lossb
# 	#print('b')
# 	if put % 10000 == 0 :
# 		print(loss)
# 		# print(lossa)
# 		# print(lossb)
# 		print('a')
# 		i = 0
# 		while i < 40:
# 			print(outputtrainmain[i])
# 			i = i+1
# 	# print(DW2[3][0])
# 	# print(DW2_a[3][0])
# 	# print(DW2_b[3][0]*100/integral)
# 	# print(loss)
# 	# print(lossa)
# 	# print(lossb)
# 	# w2[3][0] = w2[3][0]+0.00001


# 	#print(put)

# # fin = np.random.randint(1,high=150,size=50)
# # test = poisson(fin,1)

# #bit + integraterin[0]


# #testing begins
# # d = np.random.poisson(para, 200)
# # d = d.astype(float)
# # i = 0
# # meant1 = np.mean(d)
# # #print(mean1)
# # meant2 = np.mean(np.power(d,2))
# # #print(mean2)
# # meant3 = np.mean(np.power(d,3))
# # #print(mean3)
# # vart1 = np.std(d)
# # #print(var1)
# # vart2 = np.std(np.power(d,2))
# # #print(var2)
# # vart3 = np.std(np.power(d,3))
# # test1 = np.zeros((200,4))
# # while i < 200:
# # 	test1[i][0] = 1.0
# # 	test1[i][1] = (d[i])/vart1
# # 	test1[i][2] = (d[i]**2)/vart2
# # 	test1[i][3] = para
# # 	i = i+1
# # outt1 = np.matmul(test1,w1)
# # outt2 = 1.0/(1+np.exp(-outt1))
# # outputtestt = np.matmul(outt2,w2)
# # outputt = outputtestt/integral
# # testt = poissons(d,sigma)
# # i = 0
# # final_loss = 0
# # while i < 200:
# # 	final_loss = final_loss + (testt[i][1]-outputt[i])
# # 	i = i+1
# # print(final_loss/200.0)


# # e =np.random.poisson(para, 55)
# # i = 0
# # while i <55:
# # 	e[i] = i
# # 	i = i+1
# # e = e.astype(float)
# # i = 0
# # meant1 = np.mean(e)
# # #print(mean1)
# # meant2 = np.mean(np.power(e,2))
# # #print(mean2)
# # meant3 = np.mean(np.power(e,3))
# # #print(mean3)
# # vart1 = np.std(e)
# # #print(var1)
# # vart2 = np.std(np.power(e,2))
# # #print(var2)
# # vart3 = np.std(np.power(e,3))
# # testt1 = np.zeros((55,4))
# # while i < 55:
# # 	testt1[i][0] = 1.0
# # 	testt1[i][1] = (e[i])/vart1
# # 	testt1[i][2] = (e[i]**2)/vart2
# # 	testt1[i][3] = para
# # 	i = i+1
# # # while i < 5:
# # # 	print(output[i])
# # # 	print(test[i][1])
# # # 	i = i+1

# # out1 = np.matmul(testt1,w1)
# # #print(out1)
# # out2 = 1.0/(1+np.exp(-out1))
# # #print(out2)
# # outputtest = np.matmul(out2,w2)
# # #print(outputtest)
# # print(integral)
# # output = outputtest
# # #print(output)
# # testt = poissons(e,sigma)
# # i = 0
# # final_loss = 0
# # while i < 55:
# # 	print(output[i])
# # 	print(testt[i][1])
# # 	print(testt[i][0])
# # 	i = i+1



# # e = np.random.randint(1,high=50,size=25)
# # e = e.astype(float)
# # i = 0
# # meant1 = np.mean(c)
# # #print(mean1)
# # meant2 = np.mean(np.power(e,2))
# # #print(mean2)
# # meant3 = np.mean(np.power(e,3))
# # #print(mean3)
# # vart1 = np.std(e)
# # #print(var1)
# # vart2 = np.std(np.power(e,2))
# # #print(var2)
# # vart3 = np.std(np.power(e,3))
# # testt1 = np.zeros((25,4))
# # while i < 25:
# # 	testt1[i][0] = 1.0
# # 	testt1[i][1] = (e[i]-meant1)/vart1
# # 	testt1[i][2] = (e[i]**2-meant2)/vart2
# # 	testt1[i][3] = (e[i]**3-meant3)/vart3
# # 	i = i+1
# # while i < 5:
# # 	print(output[i])
# # 	print(test[i][1])
# # 	i = i+1

# # ou1 = np.matmul(train2,w1)
# # #print(ou1)
# # ou2 = 1.0/(1+np.exp(-ou1))
# # #print(ou2)
# # ouputtest = np.matmul(ou2,w2)
# # #print(ouputtest)
# # print(integral)
# # ouput = ouputtest/integral
# # #print(ouput)
# # testtt = poisson(c,sigma)
# # i = 0
# # final_loss = 0
# # while i < 25:
# # 	print(ouput[i])
# #  	print(testtt[i][1])
# #  	print(testtt[i][0])
# #  	i = i+1



# # e = np.random.randint(1,high=50,size=25)
# # e = e.astype(float)
# # i = 0
# # meant1 = np.mean(e)
# # #print(mean1)
# # meant2 = np.mean(np.power(e,2))
# # #print(mean2)
# # meant3 = np.mean(np.power(e,3))
# # #print(mean3)
# # vart1 = np.std(e)
# # #print(var1)
# # vart2 = np.std(np.power(e,2))
# # #print(var2)
# # vart3 = np.std(np.power(e,3))
# # testt1 = np.zeros((25,4))
# # while i < 25:
# # 	testt1[i][0] = 1.0
# # 	testt1[i][1] = (e[i]-meant1)/vart1
# # 	testt1[i][2] = (e[i]**2-meant2)/vart2
# # 	testt1[i][3] = (e[i]**3-meant3)/vart3
# # 	i = i+1
# # # while i < 5:
# # # 	print(output[i])
# # # 	print(test[i][1])
# # # 	i = i+1

# # out1 = np.matmul(testt1,w1)
# # print(out1)
# # out2 = 1.0/(1+np.exp(-out1))
# # print(out2)
# # outputtest = np.matmul(out2,w2)
# # print(outputtest)
# # print(integral)
# # output = outputtest/integral
# # print(output)
# # testt = poisson(e,sigma)
# # i = 0
# # final_loss = 0
# # while i < 25:
# # 	print(output[i])
# # 	print(testt[i][1])
# # 	i = i+1

# # put[i])
# # 	print(testtt[i][1])
# # 	i = i+1


from __future__ import print_function
import torch
import torch.distributions as tdist
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
train1 = torch.empty((5,4))
i = 0
while i < 5:
	train1[i,0] = 1.0
	train1[i,1] = (i)
	train1[i,2] = ((i)**2)
	train1[i,3] = ((i)**3)
	i = i+1
print(train1)
print(train1.mean(dim=0)[0])


		train2[i,0] = 1.0
		train2[i,1] = (c[i])/var1
		train2[i,2] = (c[i]**2)/var2
		train2[i,3] = (c[i]**3)/var3
		i = i+1


	y_pip = torch.empty(sampletrain)
	while i < sampletrain:
		if y_pred2[i] > 0:
			y_pip[i] = -torch.log(y_pred2[i])
		else :
			y_pip[i] = -y_pred2[i]
		i = i+1
	loss = y_pip.sum() 

	i = 0
	print(counter)

	if counter % 200 == 0:
		print(counter)