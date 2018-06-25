from pyspark import SparkContext, SparkConf
from multiprocessing import Process
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithSGD
import re
import time

sc = SparkContext('local','log')

file = "/home/pi/Documents/splt.txt"
f = open(file, 'r+')
text = f.read()

splitLine = text.split("\n")

data = []

for temp1 in splitLine:
	strip = temp1.strip()
	data.append(strip)

fileProgramming = "/home/pi/Documents/prg.txt"
fileOther = "/home/pi/Documents/oth.txt"

def task1():
	#Print discontinue product
	print ("-------------------------------------------")
	start = time.time()
	dataRDD = sc.textFile(file)
	mapRDD = dataRDD.map(lambda x: (x.strip(),1))
	reduceRDD = mapRDD.reduceByKey(lambda x,y : x+y)
	reducePrint = reduceRDD.collect()
	for i in reducePrint:
		if 'discontinued product' in i:
			print (i)
	end = time.time()
	elapsed = end - start
	print (elapsed)
	print ("-------------------------------------------")

def task2():
	#Print title with Machine Learning Classification
	print ("-------------------------------------------")
	startTitle = time.time()
	regex1 = re.compile(".*(title:).*")
	find1 = [m.group(0) for l in data for m in [regex1.search(l)] if m]
	title = [i.split('title: ', 1)[1] for i in find1]

	Programming = sc.textFile(fileProgramming)
	Other = sc.textFile(fileOther)

	# Create a HashingTF instance to map title text to vectors of 100,000 features.
	tf = HashingTF(numFeatures = 100000)

	# Each title is split into words, and each word is mapped to one feature.
	programmingFeatures = Programming.map(lambda title: tf.transform(title.split(" ")))
	otherFeatures = Other.map(lambda title: tf.transform(title.split(" ")))

	# Create LabeledPoint datasets for positive (programming) and negative (other) examples.
	positiveExamples = programmingFeatures.map(lambda features: LabeledPoint(1, features))
	negativeExamples = otherFeatures.map(lambda features: LabeledPoint(0, features))
	trainingData = positiveExamples.union(negativeExamples)
	trainingData.cache()

	# Run Logistic Regression using the SGD algorithm.
	model = LogisticRegressionWithSGD.train(trainingData)

	listResult = []

	for row in title:
	    test = tf.transform(row.split(" "))
	    result = "null"
	    if model.predict(test) == 1:
	    	result = "Programmings"
	    else:
	    	result = "Non-Programming"
	    joinResult = row+" = "+result
	    listResult.append(joinResult)

	for i in listResult:
		if 'Non-Programming' in i:
			print (i)

	for i in listResult:
		if 'Programmings' in i:
			print (i)

	endTitle = time.time()
	elapsedTitle = endTitle - startTitle
	print (elapsedTitle)
	print ("-------------------------------------------")

def task3():
	#Print group
	print ("-------------------------------------------")
	startGroup = time.time()
	regex2 = re.compile(".*(group:).*")
	find2 = [m.group(0) for l in data for m in [regex2.search(l)] if m]
	groupRDD = sc.parallelize(find2)
	groupMapRDD = groupRDD.map(lambda x: (x,1))
	groupReduceRDD = groupMapRDD.reduceByKey(lambda x,y : x+y)
	groupSort = groupReduceRDD.sortBy(lambda a: a[1]).collect()
	for row in groupSort:
		print (row)
	endGroup = time.time()
	elapsedGroup = endGroup - startGroup
	print (elapsedGroup)
	print ("-------------------------------------------")

def task4():
	#Print customer
	print ("-------------------------------------------")
	startCustomer = time.time()
	regex3 = re.compile(".*(cutomer:).*")
	find3 = [m.group(0) for l in data for m in [regex3.search(l)] if m]
	customer = [i.split('cutomer:', 1)[1] for i in find3]
	customer1 = [i.split('rating:', 1)[0] for i in customer]
	customer2 = []
	for splt in customer1:
		strip = splt.strip()
		customer2.append(strip)
	customerRDD = sc.parallelize(customer2)
	customerMapRDD = customerRDD.map(lambda x: (x,1))
	customerReduceRDD = customerMapRDD.reduceByKey(lambda x,y : x+y)
	customerSort = customerReduceRDD.sortBy(lambda a: a[1]).collect()
	for row in customerSort:
		print (row)
	endCustomer = time.time()
	elapsedCustomer = endCustomer - startCustomer
	print (elapsedCustomer)
	print ("-------------------------------------------")

def task5():
	#Print count ID
	print ("-------------------------------------------")
	startID= time.time()
	regex4 = re.compile(".*(Id:).*")
	find4 = [m.group(0) for l in data for m in [regex4.search(l)] if m]
	IDRDD = sc.parallelize(find4)
	IDCount = IDRDD.count()
	endID = time.time()
	print (IDCount)
	elapsedID = endID - startID
	print (elapsedID)
	print ("-------------------------------------------")

def Main():
	startT = time.time()

	p1 = Process(target=task1)
	p2 = Process(target=task2)
	p3 = Process(target=task3)
	p4 = Process(target=task4)

	p1.start()
	p2.start()
	p3.start()
	p4.start()

	p1.join()
	p2.join()
	p3.join()
	p4.join()

	# task1()
	# task2()
	# task3()
	# task4()
	task5()

	endT = time.time()
	elapsedT = endT - startT
	print (elapsedT)

if __name__ == '__main__':
	Main()

f.close()
