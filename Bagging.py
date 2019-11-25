import weka.core.jvm as jvm #weka requires java toolkit
import weka.core.converters as con #for converting the data set
from weka.clusterers import Clusterer #for clustering
from weka.classifiers import Classifier
from weka.classifiers import KernelClassifier, MultipleClassifiersCombiner
from weka.core.dataset import Instances
from weka.core.dataset import Instance
from weka.classifiers import Evaluation, PredictionOutput
from weka.core.classes import JavaObject, Random
import itertools
import javabridge
import numpy
import random



jvm.start() 
data = con.load_any_file("Train_BiometrikaFullPPR80.arff") #to load the required file
data_copy = Instances.copy_instances(data)
test = con.load_any_file("Test_BiometrikaPP.arff")
test_copy = Instances.copy_instances(test)
test.delete_last_attribute()
#separate_test = Instances.template_instances(test_copy)
data_copy.delete_last_attribute()

#dataset = con.load_any_file("credit.g.arff") 
#data, test = dataset.train_test_split(66.0, Random(1))
#data_copy = Instances.copy_instances(data)
#test_copy = Instances.copy_instances(test)
#test.delete_last_attribute()
#data_copy.delete_last_attribute()
validation = con.load_any_file("Train_BiometrikaFullPPR20.arff")
validation.class_is_last()
data.class_is_last()
test_copy.class_is_last()

# randomize data
folds = 10
seed = 1
rnd = Random(seed)
rand_data = Instances.copy_instances(data)
rand_data.randomize(rnd)
if rand_data.class_attribute.is_nominal:
	rand_data.stratify(folds)

class Instances(JavaObject):
	def __init__(self, jobject):
		self.__num_attributes = javabridge.make_call(self.jobject, "numAttributes", "()I")


	def num_attributes(self):
		return self.__num_attributes()


	num_of_attr = num_attributes(data)
	print("number of Attributes:",num_of_attr)


	#clusterer = Clusterer(classname="weka.clusterers.EM")#no of clusters = 3
	clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])
	clusterer.build_clusterer(data_copy) #generates a cluster
	clusters = dict()


	# cluster the data
	for inst in data_copy:
		cl = clusterer.cluster_instance(inst)#classifies a given instance-returns which cluster it belongs to

		if cl not in clusters:
		    clusters[cl]=1
		else:
		    clusters[cl] = clusters[cl]+1
	#clusters is a list containing cluster number as key and no of instances present in that cluster as value
	print(clusters)
	print("............................................................................")




	num_of_trees = len(clusters)
	num_of_clusters = len(clusters)
	cluster_data = [Instances.template_instances(data) for i in range(0, num_of_clusters)]
	count_instances = 0
	for inst in data_copy:
		count_instances = count_instances + 1

	for i in range(count_instances):
		cl = clusterer.cluster_instance(data_copy.get_instance(i))
		#dist = clusterer.distribution_for_instance(data.get_instance(i))
		cluster_data[cl].add_instance(data.get_instance(i))

	for d in cluster_data:
		d.class_is_last()

	def accuracy(trees, num_of_trees, validation):
		acc = list()
		for obj in range(num_of_trees):
			final_class_count = [[0,0] for i in range(len(validation))] 
			separate_test = [Instances.template_instances(validation) for i in range(len(validation))] 									
			for i in range(0,len(validation)):
				separate_test[i].add_instance(validation.get_instance(i))
				separate_test[i].class_is_last()


			for j in range(len(validation)):
				evl = Evaluation(separate_test[j])
				evl.test_model(trees[obj], separate_test[j])
				if(int(evl.correct)==1):
					final_class_count[j][0] = final_class_count[j][0] + 1
				elif(int(evl.correct)==0):
					final_class_count[j][1] = final_class_count[j][1] + 1
			correct_class = 0
			for i in range(len(validation)):
				if(final_class_count[i][0]>final_class_count[i][1]):
					correct_class = correct_class + 1
			del separate_test
			del final_class_count

			print("Total number of instances",len(validation))
			print("No of correct predictions",correct_class)
			a = float(correct_class)/float(len(validation))
			print("Accuracy", a)
			acc.append(a)	
			#print(acc)
		return acc

	def predicted(trees, num_of_attr, test):
		count = 0
		for index, inst in enumerate(test):
			pred = trees.classify_instance(inst)
			predicted_value = inst.class_attribute.value(int(pred))
			actual = inst.get_value(num_of_attr-1)
			if actual == pred:
				count = count+1	
				
		return count

	def crossvalidation(classifier, rand_data, folds, num_of_attr, num_of_trees):
		
		def predicted(trees, num_of_attr, test):
			count = 0
			for index, inst in enumerate(test):
				pred = trees.classify_instance(inst)
				predicted_value = inst.class_attribute.value(int(pred))
				actual = inst.get_value(num_of_attr-1)
				if actual == pred:
					count = count+1	
				
			return count
		accuracy = list()
		
		predicted_data = None
	    	evaluation = Evaluation(rand_data)
		for j in range(num_of_trees):
			summation = 0
		    	for i in xrange(folds):
				train = rand_data.train_cv(folds, i)
				test = rand_data.test_cv(folds, i)
				cls = Classifier.make_copy(classifier[j])
				#cls.build_classifier(train)
				#evaluation.test_model(cls, test)
				count = predicted(cls, num_of_attr, test)
				count = float(count)
				total = len(test)
				total = float(total)
				acc = count/total
				summation = summation+acc
		
			length = float(folds)
			accuracy.append(summation/length)
			print("Accuracy of Cluster", j, "is", summation/length)
		return accuracy
	
	
	
	trees = list()
	final_trees = list()
	
	trees = [Classifier(classname= "weka.classifiers.functions.SMO") for i in range(0, num_of_trees)]
	for i in range(0,num_of_clusters):
		classifier1 = trees[i].build_classifier(cluster_data[i])
	print("When validation data is passed through Classifier 0")
	a1 = accuracy(trees, num_of_trees, validation)
	final_trees.append(trees)

	print("----------------------------------------------------------------------------------")
	print("When crossvalidation is done on Classifier 0")
	ac0 = crossvalidation(trees, rand_data, folds, num_of_attr, num_of_trees)
	#print("Accuracy of Classifier 0 on crossvalidation", ac1)


	print("----------------------------------------------------------------------------------")
	print("When test data is passed through Classifier 0")
	accuracy(trees, num_of_trees, test_copy)


	print("----------------------------------------------------------------------------------")
	trees = [Classifier(classname="weka.classifiers.functions.SMO" )for i in range(0, num_of_trees)]
	for i in range(0, num_of_clusters):
		classifier2 = trees[i].build_classifier(cluster_data[i])
	print("When validation data is passed through Classifier 1")
	a2 = accuracy(trees, num_of_trees, validation)
	final_trees.append(trees)
	
	print("----------------------------------------------------------------------------------")
	print("When crossvalidation is done on Classifier 1")
	ac1 = crossvalidation(trees, rand_data, folds, num_of_attr, num_of_trees)

	print("----------------------------------------------------------------------------------")
	print("When test data is passed through Classifier 1")
	accuracy(trees, num_of_trees, test_copy)
	print("----------------------------------------------------------------------------------")
	trees = [Classifier(classname="weka.classifiers.functions.SMO" )for i in range(0, num_of_trees)]
	for i in range(0, num_of_clusters):
		classifier3 = trees[i].build_classifier(cluster_data[i])
	print("When validation data passed through Classifier 2")
	a3 = accuracy(trees, num_of_trees, validation)
	final_trees.append(trees)

	print("----------------------------------------------------------------------------------")
	print("When crossvalidation is done on Classifier 2")
	ac2 = crossvalidation(trees, rand_data, folds, num_of_attr, num_of_trees)

	print("----------------------------------------------------------------------------------")
	print("When test data is passed through Classifier 2")
	accuracy(trees, num_of_trees, test_copy)
	encode = {0:classifier1, 1:classifier2, 2:classifier3}

	print("----------------------------------------------------------------------------------")
	final = list()
	for i in range(0, num_of_trees):
		item = max(a1[i],a2[i],a3[i])
		if item == a1[i]: final.append(0)
		elif item == a2[i]: final.append(1)
		elif item == a3[i]: final.append(2)
	for item in range(0, num_of_clusters):
		print("For cluster number", item, "classifier", final[item], "is selected in validation")

	print("----------------------------------------------------------------------------------")
	finalc = list()
	for i in range(0, num_of_trees):
		item = max(ac0[i],ac1[i],ac2[i])
		if item == ac0[i]: finalc.append(0)
		elif item == ac1[i]: finalc.append(1)
		elif item == ac2[i]: finalc.append(2)

	for item in range(0, num_of_clusters):
		print("For cluster number", item, "classifier", finalc[item], "is selected in cross validation")
	
	
	meta = MultipleClassifiersCombiner(classname="weka.classifiers.meta.Vote")
	classifiers = list()
	for index1 in range(0, num_of_clusters):
			index2 = final[index1]
			classifiers.append(final_trees[index2][index1])
	
    	#classifiers = [Classifier(classname="weka.classifiers.trees.RandomForest" )]
	meta.classifiers = classifiers

	meta.build_classifier(data)
	print(meta.to_commandline())
	
	
	print("----------------------------------------------------------------------------------")
	print("TESTING THE DATA ON META CLASSIFIERS CHOOSING THE CLASSIFIERS BASED ON VALIDATION ACCURACIES")
	evaluation = Evaluation(test_copy)
	evl = evaluation.test_model(meta, test_copy)
	print(evaluation.summary())
	
	del meta, classifiers
	
	meta = MultipleClassifiersCombiner(classname="weka.classifiers.meta.Vote")
	classifiers = list()
	for index1 in range(0, num_of_clusters):
			index2 = finalc[index1]
			classifiers.append(final_trees[index2][index1])
	
    	#classifiers = [Classifier(classname="weka.classifiers.trees.RandomForest" )]
	meta.classifiers = classifiers

	meta.build_classifier(data)
	print(meta.to_commandline())
	
	
	print("----------------------------------------------------------------------------------")
	print("TESTING THE DATA ON META CLASSIFIERS CHOOSING THE CLASSIFIERS BASED ON CROSS-VALIDATION ACCURACIES")
	evaluation = Evaluation(test_copy)
	evl = evaluation.test_model(meta, test_copy)
	print(evaluation.summary())
        print(evaluation.matrix())

	print("----------------------------------------------------------------------------------")
	print("TESTING THE DATA ON META CLASSIFIERS CHOOSING THE CLASSIFIERS BASED ON COMBINATIONS")
	list1 = list()
	list2 = list()
	#index1 = classifier number
	#index2 = cluster number
	for index1 in range(len(final_trees)):
			list1.append(final_trees[index1][0])
			list2.append(final_trees[index1][1])


		
	accuracies = list()
	combo = list()
	for combination in itertools.product(list1, list2):
		classifiers = list()
		for item in combination:
			classifiers.append(item)
		meta = MultipleClassifiersCombiner(classname="weka.classifiers.meta.Vote" )
		
		meta.classifiers = classifiers

		meta.build_classifier(data)
                #print(false_positive_rate(1))
                #print(meta.classifiers.class_details())
		#print(meta)
		#evaluation = Evaluation(test_copy)
		#evl = evaluation.test_model(meta, test_copy)
		#print(evaluation.summary())
		count = predicted(meta, num_of_attr, test_copy)
		count = float(count)
		total = len(test_copy)
		total = float(total)
		print("Accuracy", count/total)
		accuracies.append(count/total)
		combo.append(combination)
		del meta, classifiers
	#ind = accuracies.index(max(accuracies))
	
	m = max(accuracies)
	ind = [i for i, j in enumerate(accuracies) if j == m]
	class_name =list()
	for v in range(len(ind)):
		for item in combo[ind[v]]:
			for i in range(len(final_trees)):
				for j in range(num_of_clusters):
					if item == final_trees[i][j]:
						class_name.append([i,j])
					
				
	print("Maximum accuracy achieved", max(accuracies))
	print("With a combination of")	
	count = 0
	for item in class_name:
		if(count ==2):
			print("--------------------------------------------------")
			print("OR")
			count = 0
		print("For cluster", item[1], "Classifier", item[0])
		count = count+1
		
		#print(item)
	print("------------------------------------------------")
	print("Classifier 0 is weka.classifiers.functions.SMO")
	print("Classifier 1 is weka.classifiers.functions.VotedPerceptron")
	print("Classifier 2 is weka.classifiers.trees.RandomForest")
	jvm.stop()

