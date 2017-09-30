def get_train_batch():
	train_file = open('./Data/train/talker','r')
	counter = 0 
	tourist = [-1,-1,-1]
	guide = [-1,-1,-1]
	number = [1186,2553,3643,4668,5834,6867,8058,8964,10627,11407,12818,13812,14832]
	batch = []
	mapping = []
	for talker in train_file:
		if counter in number:
			tourist = [-1,-1,-1]
			guide = [-1,-1,-1]
		tmp = []
		if talker.strip() == "Guide":
			for i in tourist:
				tmp.append(i)
			for i in guide:
				tmp.append(i)
			batch.append(tmp)
			mapping.append(counter)
		if talker.strip() == "Guide":
			guide = guide[1:]
			guide.append(counter)
			counter += 1
		elif talker.strip() == "Tourist":
			tourist = tourist[1:]
			tourist.append(counter)
			counter += 1
		else:
			print ("talker error",talker)

	return batch,mapping

def get_test_batch():
	test_file = open('./Data/test/talker','r')
	counter = 0
	number = [709,1568,2813,4216,5286,6719,7979,9077]
	tourist = [-1,-1,-1]
	guide = [-1,-1,-1]
	batch = []
	for talker in test_file:
		if counter in number:
			tourist = [-1,-1,-1]
			guide = [-1,-1,-1]
		tmp = []
		if talker.strip() == "Guide":
			for i in tourist:
				tmp.append(i)
			for i in guide:
				tmp.append(i)
			batch.append(tmp)
			mapping.append(counter)
		if talker.strip() == "Guide":
			guide = guide[1:]
			guide.append(counter)
			counter += 1
		elif talker.strip() == "Tourist":
			tourist = tourist[1:]
			tourist.append(counter)
			counter += 1
		else:
			print ("talker error")

	return batch,mapping