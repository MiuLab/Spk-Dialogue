with open('intent','r') as f:
	d = dict()
	for line in f:
		line = line.strip()
		if line in d:
			continue
		else:
			d[line] = 1
with open('intent_list','w') as f:
	for key,value in d.iteritems():
		f.write(key+'\n')
with open('seq.out','r') as f:
	d = dict()
	for line in f:
		word = line.strip().split()
		for c in word:
			if c in d:
				continue
			else:
				d[c] = 1
with open('slot_list','w') as f:
	for key,value in d.iteritems():
		f.write(key+'\n')
