import numpy as np
import random

class DataPrepare(object):
  def __init__(self,path,glove):
    self.seed = 0
    self.maxlength = 40
    self.slotdict,self.rev_slotdict,self.slot_len,self.intentdict,self.rev_intentdict,self.intent_len = self.get_slots()
    self.worddict = glove
    self.encoded,self.reverse = self.set_word2vec(path[1])
    self.slotvalue = self.get_svalue(path[2])
    self.intentvalue = self.get_ivalue(path[3])
    self.sen_info = self.get_info(path[4])
    #self.talker = self.get_talker(path[5])

  def get_talker(self,tpath):
    tf = open(tpath,'r')
    l = []
    for info in tf:
      info = info.strip()
      l.append(info)
    return l

  def get_slots(self):
    BIO = ['B','I','O']
    MAIN = ['AREA','DET','FEE','FOOD','LOC','TIME','TRSP','WEATHER']
    SUBCAT = ['COUNTRY','CITY','DISTRICT','NEIGHBORHOOD','ACCESS', 'BELIEF', 'BUILDING', 'EVENT', 'PRICE', 'NATURE', 'HISTORY', 'MEAL', 'MONUMENT','STROLL','VIEW','ATTRACTION', 'SERVICES', 'PRODUCTS','TEMPLE', 'RESTAURANT', 'SHOP', 'CULTURAL', 'GARDEN', 'HOTEL', 'WATERSIDE', 'EDUCATION', 'ROAD', 'AIRPORT','DATE', 'INTERVAL', 'START', 'END', 'OPEN', 'CLOSE','STATION', 'TYPE','MAIN']
    REL = ['NEAR', 'FAR', 'NEXT', 'OPPOSITE', 'NORTH', 'SOUTH', 'EAST','WEST','BEFORE', 'AFTER', 'AROUND','NONE']
    FROM_TO = ['FROM', 'TO','NONE']
    SAC = ['QST','RES','INI','FOL','None']
    SAA = ['ACK','CANCEL','CLOSING','COMMIT','CONFIRM','ENOUGH','EXPLAIN','HOW_MUCH','HOW_TO','INFO','NEGATIVE','OPENING','POSITIVE','PREFERENCE','RECOMMEND','THANK','WHAT','WHEN','WHERE','WHICH','WHO','none']
    slot = [BIO,MAIN,FROM_TO,REL,SUBCAT]
    intent = [SAC,SAA]
    s_d = []
    s_rev_d = dict()
    i_d = []
    i_rev_d = dict()
    count = 0
    for tags in slot:
      d = dict()
      for tag in tags:
        d[tag] = count
        s_rev_d[count] = tag
        count += 1
      s_d.append(d)
    slot_len = count
    count = 0
    for tags in intent:
      d = dict()
      for tag in tags:
        d[tag] = count
        i_rev_d[count] = tag
        count += 1
      i_d.append(d)
    assert slot_len == len(BIO) + len(MAIN) + len(SUBCAT) + len(REL) + len(FROM_TO)
    return s_d,s_rev_d,slot_len,i_d,i_rev_d,count

  def set_word2vec(self,seq_in):
    """
     for ***next*** only
    """
    unkbook = open('nuk','w')
    parse = ['~','\\','!']
    with open(seq_in,'r') as f:
     training_set = []
     rev_set = []
     for line in f:
      batch = []
      #0***...***6***
      line = line.strip().split('***next***')
      line = line[:-1]
      rev = []
      for i in range(len(line)):
        l = line[i].lower()
        rev.append(l)
        l = l.strip().split()
        if len(l) > self.maxlength:
          print ("opps",len(l))
        sen = []
        for word in l:
          for p in parse:
            word = word.replace(p,"")
          if word in self.worddict:
            sen.append(self.worddict[word])
          elif word.find('\'s',len(word)-2) != -1:
            word = word[:len(word)-2]
            if word in self.worddict:
              sen.append(np.add(self.worddict[word],self.worddict['\'s']))
            else:
              sen.append(self.worddict['<unk>'])
              unkbook.write(word+'\n')
          else:
            sen.append(self.worddict['<unk>'])
            unkbook.write(word+'\n')
        for _ in range(self.maxlength - len(l)):
          sen.append(self.worddict['Empty'])
        assert len(sen) == self.maxlength
        batch.append(sen)
      rev_set.append(rev)
      training_set.append(batch)
    return training_set, rev_set

  def get_svalue(self,label_p):
    training_set = []
    error = open('error_slot','w')
    with open(label_p,'r') as f:
      for line in f:
        line = line.strip().split('***next***')
        line = line[:-1]
        batches = []
        for i in range(len(line)):
          batch = []
          l = line[i].strip().split()
          for tags in l:
            vec = np.zeros(self.slot_len)
            tags = tags.strip()
            tags = tags.replace("DIRECTION-","")
            if len(tags) == 1:#O
              vec[self.slotdict[0]['O']] += 1
            else:
              tags = tags.strip().split('-')
              for i in range(len(tags)):
                if tags[i] == "":
                  vec[self.slotdict[2]["NONE"]] += 1
                else:
                  vec[self.slotdict[i][tags[i]]] += 1
            batch.append(vec)
          for _ in range(self.maxlength - len(l)):
            tmp = np.zeros(self.slot_len)
            #tmp[self.slotdict[0]['O']] += 1
            batch.append(tmp)
          assert len(batch) == self.maxlength
          batches.append(batch)
        training_set.append(batches)
    return training_set

  def get_ivalue(self,label_p):
    training_set = []
    error = open('error_intent','w')
    with open(label_p,'r') as f:
      for line in f:
        line = line.split('***next***')
        line = line[:-1]
        batch = []
        for i in range(len(line)):
          tags = line[i].strip().split('-')
          vec = np.zeros(self.intent_len)
          for tag in tags:
            if tag in self.intentdict[0]:
              vec[self.intentdict[0][tag]] += 1
            elif tag in self.intentdict[1]:
              vec[self.intentdict[1][tag]] += 1
            else:
              print ("not exist intent error",tags)
          batch.append(vec)
        training_set.append(batch)
    return training_set

  def get_info(self,info):
    info_batch = []
    with open(info,'r') as f:
      for line in f:
        batch = []
        line = line.strip().split('***next***')
        line = line[:-1]
        for l in line:
          batch.append(int(l.strip()))
        info_batch.append(batch)
    return info_batch

  # def get_batches(self):
  #   seq_in = []
  #   seq_out = []
  #   intent = []
  #   #5x30x200
  #   for sentence in self.encoded:#4sentence + guide
  #     for i in range(len(sentence)-1):#each sen
  #       for _ in range(self.maxlength - len(sentence[i])):
  #         sentence[i].append(np.zeros(200))
  #     seq_in.append(sentence)
  #   #4x30x350
  #   for out in self.slotvalue:
  #     for i in range(len(out)):
  #       for _ in range(self.maxlength - len(out[i])):
  #         out[i].append(np.zeros(len(self.slotdict)))
  #     seq_out.append(out)
  #   intent = self.intentvalue
  #   return seq_in,seq_out,intent

  def get_all(self):
    return self.encoded,self.slotvalue,self.intentvalue,self.sen_info,self.reverse#,self.talker

  def get_batch(self,batch_size,shuffle=True):
    if shuffle == True:
      r = random.random()
      random.shuffle(self.intentvalue,lambda : r)
      random.shuffle(self.sen_info,lambda : r)
    data_size = len(self.intentvalue)
    num_batches_per_epoch = int((data_size-1)/batch_size)
    for batch_index in range(num_batches_per_epoch):
      start_index = batch_index*batch_size
      end_index = min((batch_index+1)*batch_size,data_size)
      return_x = self.intentvalue[start_index:end_index]
      temp_info = self.sen_info[start_index:end_index]
      temp_info_batch = []
      for batch in temp_info:
        temp = []
        for distence in batch:
          d = []
          d.append(distence)
          temp.append(d)
        temp_info_batch.append(temp)
      return_info = temp_info_batch
      yield (return_x,return_info)
