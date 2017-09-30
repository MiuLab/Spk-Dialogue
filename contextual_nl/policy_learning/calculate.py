import os

def calculate_mean():
    try:
        f1 = open('result.txt', 'r')
    except:
        return

    s = 0
    for line in f1:
        s += float(line)
    print s * 0.2

# traverse root directory, and list directories as dirs and files as files
d = list()
for root, dirs, files in os.walk("."):
    d = dirs
    break
d = sorted(d)
original = os.getcwd()
for dirs in d:
    os.chdir(dirs)
    print dirs
    calculate_mean()
    os.chdir(original)
