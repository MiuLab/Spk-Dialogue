rm -rf result.txt
for i in {1..5}
do
  python2.7 slu.py >> result.txt
done
