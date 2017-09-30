cd BLSTM
rm -rf result.txt
for i in {1..5}
do
  python2.7 slu.py Tourist >> result.txt
done
cd ..

cd role_based
rm -rf result.txt
for i in {1..5}
do
  python2.7 slu.py Tourist >> result.txt
done
cd ..
