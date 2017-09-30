cd BLSTM
rm -rf result.txt
for i in $(seq 1 1 5)
do
  python2.7 slu.py >> result.txt
done
cd ..

cd role_based
rm -rf result.txt
for i in $(seq 1 1 5)
do
  python2.7 slu.py >> result.txt
done
cd ..

cd role_based_intermediate
rm -rf result.txt
for i in $(seq 1 1 5)
do
  python2.7 slu.py >> result.txt
done
cd ..
