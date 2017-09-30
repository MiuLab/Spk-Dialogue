cd role_based_intermediate
for i in $(seq 1 1 5)
do
  python2.7 slu.py >> result.txt
done
cd ..
