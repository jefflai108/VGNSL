cd src
#python test.py --candidate ../output/model_best.pth.tar
#python test.py --candidate ../output/6.pth.tar
#python test.py --candidate ../output/7.pth.tar

i=0
while [ $i -ne 29 ]
do
    echo ../output/${i}.pth.tar
    python test.py --candidate ../output/${i}.pth.tar
    i=$(($i+1))
done
