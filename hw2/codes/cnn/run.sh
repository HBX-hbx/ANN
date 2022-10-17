export CUDA_VISIBLE_DEVICES=1
for batch_size in 50 100 200 500
do


for learning_rate in 0.01 0.001 0.0005 0.0001
do

for drop_rate in 0.0 0.1 0.3 0.5
do
	echo\
	 --drop_rate $drop_rate \
	 --learning_rate  $learning_rate \
	 --batch_size  $batch_size
	python main.py --drop_rate $drop_rate --learning_rate $learning_rate --batch_size $batch_size
done

done
done