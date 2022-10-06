export CUDA_VISIBLE_DEVICES=7
for batch_size in 1 25 125 625 3125
do


for learning_rate in 0.01 0.001 0.0001 0.00001
do

for drop_rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
	echo\
	  $drop_rate \
	  $learning_rate \
	  $batch_size
	
done

done
done