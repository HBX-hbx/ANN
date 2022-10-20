export CUDA_VISIBLE_DEVICES=4

for num in 1 4 8 12 16 32
do
	python main.py \
		--name 'Tfmr_scratch_heads_'$num
done