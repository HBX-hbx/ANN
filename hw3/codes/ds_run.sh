export CUDA_VISIBLE_DEVICES=4

for temp in 0.7 1.0
do
	for model in 'Tfmr_scratch' 'Tfmr_finetune'
	do
		for ds in 'random' 'top-k' 'top-p'
		do
			if [ $ds = 'random' ]; then
				echo \
				--decode_strategy $ds \
				--temp $temp \
				--model $model

				python main.py \
					--test $model \
					--decode_strategy $ds \
					--temperature $temp \

			elif [ $ds = 'top-k' ]; then
				echo \
				--decode_strategy $ds \
				--temp $temp \
				--top-k 40 \
				--model $model

				python main.py \
					--test $model \
					--decode_strategy $ds \
					--temperature $temp \
					--top_k 40

			elif [ $ds = 'top-p' ]; then
				echo \
				--decode_strategy $ds \
				--temp $temp \
				--top_p 0.9 \
				--model $model

				python main.py \
					--test $model \
					--decode_strategy $ds \
					--temperature $temp \
					--top_p 0.9
			fi
		done
	done
done
