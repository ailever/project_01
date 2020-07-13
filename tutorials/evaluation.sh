CUDA_VISIBLE_DEVICES=0 python tutor_eval.py \
	--model log/log_grnet_aoa/model.pth \
	--infos_path log/log_grnet_aoa/infos_grnet_aoa.pkl \
	--dump_images 0 \
	--dump_json 1 \
	--num_images -1 \
	--language_eval 1 \
	--beam_size 2 \
	--batch_size 100 \
	--split test \
    --input_fc_dir  data/stylized_cocotalk_fc \
    --input_att_dir  data/stylized_cocotalk_att \
