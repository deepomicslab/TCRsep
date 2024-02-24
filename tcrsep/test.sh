#test basic training 
# python train.py --post_data_path data/HIP13610.csv \
#                 --save_dir results/test0
#test provide the pre_data
# python train.py --iters 1000 \
#                 --alpha 0.1 \
#                 --val_ratio 0.05 \
#                 --batchsize 1000 \
#                 --post_data_path data/HIP13610.csv \
#                 --save_dir results/test1 \
#                 --gen_model_path models/generation_model/CMV_whole/ \
#                 --pre_data_path results/test0/pre-sel.csv
#test using embeding data
# python train.py --iters 1000 \
#                 --alpha 0.1 \
#                 --val_ratio 0.05 \
#                 --batchsize 1000 \
#                 --post_data_path results/test0/post_emb.npy.gz \
#                 --save_dir results/test2 \
#                 --gen_model_path models/generation_model/CMV_whole/ \
#                 --pre_data_path results/test0/pre_emb.npy.gz

#test basic eval
# python eval.py  --data_path data/query_data.csv \
#                 --save_dir results/test4

#test add embedding
# python eval.py  --data_path data/query_data.csv \
#                 --save_dir results/test5 \
#                 --data_emb_path results/test4/query_data_embedding.npy.gz 

#test using selection model & generation model
# python eval.py  --data_path data/query_data.csv \
#                 --save_dir results/test6 \
#                 --data_emb_path results/test4/query_data_embedding.npy.gz \
#                 --sel_model_path results/test0/tcrsep.pth \
#                 --gen_model_path models/generation_model/CMV_whole/                                     