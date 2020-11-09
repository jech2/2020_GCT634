### Experiment Schedule
### Original / Aug / Seg / Aug + Seg

# Naive compare
python main.py --data_dir gtzan/spec/ --model Baseline --n_train 3 --title naive_compare --do_test
python main.py --data_dir gtzan/spec/ --model Q1 --n_train 3 --title naive_compare --do_test
python main.py --data_dir gtzan/embed/ --model Q2 --n_train 3 --title naive_compare --do_test
python main.py --data_dir gtzan/msd_embed/ --model Q2 --n_train 3 --title naive_compare --do_test
# Q3
python main.py --data_dir gtzan/spec/ --model Base2DCNN --n_train 3 --title naive_compare --do_test
python main.py --data_dir gtzan/spec/ --embed_dir gtzan/msd_embed/ --model SpecAndEmbed --model2 Q1 --n_train 3 --title naive_compare --do_test
python main.py --data_dir gtzan/spec/ --embed_dir gtzan/msd_embed/ --model SpecAndEmbed --model2 Base2DCNN --n_train 3 --title naive_compare --do_test
python main.py --data_dir gtzan/spec/ --model resnet34 --n_train 3 --title naive_compare --do_test


# Effect of aug
python main.py --data_dir gtzan/aug_spec/ --model Baseline --n_train 3 --title effect_of_aug --do_test
python main.py --data_dir gtzan/aug_spec/ --model Q1 --n_train 3 --title effect_of_aug --do_test
python main.py --data_dir gtzan/aug_embed/ --model Q2 --n_train 3 --title effect_of_aug --do_test
python main.py --data_dir gtzan/aug_msd_embed/ --model Q2 --n_train 3 --title effect_of_aug --do_test
# Q3
python main.py --data_dir gtzan/aug_spec/ --model Base2DCNN --n_train 3 --title effect_of_aug --do_test
python main.py --data_dir gtzan/aug_spec/ --embed_dir gtzan/aug_msd_embed/ --model SpecAndEmbed --model2 Q1 --n_train 3 --title effect_of_aug --do_test
python main.py --data_dir gtzan/aug_spec/ --embed_dir gtzan/aug_msd_embed/ --model SpecAndEmbed --model2 Base2DCNN --n_train 3 --title effect_of_aug --do_test
python main.py --data_dir gtzan/aug_spec/ --model resnet34 --n_train 3 --title effect_of_aug --do_test


# Effect of seg
python main.py --data_dir gtzan/seg_spec/ --model Base2DCNN --n_train 3 --title effect_of_seg --use_segment --do_test
python main.py --data_dir gtzan/seg_spec/ --embed_dir gtzan/seg_msd_embed/ --model SpecAndEmbed --model2 Base2DCNN --n_train 3 --title effect_of_seg --use_segment --do_test
python main.py --data_dir gtzan/seg_spec/ --embed_dir gtzan/seg_msd_embed/ --model SpecAndEmbed --model2 resnet34 --n_train 3 --title effect_of_seg --use_segment --do_test

# Effect of aug and seg
python main.py --data_dir gtzan/seg_aug_spec/ --model Base2DCNN --n_train 3 --title effect_of_aug_seg --use_segment --do_test
python main.py --data_dir gtzan/seg_aug_spec/ --embed_dir gtzan/seg_aug_msd_embed/ --model SpecAndEmbed --model2 Base2DCNN --n_train 3 --title effect_of_aug_seg --use_segment --do_test
python main.py --data_dir gtzan/seg_aug_spec/ --embed_dir gtzan/seg_aug_msd_embed/ --model SpecAndEmbed --model2 resnet34 --n_train 3 --title effect_of_aug_seg --use_segment --do_test


## vgg
python main.py --data_dir gtzan/seg_aug_spec/ --model vgg13 --n_train 3 --use_segment --title effect_of_aug_seg --do_test
python main.py --data_dir gtzan/seg_aug_spec/ --model vgg16 --n_train 3 --use_segment --title effect_of_aug_seg --do_test
python main.py --data_dir gtzan/seg_aug_spec/ --model vgg19 --n_train 3 --use_segment --title effect_of_aug_seg --do_test

## resnet
python main.py --data_dir gtzan/seg_aug_spec/ --model resnet18 --n_train 3 --use_segment --title effect_of_aug_seg --do_test
python main.py --data_dir gtzan/seg_aug_spec/ --model resnet34 --n_train 3 --use_segment --title effect_of_aug_seg --do_test
python main.py --data_dir gtzan/seg_aug_spec/ --model resnet50 --n_train 3 --use_segment --title effect_of_aug_seg --do_test
python main.py --data_dir gtzan/seg_aug_spec/ --model resnet101 --n_train 3 --use_segment --title effect_of_aug_seg --do_test