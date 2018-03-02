import subprocess
import numpy as np

settings = []
for p_io in [0.5,0.45,0.4,0.35,0.3,0.25,0.2]:
# for p_io in [0.4,0.35,0.3,0.25,0.2]:
	for p_s in [0.5,0.45,0.4,0.35,0.3,0.25,0.2]:
	# for p_s in [0.4,0.35,0.3,0.25,0.2]:
		#for p_e in [0.5,0.45,0.4,0.35,0.3,0.25,0.2]:
		for p_e in [0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.5,0.0]:
			for embed_size in [100]:
				for state_size in [embed_size, embed_size/2]:
					for n_layers in [2]:
						settings.append([
							'python', 
							'tf_classify.py',
							'embed_trainable', str(True),
							# '--embed_path', 'data/twitter_davidson/embeddings.word2vec.{}d.dat',
							# '--embed_path', 'random',
							# '--vocab_path', 'data/twitter_davidson/vocab.stemmed.dat',
							'--input_dropout', str(p_io),
							'--output_dropout', str(p_io),
							'--state_dropout', str(p_s),
							'--embedding_dropout', str(p_e),
							'--embedding_size', str(embed_size),
							'--state_size', str(state_size),
							'--num_layers', str(n_layers),
							'--max_gradient_norm', str(10.0),
							'--epochs', str(30),
							'--log_dir', 'randsearch_E',
							'--scoring', 'f1_macro',
							'--output_size', '3'
							# '--model_type', 'hb_append'
						])
						
tryset = np.random.choice(len(settings), 100, replace=False)
for i in tryset:
	print settings[i]
	subprocess.call(settings[i])
