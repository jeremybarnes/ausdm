# loadbuild makefile for Jeremy's AUSDM entry

# Create a directory
 %/.dir_exists:
	@mkdir $@


# train_top_n
# Trains models for the top "n" performance for the given dataset
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define train_top_n

models/top_n/${1}_${2}_Score_top_n.txt:
	

endef