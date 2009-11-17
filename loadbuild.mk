# loadbuild makefile for Jeremy's AUSDM entry

# Create a directory
 %/.dir_exists:
	@mkdir $(basename $@)
	@touch $@

# Perform an SVD decomposition
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_svd_decomposition
loadbuild/$(1)_$(2)_SVD.dat: loadbuild/.dir_exists
	$(BIN)/decompose -T SVD -S $(1) -t $(2) -o $$@~ 2>&1 | tee $$@.log
	mv $$@~ $$@

SVD: loadbuild/$(1)_$(2)_SVD.dat
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_svd_decomposition,$(size),$(type)))))

# train_top_n
# Trains models for the top "n" performance for the given dataset
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define train_top_n

loadbuild/top_n/${1}_${2}_Score_top_n.txt:
	$(BIN)/ausdm -S $(1) -t $(2) --decomposition "" -c top_n_config.txt -o $@~

endef