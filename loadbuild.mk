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
	/usr/bin/time \
	$(BIN)/decompose -T SVD -S $(1) -t $(2) -o $$@~ 2>&1 | tee $$@.log
	mv $$@~ $$@

SVD: loadbuild/$(1)_$(2)_SVD.dat
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_svd_decomposition,$(size),$(type)))))


# Create a denoising autoencoder decomposition
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

# Sample which proportion of the training examples per iteration
dnae_sp_S := 0.8
dnae_sp_M := 0.5
dnae_sp_L := 0.1

# Test every x iterations
dnae_te_S := 10
dnae_te_M := 10
dnae_te_L := 50

# Learning rates
dnae_lr_S := 0.3
dnae_lr_M := 0.2
dnae_lr_L := 0.1

define do_dnae_decomposition
loadbuild/$(1)_$(2)_DNAE.dat: loadbuild/.dir_exists
	/usr/bin/time \
	$(BIN)/decompose \
		-T DNAE \
		-S $(1) -t $(2) \
		niter=500 \
		prob_cleared=0.1 \
		layer_sizes=250,150,100,50 \
		minibatch_size=256 \
		test_every=$(dnae_te_$(1)) \
		sample_proportion=$(dnae_sp_$(1)) \
		learning_rate=$(dnae_lr_$(1)) \
		-o $$@~ \
	2>&1 | tee $$@.log
	mv $$@~ $$@

DNAE: loadbuild/$(1)_$(2)_DNAE.dat
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_dnae_decomposition,$(size),$(type)))))



# Top-n model
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_topn
loadbuild/topn/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/topn/.dir_exists
	$(BIN)/ausdm -S $(1) -t $(2) -T 0.20 --decomposition "" -o loadbuild/topn/$(1)_$(2)_merge.txt~ -O $$@~ -n topn.$(2).$(1) 2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/topn/$(1)_$(2)_merge.txt~ loadbuild/topn/$(1)_$(2)_merge.txt

topn: loadbuild/topn/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_topn,$(size),$(type)))))

# Gated model
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_gated
loadbuild/gated/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/gated/.dir_exists loadbuild/$(1)_$(2)_SVD.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "loadbuild/$(1)_$(2)_SVD.dat" \
		order=200 \
		-o loadbuild/gated/$(1)_$(2)_merge.txt~ \
		-O $$@~ -n gated \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/gated/$(1)_$(2)_merge.txt~ loadbuild/gated/$(1)_$(2)_merge.txt

gated: loadbuild/gated/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_gated,$(size),$(type)))))

# Gated model with no decomposition
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_gated2
loadbuild/gated2/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/gated2/.dir_exists
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "" \
		-o loadbuild/gated2/$(1)_$(2)_merge.txt~ \
		-O $$@~ -n gated \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/gated2/$(1)_$(2)_merge.txt~ loadbuild/gated2/$(1)_$(2)_merge.txt

gated2: loadbuild/gated2/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_gated2,$(size),$(type)))))

# Gated model with denoising autoencoder decomposition
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_gated3
loadbuild/gated3/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/gated3/.dir_exists loadbuild/$(1)_$(2)_DNAE.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "" \
		-o loadbuild/gated3/$(1)_$(2)_merge.txt~ \
		-O $$@~ -n gated \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/gated3/$(1)_$(2)_merge.txt~ loadbuild/gated3/$(1)_$(2)_merge.txt

gated3: loadbuild/gated3/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_gated3,$(size),$(type)))))



mr_nfeat_S := 100
mr_nfeat_M := 150
mr_nfeat_L := 200

mr_order_S := 100
mr_order_M := 150
mr_order_L := 200


# Multiple regression model 1 (no extra data)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_mr1
loadbuild/mr1/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/mr1/.dir_exists
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "" \
		-o loadbuild/mr1/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-n mr \
		mr.type=multiple_regression \
		mr.num_iter=500 \
		mr.num_examples=5000 \
		mr.num_features=$(mr_nfeat_$(1)) \
		mr.use_decomposition_features=false \
		mr.use_extra_features=false \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/mr1/$(1)_$(2)_merge.txt~ loadbuild/mr1/$(1)_$(2)_merge.txt

mr1: loadbuild/mr1/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_mr1,$(size),$(type)))))

# Multiple regression model 2 (SVD decomposition features)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_mr2
loadbuild/mr2/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/mr2/.dir_exists loadbuild/$(1)_$(2)_SVD.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "loadbuild/$(1)_$(2)_SVD.dat" \
		-o loadbuild/mr2/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-n mr \
		mr.type=multiple_regression \
		mr.num_iter=500 \
		mr.num_examples=6000 \
		mr.num_features=$(mr_nfeat_$(1)) \
		mr.use_decomposition_features=true \
		mr.use_extra_features=false \
		order=$(mr_order_$(1)) \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/mr2/$(1)_$(2)_merge.txt~ loadbuild/mr2/$(1)_$(2)_merge.txt

mr2: loadbuild/mr2/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_mr2,$(size),$(type)))))

# Multiple regression model 3 (SVD decomposition features + extra features)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_mr3
loadbuild/mr3/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/mr3/.dir_exists loadbuild/$(1)_$(2)_SVD.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "loadbuild/$(1)_$(2)_SVD.dat" \
		-o loadbuild/mr3/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-n mr \
		mr.type=multiple_regression \
		mr.num_iter=500 \
		mr.num_examples=6000 \
		mr.num_features=$(mr_nfeat_$(1)) \
		mr.use_decomposition_features=true \
		mr.use_extra_features=true \
		order=$(mr_order_$(1)) \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/mr3/$(1)_$(2)_merge.txt~ loadbuild/mr3/$(1)_$(2)_merge.txt

mr3: loadbuild/mr3/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_mr3,$(size),$(type)))))

# Multiple regression model 4 (DNAE decomposition features)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_mr4
loadbuild/mr4/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/mr4/.dir_exists loadbuild/$(1)_$(2)_SVD.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "loadbuild/$(1)_$(2)_DNAE.dat" \
		-o loadbuild/mr4/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-n mr \
		mr.type=multiple_regression \
		mr.num_iter=500 \
		mr.num_examples=6000 \
		mr.num_features=$(mr_nfeat_$(1)) \
		mr.use_decomposition_features=true \
		mr.use_extra_features=false \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/mr4/$(1)_$(2)_merge.txt~ loadbuild/mr4/$(1)_$(2)_merge.txt

mr4: loadbuild/mr4/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_mr4,$(size),$(type)))))

# Multiple regression model 3 (SVD decomposition features + extra features)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_mr5
loadbuild/mr5/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/mr5/.dir_exists loadbuild/$(1)_$(2)_SVD.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "loadbuild/$(1)_$(2)_DNAE.dat" \
		-o loadbuild/mr5/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-n mr \
		mr.type=multiple_regression \
		mr.num_iter=500 \
		mr.num_examples=6000 \
		mr.num_features=$(mr_nfeat_$(1)) \
		mr.use_decomposition_features=true \
		mr.use_extra_features=true \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/mr5/$(1)_$(2)_merge.txt~ loadbuild/mr5/$(1)_$(2)_merge.txt

mr5: loadbuild/mr5/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_mr5,$(size),$(type)))))

