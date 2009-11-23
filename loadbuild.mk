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
# NOTE: these were originally created with a buggy auto encoder implementation

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

define do_dnae_decomposition2
loadbuild/$(1)_$(2)_DNAE2.dat: loadbuild/.dir_exists
	/usr/bin/time \
	$(BIN)/decompose \
		-T DNAE \
		-S $(1) -t $(2) \
		niter=800 \
		prob_cleared=0.1 \
		layer_sizes=250,150,100,50 \
		minibatch_size=256 \
		test_every=$(dnae_te_$(1)) \
		sample_proportion=$(dnae_sp_$(1)) \
		learning_rate=$(dnae_lr_$(1)) \
		-o $$@~ \
	2>&1 | tee $$@.log
	mv $$@~ $$@

DNAE2: loadbuild/$(1)_$(2)_DNAE2.dat
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_dnae_decomposition2,$(size),$(type)))))

define do_dnae_decomposition3
loadbuild/$(1)_$(2)_DNAE3.dat: loadbuild/.dir_exists
	/usr/bin/time \
	$(BIN)/decompose \
		-T DNAE \
		-S $(1) -t $(2) \
		niter=400 \
		prob_cleared=0.1 \
		layer_sizes=250,150,100,50 \
		minibatch_size=256 \
		test_every=$(dnae_te_$(1)) \
		sample_proportion=$(dnae_sp_$(1)) \
		learning_rate=$(dnae_lr_$(1)) \
		stack_backprop_iter=200 \
		-o $$@~ \
	2>&1 | tee $$@.log
	mv $$@~ $$@

DNAE3: loadbuild/$(1)_$(2)_DNAE3.dat
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_dnae_decomposition3,$(size),$(type)))))


# Top-1 model
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_top1
loadbuild/top1/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/top1/.dir_exists
	$(BIN)/ausdm \
		-S $(1) -t $(2) \
		-T 0.20 \
		--decomposition "" \
		-o loadbuild/top1/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-n top1 top1.type=linear top1.mode=best_n top1.num_models=1 \
		2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/top1/$(1)_$(2)_merge.txt~ loadbuild/top1/$(1)_$(2)_merge.txt

top1: loadbuild/top1/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_top1,$(size),$(type)))))

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
		--decomposition loadbuild/$(1)_$(2)_DNAE.dat \
		-o loadbuild/gated3/$(1)_$(2)_merge.txt~ \
		-O $$@~ -n gated \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/gated3/$(1)_$(2)_merge.txt~ loadbuild/gated3/$(1)_$(2)_merge.txt

gated3: loadbuild/gated3/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_gated3,$(size),$(type)))))

# Gated model with denoising autoencoder decomposition (fixed)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_gated4
loadbuild/gated4/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/gated4/.dir_exists loadbuild/$(1)_$(2)_DNAE2.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition loadbuild/$(1)_$(2)_DNAE2.dat \
		-o loadbuild/gated4/$(1)_$(2)_merge.txt~ \
		-O $$@~ -n gated \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/gated4/$(1)_$(2)_merge.txt~ loadbuild/gated4/$(1)_$(2)_merge.txt

gated4: loadbuild/gated4/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_gated4,$(size),$(type)))))


# Gated model with denoising autoencoder decomposition (fixed)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_gated5
loadbuild/gated5/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/gated5/.dir_exists loadbuild/$(1)_$(2)_DNAE3.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition loadbuild/$(1)_$(2)_DNAE3.dat \
		-o loadbuild/gated5/$(1)_$(2)_merge.txt~ \
		-O $$@~ -n gated \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/gated5/$(1)_$(2)_merge.txt~ loadbuild/gated5/$(1)_$(2)_merge.txt

gated5: loadbuild/gated5/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_gated5,$(size),$(type)))))


# Gated model with denoising autoencoder decomposition, regression tree final
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_gated6
loadbuild/gated6/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/gated6/.dir_exists loadbuild/$(1)_$(2)_DNAE3.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition loadbuild/$(1)_$(2)_DNAE3.dat \
		gated.blender.trainer_name=regression_trees \
		-o loadbuild/gated6/$(1)_$(2)_merge.txt~ \
		-O $$@~ -n gated \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/gated6/$(1)_$(2)_merge.txt~ loadbuild/gated6/$(1)_$(2)_merge.txt

gated6: loadbuild/gated6/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,rmse,$(eval $(call do_gated6,$(size),$(type)))))

# Gated model with denoising autoencoder decomposition, regression tree final
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_gated7
loadbuild/gated7/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/gated7/.dir_exists loadbuild/$(1)_$(2)_DNAE3.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition loadbuild/$(1)_$(2)_DNAE3.dat \
		gated.blend_with_classifier=false \
		-o loadbuild/gated7/$(1)_$(2)_merge.txt~ \
		-O $$@~ -n gated \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/gated7/$(1)_$(2)_merge.txt~ loadbuild/gated7/$(1)_$(2)_merge.txt

gated7: loadbuild/gated7/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,rmse,$(eval $(call do_gated7,$(size),$(type)))))



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

# Multiple regression model 5 (DNAE decomposition features + extra features)
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

# Multiple regression model 6 (DNAE decomposition features 2)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_mr6
loadbuild/mr6/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/mr6/.dir_exists loadbuild/$(1)_$(2)_DNAE2.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition loadbuild/$(1)_$(2)_DNAE2.dat \
		-o loadbuild/mr6/$(1)_$(2)_merge.txt~ \
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
	mv loadbuild/mr6/$(1)_$(2)_merge.txt~ loadbuild/mr6/$(1)_$(2)_merge.txt

mr6: loadbuild/mr6/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_mr6,$(size),$(type)))))

# Multiple regression model 5 (DNAE decomposition features + extra features)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_mr7
loadbuild/mr7/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/mr7/.dir_exists loadbuild/$(1)_$(2)_DNAE2.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition loadbuild/$(1)_$(2)_DNAE2.dat \
		-o loadbuild/mr7/$(1)_$(2)_merge.txt~ \
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
	mv loadbuild/mr7/$(1)_$(2)_merge.txt~ loadbuild/mr7/$(1)_$(2)_merge.txt

mr7: loadbuild/mr7/$(1)_$(2)_official.txt
endef

# Multiple regression model 8 (DNAE decomposition features 3)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_mr8
loadbuild/mr8/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/mr8/.dir_exists loadbuild/$(1)_$(2)_DNAE3.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition loadbuild/$(1)_$(2)_DNAE3.dat \
		-o loadbuild/mr8/$(1)_$(2)_merge.txt~ \
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
	mv loadbuild/mr8/$(1)_$(2)_merge.txt~ loadbuild/mr8/$(1)_$(2)_merge.txt

mr8: loadbuild/mr8/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_mr8,$(size),$(type)))))

# Multiple regression model 9 (DNAE decomposition features 3 + extra features)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_mr9
loadbuild/mr9/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/mr9/.dir_exists loadbuild/$(1)_$(2)_DNAE3.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition loadbuild/$(1)_$(2)_DNAE3.dat \
		-o loadbuild/mr9/$(1)_$(2)_merge.txt~ \
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
	mv loadbuild/mr9/$(1)_$(2)_merge.txt~ loadbuild/mr9/$(1)_$(2)_merge.txt

mr9: loadbuild/mr9/$(1)_$(2)_official.txt
endef



$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_mr9,$(size),$(type)))))

# Deep net model 1
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_dn1
loadbuild/dn1/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/dn1/.dir_exists loadbuild/$(1)_$(2)_DNAE2.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "" \
		-o loadbuild/dn1/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-n dn1 \
		dn1.type=deep_net \
		dn1.niter=500 \
		dn1.model_base=loadbuild/$(1)_$(2)_DNAE2.dat \
		dn1.sample_proportion=$(dnae_sp_$(1)) \
		dn1.learning_rate=0.05 \
		dn1.use_extra_features=true \
		dn1.hold_out=0.3 \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/dn1/$(1)_$(2)_merge.txt~ loadbuild/dn1/$(1)_$(2)_merge.txt

dn1: loadbuild/dn1/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_dn1,$(size),$(type)))))

# Deep net model 2 (no extra features)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_dn2
loadbuild/dn2/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/dn2/.dir_exists loadbuild/$(1)_$(2)_DNAE2.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "" \
		-o loadbuild/dn2/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-n dn2 \
		dn2.type=deep_net \
		dn2.niter=500 \
		dn2.model_base=loadbuild/$(1)_$(2)_DNAE2.dat \
		dn2.sample_proportion=$(dnae_sp_$(1)) \
		dn2.learning_rate=0.05 \
		dn2.use_extra_features=false \
		dn2.hold_out=0.3 \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/dn2/$(1)_$(2)_merge.txt~ loadbuild/dn2/$(1)_$(2)_merge.txt

dn2: loadbuild/dn2/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_dn2,$(size),$(type)))))

# Deep net model 3 (no extra features)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_dn3
loadbuild/dn3/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/dn3/.dir_exists loadbuild/$(1)_$(2)_DNAE3.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "" \
		-o loadbuild/dn3/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-n dn3 \
		dn3.type=deep_net \
		dn3.niter=500 \
		dn3.model_base=loadbuild/$(1)_$(2)_DNAE3.dat \
		dn3.sample_proportion=$(dnae_sp_$(1)) \
		dn3.learning_rate=0.2 \
		dn3.use_extra_features=false \
		dn3.hold_out=0.3 \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/dn3/$(1)_$(2)_merge.txt~ loadbuild/dn3/$(1)_$(2)_merge.txt

dn3: loadbuild/dn3/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_dn4,$(size),$(type)))))

# Deep net model 3 (no extra features)
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_dn4
loadbuild/dn4/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/dn4/.dir_exists loadbuild/$(1)_$(2)_DNAE3.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "" \
		-o loadbuild/dn4/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-n dn4 \
		dn4.type=deep_net \
		dn4.niter=800 \
		dn4.model_base=loadbuild/$(1)_$(2)_DNAE3.dat \
		dn4.sample_proportion=$(dnae_sp_$(1)) \
		dn4.learning_rate=0.05 \
		dn4.use_extra_features=true \
		dn4.hold_out=0.3 \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/dn4/$(1)_$(2)_merge.txt~ loadbuild/dn4/$(1)_$(2)_merge.txt

dn4: loadbuild/dn4/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,auc rmse,$(eval $(call do_dn4,$(size),$(type)))))

# Regression trees
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_rtrees
loadbuild/rtrees/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/rtrees/.dir_exists loadbuild/$(1)_$(2)_DNAE3.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "loadbuild/$(1)_$(2)_DNAE3.dat" \
		-o loadbuild/rtrees/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-n rtrees \
		rtrees.type=classifier \
		rtrees.trainer_name=regression_trees \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/rtrees/$(1)_$(2)_merge.txt~ loadbuild/rtrees/$(1)_$(2)_merge.txt

rtrees: loadbuild/rtrees/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,rmse,$(eval $(call do_rtrees,$(size),$(type)))))

# Multiple class
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_mclass
loadbuild/mclass/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/mclass/.dir_exists loadbuild/$(1)_$(2)_DNAE3.dat
	/usr/bin/time \
	$(BIN)/ausdm \
		-S $(1) -t $(2) -T 0.20 \
		--decomposition "loadbuild/$(1)_$(2)_DNAE3.dat" \
		-o loadbuild/mclass/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-n mclass \
		mclass.type=classifier \
		mclass.use_regression=false \
		mclass.trainer_name=bbdt3 \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/mclass/$(1)_$(2)_merge.txt~ loadbuild/mclass/$(1)_$(2)_merge.txt

mclass: loadbuild/mclass/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,rmse,$(eval $(call do_mclass,$(size),$(type)))))

merged_auc_models := dn1 dn2 dn3 dn4 gated gated2 gated3 gated4 gated5 mr1 mr2 mr3 mr4 mr5 mr6 mr7 mr8 mr9
merged_rmse_models := dn1 dn2 dn3 dn4 gated gated2 gated3 gated4 gated5 mr1 mr2 mr3 mr4 mr5 mr6 mr7 mr8 mr9 gated6 gated7 rtrees mclass

merged_nf_auc := 15
merged_nf_rmse := 15

merged_nx_S := 2200
merged_nx_M := 3000
merged_nx_L := 6000



# Merged model
# $(1): S, M or L (dataset size)
# $(2): auc or rmse

define do_merged
loadbuild/merged/$(1)_$(2)_official.txt: loadbuild/.dir_exists loadbuild/merged/.dir_exists loadbuild/$(1)_$(2)_DNAE3.dat
	/usr/bin/time \
	$(BIN)/merge \
		-S $(1) -t $(2) \
		--decomposition "" \
		-o loadbuild/merged/$(1)_$(2)_merge.txt~ \
		-O $$@~ \
		-r false \
		-n mr \
		mr.type=multiple_regression \
		mr.use_extra_features=false \
		mr.num_features=$(merged_nf_$(2)) \
		mr.num_examples=$(merged_nx_$(1)) \
		$(merged_$(2)_models) \
	2>&1 | tee $$@.log
	mv $$@~ $$@
	mv loadbuild/merged/$(1)_$(2)_merge.txt~ loadbuild/merged/$(1)_$(2)_merge.txt

merged: loadbuild/merged/$(1)_$(2)_official.txt
endef

$(foreach size,S M L,$(foreach type,rmse auc,$(eval $(call do_merged,$(size),$(type)))))
