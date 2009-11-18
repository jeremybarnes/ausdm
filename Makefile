-include local.mk

default: all
.PHONY: default

BUILD   ?= build
ARCH    ?= $(shell uname -m)
OBJ     := $(BUILD)/$(ARCH)/obj
BIN     := $(BUILD)/$(ARCH)/bin
TESTS   := $(BUILD)/$(ARCH)/tests
SRC     := .

JML_TOP := jml

include $(JML_TOP)/arch/$(ARCH).mk

CXXFLAGS += -Ijml -Wno-deprecated
CXXLINKFLAGS += -Ljml/../build/$(ARCH)/bin -Wl,--rpath,jml/../build/$(ARCH)/bin

ifeq ($(MAKECMDGOALS),failed)
include .target.mk
failed:
        +make $(FAILED) $(GOALS)
else

include $(JML_TOP)/functions.mk
include $(JML_TOP)/rules.mk

$(shell echo GOALS := $(MAKECMDGOALS) > .target.mk)
endif



LIBAUSDM_SOURCES := \
	data.cc blender.cc boosting_blender.cc gated_blender.cc \
	decomposition.cc svd_decomposition.cc dnae_decomposition.cc \
	deep_net_blender.cc multiple_regression_blender.cc utils.cc \
	classifier_blender.cc

LIBAUSDM_LINK := \
	utils ACE boost_date_time-mt db arch boosting algebra neural

$(eval $(call library,ausdm,$(LIBAUSDM_SOURCES),$(LIBAUSDM_LINK)))

$(eval $(call add_sources,exception_hook.cc))

$(eval $(call program,ausdm,ausdm utils ACE boost_program_options-mt db arch boosting neural,ausdm.cc exception_hook.cc,tools))

$(eval $(call program,decompose,ausdm utils ACE boost_program_options-mt db arch boosting neural,decompose.cc exception_hook.cc,tools))

$(eval $(call program,dnae_decomposition_test,ausdm utils ACE boost_program_options-mt db arch boosting neural,dnae_decomposition_test.cc exception_hook.cc,tools))

$(eval $(call test,dnae_unit_tests,ausdm,boost))

include loadbuild.mk
