CXX := g++
FC := gfortran
NODEJS_ENABLED:=1
PYTHON_ENABLED:=1


-include local.mk

default: all
.PHONY: default

BUILD   ?= build
ARCH    ?= $(shell uname -m)
OBJ     := $(BUILD)/$(ARCH)/obj
BIN     := $(BUILD)/$(ARCH)/bin
TESTS   := $(BUILD)/$(ARCH)/tests
SRC     := .
TEST_TMP := $(TESTS)

JML_TOP := jml
INCLUDE := -I.

export JML_TOP
export BIN
export BUILD
export TEST_TMP

include $(JML_TOP)/arch/$(ARCH).mk
CFLAGS += -fno-strict-overflow

CXXFLAGS += -Wno-deprecated -Wno-uninitialized -Winit-self -fno-omit-frame-pointer -I.
CXXLINKFLAGS += -Wl,--copy-dt-needed-entries

ifeq ($(MAKECMDGOALS),failed)
include .target.mk
failed:
        +make $(FAILED) $(GOALS)
else

include $(JML_TOP)/functions.mk
include $(JML_TOP)/rules.mk

$(shell echo GOALS := $(MAKECMDGOALS) > .target.mk)
endif

$(eval $(call include_sub_make,jml,jml,jml.mk))

CWD:=

LIBAUSDM_SOURCES := \
	data.cc blender.cc boosting_blender.cc gated_blender.cc \
	decomposition.cc svd_decomposition.cc dnae_decomposition.cc \
	deep_net_blender.cc multiple_regression_blender.cc \
	classifier_blender.cc

LIBAUSDM_LINK := \
	utils ACE boost_date_time-mt db arch boosting algebra neural

$(eval $(call library,ausdm,$(LIBAUSDM_SOURCES),$(LIBAUSDM_LINK)))

$(eval $(call program,ausdm,ausdm utils ACE boost_program_options-mt db arch boosting neural,ausdm.cc exception_hook.cc,tools))

$(eval $(call program,decompose,ausdm utils ACE boost_program_options-mt db arch boosting neural,decompose.cc exception_hook.cc,tools))

$(eval $(call program,merge,ausdm utils ACE boost_program_options-mt db arch boosting neural,merge.cc exception_hook.cc,tools))

$(eval $(call program,dnae_decomposition_test,ausdm utils ACE boost_program_options-mt db arch boosting neural,dnae_decomposition_test.cc exception_hook.cc,tools))

$(eval $(call test,dnae_unit_tests,ausdm,boost))

$(eval $(call program,compare_decompositions,ausdm utils ACE boost_program_options-mt db arch boosting neural,compare_decompositions.cc exception_hook.cc,tools))

include loadbuild.mk

