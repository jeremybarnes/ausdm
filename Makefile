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
	decomposition.cc svd_decomposition.cc dnae_decomposition.cc

LIBAUSDM_LINK := \
	utils ACE boost_date_time-mt db arch boosting svdlibc algebra

$(eval $(call library,ausdm,$(LIBAUSDM_SOURCES),$(LIBAUSDM_LINK)))

$(eval $(call add_sources,exception_hook.cc))

$(eval $(call program,ausdm,ausdm utils ACE boost_program_options-mt db arch boosting svdlibc,ausdm.cc exception_hook.cc,tools))

$(eval $(call program,decompose,ausdm utils ACE boost_program_options-mt db arch boosting svdlibc,decompose.cc exception_hook.cc,tools))

$(eval $(call include_sub_makes,svdlibc))

include loadbuild.mk
