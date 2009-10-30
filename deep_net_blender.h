/* deep_net_blender.h                                              -*- C++ -*-
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender that trains on a deep neural network.
*/

#ifndef __ausdm__deep_net_blender_h__
#define __ausdm__deep_net_blender_h__


#include "blender.h"
#include "boosting/dense_features.h"
#include "boosting/classifier.h"
#include "algebra/irls.h"
#include "utils/filter_streams.h"
#include "dnae_decomposition.h"


/*****************************************************************************/
/* DEEP_NET_BLENDER                                                          */
/*****************************************************************************/

struct Deep_Net_Blender : public Blender {

    Deep_Net_Blender();

    virtual ~Deep_Net_Blender();

    virtual void configure(const ML::Configuration & config,
                           const std::string & name,
                           int random_seed,
                           Target target);
    
    virtual void init(const Data & training_data,
                      const ML::distribution<float> & example_weights);

    virtual float predict(const ML::distribution<float> & models) const;

    virtual std::string explain(const ML::distribution<float> & models) const;

    ML::Configuration config;

    std::string model_base;
    int random_seed;

    ML::DNAE_Stack stack;
};

#endif /* __ausdm__deep_net_blender_h__ */

