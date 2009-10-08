/* boosting_blender.h                                              -*- C++ -*-
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender that uses a boosting-like algorithm to choose a linear blender.
*/

#ifndef __ausdm__boosting_blender_h__
#define __ausdm__boosting_blender_h__


#include "blender.h"


/*****************************************************************************/
/* BOOSTING_BLENDER                                                          */
/*****************************************************************************/

struct Boosting_Blender : public Blender {

    Boosting_Blender();

    virtual ~Boosting_Blender();
    
    virtual void configure(const ML::Configuration & config,
                           const std::string & name,
                           int random_seed,
                           Target target);
    
    virtual void init(const Data & training_data,
                      const ML::distribution<float> & example_weights);

    virtual float predict(const ML::distribution<float> & models) const;

    std::vector<boost::shared_ptr<Blender> > submodels;
    ML::distribution<float> weights;

    ML::Configuration config;
    std::string weaklearner_name;
    Target target;
    int random_seed;

    int num_iter;
};

#endif /* __ausdm__boosting_blender_h__ */

