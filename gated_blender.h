/* gated_blender.h                                              -*- C++ -*-
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender that uses gating to control which algorithms talk.
*/

#ifndef __ausdm__gated_blender_h__
#define __ausdm__gated_blender_h__


#include "blender.h"
#include "boosting/dense_features.h"



/*****************************************************************************/
/* GATED_BLENDER                                                             */
/*****************************************************************************/

struct Gated_Blender : public Blender {

    Gated_Blender();

    virtual ~Gated_Blender();

    virtual boost::shared_ptr<ML::Dense_Feature_Space>
    conf_feature_space() const;
    
    virtual void configure(const ML::Configuration & config,
                           const std::string & name,
                           int random_seed);
    
    virtual void init(const Data & training_data);

    virtual distribution<float>
    conf(const ML::distribution<float> & models) const;
    
    virtual float predict(const ML::distribution<float> & models) const;

    const Data * data;
};

#endif /* __ausdm__gated_blender_h__ */

