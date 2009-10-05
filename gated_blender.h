/* gated_blender.h                                              -*- C++ -*-
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender that uses gating to control which algorithms talk.
*/

#ifndef __ausdm__gated_blender_h__
#define __ausdm__gated_blender_h__


#include "blender.h"
#include "boosting/dense_features.h"
#include "algebra/irls.h"


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
                           int random_seed,
                           Target target);
    
    virtual void init(const Data & training_data);

    virtual distribution<float>
    conf(const ML::distribution<float> & models) const;
    
    virtual float predict(const ML::distribution<float> & models) const;

    void train_model(int model, const Data & training_data);

    distribution<float>
    get_model_features(int model,
                       const distribution<float> & model_outputs,
                       const distribution<double> & target_singular,
                       const Target_Stats & stats) const;

    ML::Link_Function link_function;

    const Data * data;
    std::vector<ML::distribution<float> > model_coefficients;
    Target target;

    bool debug_predict;
};

#endif /* __ausdm__gated_blender_h__ */

