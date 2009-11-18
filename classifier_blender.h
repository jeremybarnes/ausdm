/* classifier_blender.h                                            -*- C++ -*-
   Jeremy Barnes, 1 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Dump the data off to a classifier.
*/

#ifndef __ausdm__classifier_blender_h__
#define __ausdm__classifier_blender_h__

#include "blender.h"
#include "boosting/dense_features.h"
#include "boosting/classifier.h"
#include "boosting/probabilizer.h"


/*****************************************************************************/
/* CLASSIFIER_BLENDER                                                        */
/*****************************************************************************/

struct Classifier_Blender : public Blender {

    Classifier_Blender();

    virtual ~Classifier_Blender();

    virtual void configure(const ML::Configuration & config,
                           const std::string & name,
                           int random_seed,
                           Target target);
    
    virtual void init(const Data & training_data,
                      const ML::distribution<float> & example_weights);

    virtual float predict(const ML::distribution<float> & models) const;

    virtual std::string explain(const ML::distribution<float> & models) const;

    virtual distribution<float>
    get_features(const ML::distribution<float> & models) const;

    virtual distribution<float>
    get_features(const ML::distribution<float> & models,
                 const ML::distribution<float> & decomp,
                 const Target_Stats & stats) const;

    virtual boost::shared_ptr<ML::Dense_Feature_Space>
    feature_space() const;
    
    std::string trainer_config_file;
    std::string trainer_name;
    
    int random_seed;

    boost::shared_ptr<ML::Dense_Feature_Space> fs;
    boost::shared_ptr<ML::Classifier_Impl>     classifier;
    boost::shared_ptr<ML::GLZ_Probabilizer>    probabilizer;

    Target target;

    const Decomposition * decomposition;
    int nm;
    int nv;
    int ndecomposed;
    std::vector<int> recomposition_orders;
    std::vector<Model_Stats> model_stats;
    std::vector<std::string> model_names;
};

#endif /* __ausdm__classifier_blender_h__ */


