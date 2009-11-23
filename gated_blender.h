/* gated_blender.h                                              -*- C++ -*-
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender that uses gating to control which algorithms talk.
*/

#ifndef __ausdm__gated_blender_h__
#define __ausdm__gated_blender_h__


#include "blender.h"
#include "boosting/dense_features.h"
#include "boosting/classifier.h"
#include "boosting/thread_context.h"
#include "algebra/irls.h"
#include "utils/filter_streams.h"


/*****************************************************************************/
/* GATED_BLENDER                                                             */
/*****************************************************************************/

struct Gated_Blender : public Blender {

    Gated_Blender();

    virtual ~Gated_Blender();

    virtual boost::shared_ptr<ML::Dense_Feature_Space>
    conf_feature_space() const;
    
    virtual boost::shared_ptr<ML::Dense_Feature_Space>
    blend_feature_space() const;
    
    virtual void configure(const ML::Configuration & config,
                           const std::string & name,
                           int random_seed,
                           Target target);
    
    virtual void init(const Data & training_data,
                      const ML::distribution<float> & example_weights);

    virtual distribution<float>
    conf(const ML::distribution<float> & models,
         const ML::distribution<float> & target_singular,
         const Target_Stats & stats) const;
    
    virtual float predict(const ML::distribution<float> & models) const;

    virtual std::string explain(const ML::distribution<float> & models) const;

    void train_conf(int model,
                    const Data & training_data,
                    const Data & testing_data,
                    const ML::distribution<float> & example_weights);

    distribution<float>
    get_conf_features(int model,
                      const distribution<float> & model_outputs,
                      const distribution<float> & target_singular,
                      const Target_Stats & stats) const;

    distribution<float>
    get_blend_features(const distribution<float> & model_outputs,
                       const distribution<float> & model_conf,
                       const distribution<float> & target_singular,
                       const Target_Stats & stats) const;

    distribution<float>
    train_blender_model(const Data & data,
                        ML::Thread_Context & thread_context,
                        int num_examples,
                        std::vector<distribution<float> *>
                            & example_features) const;

    ML::Link_Function link_function, blend_link_function;
    int num_models_to_train;
    bool debug_conf;
    bool debug_predict;
    bool blend_with_classifier;
    
    std::vector<int> recomposition_sizes;

    std::vector<ML::distribution<float> > model_coefficients;
    std::vector<distribution<double> > blend_coefficients;
    boost::shared_ptr<ML::Dense_Feature_Space> blender_fs;
    boost::shared_ptr<ML::Classifier_Impl> blender;
    Target target;
    int random_seed;

    std::string blender_trainer_config_file;
    std::string blender_trainer_name;

    const Data * data;

    std::string dump_predict_features, dump_training_features;
    mutable ML::filter_ostream predict_feature_file, training_feature_file;
    mutable Lock predict_feature_lock;
};

#endif /* __ausdm__gated_blender_h__ */

