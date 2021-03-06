/* blender.h                                                       -*- C++ -*-
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Abstract class for a blender.
*/

#ifndef __ausdm__blender_h__
#define __ausdm__blender_h__


#include <boost/shared_ptr.hpp>
#include "data.h"
#include "jml/utils/configuration.h"
#include "jml/stats/distribution.h"


extern __thread float correct_prediction;


/*****************************************************************************/
/* BLENDER                                                                   */
/*****************************************************************************/

struct Blender {
    Blender();

    virtual ~Blender();

    virtual void configure(const ML::Configuration & config,
                           const std::string & name,
                           int random_seed,
                           Target target) = 0;
    
    virtual void init(const Data & training_data,
                      const ML::distribution<float> & example_weights) = 0;

    virtual float predict(const ML::distribution<float> & models) const = 0;

    virtual std::string explain(const ML::distribution<float> & models) const;
};


/*****************************************************************************/
/* LINEAR_BLENDER                                                            */
/*****************************************************************************/

struct Linear_Blender : public Blender {
    Linear_Blender();

    virtual ~Linear_Blender();

    virtual void configure(const ML::Configuration & config,
                           const std::string & name,
                           int random_seed,
                           Target target);
    
    virtual void init(const Data & training_data,
                      const ML::distribution<float> & example_weights);

    virtual float predict(const ML::distribution<float> & models) const;

    std::string mode;
    int num_models;

    ML::distribution<float> model_weights;
};


/*****************************************************************************/
/* UTILITY FUNCTIONS                                                         */
/*****************************************************************************/

boost::shared_ptr<Blender>
get_blender(const ML::Configuration & config,
            const std::string & name,
            const Data & data,
            const ML::distribution<float> & example_weights,
            int random_seed,
            Target target);

   
#endif /* __ausdm__blender_h__ */
