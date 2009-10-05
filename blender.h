/* blender.h                                                       -*- C++ -*-
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Abstract class for a blender.
*/

#ifndef __ausdm__blender_h__
#define __ausdm__blender_h__


#include <boost/shared_ptr.hpp>
#include "data.h"
#include "utils/configuration.h"
#include "stats/distribution.h"


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
    
    virtual void init(const Data & training_data) = 0;

    virtual float predict(const ML::distribution<float> & models) const = 0;
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
    
    virtual void init(const Data & training_data);

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
            int random_seed,
            Target target);

   
#endif /* __ausdm__blender_h__ */
