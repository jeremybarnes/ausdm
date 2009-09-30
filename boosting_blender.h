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

struct Boosting_Blender : public Linear_Blender {

    Boosting_Blender();

    virtual ~Boosting_Blender();
    
    virtual void configure(const ML::Configuration & config,
                           const std::string & name,
                           int random_seed);
    
    virtual void init(const Data & training_data);
};

#endif /* __ausdm__boosting_blender_h__ */

