/* decomposer.h                                                    -*- C++ -*-
   Jeremy Barnes, 7 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Various schemes to decompose the training data.
*/

#ifndef __ausdm__decomposer_h__
#define __ausdm__decomposer_h__


#include "db/persistent_fwd.h"
#include "data.h"


/*****************************************************************************/
/* DECOMPOSER                                                                */
/*****************************************************************************/

struct Decomposer {

    virtual distribution<float>
    decompose(const distribution<float> & vals) const = 0;

    void serialize(DB::Store_Writer & store) const;

    static boost::shared_ptr<Decomposer>
    reconstitute(DB::Store_Reader & store);

    virtual void serialize_data(DB::Store_Writer & store) const = 0;
    virtual void reconstitute_data(DB::Store_Reader & store) = 0;

    virtual std::string type() const = 0;
};


/*****************************************************************************/
/* SVD_DECOMPOSER                                                            */
/*****************************************************************************/

struct SVD_Decomposer : public Decomposer {
};


/*****************************************************************************/
/* DNAE_DECOMPOSER                                                           */
/*****************************************************************************/

struct DNAE_Decomposer : public Decomposer {
};


#endif /* __ausdm__decomposer_h__ */
