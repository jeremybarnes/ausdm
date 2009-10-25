/* decomposition.h                                                    -*- C++ -*-
   Jeremy Barnes, 7 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Various schemes to decompose the training data.
*/

#ifndef __ausdm__decomposition_h__
#define __ausdm__decomposition_h__


#include "db/persistent_fwd.h"
#include "data.h"


/*****************************************************************************/
/* DECOMPOSITION                                                             */
/*****************************************************************************/

struct Decomposition {
    virtual ~Decomposition() {}

    virtual distribution<float>
    decompose(const distribution<float> & vals) const = 0;

    void poly_serialize(ML::DB::Store_Writer & store) const;
    static boost::shared_ptr<Decomposition>
    poly_reconstitute(ML::DB::Store_Reader & store);

    void save(const std::string & filename) const;
    static boost::shared_ptr<Decomposition> load(const std::string & filename);

    virtual void serialize(ML::DB::Store_Writer & store) const = 0;
    virtual void reconstitute(ML::DB::Store_Reader & store) = 0;

    virtual std::string class_id() const = 0;
};


#endif /* __ausdm__decomposition_h__ */
