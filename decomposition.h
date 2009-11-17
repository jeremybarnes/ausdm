/* decomposition.h                                                    -*- C++ -*-
   Jeremy Barnes, 7 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Various schemes to decompose the training data.
*/

#ifndef __ausdm__decomposition_h__
#define __ausdm__decomposition_h__


#include "db/persistent_fwd.h"
#include "data.h"
#include "utils/configuration.h"


/*****************************************************************************/
/* DECOMPOSITION                                                             */
/*****************************************************************************/

struct Decomposition {
    virtual ~Decomposition() {}

    virtual distribution<float>
    decompose(const distribution<float> & model_outputs) const = 0;

    /** Perform a decomposition to the given order (number of values) and
        then reconstruct, returning the reconstructed version.  If order is
        -1, then it will be done to the natural order of the decomposition.
    */
    virtual distribution<float>
    recompose(const distribution<float> & model_outputs,
              const distribution<float> & decomposition,
              int order) const = 0;
    
    /** Which order values should be passed to recompose? */
    virtual std::vector<int> recomposition_orders() const = 0;

    void poly_serialize(ML::DB::Store_Writer & store) const;

    static boost::shared_ptr<Decomposition>
    poly_reconstitute(ML::DB::Store_Reader & store);

    void save(const std::string & filename) const;
    static boost::shared_ptr<Decomposition> load(const std::string & filename);

    static boost::shared_ptr<Decomposition>
    create(const std::string & type);

    static bool known_type(const std::string & type);

    virtual void serialize(ML::DB::Store_Writer & store) const = 0;
    virtual void reconstitute(ML::DB::Store_Reader & store) = 0;

    virtual std::string class_id() const = 0;

    virtual void train(const Data & training_data,
                       const Data & testing_data,
                       const ML::Configuration & config) = 0;

    /** The dimensionality of the decomposition */
    virtual size_t size() const = 0;
};


#endif /* __ausdm__decomposition_h__ */
