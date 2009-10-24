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


/*****************************************************************************/
/* SVD_DECOMPOSITION                                                         */
/*****************************************************************************/

struct SVD_Decomposition : public Decomposition {

    SVD_Decomposition();

    /// Singular values of SVD on rankings; full rank
    distribution<float> singular_values;
    boost::multi_array<float, 2> lvectors;
    boost::multi_array<float, 2> rvectors;

    int order;
    int nvalues;
    int nm, nx;

    /// Subset of singular values for the given order
    distribution<float> singular_values_order;
        
    /// Singular representation of each model; reduced to order
    std::vector<distribution<float> > singular_models;

    void train(const std::vector<distribution<float> > & data,
               int order = -1);

    // Set the order of the model and extract things based upon it
    void extract_for_order(int order);

    /// Apply the decomposition, returning the decomposed element
    virtual distribution<float>
    decompose(const distribution<float> & vals) const;

    virtual void serialize(ML::DB::Store_Writer & store) const;

    virtual void reconstitute(ML::DB::Store_Reader & store);

    virtual std::string class_id() const;
};



/*****************************************************************************/
/* DNAE_DECOMPOSITION                                                        */
/*****************************************************************************/

struct DNAE_Decomposition : public Decomposition {
};


#endif /* __ausdm__decomposition_h__ */
