/* svd_decomposition.h                                             -*- C++ -*-
   Jeremy Barnes, 24 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Decomposition using the SVD.
*/

#ifndef __ausdm__svd_decomposition_h__
#define __ausdm__svd_decomposition_h__

#include "decomposition.h"


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

    virtual void train(const Data & training_data,
                       const Data & testing_data,
                       const ML::Configuration & config);

    void train(const Data & data,
               int order = -1);

    // Set the order of the model and extract things based upon it
    void extract_for_order(int order);

    /// Apply the decomposition, returning the decomposed element
    virtual distribution<float>
    decompose(const distribution<float> & vals) const;

    virtual distribution<float>
    recompose(const distribution<float> & model_outputs,
              const distribution<float> & decomposition,
              int order) const;

    virtual std::vector<int> recomposition_orders() const;

    virtual void serialize(ML::DB::Store_Writer & store) const;

    virtual void reconstitute(ML::DB::Store_Reader & store);

    virtual std::string class_id() const;

    virtual size_t size() const { return order; }
};




#endif /* __ausdm__svd_decomposition_h__ */
