/* dnae_decomposition.h                                            -*- C++ -*-
   Jeremy Barnes, 24 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Denoising Auto Encoder decomposition.
*/

#ifndef __ausdm__dnae_decomposition_h__
#define __ausdm__dnae_decomposition_h__


#include "decomposition.h"
#include "neural/auto_encoder_stack.h"
#include "arch/threads.h"


typedef float LFloat;


namespace ML {


} // namespace ML


/*****************************************************************************/
/* DNAE_DECOMPOSITION                                                        */
/*****************************************************************************/

/** Object that decomposes a set of model predictions into a denoised
    autoencoder representation. */

struct DNAE_Decomposition : public Decomposition {

    DNAE_Decomposition();

    /// The auto-encoder stack that implements the decomposition
    ML::Auto_Encoder_Stack stack;

    /// Apply the decomposition, returning the decomposed element
    virtual distribution<float>
    decompose(const distribution<float> & model_outputs) const;

    virtual distribution<float>
    recompose(const distribution<float> & model_outputs,
              const distribution<float> & decomposition,
              int order = -1) const;

    /** Which order values should be passed to recompose? */
    virtual std::vector<int> recomposition_orders() const;

    virtual void serialize(ML::DB::Store_Writer & store) const;

    virtual void reconstitute(ML::DB::Store_Reader & store);

    virtual std::string class_id() const;

    virtual void train(const Data & training_data,
                       const Data & testing_data,
                       const ML::Configuration & config);

    bool operator == (const DNAE_Decomposition & other) const;
};

#endif /* __ausdm__dnaw_decomposition_h__ */
