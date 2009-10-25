/* dnae_decomposition.h                                            -*- C++ -*-
   Jeremy Barnes, 24 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Denoising Auto Encoder decomposition.
*/

#ifndef __ausdm__dnae_decomposition_h__
#define __ausdm__dnaw_decomposition_h__

#include "decomposition.h"
#include "boosting/layer.h"

typedef double LFloat;

namespace ML {

/*****************************************************************************/
/* TWOWAY_LAYER                                                              */
/*****************************************************************************/

/** A perceptron layer that has both a forward and a reverse direction.  It's
    as suck both a discriminative model (in the forward direction) and a
    generative model (in the reverse direction).
*/

struct Twoway_Layer : public Dense_Layer<LFloat> {
    Twoway_Layer();

    Twoway_Layer(size_t inputs, size_t outputs,
                 Transfer_Function_Type transfer,
                 Thread_Context & context,
                 float limit = -1.0);

    Twoway_Layer(size_t inputs, size_t outputs,
                 Transfer_Function_Type transfer);

    /// Bias for the reverse direction
    distribution<LFloat> ibias;

    distribution<double> iapply(const distribution<double> & output) const;
    distribution<float> iapply(const distribution<float> & output) const;

    distribution<double> ipreprocess(const distribution<double> & output) const;
    distribution<float> ipreprocess(const distribution<float> & output) const;

    distribution<double> iactivation(const distribution<double> & output) const;
    distribution<float> iactivation(const distribution<float> & output) const;

    distribution<double>
    itransfer(const distribution<double> & activation) const;
    distribution<float>
    itransfer(const distribution<float> & activation) const;

    distribution<double> iderivative(const distribution<double> & input) const;
    distribution<float> iderivative(const distribution<float> & input) const;

    void update(const Twoway_Layer & updates, double learning_rate);

    virtual void random_fill(float limit, Thread_Context & context);

    virtual void zero_fill();

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);
    
    /** Trains a single iteration on the given data with the selected
        parameters.  Returns a moving estimate of the RMSE on the
        training set. */
    double train_iter(const std::vector<distribution<float> > & data,
                      float prob_cleared,
                      Thread_Context & thread_context,
                      int minibatch_size, float learning_rate);

    /** Tests on the given dataset, returning the exact and noisy RMSE.  If
        data_out is non-empty, then it will also fill it in with the
        hidden representations for each of the inputs (with no noise added).
        This information can be used to train the next layer. */
    std::pair<double, double>
    test_and_update(const std::vector<distribution<float> > & data_in,
                    std::vector<distribution<float> > & data_out,
                    float prob_cleared,
                    Thread_Context & thread_context) const;

    /** Tests on the given dataset, returning the exact and noisy RMSE. */
    std::pair<double, double>
    test(const std::vector<distribution<float> > & data,
         float prob_cleared,
         Thread_Context & thread_context) const
    {
        std::vector<distribution<float> > dummy;
        return test_and_update(data, dummy, prob_cleared, thread_context);
    }
};


/*****************************************************************************/
/* DNAE_STACK                                                                */
/*****************************************************************************/

/** A stack of denoising autoencoder layers, to create a deep encoder. */

struct DNAE_Stack : public std::vector<Twoway_Layer> {

    distribution<float> apply(const distribution<float> & input) const;
    distribution<double> apply(const distribution<double> & input) const;

    distribution<float> iapply(const distribution<float> & output) const;
    distribution<double> iapply(const distribution<double> & output) const;

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    void train(const std::vector<distribution<float> > & training_data,
               const std::vector<distribution<float> > & testing_data,
               const Configuration & config,
               ML::Thread_Context & thread_context);

    /** Tests on both pristine and noisy inputs.  The first returned is the
        error on pristine inputs.  The second is the error on noisy inputs.
        the prob_cleared parameter describes the probability that noise will
        be added to any given input.
    */
    std::pair<double, double>
    test(const std::vector<distribution<float> > & data,
         float prob_cleared,
         ML::Thread_Context & thread_context) const;
};

} // namespace ML


/*****************************************************************************/
/* DNAE_DECOMPOSITION                                                        */
/*****************************************************************************/

/** Object that decomposes a set of model predictions into a denoised
    autoencoder representation. */

struct DNAE_Decomposition : public Decomposition {

    ML::DNAE_Stack stack;

    /// Apply the decomposition, returning the decomposed element
    virtual distribution<float>
    decompose(const distribution<float> & vals) const;

    virtual void serialize(ML::DB::Store_Writer & store) const;

    virtual void reconstitute(ML::DB::Store_Reader & store);

    virtual std::string class_id() const;

    virtual void train(const Data & training_data,
                       const Data & testing_data,
                       const ML::Configuration & config);
};





#endif /* __ausdm__dnaw_decomposition_h__ */
