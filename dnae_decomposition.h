/* dnae_decomposition.h                                            -*- C++ -*-
   Jeremy Barnes, 24 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Denoising Auto Encoder decomposition.
*/

#ifndef __ausdm__dnae_decomposition_h__
#define __ausdm__dnae_decomposition_h__

#include "decomposition.h"
#include "neural/layer.h"
#include "arch/threads.h"

typedef float LFloat;


namespace ML {


/*****************************************************************************/
/* DENSE_MISSING_LAYER                                                       */
/*****************************************************************************/

struct Dense_Missing_Layer : public Dense_Layer<LFloat> {
    typedef Dense_Layer<LFloat> Base;

    Dense_Missing_Layer();

    Dense_Missing_Layer(bool use_dense_missing,
                        size_t inputs, size_t outputs,
                        Transfer_Function_Type transfer,
                        Thread_Context & context,
                        float limit = -1.0);

    Dense_Missing_Layer(bool use_dense_missing,
                        size_t inputs, size_t outputs,
                        Transfer_Function_Type transfer);

    /// Do we use the dense missing values?
    bool use_dense_missing;

    /// Values to use for input when the value is missing (NaN)
    std::vector<distribution<LFloat> > missing_activations;

    virtual void preprocess(const float * input,
                            float * preprocessed) const;

    virtual void preprocess(const double * input,
                            double * preprocessed) const;

    using Layer::preprocess;

    virtual void activation(const float * preprocessed,
                            float * activation) const;

    virtual void activation(const double * preprocessed,
                            double * activation) const;

    using Layer::activation;

    virtual void random_fill(float limit, Thread_Context & context);

    virtual void zero_fill();

    virtual size_t parameter_count() const;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);

    virtual std::string print() const;

    virtual Dense_Missing_Layer * make_copy() const
    {
        return new Dense_Missing_Layer(*this);
    }

    bool operator == (const Dense_Missing_Layer & other) const;
};

IMPL_SERIALIZE_RECONSTITUTE(Dense_Missing_Layer);


/*****************************************************************************/
/* TWOWAY_LAYER_UPDATES                                                      */
/*****************************************************************************/

struct Twoway_Layer;

struct Twoway_Layer_Updates {

    Twoway_Layer_Updates();

    Twoway_Layer_Updates(bool train_generative,
                         const Twoway_Layer & layer);

    Twoway_Layer_Updates(bool use_dense_missing,
                         bool train_generative,
                         int inputs, int outputs);

    void zero_fill();

    void init(bool train_generative,
              const Twoway_Layer & layer);

    void init(bool use_dense_missing, bool train_generative,
              int inputs, int outputs);

    int inputs() const { return weights.shape()[0]; }
    int outputs() const { return weights.shape()[1]; }

    Twoway_Layer_Updates & operator += (const Twoway_Layer_Updates & other);

    bool use_dense_missing;
    bool train_generative;
    boost::multi_array<double, 2> weights;
    distribution<double> bias;
    distribution<double> missing_replacements;
    std::vector<distribution<double> > missing_activations;
    distribution<double> ibias;
    distribution<double> iscales;
    distribution<double> hscales;
};


/*****************************************************************************/
/* TWOWAY_LAYER                                                              */
/*****************************************************************************/

typedef Dense_Missing_Layer Twoway_Layer_Base;


/** A perceptron layer that has both a forward and a reverse direction.  It's
    both a discriminative model (in the forward direction) and a generative
    model (in the reverse direction).
*/

struct Twoway_Layer : public Twoway_Layer_Base {
    typedef Twoway_Layer_Base Base;

    Twoway_Layer();

    Twoway_Layer(bool use_dense_missing,
                 size_t inputs, size_t outputs,
                 Transfer_Function_Type transfer,
                 Thread_Context & context,
                 float limit = -1.0);

    Twoway_Layer(bool use_dense_missing,
                 size_t inputs, size_t outputs,
                 Transfer_Function_Type transfer);

    /// Bias for the reverse direction
    distribution<LFloat> ibias;

    /// Scaling factors for the reverse direction
    distribution<LFloat> iscales;
    distribution<LFloat> hscales;

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

    void update(const Twoway_Layer_Updates & updates, double learning_rate);

    virtual void random_fill(float limit, Thread_Context & context);

    virtual void zero_fill();

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);

    /** Dump as ASCII.  This will be big. */
    virtual std::string print() const;
    
    /** Backpropagate the given example.  The gradient will be acculmulated in
        the output.  Fills in the errors for the next stage at input_errors. */
    void backprop_example(const distribution<double> & outputs,
                          const distribution<double> & output_deltas,
                          const distribution<double> & inputs,
                          distribution<double> & input_deltas,
                          Twoway_Layer_Updates & updates) const;

    /** Inverse direction backpropagation of the given example.  Again, the
        gradient will be acculmulated in the output.  Fills in the errors for
        the next stage at input_errors. */
    void ibackprop_example(const distribution<double> & outputs,
                           const distribution<double> & output_deltas,
                           const distribution<double> & inputs,
                           distribution<double> & input_deltas,
                           Twoway_Layer_Updates & updates) const;

    /** Trains a single iteration on the given data with the selected
        parameters.  Returns a moving estimate of the RMSE on the
        training set. */
    std::pair<double, double>
    train_iter(const std::vector<distribution<float> > & data,
               float prob_cleared,
               Thread_Context & thread_context,
               int minibatch_size, float learning_rate,
               int verbosity,
               float sample_proportion,
               bool randomize_order);

    /** Tests on the given dataset, returning the exact and noisy RMSE.  If
        data_out is non-empty, then it will also fill it in with the
        hidden representations for each of the inputs (with no noise added).
        This information can be used to train the next layer. */
    std::pair<double, double>
    test_and_update(const std::vector<distribution<float> > & data_in,
                    std::vector<distribution<float> > & data_out,
                    float prob_cleared,
                    Thread_Context & thread_context,
                    int verbosity) const;

    /** Tests on the given dataset, returning the exact and noisy RMSE. */
    std::pair<double, double>
    test(const std::vector<distribution<float> > & data,
         float prob_cleared,
         Thread_Context & thread_context,
         int verbosity) const
    {
        std::vector<distribution<float> > dummy;
        return test_and_update(data, dummy, prob_cleared, thread_context,
                               verbosity);
    }

    bool operator == (const Twoway_Layer & other) const;
};

IMPL_SERIALIZE_RECONSTITUTE(Twoway_Layer);


/*****************************************************************************/
/* DNAE_STACK_UPDATES                                                        */
/*****************************************************************************/

struct DNAE_Stack;

struct DNAE_Stack_Updates : public std::vector<Twoway_Layer_Updates> {

    DNAE_Stack_Updates();

    DNAE_Stack_Updates(const DNAE_Stack & stack);

    void init(const DNAE_Stack & stack);

    DNAE_Stack_Updates & operator += (const DNAE_Stack_Updates & updates);
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

    /** Train a single example.  Returns the RMSE in the first and the
        output value (which can be used to calculate the AUC) in the
        second.
    */
    std::pair<double, double>
    train_discrim_example(const distribution<float> & data,
                          float label,
                          DNAE_Stack_Updates & udpates) const;

    /** Trains a single iteration on the given data with the selected
        parameters.  Returns a moving estimate of the RMSE on the
        training set. */
    std::pair<double, double>
    train_discrim_iter(const std::vector<distribution<float> > & data,
                       const std::vector<float> & labels,
                       Thread_Context & thread_context,
                       int minibatch_size, float learning_rate,
                       int verbosity,
                       float sample_proportion,
                       bool randomize_order);

    /** Perform backpropagation given an error gradient.  Note that doing
        so will adversely affect the performance of the autoencoder, as
        the reverse weights aren't modified when performing this training.

        Returns the best training and testing error.
    */
    std::pair<double, double>
    train_discrim(const std::vector<distribution<float> > & training_data,
                  const std::vector<float> & training_labels,
                  const std::vector<distribution<float> > & testing_data,
                  const std::vector<float> & testing_labels,
                  const Configuration & config,
                  ML::Thread_Context & thread_context);

    /** Update given the learning rate and the gradient. */
    void update(const DNAE_Stack_Updates & updates, double learning_rate);
    
    /** Test the discriminative power of the network.  Returns the RMS error
        or AUC depending upon whether it's a regression or classification
        task.
    */
    std::pair<double, double>
    test_discrim(const std::vector<distribution<float> > & data,
                 const std::vector<float> & labels,
                 ML::Thread_Context & thread_context,
                 int verbosity);
    

    /** Train (unsupervised) as a stack of denoising autoencoders. */
    void train_dnae(const std::vector<distribution<float> > & training_data,
                    const std::vector<distribution<float> > & testing_data,
                    const Configuration & config,
                    ML::Thread_Context & thread_context);

    /** Tests on both pristine and noisy inputs.  The first returned is the
        error on pristine inputs.  The second is the error on noisy inputs.
        the prob_cleared parameter describes the probability that noise will
        be added to any given input.
    */
    std::pair<double, double>
    test_dnae(const std::vector<distribution<float> > & data,
              float prob_cleared,
              ML::Thread_Context & thread_context,
              int verbosity) const;
    
    bool operator == (const DNAE_Stack & other) const;
};


IMPL_SERIALIZE_RECONSTITUTE(DNAE_Stack);

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
