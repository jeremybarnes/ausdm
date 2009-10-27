/* dnae_decomposition.cc
   Jeremy Barnes, 24 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Denoising autoencoder decomposition.
*/

#include "dnae_decomposition.h"
#include "svd_decomposition.h"

#include "algebra/matrix_ops.h"
#include "math/xdiv.h"
#include "algebra/lapack.h"
#include "arch/atomic_ops.h"

#include <boost/progress.hpp>
#include <boost/bind.hpp>
#include <boost/tuple/tuple.hpp>

#include "boosting/worker_task.h"
#include "boosting/registry.h"

#include "arch/timers.h"
#include "utils/info.h"
#include "utils/guard.h"
#include "arch/threads.h"
#include "arch/atomic_ops.h"

#include "db/persistent.h"

#include <limits>

using namespace std;
using namespace ML;
using namespace ML::DB;

namespace ML {

namespace {
static const float NaN = numeric_limits<float>::quiet_NaN();
} // file scope

void calc_W_updates(double k1, const double * x, double k2, const double * y,
                    const double * z, double * r, size_t n)
{
    return SIMD::vec_k1_x_plus_k2_y_z(k1, x, k2, y, z, r, n);
}

#define CHECK_NO_NAN(x) \
    { for (unsigned i = 0;  i < x.size();  ++i) { if (isnan(x[i])) throw Exception(format("element %d of %s is Nan in %s %s:%d", i, #x, __PRETTY_FUNCTION__, __FILE__, __LINE__)); } }


/*****************************************************************************/
/* DENSE_MISSING_LAYER                                                       */
/*****************************************************************************/

Dense_Missing_Layer::
Dense_Missing_Layer()
    : use_dense_missing(true)
{
}

Dense_Missing_Layer::
Dense_Missing_Layer(bool use_dense_missing,
                    size_t inputs, size_t outputs,
                    Transfer_Function_Type transfer,
                    Thread_Context & context,
                    float limit)
    : Base(inputs, outputs, transfer),
      use_dense_missing(use_dense_missing),
      missing_activations(inputs, distribution<LFloat>(outputs))
{
    if (limit == -1.0)
        limit = 1.0 / sqrt(inputs);
    random_fill(limit, context);
}

Dense_Missing_Layer::
Dense_Missing_Layer(bool use_dense_missing,
                    size_t inputs, size_t outputs,
                    Transfer_Function_Type transfer)
    : Base(inputs, outputs, transfer),
      use_dense_missing(use_dense_missing),
      missing_activations(inputs, distribution<LFloat>(outputs))
{
}

void
Dense_Missing_Layer::
preprocess(const float * input,
           float * preprocessed) const
{
    if (!use_dense_missing) Base::preprocess(input, preprocessed);
    else Layer::preprocess(input, preprocessed);
}

void
Dense_Missing_Layer::
preprocess(const double * input,
           double * preprocessed) const
{
    if (!use_dense_missing) Base::preprocess(input, preprocessed);
    else Layer::preprocess(input, preprocessed);
}

void
Dense_Missing_Layer::
activation(const float * preprocessed,
           float * activation) const
{
    if (!use_dense_missing) {
        Base::activation(preprocessed, activation);
        return;
    }

    int ni = inputs(), no = outputs();
    double accum[no];  // Accumulate in double precision to improve rounding
    std::copy(bias.begin(), bias.end(), accum);

    for (unsigned i = 0;  i < ni;  ++i) {
        const LFloat * w;
        double input;
        if (isnan(preprocessed[i])) {
            input = 1.0;
            w = &missing_activations[i][0];
        }
        else {
            input = preprocessed[i];
            w = &weights[i][0];
        }

        SIMD::vec_add(accum, input, w, accum, no);
    }
    
    std::copy(accum, accum + no, activation);
}

void
Dense_Missing_Layer::
activation(const double * preprocessed,
           double * activation) const
{
    if (!use_dense_missing) {
        Base::activation(preprocessed, activation);
        return;
    }

    int ni = inputs(), no = outputs();
    double accum[no];  // Accumulate in double precision to improve rounding
    std::copy(bias.begin(), bias.end(), accum);

    for (unsigned i = 0;  i < ni;  ++i) {
        const LFloat * w;
        double input;
        if (isnan(preprocessed[i])) {
            input = 1.0;
            w = &missing_activations[i][0];
        }
        else {
            input = preprocessed[i];
            w = &weights[i][0];
        }

        SIMD::vec_add(accum, input, w, accum, no);
    }
    
    std::copy(accum, accum + no, activation);
}

void
Dense_Missing_Layer::
random_fill(float limit, Thread_Context & context)
{
    Base::random_fill(limit, context);

    for (unsigned i = 0;  i < inputs();  ++i)
        for (unsigned o = 0;  o < outputs();  ++o)
            missing_activations[i][o]
                = limit * (context.random01() * 2.0f - 1.0f);
}

void
Dense_Missing_Layer::
zero_fill()
{
    Base::zero_fill();
    for (unsigned i = 0;  i < inputs();  ++i)
        missing_activations[i].fill(0.0);
}

size_t
Dense_Missing_Layer::
parameter_count() const
{
    size_t result = Base::parameter_count();

    if (use_dense_missing)
        result += inputs() * (outputs() - 1);

    return result;
}



/*****************************************************************************/
/* TWOWAY_LAYER                                                              */
/*****************************************************************************/

Twoway_Layer::
Twoway_Layer()
{
}

Twoway_Layer::
Twoway_Layer(bool use_dense_missing,
             size_t inputs, size_t outputs,
             Transfer_Function_Type transfer,
             Thread_Context & context,
             float limit)
    : Base(use_dense_missing, inputs, outputs, transfer), ibias(inputs)
{
    if (limit == -1.0)
        limit = 1.0 / sqrt(inputs);
    random_fill(limit, context);
}

Twoway_Layer::
Twoway_Layer(bool use_dense_missing,
             size_t inputs, size_t outputs,
             Transfer_Function_Type transfer)
    : Base(use_dense_missing, inputs, outputs, transfer), ibias(inputs)
{
}

distribution<double>
Twoway_Layer::
iapply(const distribution<double> & output) const
{
    CHECK_NO_NAN(output);
    distribution<double> activation = weights * output;
    activation += ibias;
    transfer(&activation[0], &activation[0], inputs(), transfer_function);
    return activation;
}

distribution<float>
Twoway_Layer::
iapply(const distribution<float> & output) const
{
    CHECK_NO_NAN(output);
    distribution<float> activation = multiply_r<float>(weights, output);
    activation += ibias;
    transfer(&activation[0], &activation[0], inputs(), transfer_function);
    return activation;
}

distribution<double>
Twoway_Layer::
ipreprocess(const distribution<double> & input) const
{
    return input;
}

distribution<float>
Twoway_Layer::
ipreprocess(const distribution<float> & input) const
{
    return input;
}

distribution<double>
Twoway_Layer::
iactivation(const distribution<double> & output) const
{
    CHECK_NO_NAN(output);
    distribution<double> activation = weights * output;
    activation += ibias;
    return activation;
}

distribution<float>
Twoway_Layer::
iactivation(const distribution<float> & output) const
{
    CHECK_NO_NAN(output);
    distribution<float> activation = multiply_r<float>(weights, output);
    activation += ibias;
    return activation;
}

distribution<double>
Twoway_Layer::
itransfer(const distribution<double> & activation) const
{
    CHECK_NO_NAN(activation);
    int ni = inputs();
    if (activation.size() != ni)
        throw Exception("invalid sizes in itransfer");
    distribution<double> result(ni);
    transfer(&activation[0], &result[0], ni, transfer_function);
    return activation;
}

distribution<float>
Twoway_Layer::
itransfer(const distribution<float> & activation) const
{
    CHECK_NO_NAN(activation);
    int ni = inputs();
    if (activation.size() != ni)
        throw Exception("invalid sizes in itransfer");
    distribution<float> result(ni);
    transfer(&activation[0], &result[0], ni, transfer_function);
    return activation;
}

distribution<double>
Twoway_Layer::
iderivative(const distribution<double> & input) const
{
    CHECK_NO_NAN(input);
    if (input.size() != this->inputs())
        throw Exception("iderivative(): wrong size");
    int ni = this->inputs();
    distribution<double> result(ni);
    derivative(&input[0], &result[0], ni, transfer_function);
    return result;
}

distribution<float>
Twoway_Layer::
iderivative(const distribution<float> & input) const
{
    CHECK_NO_NAN(input);
    if (input.size() != this->inputs())
        throw Exception("iderivative(): wrong size");
    int ni = this->inputs();
    distribution<float> result(ni);
    derivative(&input[0], &result[0], ni, transfer_function);
    return result;
}

void
Twoway_Layer::
update(const Twoway_Layer & updates, double learning_rate)
{
    int ni = inputs();
    int no = outputs();
    
    ibias -= learning_rate * updates.ibias;
    bias -= learning_rate * updates.bias;
    
    for (unsigned i = 0;  i < ni;  ++i)
        SIMD::vec_add(&weights[i][0], -learning_rate,
                      &updates.weights[i][0],
                      &weights[i][0], no);

    if (use_dense_missing) {
        for (unsigned i = 0;  i < ni;  ++i)
            SIMD::vec_add(&missing_activations[i][0],
                          -learning_rate,
                          &updates.missing_activations[i][0],
                          &missing_activations[i][0], no);
    }
    else 
        missing_replacements
            -= 100.0 * learning_rate * updates.missing_replacements;
}

void
Twoway_Layer::
random_fill(float limit, Thread_Context & context)
{
    Dense_Layer<LFloat>::random_fill(limit, context);
    for (unsigned i = 0;  i < ibias.size();  ++i)
        ibias[i] = limit * (context.random01() * 2.0f - 1.0f);
}

void
Twoway_Layer::
zero_fill()
{
    Dense_Layer<LFloat>::zero_fill();
    ibias.fill(0.0);
}

void
Twoway_Layer::
serialize(DB::Store_Writer & store) const
{
    Dense_Layer<LFloat>::serialize(store);
    store << ibias;
}

void
Twoway_Layer::
reconstitute(DB::Store_Reader & store)
{
    Dense_Layer<LFloat>::reconstitute(store);
    store >> ibias;
}


/*****************************************************************************/
/* DNAE_STACK                                                                */
/*****************************************************************************/



// Float type to use for calculations
typedef double CFloat;

template<typename Float>
distribution<Float>
add_noise(const distribution<Float> & inputs,
          Thread_Context & context,
          float prob_cleared)
{
    distribution<Float> result = inputs;

    for (unsigned i = 0;  i < inputs.size();  ++i)
        if (context.random01() < prob_cleared)
            result[i] = NaN;
    
    return result;
}

pair<double, double>
train_example(const Twoway_Layer & layer,
              const vector<distribution<float> > & data,
              int example_num,
              float max_prob_cleared,
              Thread_Context & thread_context,
              Twoway_Layer & updates,
              Lock & update_lock,
              int verbosity)
{
    int ni JML_UNUSED = layer.inputs();
    int no JML_UNUSED = layer.outputs();

    // Present this input
    distribution<CFloat> model_input(data.at(example_num));

    CHECK_NO_NAN(model_input);
    
    if (model_input.size() != ni) {
        cerr << "model_input.size() = " << model_input.size() << endl;
        cerr << "ni = " << ni << endl;
        throw Exception("wrong sizes");
    }

    // Add noise up to the threshold
    // We don't add a uniform amount as this causes a bias in things like the
    // total.
    //float prob_cleared = thread_context.random01() * max_prob_cleared;
    //float prob_cleared = thread_context.random01() < 0.5 ? max_prob_cleared : 0.0;
    float prob_cleared = max_prob_cleared;

    distribution<CFloat> noisy_input
        = add_noise(model_input, thread_context, prob_cleared);

    distribution<CFloat> noisy_pre
        = layer.preprocess(noisy_input);

    distribution<CFloat> hidden_act
        = layer.activation(noisy_pre);

    CHECK_NO_NAN(hidden_act);
            
    // Apply the layer
    distribution<CFloat> hidden_rep
        = layer.transfer(hidden_act);

    CHECK_NO_NAN(hidden_rep);
            
    // Reconstruct the input
    distribution<CFloat> denoised_input
        = layer.iapply(hidden_rep);

    CHECK_NO_NAN(denoised_input);
            
    // Error signal
    distribution<CFloat> diff
        = model_input - denoised_input;
    
    // Overall error
    double error = pow(diff.two_norm(), 2);
    

    double error_exact = pow((model_input - layer.iapply(layer.apply(model_input))).two_norm(), 2);

    if (example_num < 10 && false) {
        cerr << " ex " << example_num << endl;
        cerr << "  input: " << distribution<float>(model_input.begin(),
                                                   model_input.begin() + 10)
             << endl;
        cerr << "  noisy input: " << distribution<float>(noisy_input.begin(),
                                                         noisy_input.begin() + 10)
             << endl;
        cerr << "  output: " << distribution<float>(denoised_input.begin(),
                                                    denoised_input.begin() + 10)
             << endl;
        cerr << "  diff: " << distribution<float>(diff.begin(),
                                                  diff.begin() + 10)
             << endl;
        cerr << "error: " << error << endl;
        cerr << endl;
        
    }
        
    // Now we solve for the gradient direction for the two biases as
    // well as for the weights matrix
    //
    // If f() is the activation function for the forward direction and
    // g() is the activation function for the reverse direction, we can
    // write
    //
    // h = f(Wi1 + b)
    //
    // where i1 is the (noisy) inputs, h is the hidden unit outputs, W
    // is the weight matrix and b is the forward bias vector.  Going
    // back again, we then take
    // 
    // i2 = g(W*h + c) = g(W*f(Wi + b) + c)
    //
    // where i2 is the denoised approximation of the true input weights
    // (i) and W* is W transposed.
    //
    // Using the MSE, we get
    //
    // e = sqr(||i2 - i||) = sum(sqr(i2 - i))
    //
    // where e is the MSE.
    //
    // Differentiating with respect to i2, we get
    //
    // de/di2 = 2(i2 - i)
    //
    // Finally, we want to know the gradient direction for each of the
    // parameters W, b and c.  Taking c first, we get
    //
    // de/dc = de/di2 di2/dc
    //       = 2 (i2 - i) g'(i2)
    //
    // As for b, we get
    //
    // de/db = de/di2 di2/db
    //       = 2 (i2 - i) g'(i2) W* f'(Wi + b)
    //
    // And for W:
    //
    // de/dW = de/di2 di2/dW
    //       = 2 (i2 - i) g'(i2) [ h + W* f'(Wi + b) i ]
    //
    // Since we want to minimise the reconstruction error, we use the
    // negative of the gradient.

    // NOTE: here, the activation function for the input and the output
    // are the same.
        
    const boost::multi_array<LFloat, 2> & W
        = layer.weights;

    const distribution<LFloat> & b JML_UNUSED = layer.bias;
    const distribution<LFloat> & c JML_UNUSED = layer.ibias;

    distribution<CFloat> c_updates
        = -2 * diff * layer.iderivative(denoised_input);

    CHECK_NO_NAN(c_updates);

#if 0
    Twoway_Layer layer2 = layer;

    // Calculate numerically the c updates
    for (unsigned i = 0;  i < ni;  ++i) {

        float epsilon = 1e-8;
        double old = layer2.ibias[i];
        layer2.ibias[i] += epsilon;

        // Apply the layer
        distribution<CFloat> hidden_rep2
            = layer2.apply(noisy_input);
        
        distribution<CFloat> denoised_input2
            = layer2.iapply(hidden_rep2);

        // Error signal
        distribution<CFloat> diff2
            = model_input - denoised_input2;
            
        // Overall error
        double error2 = pow(diff2.two_norm(), 2);

        double delta = error2 - error;

        double deriv  = c_updates[i];
        double deriv2 = xdiv(delta, epsilon);

        cerr << format("%3d %7.4f %9.5f %9.5f %9.5f %8.5f\n",
                       i,
                       100.0 * xdiv(abs(deriv - deriv2),
                                    max(abs(deriv), abs(deriv2))),
                       abs(deriv - deriv2),
                       deriv, deriv2, noisy_input[i]);

        layer2.ibias[i] = old;
    }
#endif

    distribution<CFloat> hidden_deriv
        = layer.derivative(hidden_rep);

    // Check hidden_deriv numerically
#if 0
    distribution<CFloat> hidden_act2 = hidden_act;

    cerr << "hidden_act = " << hidden_act << endl;
    cerr << "hidden_deriv = " << hidden_deriv << endl;

    // Calculate numerically the c updates
    for (unsigned i = 0;  i < no;  ++i) {

        float epsilon = 1e-8;
        double old = hidden_act2[i];
        hidden_act2[i] += epsilon;

        // Apply the layer
        distribution<CFloat> hidden_rep2
            = layer.transfer(hidden_act2);
        
        // Error signal

        // Overall error
        double delta = hidden_rep2[i] - hidden_rep[i];

        double deriv  = hidden_deriv[i];
        double deriv2 = xdiv(delta, epsilon);

        cerr << format("%3d %7.4f %9.5f %9.5f %9.5f %8.5f\n",
                       i,
                       100.0 * xdiv(abs(deriv - deriv2),
                                    max(abs(deriv), abs(deriv2))),
                       abs(deriv - deriv2),
                       deriv, deriv2, hidden_act[i]);

        hidden_act2[i] = old;
    }
#endif
    

    CHECK_NO_NAN(hidden_deriv);

    distribution<CFloat> b_updates
        = multiply_r<CFloat>(c_updates, W) * hidden_deriv;

    CHECK_NO_NAN(b_updates);

#if 0
    Twoway_Layer layer2 = layer;

    // Calculate numerically the c updates
    for (unsigned i = 0;  i < no;  ++i) {

        float epsilon = 1e-8;
        double old = layer2.bias[i];
        layer2.bias[i] += epsilon;

        // Apply the layer
        distribution<CFloat> hidden_rep2
            = layer2.apply(noisy_input);
        
        distribution<CFloat> denoised_input2
            = layer2.iapply(hidden_rep2);

        // Error signal
        distribution<CFloat> diff2
            = model_input - denoised_input2;
            
        // Overall error
        double error2 = pow(diff2.two_norm(), 2);

        double delta = error2 - error;

        double deriv  = b_updates[i];
        double deriv2 = xdiv(delta, epsilon);

        cerr << format("%3d %7.4f %9.5f %9.5f %9.5f %8.5f\n",
                       i,
                       100.0 * xdiv(abs(deriv - deriv2),
                                    max(abs(deriv), abs(deriv2))),
                       abs(deriv - deriv2),
                       deriv, deriv2, noisy_input[i]);

        layer2.bias[i] = old;
    }
#endif

    boost::multi_array<double, 2> W_updates(boost::extents[ni][no]);
    vector<distribution<double> >
        missing_act_updates(ni, distribution<double>(no));

    distribution<double> factor_totals(no);

    for (unsigned i = 0;  i < ni;  ++i)
        SIMD::vec_add(&factor_totals[0], c_updates[i], &W[i][0],
                      &factor_totals[0], no);

    for (unsigned i = 0;  i < ni;  ++i) {

        if (!layer.use_dense_missing
            || !isnan(noisy_input[i]))

            // We use the W value for both the input and the output, so we
            // need to accumulate it's total effect on the derivative
            calc_W_updates(c_updates[i],
                           &hidden_rep[0],
                           model_input[i],
                           &factor_totals[0],
                           &hidden_deriv[0],
                           &W_updates[i][0],
                           no);
        else {
            // The weight updates are simpler, but we also have to calculate
            // the missing activation updates

            // W value only used on the way out; simpler calculation
            SIMD::vec_add(&W_updates[i][0], c_updates[i],
                          &hidden_rep[0], &W_updates[i][0], no);

            // Missing values were used on the way in
            missing_act_updates[i] = factor_totals * hidden_deriv;
        }
    }
       
    
#if 0  // test numerically
    Twoway_Layer layer2 = layer;

    for (unsigned i = 0;  i < ni;  ++i) {

        for (unsigned j = 0;  j < no;  ++j) {
            double epsilon = 1e-8;

            double old_W = layer2.weights[i][j];
            layer2.weights[i][j] += epsilon;

            // Apply the layer
            distribution<CFloat> hidden_rep2
                = layer2.apply(noisy_input);

            distribution<CFloat> denoised_input2
                = layer2.iapply(hidden_rep2);
            
            // Error signal
            distribution<CFloat> diff2
                = model_input - denoised_input2;
                    
            //cerr << "diff = " << diff << endl;
            //cerr << "diff2 = " << diff2 << endl;
                    
            // Overall error
            double error2 = pow(diff2.two_norm(), 2);
                    
            double delta = error2 - error;

            double deriv2 = xdiv(delta, epsilon);

            double deriv = W_updates[i][j];

            cerr << format("%3d %3d %7.4f %9.5f %9.5f %9.5f %8.5f\n",
                           i, j,
                           100.0 * xdiv(abs(deriv - deriv2),
                                        max(abs(deriv), abs(deriv2))),
                           abs(deriv - deriv2),
                           deriv, deriv2, noisy_input[i]);

            //cerr << "error = " << error << " error2 = " << error2
            //     << " delta = " << delta
            //    << " deriv " << W_updates[i][j]
            //     << " deriv2 " << deriv2 << endl;


            layer2.weights[i][j] = old_W;
        }
    }
#endif  // if one/zero

#if 0  // test numerically the missing activations
    Twoway_Layer layer2 = layer;

    for (unsigned i = 0;  i < ni;  ++i) {
        if (!isnan(noisy_input[i])) continue;  // will be zero

        for (unsigned j = 0;  j < no;  ++j) {
            double epsilon = 1e-8;

            double old_W = layer2.missing_activations[i][j];
            layer2.missing_activations[i][j] += epsilon;

            // Apply the layer
            distribution<CFloat> hidden_rep2
                = layer2.apply(noisy_input);

            distribution<CFloat> denoised_input2
                = layer2.iapply(hidden_rep2);
            
            // Error signal
            distribution<CFloat> diff2
                = model_input - denoised_input2;
                    
            //cerr << "diff = " << diff << endl;
            //cerr << "diff2 = " << diff2 << endl;
                    
            // Overall error
            double error2 = pow(diff2.two_norm(), 2);
                    
            double delta = error2 - error;

            double deriv2 = xdiv(delta, epsilon);

            double deriv = missing_act_updates[i][j];

            cerr << format("%3d %3d %7.4f %9.5f %9.5f %9.5f %8.5f\n",
                           i, j,
                           100.0 * xdiv(abs(deriv - deriv2),
                                        max(abs(deriv), abs(deriv2))),
                           abs(deriv - deriv2),
                           deriv, deriv2, noisy_input[i]);

            layer2.missing_activations[i][j] = old_W;
        }
    }
#endif  // if one/zero

    distribution<double> cleared_value_updates(ni);
    
    if (!layer.use_dense_missing)
        cleared_value_updates = W * b_updates;

#if 0  // test numerically
    for (unsigned i = 0;  i < ni;  ++i) {
        double epsilon = 1e-6;

        distribution<CFloat> noisy_pre2 = noisy_pre;
        noisy_pre2[i] += epsilon;

        // Apply the layer
        distribution<CFloat> hidden_act2
            = layer.activation(noisy_pre2);

        distribution<CFloat> hidden_rep2
            = layer.transfer(hidden_act2);
                    
        distribution<CFloat> denoised_input2
            = layer.iapply(hidden_rep2);
                    
        // Error signal
        distribution<CFloat> diff2
            = model_input - denoised_input2;
                    
        // Overall error
        double error2 = pow(diff2.two_norm(), 2);
                    
        double delta = error2 - error;

        double deriv2 = xdiv(delta, epsilon);

        cerr << "error = " << error << " error2 = " << error2
             << " delta = " << delta
             << " deriv " << cleared_value_updates[i]
             << " deriv2 " << deriv2 << endl;
    }
#endif // test numerically

#if 0
    cerr << "cleared_value_updates.size() = " << cleared_value_updates.size()
         << endl;
    cerr << "ni = " << ni << endl;
    cerr << "b_updates.size() = " << b_updates.size() << endl;
#endif

    if (true) {  // faster, despite the lock
        Guard guard(update_lock);
        
        for (unsigned i = 0;  i < ni;  ++i)
            if (isnan(noisy_input[i]))
                updates.missing_replacements[i] += cleared_value_updates[i];
        
        updates.bias += b_updates;
        updates.ibias += c_updates;
        
        for (unsigned i = 0;  i < ni;  ++i) {
            SIMD::vec_add(&updates.weights[i][0],
                          &W_updates[i][0],
                          &updates.weights[i][0], no);
            
            if (layer.use_dense_missing && isnan(noisy_input[i]))
                updates.missing_activations[i] += missing_act_updates[i];
        }
    }
    else {
        for (unsigned i = 0;  i < ni;  ++i)
            if (isnan(noisy_input[i]))
                atomic_accumulate(updates.missing_replacements[i],
                                  cleared_value_updates[i]);

        atomic_accumulate(&updates.bias[0], &b_updates[0], no);
        atomic_accumulate(&updates.ibias[0], &c_updates[0], ni);

        for (unsigned i = 0;  i < ni;  ++i) {
            atomic_accumulate(&updates.weights[i][0], &W_updates[i][0], no);

            if (layer.use_dense_missing && isnan(noisy_input[i]))
                atomic_accumulate(&updates.missing_activations[i][0],
                                  &missing_act_updates[i][0],
                                  no);
        }
    }

    return make_pair(error_exact, error);
}

struct Train_Examples_Job {

    const Twoway_Layer & layer;
    const vector<distribution<float> > & data;
    int first;
    int last;
    float prob_cleared;
    const Thread_Context & context;
    int random_seed;
    Twoway_Layer & updates;
    Lock & update_lock;
    double & error_exact;
    double & error_noisy;
    boost::progress_display * progress;
    int verbosity;

    Train_Examples_Job(const Twoway_Layer & layer,
                       const vector<distribution<float> > & data,
                       int first, int last,
                       float prob_cleared,
                       const Thread_Context & context,
                       int random_seed,
                       Twoway_Layer & updates,
                       Lock & update_lock,
                       double & error_exact,
                       double & error_noisy,
                       boost::progress_display * progress,
                       int verbosity)
        : layer(layer), data(data), first(first), last(last),
          prob_cleared(prob_cleared),
          context(context), random_seed(random_seed), updates(updates),
          update_lock(update_lock),
          error_exact(error_exact), error_noisy(error_noisy),
          progress(progress), verbosity(verbosity)
    {
    }

    void operator () ()
    {
        Thread_Context thread_context(context);
        thread_context.seed(random_seed);

        double total_error_exact = 0.0, total_error_noisy = 0.0;
        for (unsigned x = first;  x < last;  ++x) {
            double eex, eno;
            boost::tie(eex, eno)
                = train_example(layer, data, x,
                                prob_cleared, thread_context,
                                updates, update_lock,
                                verbosity);

            total_error_exact += eex;
            total_error_noisy += eno;
        }

        Guard guard(update_lock);
        error_exact += total_error_exact;
        error_noisy += total_error_noisy;
        

        if (progress && verbosity >= 3)
            (*progress) += (last - first);
    }
};

std::pair<double, double>
Twoway_Layer::
train_iter(const vector<distribution<float> > & data,
           float prob_cleared,
           Thread_Context & thread_context,
           int minibatch_size, float learning_rate,
           int verbosity)
{
    Worker_Task & worker = thread_context.worker();

    int nx = data.size();
    int ni JML_UNUSED = inputs();
    int no JML_UNUSED = outputs();

    int microbatch_size = minibatch_size / (num_cpus() * 4);
            
    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx, cerr));
    Lock update_lock;

    double total_mse_exact = 0.0, total_mse_noisy = 0.0;
    
    for (unsigned x = 0;  x < nx;  x += minibatch_size) {
                
        Twoway_Layer updates(use_dense_missing, ni, no, TF_IDENTITY);
                
        // Now, submit it as jobs to the worker task to be done
        // multithreaded
        int group;
        {
            int parent = -1;  // no parent group
            group = worker.get_group(NO_JOB, "dump user results task",
                                     parent);
                    
            // Make sure the group gets unlocked once we've populated
            // everything
            Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                         boost::ref(worker),
                                         group));
                    
                    
            for (unsigned x2 = x;  x2 < nx && x2 < x + minibatch_size;
                 x2 += microbatch_size) {
                        
                Train_Examples_Job job(*this,
                                       data,
                                       x2,
                                       min<int>(nx,
                                                min(x + minibatch_size,
                                                    x2 + microbatch_size)),
                                       prob_cleared,
                                       thread_context,
                                       thread_context.random(),
                                       updates,
                                       update_lock,
                                       total_mse_exact,
                                       total_mse_noisy,
                                       progress.get(),
                                       verbosity);
                // Send it to a thread to be processed
                worker.add(job, "blend job", group);
            }
        }
                
        worker.run_until_finished(group);

        update(updates, learning_rate);
    }

    return make_pair(sqrt(total_mse_exact / nx), sqrt(total_mse_noisy / nx));
}

struct Test_Examples_Job {

    const Twoway_Layer & layer;
    const vector<distribution<float> > & data_in;
    vector<distribution<float> > & data_out;
    int first;
    int last;
    float prob_cleared;
    const Thread_Context & context;
    int random_seed;
    Lock & update_lock;
    double & error_exact;
    double & error_noisy;
    boost::progress_display * progress;
    int verbosity;

    Test_Examples_Job(const Twoway_Layer & layer,
                      const vector<distribution<float> > & data_in,
                      vector<distribution<float> > & data_out,
                      int first, int last,
                      float prob_cleared,
                      const Thread_Context & context,
                      int random_seed,
                      Lock & update_lock,
                      double & error_exact,
                      double & error_noisy,
                      boost::progress_display * progress,
                      int verbosity)
        : layer(layer), data_in(data_in), data_out(data_out),
          first(first), last(last),
          prob_cleared(prob_cleared),
          context(context), random_seed(random_seed),
          update_lock(update_lock),
          error_exact(error_exact), error_noisy(error_noisy),
          progress(progress), verbosity(verbosity)
    {
    }

    void operator () ()
    {
        Thread_Context thread_context(context);
        thread_context.seed(random_seed);

        double test_error_exact = 0.0, test_error_noisy = 0.0;

        for (unsigned x = first;  x < last;  ++x) {
            int ni JML_UNUSED = layer.inputs();
            int no JML_UNUSED = layer.outputs();

            // Present this input
            distribution<CFloat> model_input(data_in[x]);
            
            distribution<bool> was_cleared;

            // Add noise
            distribution<CFloat> noisy_input
                = add_noise(model_input, thread_context, prob_cleared);
            
            // Apply the layer
            distribution<CFloat> hidden_rep
                = layer.apply(noisy_input);
            
            // Reconstruct the input
            distribution<CFloat> denoised_input
                = layer.iapply(hidden_rep);
            
            // Error signal
            distribution<CFloat> diff
                = model_input - denoised_input;
    
            // Overall error
            double error = pow(diff.two_norm(), 2);

            test_error_noisy += error;


            // Apply the layer
            distribution<CFloat> hidden_rep2
                = layer.apply(model_input);

            if (!data_out.empty())
                data_out.at(x) = hidden_rep2.cast<float>();
            
            // Reconstruct the input
            distribution<CFloat> reconstructed_input
                = layer.iapply(hidden_rep2);
            
            // Error signal
            distribution<CFloat> diff2
                = model_input - reconstructed_input;
    
            // Overall error
            double error2 = pow(diff2.two_norm(), 2);
    
            test_error_exact += error2;

            if (x < 5 && false) {
                Guard guard(update_lock);
                cerr << "ex " << x << " error " << error2 << endl;
                cerr << "    input " << model_input << endl;
                //cerr << "    act   " << layer.activation(model_input) << endl;
                cerr << "    rep   " << hidden_rep2 << endl;
                //cerr << "    act2  " << layer.iactivation(hidden_rep2) << endl;
                cerr << "    ibias " << layer.ibias << endl;
                cerr << "    out   " << reconstructed_input << endl;
                cerr << "    diff  " << diff2 << endl;
                cerr << endl;
            }
        }

        Guard guard(update_lock);
        error_exact += test_error_exact;
        error_noisy += test_error_noisy;
        if (progress && verbosity >= 3)
            (*progress) += (last - first);
    }
};

pair<double, double>
Twoway_Layer::
test_and_update(const vector<distribution<float> > & data_in,
                vector<distribution<float> > & data_out,
                float prob_cleared,
                Thread_Context & thread_context,
                int verbosity) const
{
    Lock update_lock;
    double error_exact = 0.0;
    double error_noisy = 0.0;

    int nx = data_in.size();

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx, cerr));

    Worker_Task & worker = thread_context.worker();
            
    // Now, submit it as jobs to the worker task to be done
    // multithreaded
    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "dump user results task",
                                 parent);
        
        // Make sure the group gets unlocked once we've populated
        // everything
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        // 20 jobs per CPU
        int batch_size = nx / (num_cpus() * 20);
        
        for (unsigned x = 0; x < nx;  x += batch_size) {
            
            Test_Examples_Job job(*this, data_in, data_out,
                                  x, min<int>(x + batch_size, nx),
                                  prob_cleared,
                                  thread_context,
                                  thread_context.random(),
                                  update_lock,
                                  error_exact, error_noisy,
                                  progress.get(),
                                  verbosity);
            
            // Send it to a thread to be processed
            worker.add(job, "blend job", group);
        }
    }

    worker.run_until_finished(group);

    return make_pair(sqrt(error_exact / nx),
                     sqrt(error_noisy / nx));
}

/*****************************************************************************/
/* DNAE_STACK                                                                */
/*****************************************************************************/

distribution<float>
DNAE_Stack::
apply(const distribution<float> & input) const
{
    distribution<float> output = input;
    
    // Go down the stack
    for (unsigned l = 0;  l < size();  ++l)
        output = (*this)[l].apply(output);
    
    return output;
}

distribution<double>
DNAE_Stack::
apply(const distribution<double> & input) const
{
    distribution<double> output = input;
    
    // Go down the stack
    for (unsigned l = 0;  l < size();  ++l)
        output = (*this)[l].apply(output);
    
    return output;
}

distribution<float>
DNAE_Stack::
iapply(const distribution<float> & output) const
{
    distribution<float> input = output;
    
    // Go down the stack
    for (int l = size() - 1;  l >= 0;  --l)
        input = (*this)[l].iapply(input);
    
    return input;
}

distribution<double>
DNAE_Stack::
iapply(const distribution<double> & output) const
{
    distribution<double> input = output;
    
    // Go down the stack
    for (int l = size() - 1;  l >= 0;  --l)
        input = (*this)[l].iapply(input);
    
    return input;
}

void
DNAE_Stack::
serialize(ML::DB::Store_Writer & store) const
{
    store << (char)1; // version
    store << compact_size_t(size());
    for (unsigned i = 0;  i < size();  ++i)
        (*this)[i].serialize(store);
}

void
DNAE_Stack::
reconstitute(ML::DB::Store_Reader & store)
{
    int version;
    store >> version;
    if (version != 1) {
        cerr << "version = " << (int)version << endl;
        throw Exception("DNAE_Stack::reconstitute(): invalid version");
    }
    compact_size_t sz(store);
    resize(sz);

    for (unsigned i = 0;  i < sz;  ++i)
        (*this)[i].reconstitute(store);
}

struct Test_Stack_Job {

    const DNAE_Stack & stack;
    const vector<distribution<float> > & data;
    int first;
    int last;
    float prob_cleared;
    const Thread_Context & context;
    int random_seed;
    Lock & update_lock;
    double & error_exact;
    double & error_noisy;
    boost::progress_display * progress;
    int verbosity;

    Test_Stack_Job(const DNAE_Stack & stack,
                   const vector<distribution<float> > & data,
                   int first, int last,
                   float prob_cleared,
                   const Thread_Context & context,
                   int random_seed,
                   Lock & update_lock,
                   double & error_exact,
                   double & error_noisy,
                   boost::progress_display * progress, int verbosity)
        : stack(stack), data(data),
          first(first), last(last),
          prob_cleared(prob_cleared),
          context(context), random_seed(random_seed),
          update_lock(update_lock),
          error_exact(error_exact), error_noisy(error_noisy),
          progress(progress), verbosity(verbosity)
    {
    }

    void operator () ()
    {
        Thread_Context thread_context(context);
        thread_context.seed(random_seed);

        double test_error_exact = 0.0, test_error_noisy = 0.0;

        for (unsigned x = first;  x < last;  ++x) {

            distribution<CFloat> input(data[x]);

            // Add noise
            distribution<CFloat> noisy_input
                = add_noise(input, thread_context, prob_cleared);
            
            distribution<CFloat>
                rep = stack.apply(input),
                noisy_rep = stack.apply(noisy_input);

            distribution<CFloat>
                output = stack.iapply(rep),
                noisy_output = stack.iapply(noisy_rep);

            // Error signal
            distribution<CFloat> diff
                = input - noisy_output;
    
            // Overall error
            double error = pow(diff.two_norm(), 2);

            test_error_noisy += error;

            // Error signal
            distribution<CFloat> diff2
                = input - output;
    
            // Overall error
            double error2 = pow(diff2.two_norm(), 2);
    
            test_error_exact += error2;

            if (x < 5) {
                Guard guard(update_lock);
                cerr << "ex " << x << " error " << error2 << endl;
                cerr << "    input " << input << endl;
                cerr << "    rep   " << rep << endl;
                cerr << "    out   " << output << endl;
                cerr << "    diff  " << diff2 << endl;
                cerr << endl;
            }

        }

        Guard guard(update_lock);
        error_exact += test_error_exact;
        error_noisy += test_error_noisy;
        if (progress && verbosity >= 3) (*progress) += (last - first);
    }
};

pair<double, double>
DNAE_Stack::
test(const vector<distribution<float> > & data,
     float prob_cleared,
     Thread_Context & thread_context,
     int verbosity) const
{
    Lock update_lock;
    double error_exact = 0.0;
    double error_noisy = 0.0;

    int nx = data.size();

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx, cerr));

    Worker_Task & worker = thread_context.worker();
            
    // Now, submit it as jobs to the worker task to be done
    // multithreaded
    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "dump user results task",
                                 parent);
        
        // Make sure the group gets unlocked once we've populated
        // everything
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        // 20 jobs per CPU
        int batch_size = nx / (num_cpus() * 20);
        
        for (unsigned x = 0; x < nx;  x += batch_size) {
            
            Test_Stack_Job job(*this, data,
                               x, min<int>(x + batch_size, nx),
                               prob_cleared,
                               thread_context,
                               thread_context.random(),
                               update_lock,
                               error_exact, error_noisy,
                               progress.get(),
                               verbosity);
            
            // Send it to a thread to be processed
            worker.add(job, "blend job", group);
        }
    }
    
    worker.run_until_finished(group);
    
    return make_pair(sqrt(error_exact / nx),
                     sqrt(error_noisy / nx));
}


void
DNAE_Stack::
train(const std::vector<distribution<float> > & training_data,
      const std::vector<distribution<float> > & testing_data,
      const Configuration & config,
      Thread_Context & thread_context)
{
    double learning_rate = 0.75;
    int minibatch_size = 256;
    int niter = 50;

    /// Probability that each input is cleared
    float prob_cleared = 0.10;

    int verbosity = 2;

    Transfer_Function_Type transfer_function = TF_TANH;

    config.get(prob_cleared, "prob_cleared");
    config.get(learning_rate, "learning_rate");
    config.get(minibatch_size, "minibatch_size");
    config.get(niter, "niter");
    config.get(verbosity, "verbosity");
    config.find(transfer_function, "transfer_function");

    int nx = training_data.size();
    int nxt = testing_data.size();

    if (nx == 0)
        throw Exception("can't train on no data");

    static const int nlayers = 4;

    int layer_sizes[nlayers] = {195, 100, 50, 30};

    vector<distribution<float> > layer_train = training_data;
    vector<distribution<float> > layer_test = testing_data;

    // Learning rate is per-example
    learning_rate /= nx;

    bool use_dense_missing = true;

    for (unsigned layer_num = 0;  layer_num < nlayers;  ++layer_num) {
        cerr << endl << endl << endl << "--------- LAYER " << layer_num
             << " ---------" << endl << endl;

        vector<distribution<float> > next_layer_train, next_layer_test;

        int ni
            = layer_num == 0
            ? training_data[0].size()
            : layer_sizes[layer_num - 1];

        if (ni != layer_train[0].size())
            throw Exception("ni is wrong");

        int nh = layer_sizes[layer_num];

        Twoway_Layer layer(use_dense_missing, ni, nh, transfer_function,
                           thread_context);
        distribution<CFloat> cleared_values(ni);

        if (ni == nh && false) {
            //layer.zero_fill();
            for (unsigned i = 0;  i < ni;  ++i) {
                layer.weights[i][i] += 1.0;
            }
        }

        // Initialize with an SVD
        SVD_Decomposition init;
        init.train(layer_train, nh);

        for (unsigned i = 0;  i < ni;  ++i) {
            distribution<CFloat> init_i(&init.lvectors[i][0],
                                       &init.lvectors[i][0] + nh);
            //init_i *= init.singular_values_order / init.singular_values_order[0];

            std::copy(init_i.begin(), init_i.end(),
                      &layer.weights[i][0]);
            layer.bias.fill(0.0);
            layer.ibias.fill(0.0);
        }

        cerr << "layer.weights: " << print_size(layer.weights) << endl;
        cerr << "init.lvectors: " << print_size(init.lvectors) << endl;

        //layer.weights = transpose(init.lvectors);

        if (verbosity == 2)
            cerr << "iter  ---- train ----  ---- test -----\n"
                 << "        exact   noisy    exact   noisy\n";

        for (unsigned iter = 0;  iter < niter;  ++iter) {
            if (verbosity >= 3)
                cerr << "iter " << iter << " training on " << nx << " examples"
                     << endl;
            else if (verbosity >= 2)
                cerr << format("%4d", iter) << flush;
            Timer timer;

#if 0
            cerr << "weights: " << endl;
            for (unsigned i = 0;  i < 10;  ++i) {
                for (unsigned j = 0;  j < 10;  ++j) {
                    cerr << format("%7.4f", layer.weights[i][j]);
                }
                cerr << endl;
            }
            
            double max_abs_weight = 0.0;
            double total_abs_weight = 0.0;
            double total_weight_sqr = 0.0;
            for (unsigned i = 0;  i < ni;  ++i) {
                for (unsigned j = 0;  j < nh;  ++j) {
                    double abs_weight = abs(layer.weights[i][j]);
                    max_abs_weight = std::max(max_abs_weight, abs_weight);
                    total_abs_weight += abs_weight;
                    total_weight_sqr += abs_weight * abs_weight;
                }
            }

            double avg_abs_weight = total_abs_weight / (ni * nh);
            double rms_avg_weight = sqrt(total_weight_sqr / (ni * nh));

            cerr << "max = " << max_abs_weight << " avg = "
                 << avg_abs_weight << " rms avg = " << rms_avg_weight
                 << endl;
#endif

            distribution<LFloat> svalues(min(ni, nh));
            boost::multi_array<LFloat, 2> layer2 = layer.weights;
            int nvalues = std::min(ni, nh);
        
            boost::multi_array<double, 2> rvectors(boost::extents[ni][nvalues]);
            boost::multi_array<double, 2> lvectorsT(boost::extents[nvalues][nh]);

            int result = LAPack::gesdd("S", nh, ni,
                                       layer2.data(), nh,
                                       &svalues[0],
                                       &lvectorsT[0][0], nh,
                                       &rvectors[0][0], nvalues);
            if (result != 0)
                throw Exception("gesdd returned non-zero");
        

            if (false) {
                boost::multi_array<double, 2> weights2
                    = rvectors * diag(svalues) * lvectorsT;
                
                cerr << "weights2: " << endl;
                for (unsigned i = 0;  i < 10;  ++i) {
                    for (unsigned j = 0;  j < 10;  ++j) {
                        cerr << format("%7.4f", weights2[i][j]);
                    }
                    cerr << endl;
                }
            }

            //if (iter == 0) layer.weights = rvectors * lvectorsT;

            //cerr << "svalues = " << svalues << endl;

            double train_error_exact, train_error_noisy;
            boost::tie(train_error_exact, train_error_noisy)
                = layer.train_iter(layer_train, prob_cleared, thread_context,
                                   minibatch_size, learning_rate,
                                   verbosity);

            if (verbosity >= 3) {
                cerr << "rmse of iteration: exact " << train_error_exact
                     << " noisy " << train_error_noisy << endl;
                if (verbosity >= 3) cerr << timer.elapsed() << endl;
            }
            else if (verbosity == 2)
                cerr << format("  %7.5f %7.5f",
                               train_error_exact, train_error_noisy)
                     << flush;

            timer.restart();
            double test_error_exact = 0.0, test_error_noisy = 0.0;
            
            if (verbosity >= 3)
                cerr << "testing on " << nxt << " examples"
                     << endl;
            boost::tie(test_error_exact, test_error_noisy)
                = layer.test(layer_test, prob_cleared, thread_context,
                             verbosity);

            if (verbosity >= 3) {
                cerr << "testing rmse of iteration: exact "
                     << test_error_exact << " noisy " << test_error_noisy
                     << endl;
                cerr << timer.elapsed() << endl;
            }
            else if (verbosity == 2)
                cerr << format("  %7.5f %7.5f",
                               test_error_exact, test_error_noisy)
                     << endl;
        }

        next_layer_train.resize(nx);
        next_layer_test.resize(nxt);

        // Calculate the inputs to the next layer
        
        if (verbosity >= 3)
            cerr << "calculating next layer training inputs on "
                 << nx << " examples" << endl;
        double train_error_exact = 0.0, train_error_noisy = 0.0;
        boost::tie(train_error_exact, train_error_noisy)
            = layer.test_and_update(layer_train, next_layer_train,
                                    prob_cleared, thread_context,
                                    verbosity);

        if (verbosity >= 2)
            cerr << "training rmse of layer: exact "
                 << train_error_exact << " noisy " << train_error_noisy
                 << endl;
        
        if (verbosity >= 3)
            cerr << "calculating next layer testing inputs on "
                 << nxt << " examples" << endl;
        double test_error_exact = 0.0, test_error_noisy = 0.0;
        boost::tie(test_error_exact, test_error_noisy)
            = layer.test_and_update(layer_test, next_layer_test,
                                    prob_cleared, thread_context,
                                    verbosity);
        
        if (verbosity >= 2)
            cerr << "testing rmse of layer: exact "
                 << test_error_exact << " noisy " << test_error_noisy
                 << endl;

        layer_train.swap(next_layer_train);
        layer_test.swap(next_layer_test);

        push_back(layer);

        // Test the layer stack
        if (verbosity >= 3)
            cerr << "calculating whole stack testing performance on "
                 << nxt << " examples" << endl;
        boost::tie(test_error_exact, test_error_noisy)
            = test(testing_data, prob_cleared, thread_context, verbosity);
        
        if (verbosity >= 2)
            cerr << "testing rmse of stack: exact "
                 << test_error_exact << " noisy " << test_error_noisy
                 << endl;
    }
}

} // namespace ML


/*****************************************************************************/
/* DNAE_DECOMPOSITION                                                        */
/*****************************************************************************/

distribution<float>
DNAE_Decomposition::
decompose(const distribution<float> & vals) const
{
    return stack.apply(vals);
}

distribution<float>
DNAE_Decomposition::
recompose(const distribution<float> & decomposition, int order) const
{
    return stack.iapply(decomposition);
}

void
DNAE_Decomposition::
serialize(ML::DB::Store_Writer & store) const
{
    store << (char)1; // version
    stack.serialize(store);
}

void
DNAE_Decomposition::
reconstitute(ML::DB::Store_Reader & store)
{
    int version;
    store >> version;
    if (version != 1)
        throw Exception("DNAE_Decomposition: version was wrong");

    stack.reconstitute(store);
}

std::string
DNAE_Decomposition::
class_id() const
{
    return "DNAE";
}

void
DNAE_Decomposition::
train(const Data & training_data,
      const Data & testing_data,
      const Configuration & config)
{
    Thread_Context thread_context;

    int nx = training_data.nx();
    int nxt = testing_data.nx();

    vector<distribution<float> > layer_train(nx), layer_test(nxt);

    for (unsigned x = 0;  x < nx;  ++x)
        layer_train[x] = 0.8f * training_data.examples[x];

    for (unsigned x = 0;  x < nxt;  ++x)
        layer_test[x] = 0.8f * testing_data.examples[x];

    stack.train(layer_train, layer_test, config, thread_context);
}

namespace {

Register_Factory<Decomposition, DNAE_Decomposition>
    DNAE_REGISTER("DNAE");

} // file scope

