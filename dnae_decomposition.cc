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


/*****************************************************************************/
/* TWOWAY_LAYER                                                              */
/*****************************************************************************/

Twoway_Layer::
Twoway_Layer()
{
}

Twoway_Layer::
Twoway_Layer(size_t inputs, size_t outputs,
             Transfer_Function_Type transfer,
             Thread_Context & context,
             float limit)
    : Dense_Layer<LFloat>(inputs, outputs, transfer)
{
    ibias.resize(inputs);
    if (limit == -1.0)
        limit = 1.0 / sqrt(inputs);
    random_fill(limit, context);
}

Twoway_Layer::
Twoway_Layer(size_t inputs, size_t outputs,
             Transfer_Function_Type transfer)
    : Dense_Layer<LFloat>(inputs, outputs, transfer)
{
    ibias.resize(inputs);
    ibias.fill(0.0);
}

distribution<double>
Twoway_Layer::
iapply(const distribution<double> & output) const
{
    distribution<double> activation = weights * output;
    activation += ibias;
    transfer(&activation[0], &activation[0], inputs(), transfer_function);
    return activation;
}

distribution<float>
Twoway_Layer::
iapply(const distribution<float> & output) const
{
    distribution<float> activation = multiply_r<float>(weights, output);
    activation += ibias;
    transfer(&activation[0], &activation[0], inputs(), transfer_function);
    return activation;
}

distribution<double>
Twoway_Layer::
iactivation(const distribution<double> & output) const
{
    distribution<double> activation = weights * output;
    activation += ibias;
    return activation;
}

distribution<float>
Twoway_Layer::
iactivation(const distribution<float> & output) const
{
    distribution<float> activation = multiply_r<float>(weights, output);
    activation += ibias;
    return activation;
}

distribution<double>
Twoway_Layer::
itransfer(const distribution<double> & activation) const
{
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

    missing_replacements -= 100.0 * learning_rate * updates.missing_replacements;
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

double train_example(const Twoway_Layer & layer,
                     const vector<distribution<float> > & data,
                     int example_num,
                     float max_prob_cleared,
                     Thread_Context & thread_context,
                     Twoway_Layer & updates,
                     Lock & update_lock)
{
    int ni JML_UNUSED = layer.inputs();
    int no JML_UNUSED = layer.outputs();

    // Present this input
    distribution<CFloat> model_input(data.at(example_num));

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

    distribution<CFloat> hidden_act
        = layer.activation(noisy_input);
            
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

    const distribution<LFloat> & b = layer.bias;
    const distribution<LFloat> & c JML_UNUSED = layer.ibias;

    distribution<CFloat> c_updates
        = -2 * diff * layer.iderivative(denoised_input);

#if 0
    cerr << "c_updates = " << c_updates << endl;

    distribution<CFloat> c_updates_numeric(ni);

    // Calculate numerically the c updates
    for (unsigned i = 0;  i < ni;  ++i) {
        // Reconstruct the input
        distribution<CFloat> denoised_activation2
            = W * hidden_rep + c;
                
        float epsilon = 1e-4;

        denoised_activation2[i] += epsilon;

        distribution<CFloat> denoised_input2 = denoised_activation2;
        layer.transfer(denoised_input2);
            
        // Error signal
        distribution<CFloat> diff2
            = model_input - denoised_input2;
            
        // Overall error
        double error2 = pow(diff2.two_norm(), 2);

        double delta = error2 - error;

        c_updates_numeric[i] = delta / epsilon;
    }

    cerr << "c_updates_numeric = " << c_updates_numeric
         << endl;
#endif

    distribution<CFloat> hidden_activation
        = multiply_r<CFloat>(noisy_input, W) + b.cast<CFloat>();

    distribution<CFloat> hidden_deriv
        = layer.derivative(hidden_activation);

    distribution<CFloat> b_updates
        = multiply_r<CFloat>(c_updates, W) * hidden_deriv;

#if 0
    cerr << "b_updates = " << c_updates << endl;

    distribution<CFloat> b_updates_numeric(no);

    // Calculate numerically the c updates
    for (unsigned i = 0;  i < no;  ++i) {

        double epsilon = 1e-9;

        // Apply the layer
        distribution<CFloat> hidden_act2
            = layer.activation(noisy_input);

        //cerr << "hidden_act[i] = " << hidden_act[i];
        hidden_act2[i] += epsilon;

        //cerr << " hidden_act2[i] = "
        //     << hidden_act2[i] << endl;

        //cerr << "hidden_act2 = " << hidden_act2 << endl;
                
        distribution<CFloat> hidden_rep2
            = layer.transfer(hidden_act2);
                
        //cerr << "hidden_rep = " << hidden_rep << endl;
        //cerr << "hidden_rep2 = " << hidden_rep2 << endl;

        distribution<CFloat> denoised_input2
            = layer.iapply(hidden_rep2);
            
        //cerr << "denoised_input = " << denoised_input << endl;
        //cerr << "denoised_input2 = " << denoised_input2 << endl;
                    

        // Error signal
        distribution<CFloat> diff2
            = model_input - denoised_input2;
            
        //cerr << "diff = " << diff << endl;
        //cerr << "diff2 = " << diff2 << endl;

        // Overall error
        double error2 = pow(diff2.two_norm(), 2);

        double delta = error2 - error;

        b_updates_numeric[i] = xdiv(delta, epsilon);

        cerr << "error = " << error << " error2 = " << error2
             << " delta = " << delta
             << " diff " << b_updates[i]
             << " diff2 " << b_updates_numeric[i] << endl;

    }

    cerr << "b_updates_numeric = " << b_updates_numeric
         << endl;
#endif

    boost::multi_array<double, 2> W_updates(boost::extents[ni][no]);

    distribution<double> factor_totals(no);

    for (unsigned i = 0;  i < ni;  ++i)
        SIMD::vec_add(&factor_totals[0], c_updates[i], &W[i][0],
                      &factor_totals[0], no);

    // TODO: why the extra factor of 2?  It doesn't make any sense...
    for (unsigned i = 0;  i < ni;  ++i) {
        calc_W_updates(c_updates[i] / 2.0,
                       &hidden_rep[0],
                       model_input[i] / 2.0,
                       &factor_totals[0],
                       &hidden_deriv[0],
                       &W_updates[i][0],
                       no);
    }
       
    
#if 0  // test numerically
    //boost::multi_array<double, 2> w2
    //    = layer.weights;

    for (unsigned i = 0;  i < ni;  ++i) {

        for (unsigned j = 0;  j < no;  ++j) {
            double epsilon = 1e-6;

            // Apply the layer
            distribution<CFloat> hidden_act2
                = layer.activation(noisy_input);
            hidden_act2[j] += epsilon * noisy_input[i];

            //cerr << "noisy_input = " << noisy_input << endl;
            //cerr << "hidden_act = " << hidden_act << endl;
            //cerr << "hidden_act2 = " << hidden_act2 << endl;

            //cerr << "diff = " << (hidden_act - hidden_act2) << endl;

            distribution<CFloat> hidden_rep2
                = layer.transfer(hidden_act2);
                    
            //cerr << "hidden_rep = " << hidden_rep << endl;
            //cerr << "hidden_rep2 = " << hidden_rep2 << endl;
                    
            distribution<CFloat> denoised_input2
                = layer.iapply(hidden_rep2);
                    
            //cerr << "denoised_input = " << denoised_input << endl;
            //cerr << "denoised_input2 = " << denoised_input2 << endl;

            // Error signal
            distribution<CFloat> diff2
                = model_input - denoised_input2;
                    
            //cerr << "diff = " << diff << endl;
            //cerr << "diff2 = " << diff2 << endl;
                    
            // Overall error
            double error2 = pow(diff2.two_norm(), 2);
                    
            double delta = error2 - error;

            double deriv2 = xdiv(delta, epsilon);

            cerr << "error = " << error << " error2 = " << error2
                 << " delta = " << delta
                 << " deriv " << W_updates[i][j]
                 << " deriv2 " << deriv2 << endl;

        }
    }
#endif  // if one/zero

    distribution<double> cleared_value_updates
        = W * b_updates;

#if 0  // test numerically
    for (unsigned i = 0;  i < ni;  ++i) {
        double epsilon = 1e-6;

        distribution<CFloat> noisy_input2 = noisy_input;
        noisy_input2[i] += epsilon;

        // Apply the layer
        distribution<CFloat> hidden_act2
            = layer.activation(noisy_input2);

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
        
        for (unsigned i = 0;  i < ni;  ++i)
            SIMD::vec_add(&updates.weights[i][0],
                          &W_updates[i][0],
                          &updates.weights[i][0], no);
    }
    else {
        for (unsigned i = 0;  i < ni;  ++i)
            if (isnan(noisy_input[i]))
                atomic_accumulate(updates.missing_replacements[i],
                                  cleared_value_updates[i]);
        atomic_accumulate(&updates.bias[0], &b_updates[0], no);
        atomic_accumulate(&updates.ibias[0], &c_updates[0], ni);

        for (unsigned i = 0;  i < ni;  ++i)
            atomic_accumulate(&updates.weights[i][0], &W_updates[i][0], no);
    }

    return error;
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
    double & error;
    boost::progress_display & progress;

    Train_Examples_Job(const Twoway_Layer & layer,
                       const vector<distribution<float> > & data,
                       int first, int last,
                       float prob_cleared,
                       const Thread_Context & context,
                       int random_seed,
                       Twoway_Layer & updates,
                       Lock & update_lock,
                       double & error,
                       boost::progress_display & progress)
        : layer(layer), data(data), first(first), last(last),
          prob_cleared(prob_cleared),
          context(context), random_seed(random_seed), updates(updates),
          update_lock(update_lock), error(error), progress(progress)
    {
    }

    void operator () ()
    {
        Thread_Context thread_context(context);
        thread_context.seed(random_seed);

        double total_error = 0.0;
        for (unsigned x = first;  x < last;  ++x)
            total_error += train_example(layer, data, x,
                                         prob_cleared, thread_context,
                                         updates, update_lock);

        Guard guard(update_lock);
        error += total_error;
        progress += (last - first);
    }
};

double
Twoway_Layer::
train_iter(const vector<distribution<float> > & data,
           float prob_cleared,
           Thread_Context & thread_context,
           int minibatch_size, float learning_rate)
{
    Worker_Task & worker = thread_context.worker();

    int nx = data.size();
    int ni JML_UNUSED = inputs();
    int no JML_UNUSED = outputs();

    int microbatch_size = minibatch_size / (num_cpus() * 4);
            
    boost::progress_display progress(nx, cerr);
    Lock progress_lock;

    double total_mse = 0.0;
    
    for (unsigned x = 0;  x < nx;  x += minibatch_size) {
                
        Twoway_Layer updates(ni, no, TF_IDENTITY);
                
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
                                       progress_lock,
                                       total_mse,
                                       progress);
                // Send it to a thread to be processed
                worker.add(job, "blend job", group);
            }
        }
                
        worker.run_until_finished(group);

        update(updates, learning_rate);
    }

    return sqrt(total_mse / nx);
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
    boost::progress_display & progress;

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
                      boost::progress_display & progress)
        : layer(layer), data_in(data_in), data_out(data_out),
          first(first), last(last),
          prob_cleared(prob_cleared),
          context(context), random_seed(random_seed),
          update_lock(update_lock),
          error_exact(error_exact), error_noisy(error_noisy),
          progress(progress)
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
        progress += (last - first);
    }
};

pair<double, double>
Twoway_Layer::
test_and_update(const vector<distribution<float> > & data_in,
                vector<distribution<float> > & data_out,
                float prob_cleared,
                Thread_Context & thread_context) const
{
    Lock update_lock;
    double error_exact = 0.0;
    double error_noisy = 0.0;

    int nx = data_in.size();

    boost::progress_display progress(nx, cerr);

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
                                  progress);
            
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
    char version;
    store >> version;
    if (version != 1)
        throw Exception("DNAE_Stack::reconstitute(): invalid version");
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
    boost::progress_display & progress;

    Test_Stack_Job(const DNAE_Stack & stack,
                   const vector<distribution<float> > & data,
                   int first, int last,
                   float prob_cleared,
                   const Thread_Context & context,
                   int random_seed,
                   Lock & update_lock,
                   double & error_exact,
                   double & error_noisy,
                   boost::progress_display & progress)
        : stack(stack), data(data),
          first(first), last(last),
          prob_cleared(prob_cleared),
          context(context), random_seed(random_seed),
          update_lock(update_lock),
          error_exact(error_exact), error_noisy(error_noisy),
          progress(progress)
    {
    }

    void operator () ()
    {
        Thread_Context thread_context(context);
        thread_context.seed(random_seed);

        double test_error_exact = 0.0, test_error_noisy = 0.0;

        for (unsigned x = first;  x < last;  ++x) {

            distribution<CFloat> input(0.8 * data[x]);

            distribution<bool> was_cleared;

            // Add noise
            distribution<CFloat> noisy_input
                = add_noise(input, thread_context, prob_cleared);
            
            distribution<CFloat> output = input, noisy_output = noisy_input;

            // Go down the stack
            for (unsigned l = 0;  l < stack.size();  ++l) {
                output = stack[l].apply(output);
                noisy_output = stack[l].apply(noisy_output);
            }

            distribution<CFloat> rep = output, noisy_rep = output;

            // Go back up the stack
            for (int l = stack.size() - 1;  l >= 0;  --l) {
                output = stack[l].iapply(output);
                noisy_output = stack[l].iapply(noisy_output);
            }
            
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
        progress += (last - first);
    }
};

pair<double, double>
DNAE_Stack::
test(const vector<distribution<float> > & data,
     float prob_cleared,
     Thread_Context & thread_context) const
{
    Lock update_lock;
    double error_exact = 0.0;
    double error_noisy = 0.0;

    int nx = data.size();

    boost::progress_display progress(nx, cerr);

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
                               progress);
            
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


    config.get(prob_cleared, "prob_cleared");
    config.get(learning_rate, "learning_rate");
    config.get(minibatch_size, "minibatch_size");
    config.get(niter, "niter");

    int nx = training_data.size();
    int nxt = testing_data.size();

    if (nx == 0)
        throw Exception("can't train on no data");

    static const int nlayers = 4;

    int layer_sizes[nlayers] = {100, 80, 50, 30};

    vector<distribution<float> > layer_train = training_data;
    vector<distribution<float> > layer_test = testing_data;

    // Do a SVD so that we can compare against it
    SVD_Decomposition svd;
    svd.train(training_data);

    // Learning rate is per-example
    learning_rate /= nx;

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

        Twoway_Layer layer(ni, nh, TF_TANH, thread_context);
        distribution<CFloat> cleared_values(ni);

        if (ni == nh && false) {
            //layer.zero_fill();
            for (unsigned i = 0;  i < ni;  ++i) {
                layer.weights[i][i] += 1.0;
            }
        }

        for (unsigned iter = 0;  iter < niter;  ++iter) {
            cerr << "iter " << iter << " training on " << nx << " examples"
                 << endl;
            Timer timer;

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

            distribution<LFloat> svalues(min(ni, nh));
            boost::multi_array<LFloat, 2> layer2 = layer.weights;

            int result = LAPack::gesdd("N",
                                       layer2.shape()[1],
                                       layer2.shape()[0],
                                       layer2.data(), layer2.shape()[1], 
                                       &svalues[0], 0, 1, 0, 1);
            
            if (result != 0)
                throw Exception("error in SVD");

            cerr << "svalues = " << svalues << endl;

            double train_error
                = layer.train_iter(layer_train, prob_cleared, thread_context,
                                   minibatch_size, learning_rate);

            cerr << "rmse of iteration: " << train_error << endl;
            cerr << timer.elapsed() << endl;


            timer.restart();
            double test_error_exact = 0.0, test_error_noisy = 0.0;
            
            cerr << "testing on " << nxt << " examples"
                 << endl;
            boost::tie(test_error_exact, test_error_noisy)
                = layer.test(layer_test, prob_cleared, thread_context);

            cerr << "testing rmse of iteration: exact "
                 << test_error_exact << " noisy " << test_error_noisy
                 << endl;
            cerr << timer.elapsed() << endl;
        }

        next_layer_train.resize(nx);
        next_layer_test.resize(nxt);

        // Calculate the inputs to the next layer
        
        cerr << "calculating next layer training inputs on "
             << nx << " examples" << endl;
        double train_error_exact = 0.0, train_error_noisy = 0.0;
        boost::tie(train_error_exact, train_error_noisy)
            = layer.test_and_update(layer_train, next_layer_train,
                                    prob_cleared, thread_context);

        cerr << "training rmse of layer: exact "
             << train_error_exact << " noisy " << train_error_noisy
             << endl;

        cerr << "calculating next layer testing inputs on "
             << nxt << " examples" << endl;
        double test_error_exact = 0.0, test_error_noisy = 0.0;
        boost::tie(test_error_exact, test_error_noisy)
            = layer.test_and_update(layer_test, next_layer_test,
                                    prob_cleared, thread_context);
        
        cerr << "testing rmse of layer: exact "
             << test_error_exact << " noisy " << test_error_noisy
             << endl;

        layer_train.swap(next_layer_train);
        layer_test.swap(next_layer_test);

        push_back(layer);

        // Test the layer stack
        cerr << "calculating whole stack testing performance on "
             << nxt << " examples" << endl;
        boost::tie(test_error_exact, test_error_noisy)
            = test(testing_data, prob_cleared, thread_context);
        
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
    char version;
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
