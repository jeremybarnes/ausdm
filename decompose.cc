/* ausdm.cc                                                        -*- C++ -*-
   Jeremy Barnes, 6 August 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   AusDM entry.
*/

#include "data.h"

#include <fstream>
#include <iterator>
#include <iostream>

#include "arch/exception.h"
#include "utils/string_functions.h"
#include "utils/pair_utils.h"
#include "utils/vector_utils.h"
#include "utils/filter_streams.h"
#include "utils/configuration.h"
#include "arch/timers.h"
#include "utils/info.h"
#include "utils/guard.h"
#include "arch/threads.h"

#include "boosting/worker_task.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/progress.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/bind.hpp>

#include "boosting/perceptron_generator.h"
#include "algebra/matrix_ops.h"
#include "math/xdiv.h"


using namespace std;
using namespace ML;

void calc_W_updates(double k1, const double * x, double k2, const double * y,
                    const double * z, double * r, size_t n)
{
    return SIMD::vec_k1_x_plus_k2_y_z(k1, x, k2, y, z, r, n);
}

template<typename Val>
void atomic_add(Val & value, const Val & increment)
{
    Val old_val = value, new_val;
    do {
        new_val = old_val + increment;
    } while (!JML_LIKELY(cmp_xchg(value, old_val, new_val)));
}

template<typename Val>
void atomic_update_vec(Val * old, const Val * increment, int n)
{
    for (unsigned i = 0;  i < n;  ++i)
        atomic_add(old[i], increment[i]);
}

typedef double LFloat;

struct Twoway_Layer : public Dense_Layer<LFloat> {
    Twoway_Layer(size_t inputs, size_t outputs,
                 Transfer_Function_Type transfer,
                 Thread_Context & context)
        : Dense_Layer<LFloat>(inputs, outputs, transfer)
    {
        ibias.resize(inputs);
        random_fill(1.0 / sqrt(inputs), context);
    }

    Twoway_Layer(size_t inputs, size_t outputs,
                 Transfer_Function_Type transfer)
        : Dense_Layer<LFloat>(inputs, outputs, transfer)
    {
        ibias.resize(inputs);
        ibias.fill(0.0);
    }
    distribution<LFloat> ibias;

    distribution<double> iapply(const distribution<double> & output) const
    {
        distribution<double> activation = weights * output;
        activation += ibias;
        transfer(&activation[0], &activation[0], inputs(), transfer_function);
        return activation;
    }

    distribution<float> iapply(const distribution<float> & output) const
    {
        distribution<float> activation = multiply_r<float>(weights, output);
        activation += ibias;
        transfer(&activation[0], &activation[0], inputs(), transfer_function);
        return activation;
    }

    distribution<double>
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
    itransfer(const distribution<float> & activation) const
    {
        int ni = inputs();
        if (activation.size() != ni)
            throw Exception("invalid sizes in itransfer");
        distribution<float> result(ni);
        transfer(&activation[0], &result[0], ni, transfer_function);
        return activation;
    }

    distribution<double> iderivative(const distribution<double> & input) const
    {
        if (input.size() != this->inputs())
            throw Exception("iderivative(): wrong size");
        int ni = this->inputs();
        distribution<double> result(ni);
        derivative(&input[0], &result[0], ni, transfer_function);
        return result;
    }

    distribution<float> iderivative(const distribution<float> & input) const
    {
        if (input.size() != this->inputs())
            throw Exception("iderivative(): wrong size");
        int ni = this->inputs();
        distribution<float> result(ni);
        derivative(&input[0], &result[0], ni, transfer_function);
        return result;
    }

    void update(const Twoway_Layer & updates, double learning_rate)
    {
        int ni = inputs();
        int no = outputs();

        ibias -= learning_rate * updates.ibias;
        bias -= learning_rate * updates.bias;
        
        for (unsigned i = 0;  i < ni;  ++i)
            SIMD::vec_add(&weights[i][0], -learning_rate,
                          &updates.weights[i][0],
                          &weights[i][0], no);
    }

    virtual void random_fill(float limit, Thread_Context & context)
    {
        Dense_Layer<LFloat>::random_fill(limit, context);
        for (unsigned i = 0;  i < ibias.size();  ++i)
            ibias[i] = limit * (context.random01() * 2.0f - 1.0f);
    }

    virtual void zero_fill()
    {
        Dense_Layer<LFloat>::zero_fill();
        ibias.fill(0.0);
    }
};

// Float type to use for calculations
typedef double CFloat;

template<typename Float>
distribution<Float>
add_noise(const distribution<Float> & inputs,
          const distribution<Float> & cleared_values,
          distribution<bool> & was_cleared,
          Thread_Context & context,
          float prob_cleared)
{
    distribution<Float> result = inputs;

    was_cleared.clear();
    was_cleared.resize(inputs.size());

    for (unsigned i = 0;  i < inputs.size();  ++i) {
        if (context.random01() < prob_cleared) {
            result[i] = cleared_values[i];
            was_cleared[i] = true;
        }
    }
    
    return result;
}


double train_example(const Twoway_Layer & layer,
                     const vector<distribution<float> > & data,
                     int example_num,
                     const distribution<CFloat> & cleared_values,
                     distribution<double> & cleared_values_update,
                     float prob_cleared,
                     Thread_Context & thread_context,
                     Twoway_Layer & updates,
                     int iter,
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

    // Which ones had noise added?
    distribution<bool> was_cleared;

    // Add noise
    distribution<CFloat> noisy_input
        = add_noise(model_input, cleared_values, was_cleared, thread_context,
                    prob_cleared);

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
        cerr << "iter " << iter << " ex " << example_num << endl;
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

        double epsilon = 1e-5;

        // Apply the layer
        distribution<CFloat> hidden_act2
            = layer.activation(noisy_input);

        hidden_act2[i] += epsilon;
                
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

    for (unsigned i = 0;  i < ni;  ++i) {
        calc_W_updates(c_updates[i],
                       &hidden_rep[0],
                       model_input[i],
                       &factor_totals[0],
                       &hidden_deriv[0],
                       &W_updates[i][0],
                       no);
    }
            

#if 0  // test numerically
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

#if 0
            // look at dh/dwij
            distribution<double> dh_dwij2
                = (hidden_rep2 - hidden_rep) / epsilon;

            cerr << "dh_dwij = " << dh_dwij << endl;
            cerr << "dh_dwij2 = " << dh_dwij2 << endl;
#endif
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
            if (was_cleared[i])
                cleared_values_update[i] += cleared_value_updates[i];
        
        updates.bias += b_updates;
        updates.ibias += c_updates;
        
        for (unsigned i = 0;  i < ni;  ++i)
            SIMD::vec_add(&updates.weights[i][0],
                          &W_updates[i][0],
                          &updates.weights[i][0], no);
    }
    else {
        for (unsigned i = 0;  i < ni;  ++i)
            if (was_cleared[i])
                atomic_add(cleared_values_update[i], cleared_value_updates[i]);
        atomic_update_vec(&updates.bias[0], &b_updates[0], no);
        atomic_update_vec(&updates.ibias[0], &c_updates[0], ni);

        for (unsigned i = 0;  i < ni;  ++i)
            atomic_update_vec(&updates.weights[i][0], &W_updates[i][0], no);
    }

    return error;
}

struct Train_Examples_Job {

    const Twoway_Layer & layer;
    const vector<distribution<float> > & data;
    int first;
    int last;
    const distribution<CFloat> & cleared_values;
    float prob_cleared;
    const Thread_Context & context;
    int random_seed;
    Twoway_Layer & updates;
    distribution<double> & cleared_values_update;
    int iter;
    Lock & update_lock;
    double & error;
    boost::progress_display & progress;

    Train_Examples_Job(const Twoway_Layer & layer,
                       const vector<distribution<float> > & data,
                       int first, int last,
                       const distribution<CFloat> & cleared_values,
                       float prob_cleared,
                       const Thread_Context & context,
                       int random_seed,
                       Twoway_Layer & updates,
                       distribution<double> & cleared_values_update,
                       int iter,
                       Lock & update_lock,
                       double & error,
                       boost::progress_display & progress)
        : layer(layer), data(data), first(first), last(last),
          cleared_values(cleared_values), prob_cleared(prob_cleared),
          context(context), random_seed(random_seed), updates(updates),
          cleared_values_update(cleared_values_update), iter(iter),
          update_lock(update_lock), error(error), progress(progress)
    {
    }

    void operator () ()
    {
        Thread_Context thread_context(context);
        thread_context.seed(random_seed);

        double total_error = 0.0;
        for (unsigned x = first;  x < last;  ++x)
            total_error += train_example(layer, data, x, cleared_values,
                                         cleared_values_update,
                                         prob_cleared, thread_context,
                                         updates, iter, update_lock);

        Guard guard(update_lock);
        error += total_error;
        progress += (last - first);
    }
};

double
train_layer(Twoway_Layer & layer,
            distribution<CFloat> & cleared_values,
            const vector<distribution<float> > & data,
            int first, int last,
            float prob_cleared,
            Thread_Context & thread_context,
            int iter, int minibatch_size, float learning_rate)
{
    static Worker_Task & worker
        = Worker_Task::instance(num_threads() - 1);

    int nx = data.size();
    int ni JML_UNUSED = layer.inputs();
    int no JML_UNUSED= layer.outputs();

    int microbatch_size = minibatch_size / (num_cpus() * 4);
            
    boost::progress_display progress(nx, cerr);
    Lock progress_lock;

    double total_mse = 0.0;
    
    for (unsigned x = 0;  x < nx;  x += minibatch_size) {
                
        Twoway_Layer updates(ni, no, TF_TANH);
        distribution<double> cleared_value_updates(ni);
                
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
                        
                Train_Examples_Job job(layer,
                                       data,
                                       x2,
                                       min(x + minibatch_size,
                                           x2 + microbatch_size),
                                       cleared_values, prob_cleared,
                                       thread_context,
                                       thread_context.random(),
                                       updates, cleared_value_updates,
                                       iter, progress_lock,
                                       total_mse,
                                       progress);
                // Send it to a thread to be processed
                worker.add(job, "blend job", group);
            }
        }
                
        worker.run_until_finished(group);


        layer.update(updates, learning_rate);
                
        //cerr << "cleared_value_updates = " << cleared_value_updates
        //     << endl;
        //cerr << "c_updates = " << updates.ibias << endl;
        
        cleared_values -= 100.0 * learning_rate * cleared_value_updates;
    }

    return sqrt(total_mse / nx);
}

struct Test_Examples_Job {

    const Twoway_Layer & layer;
    const vector<distribution<float> > & data_in;
    vector<distribution<float> > & data_out;
    int first;
    int last;
    const distribution<CFloat> & cleared_values;
    float prob_cleared;
    const Thread_Context & context;
    int random_seed;
    int iter;
    Lock & update_lock;
    double & error_exact;
    double & error_noisy;
    boost::progress_display & progress;

    Test_Examples_Job(const Twoway_Layer & layer,
                      const vector<distribution<float> > & data_in,
                      vector<distribution<float> > & data_out,
                      int first, int last,
                      const distribution<CFloat> & cleared_values,
                      float prob_cleared,
                      const Thread_Context & context,
                      int random_seed,
                      int iter,
                      Lock & update_lock,
                      double & error_exact,
                      double & error_noisy,
                      boost::progress_display & progress)
        : layer(layer), data_in(data_in), data_out(data_out),
          first(first), last(last),
          cleared_values(cleared_values), prob_cleared(prob_cleared),
          context(context), random_seed(random_seed), iter(iter),
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
                = add_noise(model_input, cleared_values, was_cleared,
                            thread_context, prob_cleared);
            
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
        }

        Guard guard(update_lock);
        error_exact += test_error_exact;
        error_noisy += test_error_noisy;
        progress += (last - first);
    }
};

pair<double, double>
test_layer(const Twoway_Layer & layer,
           const vector<distribution<float> > & data_in,
           vector<distribution<float> > & data_out,
           int first, int last,
           const distribution<CFloat> & cleared_values,
           float prob_cleared,
           Thread_Context & thread_context,
           int iter)
{
    Lock update_lock;
    double error_exact = 0.0;
    double error_noisy = 0.0;

    int nx = data_in.size();

    boost::progress_display progress(nx, cerr);

    static Worker_Task & worker
        = Worker_Task::instance(num_threads() - 1);
            
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
            
            Test_Examples_Job job(layer, data_in, data_out,
                                  x, min<int>(x + batch_size, nx),
                                  cleared_values, prob_cleared,
                                  thread_context,
                                  thread_context.random(),
                                  iter, update_lock,
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

int main(int argc, char ** argv)
{
    // Filename to dump decomposition to
    string output_file;

    // Configuration file to use
    string config_file = "config.txt";

    // Name of decomposer in config file
    string decomposer_name;

    // Extra configuration options
    vector<string> extra_config_options;

    // Probability that it's cleared
    float prob_cleared = 0.10;

    // What type of target do we predict?
    //string target_type;
    {
        using namespace boost::program_options;

        options_description config_options("Configuration");

        config_options.add_options()
            ("config-file,c", value<string>(&config_file),
             "configuration file to read configuration options from")
            ("decomposer-name,n", value<string>(&decomposer_name),
             "name of decomposer in configuration file")
            ("extra-config-option", value<vector<string> >(&extra_config_options),
             "extra configuration option=value (can go directly on command line)");

        options_description control_options("Control Options");

        control_options.add_options()
            //("target-type,t", value<string>(&target_type),
            // "select target type: auc or rmse")
            ("output-file,o",
             value<string>(&output_file),
             "dump output file to the given filename");

        positional_options_description p;
        p.add("extra-config-option", -1);

        options_description all_opt;
        all_opt
            .add(config_options)
            .add(control_options);

        all_opt.add_options()
            ("help,h", "print this message");
        
        variables_map vm;
        store(command_line_parser(argc, argv)
              .options(all_opt)
              .positional(p)
              .run(),
              vm);
        notify(vm);

        if (vm.count("help")) {
            cout << all_opt << endl;
            return 1;
        }
    }

#if 0
    Target target;
    if (target_type == "auc") target = AUC;
    else if (target_type == "rmse") target = RMSE;
    else throw Exception("target type " + target_type + " not known");

    if (decomposer_name == "")
        decomposer_name = target_type;
#endif

    // Load up configuration
    Configuration config;
    if (config_file != "") config.load(config_file);

    // Allow configuration to be overridden on the command line
    config.parse_command_line(extra_config_options);

    double learning_rate = 0.75;
    int minibatch_size = 256;

    config.get(prob_cleared, "prob_cleared");
    config.get(learning_rate, "learning_rate");
    config.get(minibatch_size, "minibatch_size");

    // Load up the data
    Timer timer;

    cerr << "loading data...";

    // The decomposition is entirely unsupervised, and so doesn't use the
    // label at all; thus we can put the training and validation sets together

    // Data[s/m/l][auc/rmse]
    Data data[3][2][2];

    const char * const size_names[3]   = { "S", "M", "L" };
    const char * const target_names[2] = { "RMSE", "AUC" };
    const char * const set_names[2]    = { "Train", "Score" };

    set<string> model_names;

    // (Small only for the moment until we get the hang of it)

    for (unsigned i = 0;  i < 1;  ++i) {
        for (unsigned j = 0;  j < 1;  ++j) {
            for (unsigned k = 0;  k < 2;  ++k) {
                string filename = format("download/%s_%s_%s.csv",
                                         size_names[i],
                                         target_names[j],
                                         set_names[k]);
                Data & this_data = data[i][j][k];

                this_data.load(filename, (Target)j, true /*(k == 0)*/ /* clear first */);
                
                model_names.insert(this_data.model_names.begin(),
                                   this_data.model_names.end());
            }
        }
    }

    cerr << "done" << endl;

    cerr << model_names.size() << " total models" << endl;

    // Now, we look at all of the column names to get an idea of the input
    // dimensions.  


    // Denoising auto encoder
    // We train a stack of layers, one at a time

    const Data & training_data = data[0][0][0];
    const Data & testing_data  = data[0][0][1];
    
    int nx = training_data.nx();
    int nxt = testing_data.nx();

    Thread_Context thread_context;
    
    vector<Twoway_Layer> layers;

    static const int nlayers = 3;

    int layer_sizes[nlayers] = {100, 50, 30};

    vector<distribution<float> > layer_train(nx), layer_test(nxt);

    for (unsigned x = 0;  x < nx;  ++x)
        layer_train[x] = 0.8f * training_data.examples[x];

    for (unsigned x = 0;  x < nxt;  ++x)
        layer_test[x] = 0.8f * testing_data.examples[x];

    // Learning rate is per-example
    learning_rate /= nx;

    for (unsigned layer_num = 0;  layer_num < nlayers;  ++layer_num) {
        cerr << endl << endl << endl << "--------- LAYER " << layer_num
             << " ---------" << endl << endl;

        vector<distribution<float> > next_layer_train, next_layer_test;

        int ni
            = layer_num == 0
            ? training_data.nm()
            : layer_sizes[layer_num - 1];

        if (ni != layer_train[0].size())
            throw Exception("ni is wrong");

        int nh = layer_sizes[layer_num];

        Twoway_Layer layer(ni, nh, TF_TANH, thread_context);
        distribution<CFloat> cleared_values(ni);

        for (unsigned iter = 0;  iter < 50;  ++iter) {
            cerr << "iter " << iter << " training on " << nx << " examples"
                 << endl;
            Timer timer;
            
            double train_error
                = train_layer(layer, cleared_values,
                              layer_train, 0, nx, prob_cleared,
                              thread_context, iter, minibatch_size,
                              learning_rate);

            cerr << "rmse of iteration: " << train_error << endl;
            cerr << timer.elapsed() << endl;


            timer.restart();
            double test_error_exact = 0.0, test_error_noisy = 0.0;
            
            cerr << "testing on " << nxt << " examples"
                 << endl;
            boost::tie(test_error_exact, test_error_noisy)
                = test_layer(layer, layer_test, next_layer_test, 0, nxt,
                             cleared_values, prob_cleared, thread_context,
                             iter);

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
            = test_layer(layer, layer_train, next_layer_train, 0, nx,
                         cleared_values, prob_cleared, thread_context,
                         -1);

        cerr << "training rmse of layer: exact "
             << train_error_exact << " noisy " << train_error_noisy
             << endl;

        cerr << "calculating next layer testing inputs on "
             << nxt << " examples" << endl;
        double test_error_exact = 0.0, test_error_noisy = 0.0;
        boost::tie(test_error_exact, test_error_noisy)
            = test_layer(layer, layer_test, next_layer_test, 0, nxt,
                         cleared_values, prob_cleared, thread_context,
                         -1);
        
        cerr << "testing rmse of layer: exact "
             << train_error_exact << " noisy " << train_error_noisy
             << endl;

        layer_train.swap(next_layer_train);
        layer_test.swap(next_layer_test);

        layers.push_back(layer);

        // Test the layer stack
        
        
    }

    cerr << timer.elapsed() << endl;
}
