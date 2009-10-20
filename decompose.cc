/* ausdm.cc                                                       -*- C++ -*-
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

using namespace std;
using namespace ML;

struct Twoway_Layer : public Layer {
    Twoway_Layer(size_t inputs, size_t outputs, Activation activation)
        : Layer(inputs, outputs, activation)
    {
        ibias.resize(inputs);
    }

    distribution<float> ibias;

    distribution<float> iapply(const distribution<float> & output) const
    {
        distribution<float> result = weights * output;
        result += ibias;
        transform(result);
        return result;
    }
};

distribution<float>
add_noise(const distribution<float> & inputs,
          const distribution<float> & cleared_values,
          Thread_Context & context,
          float prob_cleared)
{
    distribution<float> result = inputs;

    for (unsigned i = 0;  i < inputs.size();  ++i)
        if (context.random01() < prob_cleared)
            result[i] = cleared_values[i];
    
    return result;
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

    // Load up the data
    Timer timer;

    cerr << "loading data...";

    // The decomposition is entirely unsupervised, and so doesn't use the
    // label at all; thus we can put the training and validation sets together

    // Data[s/m/l][auc/rmse]
    Data data[3][2];

    const char * const size_names[3]   = { "S", "M", "L" };
    const char * const target_names[2] = { "RMSE", "AUC" };
    const char * const set_names[2]    = { "Train", "Score" };

    set<string> model_names;

    // (Small only for the moment until we get the hang of it)

    for (unsigned i = 0;  i < 1;  ++i) {
        for (unsigned j = 0;  j < 2;  ++j) {
            for (unsigned k = 0;  k < 2;  ++k) {
                string filename = format("download/%s_%s_%s.csv",
                                         size_names[i],
                                         target_names[j],
                                         set_names[k]);
                Data & this_data = data[i][j];

                this_data.load(filename, (Target)j, (k == 0) /* clear first */);
                
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

    const Data & to_train
        = data[0][0];
    
    int nx = to_train.targets.size();
    int nm = to_train.models.size();

    Twoway_Layer layer(nm, nm / 2, ACT_LOGSIG);

    bool prob_cleared = 0.25;
    distribution<float> cleared_values(nm);

    Thread_Context thread_context;
    
    double total_mse = 0.0;

    double learning_rate = 1e-5;

    for (unsigned iter = 0;  iter < 5;  ++iter) {
        cerr << "iter " << iter << " training on " << nx << " examples"
             << endl;
        boost::progress_display progress(nx, cerr);

        for (unsigned x = 0;  x < nx;  ++x, ++progress) {
            // Present this input
            distribution<float> model_input(nm);
            for (unsigned m = 0;  m < nm;  ++m)
                model_input[m] = to_train.models[m][x];
            
            // Add noise
            distribution<float> noisy_input
                = add_noise(model_input, cleared_values, thread_context,
                            prob_cleared);
            
            // Apply the layer
            distribution<float> hidden_rep
                = layer.apply(noisy_input);
            
            // Reconstruct the input
            distribution<float> denoised_input
                = layer.iapply(hidden_rep);
            
            // Error signal
            distribution<float> diff
                = model_input - denoised_input;
            
            // Overall error
            float error = pow(diff.two_norm(), 2);
            
            total_mse += error;
        
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
        
            boost::multi_array<float, 2> & W
                = layer.weights;
            distribution<float> & b = layer.bias;
            distribution<float> & c = layer.ibias;

            distribution<float> c_updates
                = 2 * diff * layer.derivative(denoised_input);

            distribution<float> hidden_activation
                = noisy_input * W + b;

            distribution<float> hidden_deriv
                = layer.derivative(hidden_activation);

            distribution<float> b_updates
                = c_updates * W * hidden_deriv;

            //boost::multi_array<float, 2> W_updates
            //    = c_updates * (hidden_rep + noisy_input * W * hidden_deriv);
        
            c -= learning_rate * c_updates;
            b -= learning_rate * b_updates;
            //W -= learning_rate * W_updates;
        }

        cerr << "rmse of iteration: " << sqrt(total_mse / nx)
             << endl;
    }
}
