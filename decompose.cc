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

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/progress.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/bind.hpp>

#include "decomposition.h"
#include "svd_decomposition.h"
#include "dnae_decomposition.h"


using namespace std;
using namespace ML;
using namespace ML::DB;


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
    int niter = 50;

    config.get(prob_cleared, "prob_cleared");
    config.get(learning_rate, "learning_rate");
    config.get(minibatch_size, "minibatch_size");
    config.get(niter, "niter");

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
    
    DNAE_Stack stack;
    distribution<CFloat> cleared_values0;

    static const int nlayers = 4;

    int layer_sizes[nlayers] = {100, 80, 50, 30};

    vector<distribution<float> > layer_train(nx), layer_test(nxt);

    for (unsigned x = 0;  x < nx;  ++x)
        layer_train[x] = 0.8f * training_data.examples[x];

    for (unsigned x = 0;  x < nxt;  ++x)
        layer_test[x] = 0.8f * testing_data.examples[x];

    SVD_Decomposition svd;
    svd.train(layer_train);

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
             << test_error_exact << " noisy " << test_error_noisy
             << endl;

        layer_train.swap(next_layer_train);
        layer_test.swap(next_layer_test);

        stack.push_back(layer);
        if (layer_num == 0)
            cleared_values0 = cleared_values;

        // Test the layer stack
        cerr << "calculating whole stack testing performance on "
             << nxt << " examples" << endl;
        boost::tie(test_error_exact, test_error_noisy)
            = test_stack(stack, testing_data.examples, 0, nxt,
                         cleared_values0, prob_cleared, thread_context,
                         -1);
        
        cerr << "testing rmse of stack: exact "
             << test_error_exact << " noisy " << test_error_noisy
             << endl;
    }

    cerr << timer.elapsed() << endl;
}
