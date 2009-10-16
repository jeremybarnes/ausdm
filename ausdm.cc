/* ausdm.cc                                                       -*- C++ -*-
   Jeremy Barnes, 6 August 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   AusDM entry.
*/

#include "data.h"
#include "blender.h"

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
#include "stats/distribution_ops.h"

#include "boosting/worker_task.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/progress.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/bind.hpp>

#include "stats/moments.h"

using namespace std;
using namespace ML;
using namespace ML::Stats;

struct Predict_Job {
    Model_Output & result;
    Model_Output & baseline_result;
    int first, last;
    const Blender & blender;
    boost::shared_ptr<Blender> baseline_blender;
    const Data & data;
    boost::progress_display & progress;
    Lock & progress_lock;

    Predict_Job(Model_Output & result,
                Model_Output & baseline_result,
                int first, int last,
                const Blender & blender,
                boost::shared_ptr<Blender> baseline_blender,
                const Data & data,
                boost::progress_display & progress,
                Lock & progress_lock)
        : result(result), baseline_result(baseline_result),
          first(first), last(last), blender(blender),
          baseline_blender(baseline_blender),
          data(data), progress(progress), progress_lock(progress_lock)
    {
    }

    void operator () ()
    {
        for (int j = first;  j < last;  ++j) {
            distribution<float> model_inputs(data.models.size());
            for (unsigned i = 0;  i < data.models.size();  ++i)
                model_inputs[i] = data.models[i][j];

            correct_prediction = data.targets[j];

            float val = blender.predict(model_inputs);
            result.at(j) = val;

            if (baseline_blender)
                baseline_result.at(j)
                    = baseline_blender->predict(model_inputs);

            Guard guard(progress_lock);
            ++progress;
        }
    }
};

template<typename T>
T sqr(T val)
{
    return val * val;
}

int main(int argc, char ** argv)
{
    // Filename to dump output data to
    string output_file;

    // Configuration file to use
    string config_file = "config.txt";

    // Name of blender in config file
    string blender_name;

    // Name of baseline in config file
    string baseline_name;

    // Extra configuration options
    vector<string> extra_config_options;

    // Do we perform a fake test (with held-out data)?
    float hold_out_data = 0.0;

    // What type of target do we predict?
    string target_type;

    // How many cross-validation trials do we perform?
    int num_trials = 1;

    // Do we train on testing data?
    bool train_on_test = false;

    // Do we avoid decomposigion?
    bool no_decomposition = false;

    {
        using namespace boost::program_options;

        options_description config_options("Configuration");

        config_options.add_options()
            ("config-file,c", value<string>(&config_file),
             "configuration file to read configuration options from")
            ("blender-name,n", value<string>(&blender_name),
             "name of blender in configuration file")
            ("baseline-name", value<string>(&baseline_name),
             "name of baseline blender in configuration file")
            ("extra-config-option", value<vector<string> >(&extra_config_options),
             "extra configuration option=value (can go directly on command line)");

        options_description control_options("Control Options");

        control_options.add_options()
            ("hold-out-data,T", value<float>(&hold_out_data),
             "run a local test and score on held out data")
            ("target-type,t", value<string>(&target_type),
             "select target type: auc or rmse")
            ("num-trials,r", value<int>(&num_trials),
             "select number of trials to perform")
            ("train-on-test", value<bool>(&train_on_test)->zero_tokens(),
             "train on testing data as well (to test biasing, etc)" )
            ("no-decomposition", value<bool>(&no_decomposition)->zero_tokens(),
             "don't perform a decomposition")
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

    Target target;
    if (target_type == "auc") target = AUC;
    else if (target_type == "rmse") target = RMSE;
    else throw Exception("target type " + target_type + " not known");

    if (blender_name == "")
        blender_name = target_type;

    if (baseline_name == "")
        baseline_name = "baseline_" + target_type;

    // Load up configuration
    Configuration config;
    if (config_file != "") config.load(config_file);

    // Allow configuration to be overridden on the command line
    config.parse_command_line(extra_config_options);

    // Load up the data
    Timer timer;

    cerr << "loading data...";

    string targ_type_uc;
    if (target == AUC) targ_type_uc = "AUC";
    else if (target == RMSE) targ_type_uc = "RMSE";
    else throw Exception("unknown target type");


    Data decompose_training_data;
    if (!no_decomposition) {
        decompose_training_data.load("download/S_"
                                     + targ_type_uc + "_Train.csv", target);
        decompose_training_data.load("download/S_"
                                     + targ_type_uc + "_Score.csv", target,
                                     false);
        
        cerr << "training decomposition" << endl;
        decompose_training_data.decompose();
        cerr << "done" << endl;
    }

    vector<double> trial_scores;

    if (hold_out_data == 0.0 && num_trials > 1)
        throw Exception("need to hold out data for multiple trials");

    Model_Output result, baseline_result;

    for (unsigned trial = 0;  trial < num_trials;  ++trial) {
        if (num_trials > 1) cerr << "trial " << trial << endl;

        int rand_seed = hold_out_data > 0.0 ? 1 + trial : 0;

        Data data_train;
        data_train.load("download/S_" + targ_type_uc + "_Train.csv", target);

        Data data_test;
        if (!train_on_test) {
            if (hold_out_data > 0.0)
                data_train.hold_out(data_test, hold_out_data, rand_seed);
            else data_test.load("download/S_"
                                + targ_type_uc + "_Score.csv", target);
        }

        // Calculate the scores necessary for the job
        data_train.calc_scores();

        if (!no_decomposition)
            data_train.apply_decomposition(decompose_training_data);
        data_train.stats();
        
        data_test.stats();

        distribution<float> example_weights(data_train.targets.size(),
                                            1.0 / data_train.targets.size());

        boost::shared_ptr<Blender> blender
            = get_blender(config, blender_name, data_train,
                          example_weights, rand_seed, target);
        

        boost::shared_ptr<Blender> baseline_blender;
        if (baseline_name != "")
            baseline_blender = get_blender(config, baseline_name, data_train,
                                           example_weights, rand_seed, target);
        
        if (train_on_test && hold_out_data > 0.0) {
            data_train.hold_out(data_test, hold_out_data, rand_seed);
            data_test.stats();
        }

        int np = data_test.targets.size();
        
        // Now run the model
        result.resize(np);
        baseline_result.resize(np);
        
        static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
        
        cerr << "processing " << np << " predictions..." << endl;
        
        boost::progress_display progress(np, cerr);
        Lock progress_lock;
        
        // Now, submit it as jobs to the worker task to be done multithreaded
        int group;
        int job_num = 0;
        {
            int parent = -1;  // no parent group
            group = worker.get_group(NO_JOB, "dump user results task", parent);
            
            // Make sure the group gets unlocked once we've populated
            // everything
            Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                         boost::ref(worker),
                                         group));
            
            for (unsigned i = 0;  i < np;  i += 100, ++job_num) {
                int last = std::min<int>(np, i + 100);
                
                // Create the job
                Predict_Job job(result, baseline_result,
                                i, last,
                                *blender, baseline_blender,
                                data_test, progress, progress_lock);
                
                // Send it to a thread to be processed
                worker.add(job, "blend job", group);
            }
        }
        
        // Add this thread to the thread pool until we're ready
        worker.run_until_finished(group);
        
        cerr << " done." << endl;
        
        cerr << timer.elapsed() << endl;

        if (hold_out_data > 0.0) {
            int npt = data_test.targets.size();

            cerr << "result = " << result << endl;
            cerr << "baseline = " << baseline_result << endl;

            double score = result.calc_score(data_test.targets, target);
            cerr << format("score: %0.4f", score);
            if (baseline_blender) {
                double baseline_score
                    = baseline_result.calc_score(data_test.targets, target);
                cerr << format("  baseline: %0.4f  diff: %0.5f",
                               baseline_score, score - baseline_score);
                cerr << endl;

                vector<pair<float, float> > ranked_targets, baseline_ranked_targets;
                for (unsigned i = 0;  i < npt;  ++i) {
                    ranked_targets.push_back(make_pair(result[i],
                                                       data_test.targets[i]));
                    baseline_ranked_targets.push_back
                        (make_pair(baseline_result[i], data_test.targets[i]));
                }

                sort_on_first_ascending(ranked_targets);
                sort_on_first_ascending(baseline_ranked_targets);

                distribution<float> pos_scores, neg_scores, bl_pos_scores, bl_neg_scores;
                int pos_total = 0, neg_total = 0;
                pos_scores.push_back(0);  neg_scores.push_back(0);
                for (unsigned i = 0;  i < npt;  ++i) {
                    if (ranked_targets[i].second == -1.0)
                        ++neg_total;
                    else ++pos_total;
                    pos_scores.push_back(pos_total);
                    neg_scores.push_back(neg_total);
                }

                pos_scores /= pos_scores.max();
                neg_scores = (1.0f - (neg_scores / neg_scores.max()));

                pos_total = 0; neg_total = 0;
                bl_pos_scores.push_back(0);  bl_neg_scores.push_back(0);
                for (unsigned i = 0;  i < npt;  ++i) {
                    if (baseline_ranked_targets[i].second == -1.0)
                        ++neg_total;
                    else ++pos_total;
                    bl_pos_scores.push_back(pos_total);
                    bl_neg_scores.push_back(neg_total);
                }

                bl_pos_scores /= bl_pos_scores.max();
                bl_neg_scores = (1.0f - (bl_neg_scores / bl_neg_scores.max()));

                distribution<float> ranked = result;
                distribution<float> baseline_ranked = baseline_result;

                std::sort(ranked.begin(), ranked.end());
                std::sort(baseline_ranked.begin(), baseline_ranked.end());

                // Look at individual error terms
                vector<pair<int, float> > improvements;

                vector<float> errors_pred, errors_bl;

                distribution<double>
                    category_errors(4),
                    category_counts(4),
                    baseline_category_errors(4),
                    category_improvements(4);

                for (unsigned i = 0;  i < npt;  ++i) {
                    float pred  = result[i];
                    float bl    = baseline_result[i];
                    float label = data_test.targets[i];

                    float improvement;
                    float error_pred, error_bl;
                    if (target == AUC) {
                        int upos, lpos, bl_upos, bl_lpos, needed;
                        lpos = std::lower_bound(ranked.begin(),
                                                ranked.end(),
                                                pred)
                            - ranked.begin();
                        bl_lpos = std::lower_bound(baseline_ranked.begin(),
                                                   baseline_ranked.end(),
                                                   bl)
                            - baseline_ranked.begin();
                        upos = std::upper_bound(ranked.begin(),
                                                ranked.end(),
                                                pred)
                            - ranked.begin();
                        bl_upos = std::upper_bound(baseline_ranked.begin(),
                                                   baseline_ranked.end(),
                                                   bl)
                            - baseline_ranked.begin();

                        needed = npt;

                        if (label == -1.0) {
                            error_pred = (pos_scores.at(lpos) + pos_scores.at(upos)) / 2.0;
                            error_bl = (bl_pos_scores.at(bl_lpos) + bl_pos_scores.at(bl_upos)) / 2.0;
                        }
                        else {
                            error_pred = (neg_scores.at(lpos) + neg_scores.at(upos)) / 2.0;
                            error_bl = (bl_neg_scores.at(bl_lpos) + bl_neg_scores.at(bl_upos)) / 2.0;
                        }

                        improvement = error_bl - error_pred;
                    }
                    else {
                        error_pred  = sqr(pred - label);
                        error_bl    = sqr(bl - label);
                        improvement = error_bl - error_pred;
                    }

                    errors_pred.push_back(error_pred);
                    errors_bl.push_back(error_bl);
                    improvements.push_back(make_pair(i, improvement));

                    int cat = data_test.target_difficulty[i].category;
                    category_errors[cat] += error_pred;
                    category_counts[cat] += 1.0;
                    baseline_category_errors[cat] += error_bl;
                    category_improvements[cat] += improvement;
                }

                distribution<double> avg_error
                    = xdiv(category_errors, category_counts);
                distribution<double> bl_avg_error
                    = xdiv(baseline_category_errors, category_counts);
                distribution<double> avg_improvement
                    = xdiv(category_improvements, category_counts);

                for (unsigned i = 0;  i < 4;  ++i) {
                    Difficulty_Category cat = (Difficulty_Category)i;
                    cerr << "category " << cat << ": count "
                         << category_counts[i]
                         << " avg error " << avg_error[i]
                         << " baseline avg error " << bl_avg_error[i]
                         << " avg improvement "
                         << avg_improvement[i] << endl;
                }
                cerr << "overall: count " << category_counts.total()
                     << " avg error "
                     << avg_error.dotprod(category_counts)
                          / category_counts.total()
                     << " baseline avg error "
                     << bl_avg_error.dotprod(category_counts)
                          / category_counts.total()
                     << " avg improvement "
                     << avg_improvement.dotprod(category_counts)
                          / category_counts.total()
                     << endl;


                sort_on_second_ascending(improvements);

                cerr << "worst entries: " << endl;
                for (unsigned ii = 0;  ii < min(npt, 50);  ++ii) {
                    int i = improvements[ii].first;

                    float pred  = result[i];
                    float bl    = baseline_result[i];
                    float label = data_test.targets[i];

                    cerr << ii << ": " << i << " " << " label: " << label
                         << " pred: " << pred << " bl: " << bl
                         << " " << data_test.target_difficulty[i].category;

                    float improvement = improvements[ii].second;
                    float error_pred  = errors_pred[i];
                    float error_bl    = errors_bl[i];

                    cerr << " error_pred: " << error_pred
                         << " error_bl: " << error_bl;

                    cerr << " improvement: " << improvement << endl;

                    distribution<float> model_inputs(data_test.models.size());
                    for (unsigned j = 0;  j < data_test.models.size();  ++j)
                        model_inputs[j] = data_test.models[j][i];

                    cerr << "    min: " << model_inputs.min()
                         << "  max: " << model_inputs.max() << " avg: "
                         << model_inputs.mean() << endl;

                    cerr << "explanation: " << endl;
                    cerr << blender->explain(model_inputs) << endl << endl;
                }

                cerr << "best entries: " << endl;
                for (unsigned ii = 0;  ii < min(npt, 50);  ++ii) {
                    int i = improvements[improvements.size() - ii - 1].first;

                    float pred  = result[i];
                    float bl    = baseline_result[i];
                    float label = data_test.targets[i];

                    cerr << ii << ": " << i << " " << " label: " << label
                         << " pred: " << pred << " bl: " << bl
                         << " " << data_test.target_difficulty[i].category;

                    float improvement = improvements[improvements.size() - ii - 1].second;
                    float error_pred  = errors_pred[i];
                    float error_bl    = errors_bl[i];

                    cerr << " error_pred: " << error_pred
                         << " error_bl: " << error_bl;

                    cerr << " improvement: " << improvement << endl;
                }

            }
            cerr << endl;

            trial_scores.push_back(score);

            vector<distribution<float> > weights(4, distribution<float>(npt, 0.0));
            for (unsigned i = 0;  i < npt;  ++i) {
                //cerr << "cat = " << data_test.target_difficulty[i].category
                //     << endl;
                weights.at(data_test.target_difficulty[i].category)[i] = 1.0;
            }

            for (unsigned i = 0;  i < 4;  ++i) {
                Difficulty_Category cat = (Difficulty_Category)i;
                cerr << "total is " << weights[i].total() << endl;
                //weights[i].normalize();
                double score = result.calc_score(data_test.targets,
                                                 weights[i],
                                                 target);
                cerr << "score for " << cat << ": "
                     << format("%.4f", score);

                if (baseline_blender) {
                    double baseline_score
                        = baseline_result.calc_score(data_test.targets,
                                                     weights[i],
                                                     target);
                    cerr << format(" baseline: %.4f diff: %6.4f",
                                   baseline_score, score - baseline_score);
                }

                cerr << endl;
            }
        }
    }

    if (hold_out_data > 0.0) {
        double mean = Stats::mean(trial_scores.begin(), trial_scores.end());
        double std = Stats::std_dev(trial_scores.begin(), trial_scores.end(),
                                    mean);
        
        cout << "scores: " << trial_scores << endl;
        cout << format("%6.4f +/- %6.4f", mean, std) << endl;
    }

    if (output_file != "" && num_trials == 1) {
        filter_ostream out(output_file);
        for (unsigned i = 0;  i < result.size();  ++i)
            out << format("%.6f", result[i] * 1000.0) << endl;
    }
}
