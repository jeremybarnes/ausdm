/* boosting_blender.cc
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender that uses boosting.
*/

#include "boosting_blender.h"
#include "utils/vector_utils.h"
#include "boosting/boosting_core.h"


using namespace ML;
using namespace std;


/*****************************************************************************/
/* BOOSTING_BLENDER                                                          */
/*****************************************************************************/

Boosting_Blender::Boosting_Blender()
{
}

Boosting_Blender::~Boosting_Blender()
{
}
    
void
Boosting_Blender::
configure(const ML::Configuration & config,
          const std::string & name,
          int random_seed)
{
    Linear_Blender::configure(config, name, random_seed);
}
    
void
Boosting_Blender::
init(const Data & training_data)
{
    // Things to do:
    // 1.  Split off a validation set for early stopping
    // 2.  Get example weights
    // 3.  Make the weak learner simply choose the best submodel according to
    //     the current weights
    // 4.  Calculate residuals at each iteration to choose weight and updates

    //ML::Boosting_Loss loss;

    Data training_set = training_data;
    Data validate_set;
    training_set.hold_out(validate_set, 0.2);

    distribution<double> example_weights(training_set.targets.size(), 1.0);
    
    distribution<double> training_predictions(training_set.targets.size(), 0.0);
    distribution<double> validate_predictions(validate_set.targets.size(), 0.0);
    Target target = training_set.target;

    int max_iter = 100;

    model_weights.clear();
    model_weights.resize(training_set.models.size(), 0.0);

    int num_possible = 0, num_impossible = 0;

    vector<int> possible(training_set.targets.size(), false);

    for (unsigned x = 0;  x < training_set.targets.size();  ++x) {
        float label = training_set.targets[x];
        int & is_possible = possible[x];
        for (unsigned m = 0;  m < training_set.models.size() && !is_possible;
             ++m) {

            float pred = (training_set.models[m][x] - 3.0) / 2.0;
            float margin = pred * label;
            if (margin > 0.0) is_possible = true;
        }

        if (!is_possible) {
            num_impossible += 1;
            example_weights[x] = 0.0;
        }
        else num_possible += 1;
    }

    cerr << "examples: possible " << num_possible << " impossible "
         << num_impossible << endl;

    example_weights.normalize();

    distribution<double> best_model_weights = model_weights;
    float best_validate_score = 0.0;
    int best_iter = -1;


    for (int iter = 0;  iter < max_iter;  ++iter) {
        // Calculate the (weighted) score for each of the weak learners
        vector<pair<int, float> > weak_scores;

#if 0
        cerr << "example weights: "
             << distribution<double>(example_weights.begin(),
                                     example_weights.begin() + 20)
                 * example_weights.size()
             << endl;
#endif
        
        for (unsigned m = 0;  m < training_set.models.size();  ++m) {
            // Calculate the model score
            double error;
            if (target == RMSE)
                error
                    = 1.0 - training_set.models[m]
                                .calc_rmse_weighted(training_set.targets,
                                                    example_weights);
            else {
                // AUC calculation; we want a normal weighted accuracy thing
                // A weighted, real AUC doesn't make much sense really
                error = 0.0;

                double correct = 0.0, incorrect = 0.0;
                double total_margin = 0.0;

                for (unsigned x = 0;  x < training_set.targets.size();  ++x) {
                    // scores are between 1.000 and 5.000, so we say that it
                    // is correct if it is on the right side of 3
                    // Might need something a bit more clever here, though
                    float pred = (training_set.models[m][x] - 3.0) / 2.0;
                    float label = training_set.targets[x];
                    float margin = pred * label;

                    //cerr << "pred = " << pred << " correct = " << correct
                    //     << " margin = " << margin << " weight "
                    //     << example_weights[x] << endl;

                    if (margin < 0.0) incorrect += example_weights[x];
                    else correct += example_weights[x];

                    total_margin += example_weights[x] * margin;
                }

                error = incorrect / (correct + incorrect);

                //cerr << "correct = " << correct << " incorrect = "
                //     << incorrect << " error " << error << endl;
                //cerr << "error = " << error << " avg_margin = "
                //     << total_margin << endl;

                //error = total_margin;
            }

            weak_scores.push_back(make_pair(m, error));
        }

        sort_on_second_ascending(weak_scores);

        int weak_model = weak_scores[0].first;
        double error = weak_scores[0].second;

        //cerr << "error = " << error << endl;

        if (error >= 0.5) break;

        double beta = error / (1.0 - error);

        double weight = -0.5 * log(beta);
        

        //cerr << "weight is " << weight << endl;

        // Update training weights

        double total_ex_weight = 0.0;
        double min_weight = INFINITY, max_weight = -INFINITY;

        double total_weight_correct = 0.0, total_weight_incorrect = 0.0;

        for (unsigned x = 0;  x < training_set.targets.size();  ++x) {
            if (!possible[x]) {
                if (example_weights[x] != 0.0)
                    throw Exception("impossible example not ignored");
                continue;
            }

            double margin;

            if (target == RMSE)
                margin = -fabs(training_set.models[weak_model][x]
                               - training_set.targets[x]);
            else {
                float pred = (training_set.models[weak_model][x] - 3.0) / 2.0;
                float correct = training_set.targets[x];
                margin = pred * correct;
                
                if (x < 20 && false)
                    cerr << "x = " << x << " pred " << pred << " correct "
                         << correct << " margin " << margin
                         << " oldwt "
                         << example_weights[x] * example_weights.size()
                         << " newwt "
                         << (example_weights[x] * example_weights.size()
                                 * exp(-weight * margin))
                         << endl;
            }

            example_weights[x] *= exp(2.0 * -weight * margin);
 
            min_weight = std::min(example_weights[x], min_weight);
            max_weight = std::max(example_weights[x], max_weight);

            total_ex_weight += example_weights[x];

            if (margin > 0.0)
                total_weight_correct += example_weights[x];
            else total_weight_incorrect += example_weights[x];
        }
        
        //cerr << "min_weight = " << (min_weight / total_ex_weight * example_weights.size()) << endl;
        //cerr << "max_weight = " << (max_weight / total_ex_weight * example_weights.size()) << endl;

        //cerr << "total_ex_weight = " << total_ex_weight << endl;

        cerr << "total_weight_correct = " << total_weight_correct << endl;
        cerr << "total_weight_incorrect = " << total_weight_incorrect << endl;

        example_weights /= total_ex_weight;

        model_weights[weak_model] += weight;
        // TODO: how to normalize the model weights?


        // Now score accuracy of training and validate examples
        for (unsigned x = 0;  x < training_set.targets.size();  ++x) {
            float pred = training_set.models[weak_model][x];
            if (target == AUC)
                pred = (pred - 3.0) / 2.0;

            training_predictions[x] += pred;
        }

        for (unsigned x = 0;  x < validate_set.targets.size();  ++x) {
            float pred = validate_set.models[weak_model][x];
            if (target == AUC)
                pred = (pred - 3.0) / 2.0;

            validate_predictions[x] += pred;
        }

        Model_Output training_output, validate_output;
        training_output.insert(training_output.end(),
                               training_predictions.begin(),
                               training_predictions.end());

        validate_output.insert(validate_output.end(),
                               validate_predictions.begin(),
                               validate_predictions.end());
        
        double training_score
            = training_output.calc_score(training_set.targets, target);
        
        double validate_score
            = validate_output.calc_score(validate_set.targets, target);

        if (validate_score >= best_validate_score) {
            best_validate_score = validate_score;
            best_model_weights = model_weights;
            best_iter = iter;
        }
            
        
        //cerr << "iter " << iter << " chose model " << weak_model << " ("
        //     << training_set.model_names[weak_model] << ") with error "
        //     << weak_scores[0].second << endl;

        //cerr << "error = " << error << endl;
        //cerr << "training score: " << training_score << endl;
        //cerr << "validate score: " << validate_score << endl;

        cerr << format("iter %4d model %5d error %6.4f train %6.4f val %6.4f wt %6.4f",
                       iter, weak_model,
                       training_set.model_names[weak_model].c_str(),
                       error, training_score, validate_score,
                       weight) << endl;
    }

    cerr << "best was on iter " << best_iter << endl;
    model_weights = best_model_weights;
}
