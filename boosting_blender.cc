/* boosting_blender.cc
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender that uses boosting.
*/

#include "boosting_blender.h"
#include "utils/vector_utils.h"


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


    Data training_set = training_data;
    Data validate_set;
    training_set.hold_out(validate_set, 0.2);

    distribution<double> example_weights(training_set.targets.size(), 1.0);
    example_weights.normalize();
    
    distribution<double> training_predictions(training_set.targets.size(), 0.0);
    distribution<double> validate_predictions(validate_set.targets.size(), 0.0);
    Target target = training_set.target;

    int max_iter = 100;

    model_weights.clear();
    model_weights.resize(training_set.models.size(), 0.0);

    for (int iter = 0;  iter < max_iter;  ++iter) {
        // Calculate the (weighted) score for each of the weak learners
        vector<pair<int, float> > weak_scores;
        
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
                for (unsigned x = 0;  x < training_set.targets.size();  ++x) {
                    // scores are between 1.000 and 5.000, so we say that it
                    // is correct if it is on the right side of 3
                    // Might need something a bit more clever here, though
                    float pred = (training_set.models[m][x] - 3.0) / 2.0;
                    float correct = training_set.targets[x];

                    if ((pred < 0.0 && correct >= 0.0)
                        || (pred > 0.0 && correct <= 0.0))
                        error += example_weights[x];
                }
            }

            weak_scores.push_back(make_pair(m, error));
        }

        sort_on_second_ascending(weak_scores);

        int weak_model = weak_scores[0].first;
        double error = weak_scores[0].second;

        //cerr << "error = " << error << endl;

        if (error >= 0.5) break;

        double weight = 0.5 * log((1.0 - error) / error);

        //cerr << "weight is " << weight << endl;

        // Update training weights

        double total_ex_weight = 0.0;
        double min_weight = INFINITY, max_weight = -INFINITY;

        for (unsigned x = 0;  x < training_set.targets.size();  ++x) {
            double margin;

            if (target == RMSE)
                margin = -fabs(training_set.models[weak_model][x]
                               - training_set.targets[x]);
            else {
                float pred = (training_set.models[weak_model][x] - 3.0) / 2.0;
                float correct = training_set.targets[x];
                margin = pred * correct;
            }

            example_weights[x] *= exp(-weight * margin);
 
            min_weight = std::min(example_weights[x], min_weight);
            max_weight = std::max(example_weights[x], max_weight);

            total_ex_weight += example_weights[x];
        }
        
        //cerr << "min_weight = " << (min_weight / total_ex_weight * example_weights.size()) << endl;
        //cerr << "max_weight = " << (max_weight / total_ex_weight * example_weights.size()) << endl;

        //cerr << "total_ex_weight = " << total_ex_weight << endl;

        example_weights /= total_ex_weight;

        model_weights[weak_model] += weight;
        // TODO: how to normalize the model weights?


        // Now score accuracy of training and validate examples
        for (unsigned x = 0;  x < training_set.targets.size();  ++x) {
            training_predictions[x]
                += weight * training_set.models[weak_model][x];
        }

        for (unsigned x = 0;  x < validate_set.targets.size();  ++x) {
            validate_predictions[x]
                += weight * validate_set.models[weak_model][x];
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

        //cerr << "iter " << iter << " chose model " << weak_model << " ("
        //     << training_set.model_names[weak_model] << ") with error "
        //     << weak_scores[0].second << endl;

        //cerr << "error = " << error << endl;
        //cerr << "training score: " << training_score << endl;
        //cerr << "validate score: " << validate_score << endl;

        cerr << format("iter %4d model %5d error %6.4f train %6.4f val %6.4f",
                       iter, weak_model,
                       training_set.model_names[weak_model].c_str(),
                       error, training_score, validate_score) << endl;
    }
}
