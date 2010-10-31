/* boosting_blender.cc
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender that uses boosting.
*/

#include "boosting_blender.h"
#include "jml/utils/vector_utils.h"
#include "jml/boosting/boosting_core.h"
#include "jml/stats/distribution_ops.h"
#include "jml/algebra/glz.h"


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
configure(const ML::Configuration & config_,
          const std::string & name,
          int random_seed,
          Target target)
{
    Configuration config(config_, name, Configuration::PREFIX_APPEND);

    this->config = config;

    weaklearner_name = "weaklearner";

    this->target = target;
    this->random_seed = random_seed;
    
    config.require(num_iter, "num_iter");
}
    
void
Boosting_Blender::
init(const Data & training_data,
     const ML::distribution<float> & example_weights_)
{
    // We dispense with early stopping...

    distribution<float> example_weights = example_weights_;

    int nx = training_data.nx();

    if (example_weights.size() != nx)
        throw Exception("invalid nx");
    
    // Scores for the entire model
    Model_Output blended_predictions;
    blended_predictions.resize(nx);

    for (unsigned i = 0;  i < num_iter;  ++i) {

        // Get this weak learner

        boost::shared_ptr<Blender> weak
            = get_blender(config, weaklearner_name, training_data,
                          example_weights, random_seed, target);

        submodels.push_back(weak);

        // Run it over the training set
        Model_Output weak_predictions;
        weak_predictions.resize(nx);

        for (unsigned x = 0;  x < nx;  ++x) {

#if 0
            float label = training_data.targets[x];
            int & is_possible = possible[x];
            for (unsigned m = 0;  m < training_data.nm() && !is_possible;
                 ++m) {
                
                float pred = (training_data.models[m][x] - 3.0) / 2.0;
                float margin = pred * label;
                if (margin > 0.0) is_possible = true;
            }
            
            if (!is_possible) {
                num_impossible += 1;
                example_weights[x] = 0.0;
            }
            else num_possible += 1;
#endif
            
            const distribution<float> & models
                = training_data.examples[x]->models;

            weak_predictions[x] = weak->predict(models);
        }

        // Calculate the score (over the training data)
        float weak_score
            = weak_predictions.calc_score(training_data.targets, target);
        float weak_score_weighted
            = weak_predictions.calc_score(training_data.targets,
                                          example_weights,
                                          target);
        
        // Calculate the weight
        float weight = 1.0 - weak_score_weighted;
        
        // Calculate the blended outputs
        weights.push_back(weight);

        blended_predictions += weak_predictions * weight;

        double factor = weights.total();

        Model_Output norm_blended_predictions = blended_predictions;
        norm_blended_predictions /= factor;

        float blended_score
            = norm_blended_predictions
            .calc_score(training_data.targets, target);
        
        cerr << "weight = " << weight << " weak_score = " << weak_score
             << " weighted = " << weak_score_weighted
             << " blended_score = " << blended_score << endl;
        
        // Update the example weights
        distribution<float> margins
            = norm_blended_predictions * training_data.targets;

        Logit_Link<float> link;
        distribution<float> lmargin = (link.inverse(margins) - 0.5) * 2.0;
        
        for (unsigned i = 0;  i < 100;  ++i) {
            cerr << "example " << i << ": label " << training_data.targets[i]
                 << " weight " << example_weights[i] * nx << " new pred "
                 << weak_predictions[i] << " combined pred "
                 << norm_blended_predictions[i] << " margin "
                 << margins[i] << " lmargin "
                 << lmargin[i] << " new weight "
                 << example_weights[i] * std::exp(-lmargin[i]) * nx << endl;
        }


        example_weights *= exp(-lmargin);

        cerr << "example_weights.total() = " << example_weights.total()
             << endl;

        example_weights.normalize();
    }
}

float
Boosting_Blender::
predict(const ML::distribution<float> & models) const
{
    distribution<float> submodel_predictions(submodels.size());
    for (unsigned i = 0;  i < submodels.size();  ++i)
        submodel_predictions[i] = submodels[i]->predict(models);
    return weights.dotprod(submodel_predictions);
}
