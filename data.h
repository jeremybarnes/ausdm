/* data.h                                                          -*- C++ -*-
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   File to hold the data.
*/

#ifndef __ausdm__data_h__
#define __ausdm__data_h__

#include <boost/multi_array.hpp>
#include <vector>
#include <string>
#include "stats/distribution.h"

/// What kind of target are we calculating?
enum Target {
    RMSE,
    AUC
};

using ML::distribution;


/*****************************************************************************/
/* MODEL_OUTPUT                                                              */
/*****************************************************************************/

/// The output of one of the models that we are blending
struct Model_Output : public distribution<float> {
    
    /// Calculate the RMSE over the given set of targets
    double calc_rmse(const distribution<float> & targets) const;

    /// Same, but taking into account weights
    double calc_rmse(const distribution<float> & targets,
                     const distribution<float> & weights) const;
    
    /// Calculate the AUC over the given set of targets
    double calc_auc(const distribution<float> & targets) const;

    /// Same, but with weights.  Not sure that it makes much sense
    /// mathematically...
    double calc_auc(const distribution<float> & targets,
                    const distribution<float> & weights) const;

    /// Calculate the score based upon the target
    double calc_score(const distribution<float> & targets,
                      Target target) const;

    /// Calculate the weighted score based upon the target
    double calc_score(const distribution<float> & targets,
                      const distribution<float> & weights,
                      Target target) const;
    

    /// Score over whatever target we are trying to calculate
    double score;
    
    /// Rank in accuracy over targets we are trying to calculate
    int rank;
};


/*****************************************************************************/
/* TARGET_STATS                                                              */
/*****************************************************************************/

/** The statistics for a given target output */

struct Target_Stats {

    Target_Stats()
        : mean(0.0), std(0.0), min(0.0), max(0.0)
    {
    }

    template<class Iterator>
    Target_Stats(Iterator first, const Iterator & last)
    {
        int n = std::distance(first, last);

        double total = 0.0;
        float tmin = INFINITY, tmax = -INFINITY;

        for (Iterator it = first; it != last;  ++it) {
            total += *it;
            tmin = std::min(tmin, *it);
            tmax = std::max(tmax, *it);
        }

        double mean = total / n;
        total = 0.0;
        for (Iterator it = first; it != last;  ++it)
            total += pow(*it - mean, 2);

        this->mean = mean;
        this->std = sqrt(total);
        this->min = tmin;
        this->max = tmax;
    }

    float mean;
    float std;
    float min;
    float max;

};

/*****************************************************************************/
/* DATA                                                                      */
/*****************************************************************************/

/** Data structure to contain the dataset that we are working on. */

struct Data {
    void load(const std::string & filename, Target target,
              bool clear_first = true);

    void calc_scores();

    void hold_out(Data & remove_to, float proportion,
                  int random_seed = 1);

    void hold_out(Data & remove_to, float proportion,
                  distribution<float> & example_weights,
                  distribution<float> & remove_to_example_weights,
                  int random_seed = 1);

    void clear();

    void swap(Data & other);

    void decompose();

    void apply_decomposition(const Data & decomposed);

    distribution<float>
    apply_decomposition(const distribution<float> & models) const;

    void stats();

    Target target;

    /// Target values to predict
    distribution<float> targets;

    /// Names of the models
    std::vector<std::string> model_names;

    /// ID values of the models
    std::vector<int> model_ids;

    std::vector<Model_Output> models;

    /// Sorted list of models in order of score
    std::vector<int> model_ranking;

    /// Singular values of SVD on rankings
    distribution<float> singular_values;

    /// Singular representation of each model
    std::vector<distribution<float> > singular_models;

    /// Singular representation of each target
    std::vector<distribution<float> > singular_targets;

    /// Statistics about models for each target
    std::vector<Target_Stats> target_stats;
};

#endif /* __ausdm__data_h__ */


