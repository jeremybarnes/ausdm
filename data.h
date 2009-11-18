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
#include <boost/shared_ptr.hpp>

/// What kind of target are we calculating?
enum Target {
    RMSE,
    AUC
};

using ML::distribution;

class Decomposition;

/*****************************************************************************/
/* DIFFICULTY                                                                */
/*****************************************************************************/

enum Difficulty_Category {
    DIF_UNKNOWN,        ///< Label is unknown so difficulty is unknown
    DIF_AUTOMATIC,      ///< Automatically correct (all models are correct)
    DIF_POSSIBLE,       ///< At least one is correct
    DIF_IMPOSSIBLE      ///< All models have misclassified it
};

std::string print(const Difficulty_Category & cat);

std::ostream & operator << (std::ostream & stream, Difficulty_Category cat);

struct Difficulty {
    Difficulty();
    Difficulty(const ML::distribution<float> & model_outputs,
               float label,
               Target target);

    Difficulty_Category category;
    float difficulty;
};


/*****************************************************************************/
/* SCORES                                                                    */
/*****************************************************************************/

struct Scores {
    distribution<float> target_values;
    operator double() const { return score; }

    double category_scores[4];  // one for each difficulty category
    double category_averages[4];
    double score;
};


/*****************************************************************************/
/* MODEL_STATS                                                               */
/*****************************************************************************/

/// The output of one of the models that we are blending
struct Model_Stats {
    /// Score over whatever target we are trying to calculate
    double score;
    
    /// Rank in accuracy over targets we are trying to calculate
    int rank;
};


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
    Data()
        : decomposition(0)
    {
    }
    
    void load(const std::string & filename, Target target,
              bool clear_first = true);

    void hold_out(Data & remove_to, float proportion,
                  int random_seed = 1);

    void hold_out(Data & remove_to, float proportion,
                  distribution<float> & example_weights,
                  distribution<float> & remove_to_example_weights,
                  int random_seed = 1);

    void clear();

    void swap(Data & other);

    void apply_decomposition(const Decomposition & decomposition);

    distribution<float>
    apply_decomposition(const distribution<float> & example) const;

    int nm() const { return models.size(); }
    int nx() const { return examples.size(); }

    Target target;

    /// Target values to predict
    distribution<float> targets;

    /// Names of the models
    std::vector<std::string> model_names;

    /// ID values of the models
    std::vector<int> model_ids;

    std::vector<Model_Stats> models;

    /// Sorted list of models in order of score
    std::vector<int> model_ranking;


    struct Example {
        Example()
            : label(0.0)
        {
        }

        Example(const distribution<float> & models, float label, Target target)
            : label(label), models(models), stats(models.begin(), models.end()),
              difficulty(models, label, target)
        {
        }

        float label;
        distribution<float> models;
        distribution<float> decomposed;
        Target_Stats stats;
        Difficulty difficulty;
    };

    std::vector<boost::shared_ptr<Example> > examples;

    const Decomposition * decomposition;

    size_t decomposition_size() const;

protected:
    void calc_scores();
};

#endif /* __ausdm__data_h__ */


