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
    
    /// Calculate the AUC over the given set of targets
    double calc_auc(const distribution<float> & targets) const;

    double calc_score(const distribution<float> & targets,
                      Target target) const;
    

    /// Calculate the RMSE over the given set of targets
    double calc_rmse_weighted(const distribution<float> & targets,
                              const distribution<double> & weights) const;
    
    /// Score over whatever target we are trying to calculate
    double score;
    
    /// Rank in accuracy over targets we are trying to calculate
    int rank;
};


/*****************************************************************************/
/* DATA                                                                      */
/*****************************************************************************/

/** Data structure to contain the dataset that we are working on. */

struct Data {
    void load(const std::string & filename, Target target);

    void calc_scores();

    void hold_out(Data & remove_to, float proportion,
                  int random_seed = 1);

    void clear();

    void swap(Data & other);

    void decompose();

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
};

#endif /* __ausdm__data_h__ */


