/* data.cc
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   File to read data for the competition.
*/

#include "data.h"
#include "utils/parse_context.h"
#include "stats/distribution_ops.h"
#include "stats/distribution_simd.h"
#include "utils/vector_utils.h"
#include "utils/pair_utils.h"
#include "algebra/lapack.h"
#include "arch/timers.h"
#include "arch/threads.h"


using namespace std;
using namespace ML;
using ML::Stats::sqr;


/*****************************************************************************/
/* DIFFICULTY                                                                */
/*****************************************************************************/

std::string print(const Difficulty_Category & cat)
{
    switch (cat) {
    case DIF_UNKNOWN:    return "UNKNOWN";
    case DIF_AUTOMATIC:  return "AUTO";
    case DIF_POSSIBLE:   return "POSS";
    case DIF_IMPOSSIBLE: return "IMP";
    default: return format("Difficulty_Category(%d)", cat);
    }
}

std::ostream & operator << (std::ostream & stream, Difficulty_Category cat)
{
    return stream << print(cat);
}

Difficulty::
Difficulty()
    : category(DIF_UNKNOWN), difficulty(0.0)
{
}

Difficulty::
Difficulty(const ML::distribution<float> & model_outputs,
           float label, Target target)
{
    // If label is zero, then the example was unlabeled
    if ((target == RMSE && label == -1.5)
        || (target == AUC && label == 0.0)) {
        difficulty = 0.0;
        category = DIF_UNKNOWN;
        return;
    }

    if (target == AUC) {
        int ncorr = (model_outputs * label >= 0.0f).count();
        int nincorr = (model_outputs.size() - ncorr);

        if (ncorr == 0) {
            category = DIF_IMPOSSIBLE;
            difficulty = model_outputs.mean() * target;
        }
        else if (nincorr == 0) {
            category = DIF_AUTOMATIC;
            difficulty = model_outputs.mean() * target;
        }
        else {
            category = DIF_POSSIBLE;
            difficulty = model_outputs.mean() * target;
        }
    } else {
        int ncorr = (abs(model_outputs - label) < 0.5).count();
        int nincorr = (model_outputs.size() - ncorr);

        if (ncorr == 0) {
            category = DIF_IMPOSSIBLE;
            difficulty = abs(model_outputs.mean() - target);
        }
        else if (nincorr == 0) {
            category = DIF_AUTOMATIC;
            difficulty = abs(model_outputs.mean() - target);
        }
        else {
            category = DIF_POSSIBLE;
            difficulty = abs(model_outputs.mean() - target);
        }

    }
}


/*****************************************************************************/
/* MODEL_OUTPUT                                                              */
/*****************************************************************************/

double
Model_Output::
calc_rmse(const distribution<float> & targets) const
{
    if (targets.size() != size())
        throw Exception("targets and predictions don't match");

    //cerr << "targets[0] = " << targets[0] << endl;
    //cerr << "this[0] = " << (*this)[0] << endl;

    return sqrt(sqr((targets - *this) * 2.0).total() * (1.0 / size()));
}

double
Model_Output::
calc_rmse(const distribution<float> & targets,
          const distribution<float> & weights) const
{
    if (targets.size() != size())
        throw Exception("targets and predictions don't match");

    if (weights.size() != size())
        throw Exception("weights and predictions don't match");

    return sqrt((sqr((targets - *this) * 2.0) * weights).total() / weights.total());
}


namespace {
        
struct AUC_Entry {
    AUC_Entry(float model = 0.0, float target = 0.0, float weight = 1.0)
        : model(model), target(target), weight(weight)
    {
    }

    float model;
    float target;
    float weight;

    bool operator < (const AUC_Entry & other) const
    {
        return model < other.model;
    }
};

double do_calc_auc(std::vector<AUC_Entry> & entries)
{
    // 1.  Total number of positive and negative
    int num_neg = 0, num_pos = 0;

    for (unsigned i = 0;  i < entries.size();  ++i) {
        if (entries[i].weight == 0.0) continue;
        if (entries[i].target == -1) ++num_neg;
        else ++num_pos;
    }

    // 2.  Sort
    std::sort(entries.begin(), entries.end());
    
    // 3.  Get (x,y) points and calculate the AUC
    int total_pos = 0, total_neg = 0;

    float prevx = 0.0, prevy = 0.0;

    double total_area = 0.0, total_weight = 0.0, current_weight = 0.0;

    for (unsigned i = 0;  i < entries.size();  ++i) {
        if (entries[i].weight > 0.0) {
            if (entries[i].target == -1) ++total_neg;
            else ++total_pos;
        }
        
        current_weight += entries[i].weight;
        total_weight += entries[i].weight;

        if (i != entries.size() - 1
            && entries[i].model == entries[i + 1].model)
            continue;
        
        if (entries[i].weight == 0.0) continue;

        float x = total_pos * 1.0 / num_pos;
        float y = total_neg * 1.0 / num_neg;

        double area = (x - prevx) * (y + prevy) * 0.5;

        total_area += /* current_weight * */ area;

        prevx = x;
        prevy = y;
        current_weight = 0.0;
    }

    // TODO: get weighted working properly...

    //cerr << "total_area = " << total_area << " total_weight = "
    //     << total_weight << endl;

    if (total_pos != num_pos || total_neg != num_neg)
        throw Exception("bad total pos or total neg");

    // 4.  Convert to gini
    //double gini = 2.0 * (total_area - 0.5);

    // 5.  Final score is absolute value.  Since we want an error, we take
    //     1.0 - the gini
    //return 1.0 - fabs(gini);
    return 1.0 - total_area;
}

} // file scope

double
Model_Output::
calc_auc(const distribution<float> & targets) const
{
    if (targets.size() != size())
        throw Exception("targets and predictions don't match");
    
    vector<AUC_Entry> entries;
    entries.reserve(size());
    for (unsigned i = 0;  i < size();  ++i)
        entries.push_back(AUC_Entry((*this)[i], targets[i]));
    
    return do_calc_auc(entries);
}

double
Model_Output::
calc_auc(const distribution<float> & targets,
         const distribution<float> & weights) const
{
    if (targets.size() != size())
        throw Exception("targets and predictions don't match");
    if (weights.size() != size())
        throw Exception("targets and weights don't match");
    
    vector<AUC_Entry> entries;
    entries.reserve(size());
    for (unsigned i = 0;  i < size();  ++i)
        entries.push_back(AUC_Entry((*this)[i], targets[i], weights[i]));
    
    return do_calc_auc(entries);
}

double
Model_Output::
calc_score(const distribution<float> & targets,
         Target target) const
{
    if (target == AUC) return calc_auc(targets);
    else if (target == RMSE) return calc_rmse(targets);
    else throw Exception("unknown target");
}

double
Model_Output::
calc_score(const distribution<float> & targets,
           const distribution<float> & weights,
           Target target) const
{
    if (target == AUC) return calc_auc(targets, weights);
    else if (target == RMSE) return calc_rmse(targets, weights);
    else throw Exception("unknown target");
}


/*****************************************************************************/
/* DATA                                                                      */
/*****************************************************************************/

void
Data::
load(const std::string & filename, Target target, bool clear_first)
{
    if (clear_first) clear();
    else if (target != this->target)
        throw Exception("Data::load(): target loaded on top isn't same");

    this->target = target;

    Parse_Context c(filename);

    // First row: header.
    c.expect_literal("RowID,Target");

    int m = 0;
    while (!c.match_eol()) {
        c.expect_literal(',');
        string model_name = c.expect_text(",\n\r");

        if (clear_first)
            model_names.push_back(model_name);
        else if (model_name != model_names.at(m++))
            throw Exception("model names don't match");
    }

    if (!clear_first && model_names.size() != m)
        throw Exception("wrong number of models");

    int nm = model_names.size();

    // Create the data structures
    //cerr << model_names.size() << " models... ";
    
    if (clear_first) {
        models.resize(model_names.size());
        for (unsigned i = 0;  i < nm;  ++i) {
            models[i].reserve(50000);
        }
    }

    examples.reserve(50000);
    
    int num_rows = 0;
    for (; c; ++num_rows) {
        int id = c.expect_int();
        model_ids.push_back(id);
        c.expect_literal(',');

        distribution<float> example(nm);

        float target_val = c.expect_int();

        if (target == RMSE) {
            // Convert into range (-1, 1)
            target_val = (target_val - 3000.0) / 2000.0;
        }

        targets.push_back(target_val);

        for (unsigned i = 0;  i < nm;  ++i) {
            c.expect_literal(',');
            int score = c.expect_int();
            float val = (score - 3000)/ 2000.0;
            example[i] = val;
            models[i].push_back(val);
        }

        examples.push_back(distribution<float>());
        examples.back().swap(example);

        c.skip_whitespace();
        c.expect_eol();
    }

    //cerr << num_rows << " rows... ";
}

void
Data::
clear()
{
    targets.clear();
    model_names.clear();
    model_ids.clear();
    models.clear();
    model_ranking.clear();
    singular_values.clear();
    singular_models.clear();
    singular_targets.clear();
    target_stats.clear();
}

void
Data::
swap(Data & other)
{
    std::swap(target, other.target);
    targets.swap(other.targets);
    model_names.swap(other.model_names);
    model_ids.swap(other.model_ids);
    models.swap(other.models);
    model_ranking.swap(other.model_ranking);
    singular_values.swap(other.singular_values);
    singular_models.swap(other.singular_models);
    singular_targets.swap(other.singular_targets);
    target_stats.swap(other.target_stats);
}

void
Data::
calc_scores()
{
    vector<pair<float, int> > model_scores;

    for (unsigned i = 0;  i < nm();  ++i) {
        if (target == RMSE)
            models[i].score = models[i].calc_rmse(targets);
        else models[i].score = models[i].calc_auc(targets);

        model_scores.push_back(make_pair(models[i].score, i));
    }

    sort_on_first_ascending(model_scores);

    for (unsigned i = 0;  i < nm();  ++i)
        models[model_scores[i].second].rank = i;

#if 0
    for (unsigned i = 0;  i < 20;  ++i) {
        int m = model_scores[i].second;
        cerr << "rank " << i << " " << model_names[m] << " score "
             << models[m].score << endl;
    }
#endif

    model_ranking.clear();
    model_ranking.insert(model_ranking.end(),
                         second_extractor(model_scores.begin()),
                         second_extractor(model_scores.end()));
}

void
Data::
hold_out(Data & remove_to, float proportion,
         int random_seed)
{
    distribution<float> example_weights(nx());
    distribution<float> remove_to_example_weights;

    hold_out(remove_to, proportion, example_weights,
             remove_to_example_weights, random_seed);
}

void
Data::
hold_out(Data & remove_to, float proportion,
         distribution<float> & example_weights,
         distribution<float> & remove_to_example_weights,
         int random_seed)
{
    static Lock lock;
    Guard guard(lock);

    srand(random_seed);

    vector<int> to_remove;
    for (unsigned i = 0;  i < nx();  ++i) {
        to_remove.push_back(i);
    }

    std::random_shuffle(to_remove.begin(), to_remove.end());

    if (proportion < 0 || proportion > 1.0)
        throw Exception("bad proportion");
    
    to_remove.erase(to_remove.begin() + proportion * to_remove.size(),
                    to_remove.end());

    vector<int> remove_me(nx(), false);
    for (unsigned i = 0;  i < to_remove.size();  ++i)
        remove_me[to_remove[i]] = true;

    remove_to.clear();
    
    Data new_me;

    new_me.target = remove_to.target = target;
    new_me.model_names = remove_to.model_names = model_names;
    new_me.singular_values = remove_to.singular_values = singular_values;
    new_me.singular_models = remove_to.singular_models = singular_models;
    
    new_me.models.resize(model_names.size());
    remove_to.models.resize(model_names.size());

    distribution<float> new_example_weights;
    new_example_weights.reserve(nx() - to_remove.size());

    remove_to_example_weights.clear();
    remove_to_example_weights.reserve(to_remove.size());

    bool has_st = !singular_targets.empty();

    for (unsigned i = 0;  i < model_names.size();  ++i) {
        new_me.models[i].reserve(nx() - to_remove.size());
        remove_to.models[i].reserve(to_remove.size());
    }

    if (has_st) {
        new_me.singular_targets.reserve(nx() - to_remove.size());
        remove_to.singular_targets.reserve(to_remove.size());
    }

    for (unsigned i = 0;  i < nx();  ++i) {
        Data & add_to = remove_me[i] ? remove_to : new_me;
        distribution<float> & weights
            = remove_me[i] ? remove_to_example_weights : new_example_weights;
        add_to.targets.push_back(targets[i]);
        add_to.model_ids.push_back(model_ids[i]);
        weights.push_back(example_weights[i]);

        for (unsigned j = 0;  j < model_names.size();  ++j)
            add_to.models[j].push_back(models[j][i]);

        if (has_st) {
            add_to.singular_targets.push_back(distribution<float>());
            add_to.singular_targets.back().swap(singular_targets[i]);
        }
    }

    remove_to.stats();
    new_me.stats();

    swap(new_me);
    example_weights.swap(new_example_weights);
}

void
Data::
decompose()
{
    int m = nm();
    int n = nx();

    boost::multi_array<float, 2> values(boost::extents[n][m]);

    for (unsigned i = 0;  i < m;  ++i)
        for (unsigned j = 0;  j < n;  ++j)
            values[j][i] = models[i][j];

    int nvalues = std::min(m, n);

    cerr << "n = " << n << " m = " << m << " nvalues = " << nvalues << endl;

    distribution<float> svalues(nvalues);
    boost::multi_array<float, 2> lvectorsT(boost::extents[nvalues][m]);
    boost::multi_array<float, 2> rvectors(boost::extents[n][nvalues]);

    int result = LAPack::gesdd("S", m, n,
                               values.data(), m,
                               &svalues[0],
                               &lvectorsT[0][0], m,
                               &rvectors[0][0], nvalues);

    if (result != 0)
        throw Exception("gesdd returned non-zero");

    // Transpose rvectors
    //boost::multi_array<float, 2> rvectors(boost::extents[nvalues][n]);
    //for (unsigned i = 0;  i < nvalues;  ++i)
    //    for (unsigned j = 0;  j < n;  ++j)
    //        rvectors[i][j] = rvectorsT[j][i];

    // Transpose lvectors
    boost::multi_array<float, 2> lvectors(boost::extents[m][nvalues]);
    for (unsigned i = 0;  i < m;  ++i)
        for (unsigned j = 0;  j < nvalues;  ++j)
            lvectors[i][j] = lvectorsT[j][i];

    int nwanted = std::min(nvalues, 200);
    //nwanted = 50;

    singular_values
        = distribution<float>(svalues.begin(), svalues.begin() + nwanted);

    //cerr << "singular_values = " << singular_values << endl;

    singular_models.resize(nm());
    
    for (unsigned i = 0;  i < nm();  ++i)
        singular_models[i]
            = distribution<float>(&lvectors[i][0],
                                  &lvectors[i][nwanted - 1] + 1);

    //cerr << "singular_models[0] = " << singular_models[0] << endl;
    //cerr << "singular_models[1] = " << singular_models[1] << endl;

    singular_targets.resize(nx());

    for (unsigned i = 0;  i < nx();  ++i)
        singular_targets[i]
            = distribution<float>(&rvectors[i][0],
                                  &rvectors[i][nwanted - 1] + 1);

    //cerr << "singular_targets[0] = " << singular_targets[0] << endl;
    //cerr << "singular_targets[1] = " << singular_targets[1] << endl;
}

void
Data::
apply_decomposition(const Data & decomposed)
{
    singular_targets.resize(nx());

    for (unsigned i = 0;  i < nx();  ++i) {
        distribution<float> mvalues(nm());
        for (unsigned j = 0;  j < nm();  ++j)
            mvalues[j] = models[j][i];

        singular_targets[i] = decomposed.apply_decomposition(mvalues);
    }

    singular_values = decomposed.singular_values;
    singular_models = decomposed.singular_models;
}

distribution<float>
Data::
apply_decomposition(const distribution<float> & models) const
{
    if (singular_values.empty())
        throw Exception("apply_decomposition(): no decomposition was done");

    // First, get the singular vector for the model
    distribution<double> target_singular(singular_values.size());

    for (unsigned i = 0;  i < nm();  ++i)
        target_singular += singular_models[i] * models[i];
    
    target_singular /= singular_values;

    return distribution<float>(target_singular.begin(), target_singular.end());
}

void
Data::
stats()
{
    target_stats.resize(nx());
    target_difficulty.resize(nx());

    distribution<float> model_vals(nm());

    double total_mean_neg = 0.0, total_mean_pos = 0.0;
    double num_neg = 0.0, num_pos = 0.0;
    double minval = INFINITY, maxval = -INFINITY;

    int nimpossible = 0, nautomatic = 0, npossible = 0, nunknown = 0;

    for (unsigned i = 0;  i < nx();  ++i) {
        for (unsigned j = 0;  j < nm();  ++j)
            model_vals[j] = models[j][i];
        target_stats[i] = Target_Stats(model_vals.begin(), model_vals.end());
        target_difficulty[i] = Difficulty(model_vals, targets[i], target);

        if (targets[i] < 0.0) {
            total_mean_neg += model_vals.mean();
            num_neg += 1.0;
        }
        else if (targets[i] > 0.0) {
            total_mean_pos += model_vals.mean();
            num_pos += 1.0;
        }

        minval = std::min<double>(minval, model_vals.min());
        maxval = std::max<double>(maxval, model_vals.max());

        switch (target_difficulty[i].category) {
        case DIF_IMPOSSIBLE: ++nimpossible;  break;
        case DIF_AUTOMATIC: ++nautomatic;  break;
        case DIF_POSSIBLE: ++npossible;  break;
        default: ++nunknown;
        }
    }

    cerr << "mean_neg = " << total_mean_neg / num_neg
         << " mean_pos = " << total_mean_pos / num_pos
         << " min = " << minval << " max = " << maxval
         << endl;

    cerr << "auto " << nautomatic << " imp " << nimpossible << " poss "
         << npossible << endl;
}
