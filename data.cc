/* data.cc
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   File to read data for the competition.
*/

#include "data.h"
#include "jml/utils/parse_context.h"
#include "jml/stats/distribution_ops.h"
#include "jml/stats/distribution_simd.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/pair_utils.h"
#include "jml/algebra/lapack.h"
#include "jml/arch/timers.h"
#include "jml/arch/threads.h"
#include "decomposition.h"
#include "jml/utils/worker_task.h"
#include "jml/utils/guard.h"
#include <boost/bind.hpp>
#include "jml/stats/rmse.h"
#include "jml/stats/auc.h"
#include "jml/utils/filter_streams.h"


using namespace std;
using namespace ML;


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
    return ML::calc_rmse(*this, targets);
}

double
Model_Output::
calc_rmse(const distribution<float> & targets,
          const distribution<float> & weights) const
{
    return ML::calc_rmse(*this, targets, weights);
}

double
Model_Output::
calc_auc(const distribution<float> & targets) const
{
    if ((targets == 0).all()) return 1;
    if ((targets == 0).any()) {
        cerr << "targets = " << targets << endl;
        cerr << "problem with calc_auc" << endl;
    }
    return ML::calc_auc(*this, targets, -1.0, 1.0);
}

double
Model_Output::
calc_auc(const distribution<float> & targets,
         const distribution<float> & weights) const
{
    return ML::calc_auc(*this, targets, weights, -1.0, 1.0);
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

    filter_istream stream(filename);
    Parse_Context c(filename, stream);

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
    }

    int num_rows = 0;
    for (; c; ++num_rows) {
        int id = c.expect_int();
        example_ids.push_back(id);
        c.expect_literal(',');

        distribution<float> model_values(nm);

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
            model_values[i] = val;
        }

        boost::shared_ptr<Example> example
            (new Example(model_values, target_val, target));
        examples.push_back(example);

        c.skip_whitespace();
        c.expect_eol();
    }

    //cerr << num_rows << " rows... ";
    calc_scores();
}

void
Data::
clear()
{
    targets.clear();
    model_names.clear();
    example_ids.clear();
    models.clear();
    model_ranking.clear();
    examples.clear();
    decomposition = 0;
}

void
Data::
swap(Data & other)
{
    std::swap(target, other.target);
    targets.swap(other.targets);
    model_names.swap(other.model_names);
    example_ids.swap(other.example_ids);
    models.swap(other.models);
    model_ranking.swap(other.model_ranking);
    examples.swap(other.examples);
    std::swap(decomposition, other.decomposition);
}

void
Data::
calc_scores()
{
    vector<pair<float, int> > model_scores;

    Model_Output outputs[16];
    for (unsigned i = 0;  i < 16;  ++i)
        outputs[i].resize(nx());
    
    for (unsigned i = 0;  i < nm();  i += 16) {
        int ii = std::min<int>(i + 16, nm());

        for (unsigned x = 0;  x < nx();  ++x) {
            const distribution<float> & m = examples[x]->models;
            
            for (unsigned j = i;  j < ii;  ++j)
                outputs[j - i][x] = m[j];
        }
        
        for (unsigned j = i;  j < ii;  ++j) {
            if (target == RMSE)
                models[j].score = outputs[j - i].calc_rmse(targets);
            else models[j].score = outputs[j - i].calc_auc(targets);
            
            model_scores.push_back(make_pair(models[j].score, j));
        }
    }

    sort_on_first_ascending(model_scores);

    for (unsigned i = 0;  i < nm();  ++i)
        models[model_scores[i].second].rank = i;

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

    vector<bool> remove_me(nx(), false);
    for (unsigned i = 0;  i < to_remove.size();  ++i)
        remove_me[to_remove[i]] = true;

    hold_out(remove_to, remove_me, example_weights,
             remove_to_example_weights);
}

void
Data::
hold_out(Data & remove_to, const vector<bool> & to_remove)
{
    distribution<float> example_weights(nx());
    distribution<float> remove_to_example_weights;

    hold_out(remove_to, to_remove, example_weights,
             remove_to_example_weights);
}

void
Data::
hold_out(Data & remove_to, const vector<bool> & remove_me,
         distribution<float> & example_weights,
         distribution<float> & remove_to_example_weights)
{
    remove_to.clear();
    
    if (remove_me.size() != nx())
        throw Exception("remove_me size was wrong");

    int num_to_remove = 0;
    for (unsigned i = 0;  i < remove_me.size();  ++i)
        if (remove_me[i]) ++num_to_remove;

    Data new_me;

    new_me.target = remove_to.target = target;
    new_me.model_names = remove_to.model_names = model_names;
    new_me.decomposition = remove_to.decomposition = decomposition;

    new_me.models.resize(model_names.size());
    remove_to.models.resize(model_names.size());

    distribution<float> new_example_weights;
    new_example_weights.reserve(nx() - num_to_remove);

    remove_to_example_weights.clear();
    remove_to_example_weights.reserve(num_to_remove);

    for (unsigned i = 0;  i < nx();  ++i) {
        Data & add_to = remove_me[i] ? remove_to : new_me;
        distribution<float> & weights
            = remove_me[i] ? remove_to_example_weights : new_example_weights;
        add_to.targets.push_back(targets[i]);
        add_to.example_ids.push_back(example_ids[i]);
        add_to.examples.push_back(examples[i]);
        weights.push_back(example_weights[i]);
    }

    new_me.calc_scores();
    remove_to.calc_scores();

    swap(new_me);
    example_weights.swap(new_example_weights);
}

struct Decompose_Job {

    vector<boost::shared_ptr<Data::Example> > & examples;
    const Decomposition & decomposition;
    int first;
    int last;

    Decompose_Job(vector<boost::shared_ptr<Data::Example> > & examples,
                  const Decomposition & decomposition,
                  int first, int last)
        : examples(examples),
          decomposition(decomposition), first(first), last(last)
    {
    }

    void operator () ()
    {
        for (unsigned x = first;  x < last;  ++x)
            examples[x]->decomposed
                = decomposition.decompose(examples[x]->models);
    }
};

void
Data::
apply_decomposition(const Decomposition & decomposition)
{
    this->decomposition = &decomposition;

    static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
        
    // Now, submit it as jobs to the worker task to be done multithreaded
    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "dump user results task", parent);
        
        // Make sure the group gets unlocked once we've populated
        // everything
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        for (unsigned i = 0;  i < nx();  i += 100) {
            int last = std::min<int>(nx(), i + 100);

            Decompose_Job job(examples,
                              decomposition,
                              i, last);

            worker.add(job, "decompose_job", group);
        }
    }
        
    // Add this thread to the thread pool until we're ready
    worker.run_until_finished(group);
}

distribution<float>
Data::
apply_decomposition(const distribution<float> & example) const
{
    if (!decomposition) return distribution<float>();
    return decomposition->decompose(example);
}

#if 0
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

#if 0
    cerr << "mean_neg = " << total_mean_neg / num_neg
         << " mean_pos = " << total_mean_pos / num_pos
         << " min = " << minval << " max = " << maxval
         << endl;

    cerr << "auto " << nautomatic << " imp " << nimpossible << " poss "
         << npossible << endl;
#endif
}
#endif

size_t
Data::
decomposition_size() const
{
    return (decomposition ? decomposition->size() : 0);
}
