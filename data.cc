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
#include "svdlibc/svdlib.h"
#include "arch/timers.h"
#include "algebra/matrix_ops.h"

using namespace std;
using namespace ML;
using ML::Stats::sqr;


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

    return sqrt(sqr(targets - *this).total() * (1.0 / size()));
}
        
double
Model_Output::
calc_auc(const distribution<float> & targets) const
{
    if (targets.size() != size())
        throw Exception("targets and predictions don't match");
    

    // 1.  Sort the predictions
    int num_neg = 0, num_pos = 0;

    vector<pair<float, float> > sorted;
    for (unsigned i = 0;  i < size();  ++i) {
        sorted.push_back(make_pair((*this)[i], targets[i]));
        if (targets[i] == -1) ++num_neg;
        else ++num_pos;
    }

    std::sort(sorted.begin(), sorted.end());
    
    // 2.  Get (x,y) points and calculate the AUC
    int total_pos = 0, total_neg = 0;

    float prevx = 0.0, prevy = 0.0;

    double total_area = 0.0;

    for (unsigned i = 0;  i < sorted.size();  ++i) {
        if (sorted[i].second == -1) ++total_neg;
        else ++total_pos;
        
        float x = total_pos * 1.0 / num_pos;
        float y = total_neg * 1.0 / num_neg;
        
        double area = (x - prevx) * (y + prevy) * 0.5;

        total_area += area;

        prevx = x;
        prevy = y;
    }


    // 4.  Convert to gini
    double gini = 2.0 * (total_area - 0.5);

    // 5.  Final score is absolute value
    return fabs(gini);
}

double
Model_Output::
calc_rmse_weighted(const distribution<float> & targets,
                   const distribution<double> & weights) const
{
    if (targets.size() != size())
        throw Exception("targets and predictions don't match");

    if (weights.size() != size())
        throw Exception("weights and predictions don't match");

    return sqrt((sqr(targets - *this) * weights).total() / weights.total());
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


/*****************************************************************************/
/* DATA                                                                      */
/*****************************************************************************/

void
Data::
load(const std::string & filename, Target target)
{
    this->target = target;

    Parse_Context c(filename);

    // First row: header.
    c.expect_literal("RowID,Target");

    while (!c.match_eol()) {
        c.expect_literal(',');
        string model_name = c.expect_text(",\n\r");
        model_names.push_back(model_name);
    }

    // Create the data structures
    cerr << model_names.size() << " models... ";
    
    models.resize(model_names.size());
    for (unsigned i = 0;  i < models.size();  ++i) {
        models[i].reserve(50000);
    }
    
    int num_rows = 0;
    for (; c; ++num_rows) {
        int id = c.expect_int();
        model_ids.push_back(id);
        c.expect_literal(',');

        float target_val = c.expect_int();

        if (target == RMSE) target_val /= 1000.0;

        targets.push_back(target_val);

        for (unsigned i = 0;  i < models.size();  ++i) {
            c.expect_literal(',');
            int score = c.expect_int();
            models[i].push_back(score / 1000.0);
        }

        c.skip_whitespace();
        c.expect_eol();
    }

    cerr << num_rows << " rows... ";
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
}

void
Data::
calc_scores()
{
    vector<pair<float, int> > model_scores;

    for (unsigned i = 0;  i < models.size();  ++i) {
        if (target == RMSE)
            models[i].score = models[i].calc_rmse(targets);
        else models[i].score = models[i].calc_auc(targets);

        model_scores.push_back(make_pair(models[i].score, i));
    }

    if (target == RMSE)
        sort_on_first_ascending(model_scores);
    else sort_on_first_descending(model_scores);

    for (unsigned i = 0;  i < models.size();  ++i)
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
    srand(random_seed);

    vector<int> to_remove;
    for (unsigned i = 0;  i < targets.size();  ++i) {
        to_remove.push_back(i);
    }

    std::random_shuffle(to_remove.begin(), to_remove.end());

    if (proportion < 0 || proportion > 1.0)
        throw Exception("bad proportion");
    
    to_remove.erase(to_remove.begin() + proportion * to_remove.size(),
                    to_remove.end());

    vector<int> remove_me(targets.size(), false);
    for (unsigned i = 0;  i < to_remove.size();  ++i)
        remove_me[to_remove[i]] = true;

    remove_to.clear();
    
    Data new_me;

    new_me.target = remove_to.target = target;
    new_me.model_names = remove_to.model_names = model_names;
    
    new_me.models.resize(model_names.size());
    remove_to.models.resize(model_names.size());

    for (unsigned i = 0;  i < model_names.size();  ++i) {
        new_me.models[i].reserve(targets.size() - to_remove.size());
        remove_to.models[i].reserve(to_remove.size());
    }

    for (unsigned i = 0;  i < targets.size();  ++i) {
        Data & add_to = remove_me[i] ? remove_to : new_me;
        add_to.targets.push_back(targets[i]);
        add_to.model_ids.push_back(model_ids[i]);

        for (unsigned j = 0;  j < model_names.size();  ++j)
            add_to.models[j].push_back(models[j][i]);
    }

    swap(new_me);
}

void
Data::
decompose()
{
    cerr << "doing decomposition" << endl;

    boost::multi_array<float, 2> values(boost::extents[models.size()][targets.size()]);

    for (unsigned m = 0;  m < models.size();  ++m)
        for (unsigned x = 0;  x < targets.size();  ++x)
            values[m][x] = models[m][x];

    int m = models.size();
    int n = targets.size();

    int minmn = std::min(m, n);

    cerr << "m = " << m << " n = " << n << " minnm = " << minmn << endl;

    distribution<float> svalues(minmn);
    
    boost::multi_array<float, 2> lvectors(boost::extents[m][m]);
    boost::multi_array<float, 2> rvectors(boost::extents[n][n]);

    Timer timer;

    int result = LAPack::gesdd("S", n, m,
                               values.data(), n,
                               &svalues[0],
                               &rvectors[0][0], n,
                               &lvectors[0][0], m);

    //lvectors = transpose(lvectors);
    //rvectors = transpose(rvectors);

    cerr << timer.elapsed() << endl;
    timer.restart();

    cerr << "result was " << result << endl;
    cerr << "svectors = " << svalues << endl;

    struct smat matrix2;
    matrix2.rows = m;
    matrix2.cols = n;
    matrix2.vals = m * n;
    matrix2.pointr = new long[n + 1];
    matrix2.rowind = new long[m * n];
    matrix2.value  = new double[m * n];

    matrix2.pointr[0] = 0;

    for (unsigned i = 0;  i < n;  ++i) {
        for (unsigned j = 0;  j < m;  ++j) {
            int idx = i * m + j;
            matrix2.rowind[idx] = j;
            matrix2.value[idx] = models[j][i];
        }
        matrix2.pointr[i + 1] = (i + 1) * m;
    }

    int nvalues = 50;


    // Run the SVD
    svdrec * svdresult = svdLAS2A(&matrix2, nvalues);

    cerr << "SVD elapsed: " << timer.elapsed() << endl;

    if (!svdresult)
        throw Exception("error performing SVD");

    //cerr << "num_valid_repos = " << num_valid_repos << endl;
    
    distribution<float> svalues2(svdresult->S, svdresult->S + nvalues);

    cerr << "highest values: " << svalues2 << endl;

    cerr << "result->Ut->rows = " << svdresult->Ut->rows << endl;
    cerr << "result->Ut->cols = " << svdresult->Ut->cols << endl;
    cerr << "result->Vt->rows = " << svdresult->Vt->rows << endl;
    cerr << "result->Vt->cols = " << svdresult->Vt->cols << endl;

    distribution<float> u0(svdresult->Ut->value[0], svdresult->Ut->value[0] + m);
    distribution<float> u02(&lvectors[0][0], &lvectors[0][0] + m);

    cerr << "u0 = " << u0 << endl;
    cerr << "u02 = " << u02 << endl;
    
    distribution<float> v0(svdresult->Vt->value[0], svdresult->Vt->value[0] + 50/*n*/);
    cerr << "v0 = " << v0 << endl;

    distribution<double> mean_rating(50);
    for (unsigned i = 0;  i < 50;  i++)
        for (unsigned j = 0;  j < m;  ++j)
            mean_rating[i] += models[j][i] / m;

    cerr << "mean ratings " << mean_rating << endl;


    // Free up memory (TODO: put into guards...)
    delete[] matrix2.pointr;
    delete[] matrix2.rowind;
    delete[] matrix2.value;
    svdFreeSVDRec(svdresult);


}
