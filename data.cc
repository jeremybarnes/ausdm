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


using namespace std;
using namespace ML;
using ML::Stats::sqr;

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
        string model_name = c.expect_text(",\n");
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

    for (unsigned i = 0;  i < 20;  ++i) {
        int m = model_scores[i].second;
        cerr << "rank " << i << " " << model_names[m] << " score "
             << models[m].score << endl;
    }
}

double
Data::Model::
calc_rmse(const distribution<float> & targets) const
{
    if (targets.size() != size())
        throw Exception("targets and predictions don't match");

    //cerr << "targets[0] = " << targets[0] << endl;
    //cerr << "this[0] = " << (*this)[0] << endl;

    return sqrt(sqr(targets - *this).total() * (1.0 / size()));
}
        
double
Data::Model::
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
    
    // 2.  Get (x,y) points
    vector<pair<float, float> > points;
    points.push_back(make_pair(0.0, 0.0));
    int total_pos = 0, total_neg = 0;

    for (unsigned i = 0;  i < sorted.size();  ++i) {
        if (sorted[i].second == -1) ++total_neg;
        else ++total_pos;

        points.push_back(make_pair(total_pos * 1.0 / num_pos,
                                   total_neg * 1.0 / num_neg));
    }

    // 3.  Calculate the AUC
    std::sort(points.begin(), points.end());

    double total = 0.0;
    for (unsigned i = 1;  i < points.size();  ++i) {
        float prevx = points[i - 1].first;
        float prevy = points[i - 1].second;
        
        float x = points[i].first;
        float y = points[i].second;

        double area = (x - prevx) * (y + prevy) * 0.5;

        total += area;
    }

    // 4.  Convert to gini
    double gini = 2.0 * (total - 0.5);

    // 5.  Final score is absolute value
    return fabs(gini);
}
