#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include <cmath>
#include <array>
#include <tuple>

#include "Eigen/Dense"

//using namespace std;
using namespace Eigen;

using model_t = std::tuple<double, double, double, double>;
//using point_t = std::tuple<double, double, double>;
using point_t = Vector3d;//std::tuple<double, double, double>;

model_t best_plane_from_points(const std::vector<point_t> &points)
{
    // copy coordinates to  matrix in Eigen format
    size_t num_atoms = points.size();
    Eigen::Matrix<point_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> coord(3, num_atoms);
    for (size_t i = 0; i < num_atoms; ++i)
        coord.col(i) = points[i];

    // calculate centroid
    point_t centroid(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());

    // subtract centroid
    coord.row(0).array() -= centroid(0);
    coord.row(1).array() -= centroid(1);
    coord.row(2).array() -= centroid(2);

    // we only need the left-singular matrix here
    //  http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    point_t plane_normal = svd.matrixU().rightCols<1>();

    const auto &a = plane_normal[0];
    const auto &b = plane_normal[1];
    const auto &c = plane_normal[2];
    const auto &x_0 = centroid[0];
    const auto &y_0 = centroid[1];
    const auto &z_0 = centroid[2];

    return std::make_tuple(a, b, c, -(a * x_0 + b * y_0 + c * z_0));
}

template<typename T>
std::pair<std::vector<T>, std::vector<T>> get_n_rand(const std::vector<T>& vec, int n)
{
    std::unordered_set<int> idxs;
    std::vector<T> not_in_vec = vec;
    std::vector<T> in_vec;
    int max_idx = vec.size();
    while (idxs.size() < std::min(n, max_idx))
    {
        size_t rnd_idx = std::rand() % max_idx;
        if (idxs.find(rnd_idx) == std::end(idxs))
        {
            in_vec.emplace_back(vec[rnd_idx]);
            not_in_vec.erase(std::begin(not_in_vec) + rnd_idx);
            idxs.emplace(rnd_idx);
        }
    }
    return std::make_pair(in_vec, not_in_vec);
}

double distance(const model_t& model, const point_t& point)
{
    const auto& a = std::get<0>(model);
    const auto& b = std::get<1>(model);
    const auto& c = std::get<2>(model);
    const auto& d = std::get<3>(model);
    static const double norm = std::sqrt(a * a + b * b + c * c);

    //return (a * std::get<0>(point) + b * std::get<1>(point) + c * std::get<2>(point) + d)/norm;
    return (a * point[0] + b * point[1] + c * point[2] + d)/norm;
}

template<typename T>
double error(const model_t& model, const std::vector<T> &points)
{
    double err = 0;

    for(const auto& point : points)
    {
        auto dist = distance(model, point);
        err += dist * dist;
    }

    return sqrt(err/points.size());
}

model_t RANSAC(const std::vector<point_t> &points,
    double threshold, size_t min_num, size_t max_iter)
{
    model_t best_fit;
    double best_error = std::numeric_limits<double>::max();
    size_t iter = 0;
    for(; iter < max_iter; iter++)
    {
        //std::cout << "Iteration " << iter + 1 << std::endl;
        //std::cout << "Getting random points" << std::endl;
        auto pts_vecs = get_n_rand(points, min_num);
        auto& mb_inliers = pts_vecs.first;
        auto& not_mb_inliers = pts_vecs.second;
        std::vector<point_t> also_inliers;
        //std::cout << "Fitting plane for rnd pts" << std::endl;
        best_fit = best_plane_from_points(mb_inliers);
        //std::cout << "Searching for also inliers" << std::endl;
        for(const auto& point : not_mb_inliers)
        {
            if(fabs(distance(best_fit, point)) < threshold)
            {
                //std::cout << "Found also inlier" << std::endl;
                also_inliers.emplace_back(point);
            }
        }
        //std::cout << "Found also inliers: " << also_inliers.size() << std::endl;
        if(also_inliers.size() > min_num)
        {
            mb_inliers.insert(std::end(mb_inliers), std::begin(also_inliers), std::end(also_inliers));
            auto better_model = best_plane_from_points(mb_inliers);
            auto new_error = error(better_model, mb_inliers);
            if(new_error < best_error)
            {
                best_fit = better_model;
                best_error = new_error;
                std::cout << "Found new candidate with RMS = " << new_error << std::endl;
                if(new_error < threshold/2)
                    break;
            }
        }
    }
    std::cout << "Resulting number of iterations: " << iter+1 << std::endl;
    return best_fit;
}

int main()
{
    std::ifstream ifs("sdc_point_cloud.txt", std::ifstream::in);
    double threshold = 0;
    double n = 0;
    std::vector<point_t> points;
    ifs >> threshold >> n;
    for(size_t i = 0; i < n; i++)
    {
        point_t point;
        //ifs >> std::get<0>(point) >> std::get<1>(point) >> std::get<2>(point);
        ifs >> point[0] >> point[1] >> point[2];
        points.emplace_back(point);
    }
    auto best_fit = RANSAC(points, threshold, 3, 30000);
    //print_model(best_fit);
    std::ofstream ofs("output.txt", std::ofstream::out);
    ofs << std::get<0>(best_fit) << " " << std::get<1>(best_fit) << " " << 
        std::get<2>(best_fit) << " " << std::get<3>(best_fit) << std::endl;
}