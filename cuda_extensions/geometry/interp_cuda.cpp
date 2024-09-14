#include <torch/extension.h>

#include <vector>

using namespace std;

// CUDA forward declarations

torch::Tensor interp_cuda_forward(
    const torch::Tensor& points,
    const torch::Tensor& queries,
    const torch::Tensor& centers,
    const torch::Tensor& children,
    const torch::Tensor& pi_flat,
    const torch::Tensor& pi_lengths,
    const torch::Tensor& pi_starts,
    const torch::Tensor& radii,
    const torch::Tensor& fan_point,
    const torch::Tensor& fan_node,
    const float& beta,
    const float& inv_delta,
    const int& threads);

std::vector<torch::Tensor> interp_cuda_backward(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& areas,
    const torch::Tensor& queries,
    const torch::Tensor& centers,
    const torch::Tensor& children,
    const torch::Tensor& pi_flat,
    const torch::Tensor& pi_lengths,
    const torch::Tensor& pi_starts,
    const torch::Tensor& ni_flat,
    const torch::Tensor& ni_lengths,
    const torch::Tensor& ni_starts,
    const torch::Tensor& radii,
    const torch::Tensor& features_point,
    const torch::Tensor& fan_point,
    const torch::Tensor& fan_node,
    const torch::Tensor& df_query,
    const float& beta,
    const float& inv_delta,
    const int& threads);

torch::Tensor interp_cuda_pos_grad(
    const torch::Tensor& points,
    const torch::Tensor& normals,
    const torch::Tensor& areas,
    const torch::Tensor& queries,
    const torch::Tensor& centers,
    const torch::Tensor& children,
    const torch::Tensor& pi_flat,
    const torch::Tensor& pi_lengths,
    const torch::Tensor& pi_starts,
    const torch::Tensor& radii,
    const torch::Tensor& fa_point,
    const torch::Tensor& fa_node,
    const float& beta,
    const float& inv_delta,
    const int& threads);

vector<torch::Tensor> initialize_features_fan_cuda(
    const torch::Tensor& features_point,
    const torch::Tensor& normals,
    const torch::Tensor& areas,
    const torch::Tensor& ni_flat,
    const torch::Tensor& ni_lengths,
    const torch::Tensor& ni_starts,
    const int num_nodes);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);


torch::Tensor interp_forward(
    torch::Tensor points,
    torch::Tensor queries,
    torch::Tensor centers,
    torch::Tensor children,
    torch::Tensor pi_flat,
    torch::Tensor pi_lengths,
    torch::Tensor pi_starts,
    torch::Tensor radii,
    torch::Tensor fan_point,
    torch::Tensor fan_node,
    float beta,
    float inv_delta,
    int threads)
{
    CHECK_INPUT(points);
    CHECK_INPUT(queries);
    CHECK_INPUT(centers);
    CHECK_INPUT(children);
    CHECK_INPUT(pi_flat);
    CHECK_INPUT(pi_lengths);
    CHECK_INPUT(pi_starts);
    CHECK_INPUT(radii);
    CHECK_INPUT(fan_point);
    CHECK_INPUT(fan_node);

    auto result = interp_cuda_forward(points, queries, centers,
        children, pi_flat, pi_lengths, pi_starts, radii, fan_point,
        fan_node, beta, inv_delta, threads);

    return result;
}


std::vector<torch::Tensor> interp_backward(
    torch::Tensor points,
    torch::Tensor normals,
    torch::Tensor areas,
    torch::Tensor queries,
    torch::Tensor centers,
    torch::Tensor children,
    torch::Tensor pi_flat,
    torch::Tensor pi_lengths,
    torch::Tensor pi_starts,
    torch::Tensor ni_flat,
    torch::Tensor ni_lengths,
    torch::Tensor ni_starts,
    torch::Tensor radii,
    torch::Tensor features_point,
    torch::Tensor fan_point,
    torch::Tensor fan_node,
    torch::Tensor df_query,
    float beta,
    float inv_delta,
    int threads)
{
    CHECK_INPUT(points);
    CHECK_INPUT(normals);
    CHECK_INPUT(areas);
    CHECK_INPUT(queries);
    CHECK_INPUT(centers);
    CHECK_INPUT(children);
    CHECK_INPUT(pi_flat);
    CHECK_INPUT(pi_lengths);
    CHECK_INPUT(pi_starts);
    CHECK_INPUT(ni_flat);
    CHECK_INPUT(ni_lengths);
    CHECK_INPUT(ni_starts);
    CHECK_INPUT(radii);
    CHECK_INPUT(features_point);
    CHECK_INPUT(fan_point);
    CHECK_INPUT(fan_node);
    CHECK_INPUT(df_query);

    auto result = interp_cuda_backward(points, normals, areas, queries,
        centers, children, pi_flat, pi_lengths, pi_starts, ni_flat,
        ni_lengths, ni_starts, radii, features_point, fan_point, fan_node,
        df_query, beta, inv_delta, threads);

    return result;
}

torch::Tensor interp_pos_grad(
    torch::Tensor points,
    torch::Tensor normals,
    torch::Tensor areas,
    torch::Tensor queries,
    torch::Tensor centers,
    torch::Tensor children,
    torch::Tensor pi_flat,
    torch::Tensor pi_lengths,
    torch::Tensor pi_starts,
    torch::Tensor radii,
    torch::Tensor fan_point,
    torch::Tensor fan_node,
    float beta,
    float inv_delta,
    int threads)
{
    CHECK_INPUT(points);
    CHECK_INPUT(normals);
    CHECK_INPUT(areas);
    CHECK_INPUT(queries);
    CHECK_INPUT(centers);
    CHECK_INPUT(children);
    CHECK_INPUT(pi_flat);
    CHECK_INPUT(pi_lengths);
    CHECK_INPUT(pi_starts);
    CHECK_INPUT(radii);
    CHECK_INPUT(fan_point);
    CHECK_INPUT(fan_node);

    auto result = interp_cuda_pos_grad(points, normals, areas, queries,
                            centers, children, pi_flat, pi_lengths,
                            pi_starts, radii, fan_point, fan_node, beta,
                            inv_delta, threads);

    return result;
}

vector<torch::Tensor> initialize_features_fan(
    torch::Tensor features_point,
    torch::Tensor normals,
    torch::Tensor areas,
    torch::Tensor ni_flat,
    torch::Tensor ni_lengths,
    torch::Tensor ni_starts,
    const int num_nodes)
{
    CHECK_INPUT(features_point);
    CHECK_INPUT(normals);
    CHECK_INPUT(areas);
    CHECK_INPUT(ni_flat);
    CHECK_INPUT(ni_lengths);
    CHECK_INPUT(ni_starts);

    auto result = initialize_features_fan_cuda(features_point, normals, areas, ni_flat,
                                               ni_lengths, ni_starts, num_nodes);

    return result;
}


tuple<
    vector<vector<int>>,
    torch::Tensor> build_octree(
    const torch::Tensor& points)
{

    const int MAX_DEPTH = 30000;

    vector<vector<int>> point_indices;
    vector<torch::Tensor> children;
    vector<torch::Tensor> centers;
    std::vector<float> widths;

    auto get_octant = [](const torch::Tensor& location,
                         const torch::Tensor& center){
        // We use a binary numbering of children. Treating the parent cell's
        // center as the origin, we number the octants in the following manner:
        // The first bit is 1 iff the octant's x coordinate is positive
        // The second bit is 1 iff the octant's y coordinate is positive
        // The third bit is 1 iff the octant's z coordinate is positive
        //
        // For example, the octant with negative x, positive y, positive z is:
        // 110 binary = 6 decimal
        int index = 0;
        if((location[0] >= center[0]).item<bool>()){
            index = index + 1;
        }
        if((location[1] >= center[1]).item<bool>()){
            index = index + 2;
        }
        if((location[2] >= center[2]).item<bool>()){
            index = index + 4;
        }
        return index;
    };


    std::function<torch::Tensor(const torch::Tensor&,
                                const float,
                                const int)>
    translate_center =
        [](const torch::Tensor& parent_center,
        const float h,
        const int child_index) {
        torch::Tensor change_vector = torch::tensor({-h, -h, -h});

        //positive x chilren are 1,3,4,7
        if(child_index % 2){
            change_vector[0] = h;
        }
        //positive y children are 2,3,6,7
        if(child_index == 2 || child_index == 3 ||
            child_index == 6 || child_index == 7){
            change_vector[1] = h;
        }
        //positive z children are 4,5,6,7
        if(child_index > 3){
            change_vector[2] = h;
        }
        torch::Tensor output = parent_center + change_vector;
        return output;
    };

    // How many cells do we have so far?
    int m = 0;

    // Useful list of number 0..7
    const auto options = torch::TensorOptions().dtype(torch::kInt32);
    const torch::Tensor zero_to_seven = torch::arange(8, options);
    const torch::Tensor neg_ones = torch::full({8}, -1, options);

    std::function< void(const int, const int) > helper;
    helper = [&](const int index, const int depth)
    {
        if(point_indices[index].size() > 1 && depth < MAX_DEPTH){
            //give the parent access to the children
            children[index] = zero_to_seven + m;
            //make the children's data in our arrays

            //Add the children to the lists, as default children
            float h = widths[index] / 2;
            torch::Tensor curr_center = centers[index];

            for(int i = 0; i < 8; i++){
                children.emplace_back(neg_ones);
                point_indices.emplace_back(std::vector<int>());
                centers.emplace_back(translate_center(curr_center, h / 2, i));
                widths.emplace_back(h);
            }

            //Split up the points into the corresponding children
            for(int j = 0; j < point_indices[index].size(); j++){
                int curr_point_index = point_indices[index][j];
                int cell_of_curr_point = get_octant(points[curr_point_index], curr_center) + m;
                point_indices[cell_of_curr_point].emplace_back(curr_point_index);
            }

            //Now increase m
            m += 8;

            // Look ma, I'm calling myself.
            for(int i = 0; i < 8; i++){
                helper(children[index][i].item<int>(), depth + 1);
            }
        }
    };

    std::vector<int> all(points.size(0));
    for(int i = 0; i < all.size(); i++) {
        all[i] = i;
    }

    point_indices.emplace_back(all);
    children.emplace_back(neg_ones);

    //Get the minimum AABB for the points
    torch::Tensor backleftbottom = get<0>(points.min(0));
    torch::Tensor frontrighttop = get<0>(points.max(0));
    torch::Tensor aabb_center = (backleftbottom + frontrighttop) / 2;
    float aabb_width = (frontrighttop - backleftbottom).max().item<float>();
    centers.emplace_back(aabb_center);

    //Widths are the side length of the cube, (not half the side length):
    widths.emplace_back(aabb_width);
    m++;
    // then you have to actually call the function
    helper(0, 0);

    torch::Tensor children_tensor = torch::stack(children, 0);

    return {point_indices, children_tensor};
}


vector<torch::Tensor> initialize_octree(
  const torch::Tensor& points,
  const torch::Tensor& areas,
  const std::vector<std::vector<int>>& point_indices,
  const torch::Tensor& children)
{
    const float PI_4 = 4.0 * M_PI;

    int num_nodes = point_indices.size();

    torch::Tensor centers = torch::zeros({num_nodes, 3});
    torch::Tensor radii = torch::zeros({num_nodes});

    std::function< void(const int, const int) > helper;
    helper = [&]
    (const int index, const int depth)-> void
    {
        torch::Tensor center = torch::zeros({3});
        float area_total = 0;

        for (int j = 0; j < point_indices[index].size(); j++) {
            int curr_point_index = point_indices[index][j];
            area_total += areas[curr_point_index].item<float>();
            center += areas[curr_point_index] * points[curr_point_index];
        }

        center = center / (area_total + 1e-8);
        centers[index] = center;

        if (point_indices[index].size() == 0) {
            radii[index] = 0;
        } else{
            torch::Tensor vecs = points.index({torch::tensor(point_indices[index])}) - center.index({torch::indexing::None, "..."});
            torch::Tensor dists = torch::sqrt((vecs * vecs).sum(-1));
            float max_norm = dists.max().item<float>();
            radii[index] = max_norm;
        }

        if(children[index][0].item<int>() != -1) {
            #pragma omp parallel for
            for (int i = 0; i < 8; i++){
                int child_index = children[index][i].item<int>();
                helper(child_index, depth + 1);
            }
        }

    };
    helper(0, 0);

    return {centers, radii};
}


torch::Tensor intersect_octree(
  const torch::Tensor& points,
  const std::vector<std::vector<int>>& point_indices,
  const torch::Tensor& children,
  const torch::Tensor& centers,
  const torch::Tensor& radii,
  const torch::Tensor& origins,
  const torch::Tensor& directions)
{
    int num_nodes = point_indices.size();
    int num_queries = origins.size(0);

    torch::Tensor isect_times = torch::zeros({num_queries, 1});

    auto intersect_sphere = []
    (const torch::Tensor& o, const torch::Tensor& d, const torch::Tensor& c, const torch::Tensor& r) -> float
    {
        const auto& oc = o - c;
        const auto& b = d.dot(oc);

        const auto& disc = b * b - (oc.dot(oc) - r * r);
        if (disc.item<float>() < 0) {
            return -1;
        }

        const auto& t0 = -b - torch::sqrt(disc);
        const auto& t1 = -b + torch::sqrt(disc);

        if (t0.item<float>() < 0) {
            return t1.item<float>();
        }

        return min(t0.item<float>(), t1.item<float>());
    };

    std::function< float(const torch::Tensor&, const torch::Tensor&, const int) > helper;
    helper = [&](const torch::Tensor& o, const torch::Tensor& d, const int index) -> float
    {
        // directly intersecting a leaf
        // or close enough to a node
        if ((children[index][0].item<int>() == -1 || radii[index].item<float>() < 0.01) &&
            point_indices[index].size() > 0)
        {
            const auto& p = centers[index];
            return d.dot(p - o).item<float>();
        }
        else
        {
            vector<pair<float, int>> t_index_pairs;
            for (int i = 0; i < 8; i++){
                const auto& child_index = children[index][i];
                float t = intersect_sphere(o, d, centers[child_index], radii[child_index]);
                if (t > 0) {
                    t_index_pairs.push_back({t, child_index.item<int>()});
                }
            }
            sort(t_index_pairs.begin(), t_index_pairs.end(), [](auto &left, auto &right) {
                return left.first < right.first;
            });

            for (const auto& [t, child_index] : t_index_pairs) {
                float t_child = helper(o, d, child_index);
                if (t_child > 0) {
                    return t_child;
                }
            }
            return -1;
        }
    };

    #pragma omp parallel for
    for (int iter = 0; iter < num_queries; iter += 1) {
        const auto& o = origins[iter];
        const auto& d = directions[iter];
        const auto& d_norm = d.norm();
        isect_times[iter] = helper(o, d / d_norm, 0) / d_norm;
    }

    // for (int iter = 0; iter < num_queries; iter += 1) {
    //   T(iter) = helper(O.row(iter), D.row(iter), 0);
    // }
    return isect_times;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("interp_forward", &interp_forward, "interp forward (CUDA)");
    m.def("interp_backward", &interp_backward, "interp backward (CUDA)");
    m.def("interp_pos_grad", &interp_pos_grad, "interp positional gradient (CUDA)");
    m.def("build_octree", &build_octree, "build octree");
    m.def("intersect_octree", &intersect_octree, "intersect octree");
    m.def("initialize_octree", &initialize_octree, "initialize octree");
    m.def("initialize_features_fan", &initialize_features_fan, "initialize node and point features (area and normal weighted)");
}
