#include "karto_sdk/contrib/ChowLiuTreeApprox.h"
#include "karto_sdk/contrib/EigenExtensions.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

namespace karto
{
namespace contrib
{

Eigen::SparseMatrix<double> ComputeMarginalInformationMatrix(
    const Eigen::SparseMatrix<double> & information_matrix,
    const Eigen::Index discarded_variable_index,
    const Eigen::Index variables_dimension)
{
  const Eigen::Index dimension = information_matrix.outerSize();
  assert(dimension == information_matrix.innerSize());  // must be square
  const Eigen::Index marginal_dimension = dimension - variables_dimension;  // 255 = 258 - 3
  const Eigen::Index last_variable_index = dimension - variables_dimension; // 255 = 258 -3
  // (1) Break up information matrix based on which are the variables
  // kept (a) and which is the variable discarded (b).
  Eigen::SparseMatrix<double>
      information_submatrix_aa, information_submatrix_ab,
      information_submatrix_ba, information_submatrix_bb;
  if (discarded_variable_index == 0) {
    information_submatrix_aa =
        information_matrix.bottomRightCorner(
            marginal_dimension, marginal_dimension);    // 255, 255
    information_submatrix_ab =
        information_matrix.bottomLeftCorner(
            marginal_dimension, variables_dimension);   // 255, 3
    information_submatrix_ba =
        information_matrix.topRightCorner(
            variables_dimension, marginal_dimension);   // 3, 255
    information_submatrix_bb =
        information_matrix.topLeftCorner(
            variables_dimension, variables_dimension);  // 3, 3
  } else if (discarded_variable_index == last_variable_index) {
std::cout << "2. else-if information_submatrix_aa / ab/ ba /bb created " << std::endl;
    information_submatrix_aa =
        information_matrix.topLeftCorner(
            marginal_dimension, marginal_dimension);
    information_submatrix_ab =
        information_matrix.topRightCorner(
            marginal_dimension, variables_dimension);
    information_submatrix_ba =
        information_matrix.bottomLeftCorner(
            variables_dimension, marginal_dimension);
    information_submatrix_bb =
        information_matrix.bottomRightCorner(
            variables_dimension, variables_dimension);
  } else {
std::cout << "2. else information_submatrix_aa / ab/ ba /bb created " << std::endl;
    const Eigen::Index next_variable_index =
        discarded_variable_index + variables_dimension;
    information_submatrix_aa = StackVertically(
        StackHorizontally(
            information_matrix.topLeftCorner(
                discarded_variable_index,
                discarded_variable_index),
            information_matrix.topRightCorner(
                discarded_variable_index,
                dimension - next_variable_index)),
        StackHorizontally(
            information_matrix.bottomLeftCorner(
                dimension - next_variable_index,
                discarded_variable_index),
            information_matrix.bottomRightCorner(
                dimension - next_variable_index,
                dimension - next_variable_index)));
    information_submatrix_ab = StackVertically(
        information_matrix.block(
            0,
            discarded_variable_index,
            discarded_variable_index,
            variables_dimension),
        information_matrix.block(
            next_variable_index,
            discarded_variable_index,
            dimension - next_variable_index,
            variables_dimension));
    information_submatrix_ba = StackHorizontally(
        information_matrix.block(
            discarded_variable_index,
            0,
            variables_dimension,
            discarded_variable_index),
        information_matrix.block(
            discarded_variable_index,
            next_variable_index,
            variables_dimension,
            dimension - next_variable_index));
    information_submatrix_bb =
        information_matrix.block(
            discarded_variable_index,
            discarded_variable_index,
            variables_dimension,
            variables_dimension);
  }

  // (2) Compute Schur's complement over the variables that are kept.
  std::cout << "information_submatrix_aa.size(): " << information_submatrix_aa.size() << "; " << information_submatrix_aa.rows() << "; " << information_submatrix_aa.cols() << std::endl;
  std::cout << "information_submatrix_ba.size(): " << information_submatrix_ba.size() << "; " << information_submatrix_ba.rows() << "; " << information_submatrix_ba.cols() << std::endl;
  std::cout << "information_submatrix_ab.size(): " << information_submatrix_ab.size() << "; " << information_submatrix_ab.rows() << "; " << information_submatrix_ab.cols() << std::endl;
  std::cout << "information_submatrix_ab.size(): " << information_submatrix_bb.size() << "; " << information_submatrix_bb.rows() << "; " << information_submatrix_bb.cols() << std::endl;
  Eigen::SparseMatrix<double> aaa = ComputeSparseInverse(information_submatrix_bb);
  std::cout << "inverse_bb:\n" << aaa << std::endl;

  Eigen::SparseMatrix<double> MarginalInformationMatrix = (information_submatrix_aa - information_submatrix_ab * ComputeSparseInverse(information_submatrix_bb) * information_submatrix_ba);
  std::cout << "ComputeSparseInverse:\n " << ComputeSparseInverse(information_submatrix_bb) << std::endl;
  
  // Record the information_matrix in information_matrix.csv
  std::ofstream file_aa("information_submatrix_aa.csv");
  if (file_aa.is_open()) {
    file_aa << information_submatrix_aa << '\n';
  }
  std::ofstream file_ab("information_submatrix_ab.csv");
  if (file_ab.is_open()) {
    file_ab << information_submatrix_ab << '\n';
  }
  std::ofstream file_bb("information_submatrix_bb.csv");
  if (file_bb.is_open()) {
    file_bb << information_submatrix_bb << '\n';
  }
  std::ofstream file_mi("MarginalInformationMatrix.csv");
  if (file_mi.is_open()) {
    file_mi << MarginalInformationMatrix << '\n';
  }

  // return (information_submatrix_aa - information_submatrix_ba *
  //         ComputeSparseInverse(information_submatrix_bb) *
  //         information_submatrix_ab);
  return MarginalInformationMatrix;
}

UncertainPose2 ComputeRelativePose2(
    Pose2 source_pose, Pose2 target_pose,
    Eigen::Matrix<double, 6, 6> joint_pose_covariance)
{
  // Computation is carried out as proposed in section 3.2 of:
  //
  //    R. Smith, M. Self and P. Cheeseman, "Estimating uncertain spatial
  //    relationships in robotics," Proceedings. 1987 IEEE International
  //    Conference on Robotics and Automation, 1987, pp. 850-850,
  //    doi: 10.1109/ROBOT.1987.1087846.
  //
  // In particular, this is a case of tail-tail composition of two spatial
  // relationships p_ij and p_ik as in: p_jk = ⊖ p_ij ⊕ p_ik
  UncertainPose2 relative_pose;
  // (1) Compute mean relative pose by simply
  // transforming mean source and target poses.
  Transform source_transform(source_pose);
  relative_pose.mean =
      source_transform.InverseTransformPose(target_pose);
  // (2) Compute relative pose covariance by linearizing
  // the transformation around mean source and target
  // poses.
  Eigen::Matrix<double, 3, 6> transform_jacobian;
  const double x_jk = relative_pose.mean.GetX();
  const double y_jk = relative_pose.mean.GetY();
  const double theta_ij = source_pose.GetHeading();
  transform_jacobian <<
      -cos(theta_ij), -sin(theta_ij),  y_jk,  cos(theta_ij), sin(theta_ij), 0.0,
       sin(theta_ij), -cos(theta_ij), -x_jk, -sin(theta_ij), cos(theta_ij), 0.0,
                 0.0,            0.0,  -1.0,            0.0,           0.0, 1.0;
  relative_pose.covariance = Matrix3(
      transform_jacobian * joint_pose_covariance *
      transform_jacobian.transpose());
  return relative_pose;
}

std::vector<Edge<LocalizedRangeScan> *> ComputeChowLiuTreeApproximation(
  const std::vector<Vertex<LocalizedRangeScan> *> & clique,
  const Eigen::SparseMatrix<double> & covariance_matrix)
{

  std::cout << "ComputeChowLiuTreeApproximation called with clique size: " << clique.size() << std::endl;
//   std::cout << "covariance_matrix:\n" << covariance_matrix << std::endl;

  // (1) Build clique subgraph, weighting edges by the *negated* mutual
  // information between corresponding variables (so as to apply
  // Kruskal's minimum spanning tree algorithm down below).
  using WeightedGraphT = boost::adjacency_list<
    boost::vecS, boost::vecS, boost::undirectedS, boost::no_property,
    boost::property<boost::edge_weight_t, double>>;
  WeightedGraphT clique_subgraph(clique.size());

  // Calculate the dense covariance submatrices for each pair of variables in the clique
  // MI(X, Y) = 0.5 * log2(det(Σ_X) / det(Σ_X - Σ_XY * Σ_Y ^ -1 * Σ_YX))
  // where Σ_X is the covariance matrix for variable X, Σ_Y is the covariance matrix for variable Y, and Σ_XY is the covariance matrix for the pair of variables (X, Y).
  for (size_t i = 0; i < clique.size() - 1; ++i) {
    for (size_t j = i + 1; j < clique.size(); ++j) {
      const auto covariance_submatrix_ii =
          Eigen::Matrix3d{covariance_matrix.block(i, i, 3, 3)};
      const auto covariance_submatrix_ij =
          Eigen::Matrix3d{covariance_matrix.block(i, j, 3, 3)};
      const auto covariance_submatrix_ji =
          Eigen::Matrix3d{covariance_matrix.block(j, i, 3, 3)};
      const auto covariance_submatrix_jj =
          Eigen::Matrix3d{covariance_matrix.block(j, j, 3, 3)};
      const double mutual_information =
          0.5 * std::log2(covariance_submatrix_ii.determinant() / 
          ( covariance_submatrix_ii - covariance_submatrix_ij * covariance_submatrix_jj.inverse() * covariance_submatrix_ji ).determinant());

      std::cout << "covariance_submatrix_ii:\n" << covariance_submatrix_ii << std::endl;
      std::cout << "covariance_submatrix_ij:\n" << covariance_submatrix_ij << std::endl;
      std::cout << "covariance_submatrix_ji:\n" << covariance_submatrix_ji << std::endl;
      std::cout << "covariance_submatrix_jj:\n" << covariance_submatrix_jj << std::endl;

      std::cout << "Determinant of covariance_submatrix_ii: " << covariance_submatrix_ii.determinant() << std::endl;
      std::cout << "Determinant of covariance_submatrix_ij: " << covariance_submatrix_ij.determinant() << std::endl;
      std::cout << "Determinant of covariance_submatrix_ji: " << covariance_submatrix_ji.determinant() << std::endl;
      std::cout << "Determinant of covariance_submatrix_jj: " << covariance_submatrix_jj.determinant() << std::endl;

      boost::add_edge(i, j, -mutual_information, clique_subgraph);
      std::cout << "Added edge between " << i << " and " << j << " with mutual information: " << mutual_information << std::endl;
    }
  }

  // (2) Find maximum mutual information spanning tree in the clique subgraph
  // (which best approximates the underlying joint probability distribution as
  // proved by Chow & Liu).
  using EdgeDescriptorT =
      boost::graph_traits<WeightedGraphT>::edge_descriptor;
  std::vector<EdgeDescriptorT> minimum_spanning_tree_edges;
  boost::kruskal_minimum_spanning_tree(
      clique_subgraph, std::back_inserter(minimum_spanning_tree_edges));
  std::cout << "Computed minimum spanning tree with " << minimum_spanning_tree_edges.size() << " edges." << std::endl;

  using VertexDescriptorT =
      boost::graph_traits<WeightedGraphT>::vertex_descriptor;

  // (3) Build tree approximation as an edge list, using the mean and
  // covariance of the marginal joint distribution between each variable
  // to recompute the nonlinear constraint (i.e. a 2D isometry) between them.
  std::vector<Edge<LocalizedRangeScan> *> chow_liu_tree_approximation;
  for (const EdgeDescriptorT & edge_descriptor : minimum_spanning_tree_edges) {
    const VertexDescriptorT i = boost::source(edge_descriptor, clique_subgraph);
    const VertexDescriptorT j = boost::target(edge_descriptor, clique_subgraph);
    std::cout << "Processing edge between " << i << " and " << j << std::endl;

    auto * edge = new Edge<LocalizedRangeScan>(clique[i], clique[j]);
    Eigen::Matrix<double, 6, 6> joint_pose_covariance_matrix;
    joint_pose_covariance_matrix <<  // marginalized from the larger matrix
        Eigen::Matrix3d{covariance_matrix.block(i, i, 3, 3)},
        Eigen::Matrix3d{covariance_matrix.block(i, j, 3, 3)},
        Eigen::Matrix3d{covariance_matrix.block(j, i, 3, 3)},
        Eigen::Matrix3d{covariance_matrix.block(j, j, 3, 3)};
    LocalizedRangeScan * source_scan = edge->GetSource()->GetObject();
    LocalizedRangeScan * target_scan = edge->GetTarget()->GetObject();
    const UncertainPose2 relative_pose =
        ComputeRelativePose2(source_scan->GetCorrectedPose(),
                             target_scan->GetCorrectedPose(),
                             joint_pose_covariance_matrix);
    edge->SetLabel(new LinkInfo(
        source_scan->GetCorrectedPose(),
        target_scan->GetCorrectedPose(),
        relative_pose.mean, relative_pose.covariance));
    chow_liu_tree_approximation.push_back(edge);
  }
  std::cout << "Chow-Liu tree approximation has " << chow_liu_tree_approximation.size() << " edges." << std::endl;
  return chow_liu_tree_approximation;
}

}  // namespace contrib
}  // namespace karto
