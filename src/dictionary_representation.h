#ifndef DICTIONARY_REPRESENTATION_H
#define DICTIONARY_REPRESENTATION_H

#include <pcl/point_types.h>
#include <vector>
#include <fstream>

class dictionary_representation
{
protected:
    float res; // the size of the voxels
    int sz; // size of voxel patch sz x sz
    int dict_size; // size of dictionary to represent depths
    int words_max; // the maximum number of dict entries for one patch
    int RGB_dict_size; // size of dictionary to represent RGB
    int RGB_words_max; // the maximum number of dict entries for one patch

    // the rotations of the patches
    std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > rotations;
    // the 3D means of the patches
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > means;
    // the means in RGB color space
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > RGB_means;

    Eigen::MatrixXf S; // the depth patches as horizontally stacked vectors
    // same for RGB, first the R vectors then G and B
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> RGB;
    // the masks showing where in the patches there are observations
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> W;

    Eigen::MatrixXf D; // depth dictionary, horizontally stacked
    Eigen::MatrixXf X; // code word weights, not all entries are used
    Eigen::MatrixXi I; // the index of the dictionary entries used
    std::vector<int> number_words; // the number of code words used for a patch

    Eigen::MatrixXf RGB_D; // RGB dictionary
    Eigen::MatrixXf RGB_X; // code word weights
    Eigen::MatrixXi RGB_I; // dictionary indices
    std::vector<int> RGB_number_words; // words used

    bool read_bool(std::ifstream& i, u_char& buffer, int& b);
    void read_dict_file(Eigen::MatrixXf& dict, const std::string& file);
    void read_from_file(const std::string& file);

    void write_bool(std::ofstream& o, u_char& buffer, int& b, bool bit);
    void close_write_bools(std::ofstream& o, u_char& buffer);
    void write_dict_file(const Eigen::MatrixXf& dict, const std::string& file);
    void write_to_file(const std::string& file);
public:
    dictionary_representation();
    dictionary_representation(float res, int sz, int dict_size, int words_max,
                              int RGB_dict_size, int RGB_words_max);
};

#endif // DICTIONARY_REPRESENTATION_H
