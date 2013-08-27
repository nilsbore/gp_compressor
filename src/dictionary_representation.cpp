#include "dictionary_representation.h"

using namespace Eigen;

dictionary_representation::dictionary_representation()
{

}

dictionary_representation::dictionary_representation(float res, int sz,int dict_size, int words_max,
                                                     int RGB_dict_size, int RGB_words_max) :
    res(res), sz(sz), dict_size(dict_size), words_max(words_max),
    RGB_dict_size(RGB_dict_size), RGB_words_max(RGB_words_max)
{

}


void dictionary_representation::read_dict_file(MatrixXf& dict, const std::string& file)
{
    std::ifstream dict_file(file, std::ios::binary);
    int cols = dict.cols();
    int rows = dict.rows();
    dict_file.read((char*)&cols, sizeof(int));
    dict_file.read((char*)&rows, sizeof(int));
    dict.resize(rows, cols);
    float value;
    for (int j = 0; j < dict.cols(); ++j) {
        for (int n = 0; n < dict.rows(); ++n) {
            dict_file.read((char*)&value, sizeof(float));
            dict(n, j) = value;
        }
    }
    dict_file.close();
}

bool dictionary_representation::read_bool(std::ifstream& i, u_char& buffer, int& b)
{
    if (b == 0 || b == 8) {
        i.read((char*)&buffer, sizeof(u_char));
        b = 0;
    }
    bool bit = (buffer >> b) & u_char(1);
    b++;
    return bit;
}

void dictionary_representation::read_from_file(const std::string& file)
{
    std::string rgbfile = file + "rgb.pcdict";
    read_dict_file(RGB_D, rgbfile);
    std::string depthfile = file + "depth.pcdict";
    read_dict_file(D, depthfile);
    std::string code = file + ".pccode";

    std::ifstream code_file(code, std::ios::binary);
    int nbr;
    code_file.read((char*)&nbr, sizeof(int)); // number of patches
    code_file.read((char*)&sz, sizeof(int));
    code_file.read((char*)&words_max, sizeof(int));
    code_file.read((char*)&RGB_words_max, sizeof(int));

    S.resize(sz*sz, nbr);
    W.resize(sz*sz, nbr);
    RGB.resize(sz*sz, 3*nbr);
    rotations.resize(nbr);
    means.resize(nbr);
    RGB_means.resize(nbr);

    X.resize(words_max, nbr);
    I.resize(words_max, nbr);
    number_words.resize(nbr);

    RGB_X.resize(RGB_words_max, 3*nbr);
    RGB_I.resize(RGB_words_max, 3*nbr);
    RGB_number_words.resize(3*nbr);

    code_file.read((char*)&dict_size, sizeof(int)); // dictionary size
    code_file.read((char*)&RGB_dict_size, sizeof(int)); // RGB dictionary size
    code_file.read((char*)&res, sizeof(float)); // size of voxels
    float value;
    for (int i = 0; i < S.cols(); ++i) { // means of patches
        for (int n = 0; n < 3; ++n) {
            code_file.read((char*)&value, sizeof(float));
            means[i](n) = value;
        }
    }
    for (int i = 0; i < S.cols(); ++i) { // rotations of patches, quaternions
        for (int n = 0; n < 4; ++n) {
            code_file.read((char*)&value, sizeof(float));
            rotations[i].coeffs()(n) = value;
        }
    }
    u_char words;
    for (int i = 0; i < S.cols(); ++i) { // number of words and codes
        code_file.read((char*)&words, sizeof(u_char));
        number_words[i] = words;
        for (int n = 0; n < words; ++n) {
            code_file.read((char*)&value, sizeof(float));
            X(n, i) = value;
        }
    }
    uint16_t word;
    for (int i = 0; i < S.cols(); ++i) { // dictionary entries used
        for (int n = 0; n < number_words[i]; ++n) {
            code_file.read((char*)&word, sizeof(uint16_t));
            I(n, i) = int(word);
        }
    }
    for (int i = 0; i < S.cols(); ++i) { // rgb means of patches
        for (int n = 0; n < 3; ++n) {
            code_file.read((char*)&value, sizeof(float));
            RGB_means[i](n) = value;
        }
    }
    for (int i = 0; i < 3*S.cols(); ++i) { // rgb number of words and codes
        code_file.read((char*)&words, sizeof(u_char));
        RGB_number_words[i] = words;
        for (int n = 0; n < words; ++n) {
            code_file.read((char*)&value, sizeof(float));
            RGB_X(n, i) = value;
        }
    }
    for (int i = 0; i < 3*S.cols(); ++i) { // rgb dictionary entries used
        for (int n = 0; n < RGB_number_words[i]; ++n) {
            code_file.read((char*)&word, sizeof(uint16_t));
            RGB_I(n, i) = word;
        }
    }
    u_char buffer = 0;
    int b = 0;
    for (int i = 0; i < S.cols(); ++i) { // masks of patches
        for (int n = 0; n < sz*sz; ++n) {
            W(n, i) = read_bool(code_file, buffer, b);
        }
    }
    code_file.close();
}

void dictionary_representation::write_dict_file(const MatrixXf& dict, const std::string& file)
{
    std::ofstream dict_file(file, std::ios::binary | std::ios::trunc);
    int cols = dict.cols();
    int rows = dict.rows();
    dict_file.write((char*)&cols, sizeof(int));
    dict_file.write((char*)&rows, sizeof(int));
    float value;
    for (int j = 0; j < dict.cols(); ++j) {
        for (int n = 0; n < dict.rows(); ++n) {
            value = dict(n, j);
            dict_file.write((char*)&value, sizeof(float));
        }
    }
    dict_file.close();
}

void dictionary_representation::write_bool(std::ofstream& o, u_char& buffer, int& b, bool bit)
{
    if (b == 8) {
        o.write((char*)&buffer, sizeof(u_char));
        buffer = 0;
        b = 0;
    }
    buffer |= u_char(bit) << b;
    b++;
}

void dictionary_representation::close_write_bools(std::ofstream& o, u_char& buffer)
{
    o.write((char*)&buffer, sizeof(u_char));
}

void dictionary_representation::write_to_file(const std::string& file)
{
    std::string rgbfile = file + "rgb.pcdict";
    write_dict_file(RGB_D, rgbfile);
    std::string depthfile = file + "depth.pcdict";
    write_dict_file(D, depthfile);
    std::string code = file + ".pccode";

    std::ofstream code_file(code, std::ios::binary | std::ios::trunc);
    int nbr = S.cols();
    code_file.write((char*)&nbr, sizeof(int)); // number of patches
    code_file.write((char*)&sz, sizeof(int));
    code_file.write((char*)&words_max, sizeof(int));
    code_file.write((char*)&RGB_words_max, sizeof(int));
    code_file.write((char*)&dict_size, sizeof(int)); // dictionary size
    code_file.write((char*)&RGB_dict_size, sizeof(int)); // RGB dictionary size
    code_file.write((char*)&res, sizeof(float)); // size of voxels
    float value;
    for (int i = 0; i < S.cols(); ++i) { // means of patches
        for (int n = 0; n < 3; ++n) {
            value = means[i](n);
            code_file.write((char*)&value, sizeof(float));
        }
    }
    for (int i = 0; i < S.cols(); ++i) { // rotations of patches, quaternions
        for (int n = 0; n < 4; ++n) {
            value = rotations[i].coeffs()(n);
            code_file.write((char*)&value, sizeof(float));
        }
    }
    u_char words;
    for (int i = 0; i < S.cols(); ++i) { // number of words and codes
        words = number_words[i];
        code_file.write((char*)&words, sizeof(u_char));
        for (int n = 0; n < words; ++n) {
            value = X(n, i);
            code_file.write((char*)&value, sizeof(float));
        }
    }
    uint16_t word;
    for (int i = 0; i < S.cols(); ++i) { // dictionary entries used
        for (int n = 0; n < number_words[i]; ++n) {
            word = I(n, i);
            code_file.write((char*)&word, sizeof(uint16_t));
        }
    }
    for (int i = 0; i < S.cols(); ++i) { // rgb means of patches
        for (int n = 0; n < 3; ++n) {
            value = RGB_means[i](n);
            code_file.write((char*)&value, sizeof(float));
        }
    }
    for (int i = 0; i < 3*S.cols(); ++i) { // rgb number of words and codes
        words = RGB_number_words[i];
        code_file.write((char*)&words, sizeof(u_char));
        for (int n = 0; n < words; ++n) {
            value = RGB_X(n, i);
            code_file.write((char*)&value, sizeof(float));
        }
    }
    for (int i = 0; i < 3*S.cols(); ++i) { // rgb dictionary entries used
        for (int n = 0; n < RGB_number_words[i]; ++n) {
            word = RGB_I(n, i);
            code_file.write((char*)&word, sizeof(uint16_t));
        }
    }
    u_char buffer = 0;
    int b = 0;
    for (int i = 0; i < S.cols(); ++i) { // masks of patches
        for (int n = 0; n < sz*sz; ++n) {
            write_bool(code_file, buffer, b, W(n, i));
        }
    }
    close_write_bools(code_file, buffer);
    code_file.close();
}
