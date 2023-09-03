#ifndef NAIVE_COMM
#define NAIVE_COMM

#include "abstract.h"

class NaiveComm : public AbstractComm {
public:
    // Declare all public members here
    void all_to_all(const torch::Tensor &, const torch::Tensor &, size_t);
    NaiveComm(std::vector<at::cuda::CUDAStream> *, std::vector<ncclComm_t>, const int &, const int &, const int &, const int &);
};

#endif // NAIVE_COMM