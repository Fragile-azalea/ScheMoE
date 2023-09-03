#ifndef DD_COMM
#define DD_COMM

#include "comm/abstract.h"
#include "jit.h"

class DdComm : public AbstractComm {
public:
    // Declare all public members here
    void all_to_all(const torch::Tensor &, const torch::Tensor &, size_t);
    DdComm(std::vector<at::cuda::CUDAStream> *, std::vector<ncclComm_t>, const int &, const int &, const int &, const int &);
};

#endif // DD_COMM