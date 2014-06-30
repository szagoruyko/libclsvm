#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

class CLSVM {
public:
  CLSVM(const cl::CommandQueue queue, int dims);

  void train (const cl::Buffer& x, const cl::Buffer& y, int batch_size = 32, int max_epochs = 800, float lambda = 1e-5f);
        
  void decision_function (const cl::Buffer& x, cl::Buffer& decision);
        
  void setRandomWeights ();
        
  float computeWeigtsNorm ();
        
  void projectOntoL2Ball (float norm);
protected:
  cl::CommandQueue queue;
  cl::Program program;
  cl::Buffer w;
  int dim, n_w;
};
