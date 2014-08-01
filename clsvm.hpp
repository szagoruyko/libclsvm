#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

class CLSVM {
public:
  CLSVM (int dims, const std::string loss = "l1", const cl::CommandQueue queue = cl::CommandQueue::getDefault());

  /*
   * train binary SVM with Hinge loss function
   */
  void train (const cl::Buffer& x, const cl::Buffer& y, int batch_size = 32, int max_epochs = 800, float lambda = 1e-5f);

  /*
   * prepare the data and call GPU train function
   */
  void train (const std::vector<float>& x_host, const std::vector<float>& y_host, int batch_size = 32, int max_epochs = 800, float lambda = 1e-5f); 

  /*
   * returns <w,xi> for each sample xi from x
   */
  void decision_function (const cl::Buffer& x, cl::Buffer& decision);

  /*
   * copy x and call GPU equvalent
   */
  void decision_function (const std::vector<float>& x_host, std::vector<float>& decision);

  /*
   * returns sign(<w,xi>) for x
   */
  void predict (const cl::Buffer& x, cl::Buffer& y);

  std::vector<float>
  getWeights();

  inline const cl::Buffer& getCLWeigths () const {  return w; }
protected:
  void setRandomWeights (float lambda);

  float computeWeigtsNorm ();

  void projectOntoL2Ball (float norm);
  
  void compute_temlate (const cl::Buffer& x, cl::Buffer& y, const std::string& kernel_name);
protected:
  cl::CommandQueue queue;
  cl::Program program;
  cl::Buffer w;
  int dim, n_w;
};
