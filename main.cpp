#include <iostream>
#include <matio.h>
#include <vector>

#include "clsvm.hpp"

enum {PLATFROM = 0, DEVICE = 2};


int main (int argc, char** argv)
{
  if (argc<2)
    return -1;
  auto matfilename = argv[1];

  mat_t* mat = Mat_Open(matfilename, MAT_ACC_RDONLY);

  matvar_t* xvar = Mat_VarRead(mat, "x");
  matvar_t* yvar = Mat_VarRead(mat, "y");
  const int n = xvar->dims[1];
  const int dims = xvar->dims[0];

  std::vector<float> x (n*dims), y (n);
  for (int i=0; i<n; ++i)
  {
    y[i] = ((double*)yvar->data)[i];
    for (int j=0; j<dims; ++j)
      x[i*dims + j] = ((double*)xvar->data)[i*dims + j];
  }

  std::vector<cl::Platform> platforms(5);
  std::vector<cl::Device> mdevices(5);

  cl::Platform::get (&platforms);
  platforms[0].getDevices (CL_DEVICE_TYPE_ALL, &mdevices);
  std::cout << "Using " << mdevices[DEVICE].getInfo<CL_DEVICE_NAME>() << std::endl;
  
  auto device = mdevices[DEVICE];
  cl::Context context (device);
  cl::CommandQueue queue (context, device);

  cl::Buffer X (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*n*dims, x.data());
  cl::Buffer Y (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*n, y.data());
  cl::Buffer D (context, CL_MEM_READ_WRITE, sizeof(float)*n);

  CLSVM svm (queue, dims-1);
  svm.train(X, Y, 64);
  svm.decision_function (X, D);

  std::vector<float> d (n);
  queue.enqueueReadBuffer(D, CL_TRUE, 0, sizeof(float)*n, d.data());
  for(int i=0; i<n; ++i)
    std::cout << (d[i]>0.f)*2.f-1.f << " " << y[i] << std::endl;

  return 0;
}
