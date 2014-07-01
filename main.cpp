#include <iostream>
#include <matio.h>
#include <vector>

#include "clsvm.hpp"

enum {PLATFROM = 0, DEVICE = 0};


int main (int argc, char** argv)
{
  if (argc<2)
    return -1;
  auto matfilename = argv[1];

  mat_t* mat = Mat_Open(matfilename, MAT_ACC_RDONLY);

  matvar_t* xvar = Mat_VarRead(mat, "x");
  matvar_t* yvar = Mat_VarRead(mat, "y");
  const int n = static_cast<int>(xvar->dims[1]);
  const int dims = static_cast<int>(xvar->dims[0]);

  // check that data is in single precision format
  if (xvar->data_type != MAT_T_SINGLE || yvar->data_type != MAT_T_SINGLE)
  {
    std::cout << "Please provide data in single format: x [n_samples x dim], y [n_samples x 1]\n";
    std::cout << "Quitting..\n";
    return -1;
  }

  std::cout << "Got " << n << " samples\n";
  std::cout << "Problem dimensionality: " << dims << std::endl;

  std::vector<cl::Platform> platforms(5);
  std::vector<cl::Device> mdevices(5);

  cl::Platform::get (&platforms);
  platforms[0].getDevices (CL_DEVICE_TYPE_ALL, &mdevices);
  std::cout << "Using " << mdevices[DEVICE].getInfo<CL_DEVICE_NAME>() << std::endl;
  
  auto device = mdevices[DEVICE];
  cl::Context context (device);
  cl::CommandQueue queue (context, device, CL_QUEUE_PROFILING_ENABLE);

  cl::Buffer X (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*n*dims, xvar->data);
  cl::Buffer Y (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*n, yvar->data);
  cl::Buffer D (context, CL_MEM_READ_WRITE, sizeof(float)*n);

  CLSVM svm (dims, queue);
  svm.train(X, Y, 64, 2000, 1e+0f);
  svm.predict (X, D);

  std::vector<float> d (n);
  queue.enqueueReadBuffer(D, CL_TRUE, 0, sizeof(float)*n, d.data());
  int missclassified = 0;
  for(int i=0; i<n; ++i)
    if (d[i] != ((float*)yvar->data)[i])
      missclassified++;
  std::cout << missclassified << "/" << n << std::endl;
  return 0;
}
