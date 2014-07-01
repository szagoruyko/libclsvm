#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>

#include "clsvm.hpp"


enum {CTX = 32, CTU = 32};

// temporary profiling function
void printProfilingInfo (const std::string& message, const cl::Event& event)
{
  auto st = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  auto fn = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  std::cout << message << " time: " << float(fn - st)/1000000.0 << " ms" << std::endl;
}


CLSVM::CLSVM(const cl::CommandQueue queue, int dims) : queue(queue), dim(dims), n_w(dims+1) {
  const char source_name[] = "sgd.cl";
  printf ("Loading opencl source (%s)...\n", source_name);

  std::ifstream source_file (source_name);
  if(!source_file.is_open())
    std::cout<< "File " << source_name << " not found" <<std::endl;
      
  auto context = queue.getInfo<CL_QUEUE_CONTEXT>();
  std::vector<cl::Device> devices (1, queue.getInfo<CL_QUEUE_DEVICE>());
  std::string code (std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
  program = cl::Program (context, code);
  program.build (devices, "");
  
  std::cout<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) <<std::endl;
  
  w = cl::Buffer (context, CL_MEM_READ_WRITE, sizeof(float)*n_w);
}


void
CLSVM::train (const cl::Buffer& x, const cl::Buffer& y, int batch_size, int max_epochs, float lambda)
{
  const int k = batch_size;
  const int n = static_cast<int>(y.getInfo<CL_MEM_SIZE>())/sizeof(float);

  auto context = queue.getInfo<CL_QUEUE_CONTEXT> ();	
  cl::Buffer idx (context, CL_MEM_READ_ONLY, sizeof(int)*k);
  cl::Buffer res (context, CL_MEM_READ_WRITE, sizeof(float)*k);
  
  std::vector<float> hy (n);
  queue.enqueueReadBuffer (y, CL_TRUE, 0, sizeof(float)*n, hy.data());
  
  setRandomWeights(lambda);
  
  auto kernel = cl::make_kernel<const cl::Buffer&, const cl::Buffer&, const cl::Buffer&, cl::Buffer&, int> (program, "compute_kernel");
  auto update = cl::make_kernel<const cl::Buffer&, const cl::Buffer&, const cl::Buffer&, cl::Buffer&, int, float, float, int> (program, "update_weights");
  
  for (int t=1; t<max_epochs; t++)
  {
    // generate candidates for the batch
    std::vector<int> candidates (k);
    for (auto& it: candidates)
      it = rand()%n;
    queue.enqueueWriteBuffer (idx, CL_TRUE, 0, sizeof(int)*k, candidates.data());
    
    // compute decision function x*w
    kernel (cl::EnqueueArgs (queue, cl::NDRange (k)), idx, x, w, res, dim);
    
    // select samples with y*x*w < 1
    std::vector<float> hres (k);
    queue.enqueueReadBuffer (res, CL_TRUE, 0, sizeof(float)*k, hres.data());
    
    std::vector<int> selected;
    for (int i=0; i<k; ++i)
      if (hres[i]*hy[candidates[i]] < 1.f)
        selected.push_back (candidates[i]);
    if (selected.empty())
      continue;
    
    // if found some samples, compute subgradients and update the weights
    queue.enqueueWriteBuffer (idx, CL_TRUE, 0, sizeof(int)*selected.size(), selected.data());
    
    float etat = 1.0f/(lambda*float(t));
    
    int n_selected = static_cast<int> (selected.size());
    int nwrk = (n_w + CTU -1)/CTU;
    cl::Event event = update (cl::EnqueueArgs (queue, cl::NDRange(nwrk*CTU), cl::NDRange(CTU)), idx, x, y, w, dim, etat, lambda, n_selected);
    
    // compute norm and project onto l2 norm ball
    float norm = computeWeigtsNorm();
    if (1.f/sqrt(lambda)/norm <= 1.f)
      projectOntoL2Ball(norm);
  }
}


void
CLSVM::compute_temlate(const cl::Buffer& x, cl::Buffer &y, const std::string& kernel_name)
{
  int n_samples = static_cast<int>(x.getInfo<CL_MEM_SIZE>())/sizeof(float)/dim;
  auto kernel = cl::make_kernel<const cl::Buffer&, const cl::Buffer&, cl::Buffer&, int, int> (program, kernel_name);

  int nwrk = (n_samples + CTX -1)/CTX;
  cl::Event event = kernel (cl::EnqueueArgs(queue, cl::NDRange (nwrk*CTX), cl::NDRange(CTX)), x, w, y, dim, n_samples);
  event.wait();
  printProfilingInfo(kernel_name + " exec time", event);
}


void
CLSVM::decision_function (const cl::Buffer& x, cl::Buffer& decision)
{
  compute_temlate(x, decision, "decision_function");
}


void
CLSVM::predict (const cl::Buffer& x, cl::Buffer& y)
{
  compute_temlate(x, y, "predict");
}


void
CLSVM::setRandomWeights (float lambda)
{
  std::vector<float> winit(n_w);
  for (auto &it: winit)
    it = float(rand()%RAND_MAX)/float(RAND_MAX);
  queue.enqueueWriteBuffer (w, CL_TRUE, 0, sizeof(float)*n_w, winit.data());
  float norm = computeWeigtsNorm();
  if (1.f/sqrt(lambda)/norm <= 1.f)
    projectOntoL2Ball(norm);
}


float
CLSVM::computeWeigtsNorm ()
{
  auto context = queue.getInfo<CL_QUEUE_CONTEXT>();
  cl::Buffer l2norm (context, CL_MEM_READ_WRITE, sizeof(float));
  auto kernel = cl::make_kernel<const cl::Buffer&, cl::Buffer&, int> (program, "compute_l2norm");
  kernel (cl::EnqueueArgs (queue, cl::NDRange (1)), w, l2norm, n_w);
  queue.finish();
  float ret = 1.0f;
  queue.enqueueReadBuffer (l2norm, CL_TRUE, 0, sizeof(float), &ret);
  return ret;
}


void
CLSVM::projectOntoL2Ball (float norm)
{
  auto kernel = cl::make_kernel<cl::Buffer&, float> (program, "projectOntoL2Ball");
  kernel (cl::EnqueueArgs (queue, cl::NDRange (dim)), w, norm);
}


std::vector<float>
CLSVM::getWeights()
{
  std::vector<float> ret(n_w);
  queue.enqueueReadBuffer(w, CL_TRUE, 0, sizeof(float)*n_w, ret.data());
  return ret;
}

