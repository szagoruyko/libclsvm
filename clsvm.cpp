#include <vector>
#include <fstream>
#include <iostream>
#include <Eigen/Core>

#include "clsvm.hpp"

CLSVM::CLSVM(const cl::CommandQueue queue, int dims) : queue(queue), dim(dims+1) {
  const char source_name[] = "sgd.cl";
  printf ("Loading opencl source (%s)...\n", source_name);
        
  std::ifstream source_file (source_name);
  if(!source_file.is_open())
    std::cout<< "File " << source_name << " not found" <<std::endl;
      
  auto context = queue.getInfo<CL_QUEUE_CONTEXT>();
  std::vector<cl::Device> devices (1, queue.getInfo<CL_QUEUE_DEVICE>());
  std::string code (std::istreambuf_iterator<char>(source_file),
		   (std::istreambuf_iterator<char>()));
  program = cl::Program (context, code);
  program.build (devices, "");
  
  std::cout<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) <<std::endl;
  
  w = cl::Buffer (context, CL_MEM_READ_WRITE, sizeof(float)*dim);
}
    
void
CLSVM::train (const cl::Buffer& x, const cl::Buffer& y, int batch_size, int max_epochs, float lambda)
{
  const int k = batch_size;
  const int n = y.getInfo<CL_MEM_SIZE>()/sizeof(float);

  auto context = queue.getInfo<CL_QUEUE_CONTEXT> ();	
  cl::Buffer idx (context, CL_MEM_READ_ONLY, sizeof(int)*k);
  cl::Buffer res (context, CL_MEM_READ_WRITE, sizeof(float)*k);
  
  std::vector<float> hy (n);
  queue.enqueueReadBuffer (y, CL_TRUE, 0, sizeof(float)*n, hy.data());
  
  setRandomWeights();
  
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
    update (cl::EnqueueArgs (queue, cl::NDRange(dim)), idx, x, y, w, dim, etat, lambda, n_selected);
    
    // compute norm and project onto l2 norm ball
    float norm = computeWeigtsNorm();
    if (1.f/sqrt(lambda)/norm <= 1.f)
      projectOntoL2Ball(norm);
  }
}
    
void
CLSVM::decision_function (const cl::Buffer& x, cl::Buffer& decision)
{
  size_t n_samples = x.getInfo<CL_MEM_SIZE>()/sizeof(float)/dim;
  auto kernel = cl::make_kernel<const cl::Buffer&, const cl::Buffer&, cl::Buffer&, int> (program, "decision_function");
  kernel (cl::EnqueueArgs(queue, cl::NDRange (n_samples)), x, w, decision, dim);
}
    
void
CLSVM::setRandomWeights ()
{
  Eigen::VectorXf winit = Eigen::VectorXf::Random (dim);
  //std::cout << winit.norm() << " " << 1.f/sqrt(lambda) << std::endl;
  queue.enqueueWriteBuffer (w, CL_TRUE, 0, sizeof(float)*dim, winit.data());
}
    
float
CLSVM::computeWeigtsNorm ()
{
  cl::Buffer l2norm (queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, sizeof(float));
  auto kernel = cl::make_kernel<const cl::Buffer&, cl::Buffer&, int> (program, "compute_l2norm");
  kernel (cl::EnqueueArgs (queue, cl::NDRange (1)), w, l2norm, dim);
  float ret = 1.0f;
  cl::enqueueReadBuffer (l2norm, CL_TRUE, 0, sizeof(float), &ret);
  return ret;
}
    
void
CLSVM::projectOntoL2Ball (float norm)
{
  auto kernel = cl::make_kernel<cl::Buffer&, float> (program, "projectOntoL2Ball");
  kernel (cl::EnqueueArgs (queue, cl::NDRange (dim)), w, norm);
}

