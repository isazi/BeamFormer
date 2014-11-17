// Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ctime>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <BeamFormer.hpp>

typedef float dataType;
std::string typeName("float");


int main(int argc, char *argv[]) {
  bool print = false;
  bool random = false;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int nrSamplesPerBlock = 0;
	unsigned int nrBeamsPerBlock = 0;
  unsigned int nrSamplesPerThread = 0;
  unsigned int nrBeamsPerThread = 0;
  long long unsigned int wrongSamples = 0;
  AstroData::Observation observation;

  try {
    isa::utils::ArgumentList args(argc, argv);
    print = args.getSwitch("-print");
    random = args.getSwitch("-random");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
		nrSamplesPerBlock = args.getSwitchArgument< unsigned int >("-sb");
		nrBeamsPerBlock = args.getSwitchArgument< unsigned int >("-bb");
		nrSamplesPerThread = args.getSwitchArgument< unsigned int >("-st");
		nrBeamsPerThread = args.getSwitchArgument< unsigned int >("-bt");
    observation.setNrBeams(args.getSwitchArgument< unsigned int >("-beams"));
    observation.setNrStations(args.getSwitchArgument< unsigned int >("-stations"));
    observation.setFrequencyRange(args.getSwitchArgument< unsigned int >("-channels"), 0, 0);
		observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
	} catch  ( isa::utils::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " [-print] [-random] -opencl_platform ... -opencl_device ... -padding ... -sb ... -bb ... -st ... -bt ... -beams ... -stations ... -samples ... -channels ..." << std::endl;
		return 1;
	}

	// Initialize OpenCL
	cl::Context * clContext = new cl::Context();
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate host memory
  std::vector< dataType > samples = std::vector< dataType >(observation.getNrChannels() * observation.getNrStations() * observation.getNrSamplesPerPaddedSecond() * 4);
  std::vector< dataType > output = std::vector< dataType >(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerPaddedSecond() * 4);
  std::vector< dataType > output_c = std::vector< dataType >(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerPaddedSecond() * 4);
  std::vector< float > weights = std::vector< float >(observation.getNrChannels() * observation.getNrStations() * observation.getNrPaddedBeams() * 2);
  if ( random ) {
    std::srand(time(0));
  } else {
    std::srand(42);
  }
  std::fill(weights.begin(), weights.end(), std::rand() % 100);
  std::fill(samples.begin(), samples.end(), std::rand() % 1000);

  // Allocate device memory
  cl::Buffer samples_d, output_d, weights_d;
  try {
    samples_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, samples.size() * sizeof(dataType), 0, 0);
    output_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerPaddedSecond() * 4 * sizeof(dataType), 0, 0);
    weights_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, weights.size() * sizeof(float), 0, 0);
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString(err.err()) << "." << std::endl;
    return 1;
  }

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(weights_d, CL_FALSE, 0, weights.size() * sizeof(float), reinterpret_cast< void * >(weights.data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(samples_d, CL_FALSE, 0, samples.size() * sizeof(dataType), reinterpret_cast< void * >(samples.data()));
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString(err.err()) << "." << std::endl;
    return 1;
  }

	// Generate kernel
  std::string * code = RadioAstronomy::getBeamFormerOpenCL(nrSamplesPerBlock, nrBeamsPerBlock, nrSamplesPerThread, nrBeamsPerThread, typeName, observation);
  cl::Kernel * kernel;
  if ( print ) {
    std::cout << *code << std::endl;
  }
	try {
    kernel = isa::OpenCL::compile("beamFormer", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
	} catch ( isa::OpenCL::OpenCLError &err ) {
    std::cerr << err.what() << std::endl;
		return 1;
	}

  // Run OpenCL kernel and CPU control
  try {
    cl::NDRange global(observation.getNrSamplesPerPaddedSecond() / nrSamplesPerThread, observation.getNrBeams() / nrBeamsPerThread, observation.getNrChannels());
    cl::NDRange local(nrSamplesPerBlock, nrBeamsPerBlock, 1);

    kernel->setArg(0, samples_d);
    kernel->setArg(1, output_d);
    kernel->setArg(2, weights_d);
    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local);
    RadioAstronomy::beamFormer< dataType >(observation, samples, output_c, weights);
    clQueues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(dataType), reinterpret_cast< void * >(output.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error kernel execution: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ ) {
    for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ ) {
      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
        for ( unsigned int item = 0; item < 4; item++ ) {
          if ( !isa::utils::same(output[(beam * observation.getNrChannels() * observation.getNrSamplesPerPaddedSecond() * 4) + (channel * observation.getNrSamplesPerPaddedSecond() * 4) + (sample * 4) + item], output_c[(beam * observation.getNrChannels() * observation.getNrSamplesPerPaddedSecond() * 4) + (channel * observation.getNrSamplesPerPaddedSecond() * 4) + (sample * 4) + item]) ) {
            wrongSamples++;
          }
        }
      }
    }
  }

  if ( wrongSamples > 0 ) {
    std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / (static_cast< long long unsigned int >(observation.getNrBeams()) *observation.getNrChannels() * observation.getNrSamplesPerSecond() * 4) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

	return 0;
}

