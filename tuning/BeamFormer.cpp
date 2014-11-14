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
#include <algorithm>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <BeamFormer.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>

typedef float dataType;
std::string typeName("float");


int main(int argc, char * argv[]) {
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int minThreads = 0;
  unsigned int maxThreads = 0;
	unsigned int maxRows = 0;
	unsigned int maxColumns = 0;
  unsigned int threadUnit = 0;
  unsigned int threadIncrement = 0;
  unsigned int maxItems = 0;
  AstroData::Observation observation;

	try {
    isa::utils::ArgumentList args(argc, argv);

		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    threadUnit = args.getSwitchArgument< unsigned int >("-thread_unit");
		minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
		maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
		maxRows = args.getSwitchArgument< unsigned int >("-max_rows");
		maxColumns = args.getSwitchArgument< unsigned int >("-max_columns");
    threadIncrement = args.getSwitchArgument< unsigned int >("-thread_increment");
		maxItems = args.getSwitchArgument< unsigned int >("-max_items");
    observation.setNrBeams(args.getSwitchArgument< unsigned int >("-beams"));
    observation.setNrStations(args.getSwitchArgument< unsigned int >("-stations"));
    observation.setFrequencyRange(args.getSwitchArgument< unsigned int >("-channels"), 0, 0);
		observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
	} catch ( isa::utils::EmptyCommandLine & err ) {
		std::cerr << argv[0] << " -iterations ... -opencl_platform ... -opencl_device ... -padding ... -thread_unit ... -min_threads ... -max_threads ... -max_items ... -max_columns ... -max_rows ... -thread_increment ... -beams ... -stations ... -samples ... -channels ..." << std::endl;
		return 1;
	} catch ( std::exception & err ) {
		std::cerr << err.what() << std::endl;
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
  std::vector< float > weights = std::vector< float >(observation.getNrChannels() * observation.getNrStations() * observation.getNrPaddedBeams() * 2);
  std::srand(time(0));
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

	// Find the parameters
	std::vector< unsigned int > samplesPerBlock;
	for ( unsigned int samples = minThreads; samples <= maxColumns; samples += threadIncrement ) {
		if ( (observation.getNrSamplesPerPaddedSecond() % samples) == 0 ) {
			samplesPerBlock.push_back(samples);
		}
	}
	std::vector< unsigned int > beamsPerBlock;
	for ( unsigned int beams = 1; beams <= maxRows; beams++ ) {
		if ( (observation.getNrBeams() % beams) == 0 ) {
			beamsPerBlock.push_back(beams);
		}
	}

	std::cout << std::fixed << std::endl;
	std::cout << "# nrBeams nrStations nrChannels nrSamples samplesPerBlock beamsPerBlock samplesPerThread beamsPerThread GFLOP/s GB/s time stdDeviation COV" << std::endl << std::endl;

	for ( std::vector< unsigned int >::iterator samples = samplesPerBlock.begin(); samples != samplesPerBlock.end(); ++samples ) {
		for ( std::vector< unsigned int >::iterator beams = beamsPerBlock.begin(); beams != beamsPerBlock.end(); ++beams ) {
			if ( ((*samples) * (*beams)) > maxThreads ) {
				break;
			} else if ( ((*samples) * (*beams)) % threadUnit != 0 ) {
        continue;
      }

			for ( unsigned int samplesPerThread = 1; samplesPerThread <= maxItems; samplesPerThread++ ) {
				if ( (observation.getNrSamplesPerPaddedSecond() % ((*samples) * samplesPerThread)) != 0 ) {
					continue;
				}

				for ( unsigned int beamsPerThread = 1; beamsPerThread <= maxItems; beamsPerThread++ ) {
					if ( (observation.getNrBeams() % ((*beams) * beamsPerThread)) != 0 ) {
						continue;
					} else if ( (samplesPerThread + (samplesPerThread * beamsPerThread * 4) + 8) > maxItems ) {
						break;
					}

          // Generate kernel
          double gflops = isa::utils::giga((static_cast< long long unsigned int >(observation.getNrBeams()) * observation.getNrChannels() * observation.getNrSamplesPerSecond() * observation.getNrStations() * 16) + (static_cast< long long unsigned int >(observation.getNrBeams()) * observation.getNrChannels() * observation.getNrSamplesPerSecond() * 4));
          double gbs = isa::utils::giga((static_cast< long long unsigned int >(observation.getNrChannels()) * observation.getNrSamplesPerSecond() * observation.getNrStations() * (observation.getNrBeams() / (beamsPerThread * *beams)) * 4 * sizeof(dataType)) + (static_cast< long long unsigned int >(observation.getNrBeams()) * observation.getNrChannels() * observation.getNrSamplesPerSecond() * 4 * sizeof(dataType)) + (observation.getNrChannels() * observation.getNrStations() * (observation.getNrBeams() / (beamsPerThread * *beams)) * 2 * sizeof(float)));
          isa::utils::Timer timer;
          cl::Event event;
          cl::Kernel * kernel;
          std::string * code = RadioAstronomy::getBeamFormerOpenCL(*samples, *beams, samplesPerThread, beamsPerThread, typeName, observation);

          try {
            kernel = isa::OpenCL::compile("beamFormer", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
          } catch ( isa::OpenCL::OpenCLError & err ) {
            std::cerr << err.what() << std::endl;
            continue;
          }

          cl::NDRange global(observation.getNrSamplesPerPaddedSecond() / samplesPerThread, observation.getNrBeams() / beamsPerThread, observation.getNrChannels());
          cl::NDRange local(*samples, *beams, 1);

          kernel->setArg(0, samples_d);
          kernel->setArg(1, output_d);
          kernel->setArg(2, weights_d);

          // Warm-up run
          try {
            clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
            event.wait();
          } catch ( cl::Error & err ) {
            std::cerr << "OpenCL error kernel execution: " << isa::utils::toString(err.err()) << "." << std::endl;
            continue;
          }
          // Tuning runs
          try {
            for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
              timer.start();
              clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
              event.wait();
              timer.stop();
            }
          } catch ( cl::Error & err ) {
            std::cerr << "OpenCL error kernel execution: " << isa::utils::toString(err.err()) << "." << std::endl;
            continue;
          }

          std::cout << observation.getNrBeams() << " " << observation.getNrStations() << " " << observation.getNrChannels() << " " << observation.getNrSamplesPerSecond() << " " << *samples << " " << *beams << " " << samplesPerThread << " " << beamsPerThread << " ";
          std::cout << std::setprecision(3);
          std::cout << gflops / timer.getAverageTime() << " ";
          std::cout << gbs / timer.getAverageTime() << " ";
          std::cout << std::setprecision(6);
          std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " ";
          std::cout << timer.getCoefficientOfVariation() <<  std::endl;
				}
			}
		}
	}

	std::cout << std::endl;

	return 0;
}

