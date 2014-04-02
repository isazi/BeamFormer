// Copyright 2012 Alessio Sclocco <a.sclocco@vu.nl>
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

#include <vector>
using std::vector;
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
#include <iomanip>
using std::setprecision;
#include <cmath>

#include <ArgumentList.hpp>
using isa::utils::ArgumentList;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <BeamFormer.hpp>
using LOFAR::RTCP::BeamFormer;
#include <InitializeOpenCL.hpp>
using isa::OpenCL::initializeOpenCL;
#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
#include <Observation.hpp>
using AstroData::Observation;

const unsigned int nrPolarizations = 2;


int main(int argc, char *argv[]) {
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int minThreads = 0;
	unsigned int maxThreadsPerBlock = 0;
	unsigned int maxItemsPerThread = 0;
	Observation< float > observation("DedispersionTuning", "float");
	CLData< float > * input = 0;
	CLData< float > * output = 0;
	CLData< float > * weights = 0;

	try {
		ArgumentList args(argc, argv);

		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
		minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
		maxThreadsPerBlock = args.getSwitchArgument< unsigned int >("-max_threads");
		maxItemsPerThread = args.getSwitchArgument< unsigned int >("-max_items");
		observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
		observation.setNrChannels(args.getSwitchArgument< unsigned int >("-channels"));
		observation.setNrStations(args.getSwitchArgument< unsigned int >("-stations"));
		observation.setNrBeams(args.getSwitchArgument< unsigned int >("-beams"));
	} catch ( EmptyCommandLine err ) {
		cerr << argv[0] << " -iterations ... -opencl_platform ... -opencl_device ... -padding ... -min_threads ... -max_threads ... -max_items ... -samples ... -channels ... -stations ... -beams ..." << endl;
		return 1;
	} catch ( exception &err ) {
		cerr << err.what() << endl;
		return 1;
	}

	vector< cl::Platform > *oclPlatforms = new vector< cl::Platform >();
	cl::Context *oclContext = new cl::Context();
	vector< cl::Device > *oclDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > *oclQueues = new vector< vector< cl::CommandQueue > >();

	initializeOpenCL(clPlatformID, 1, oclPlatforms, oclContext, oclDevices, oclQueues);

	cout << fixed << endl;
	cout << "# nrStations nrBeams nrSamplesPerSecond nrChannels nrSamplesPerBlock beamsBlock GFLOP/s std.dev. time std.dev." << endl << endl;

	input = new CLData< float >("Input", true);
	input->allocateHostData(observation.getNrStations() * observation.getNrChannels() * observation.getNrSamplesPerPaddedSecond() * nrPolarizations * 2);
	input->setCLContext(oclContext);
	input->setCLQueue(&(oclQueues->at(clDeviceID)[0]));
	input->setDeviceReadOnly();
	input->allocateDeviceData();

	output = new CLData< float >("Output", true);
	output->allocateHostData(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerPaddedSecond() * nrPolarizations * 2);
	output->setCLContext(oclContext);
	output->setCLQueue(&(oclQueues->at(clDeviceID)[0]));
	output->setDeviceWriteOnly();
	output->allocateDeviceData();

	weights = new CLData< float >("Weights", true);
	weights->allocateHostData(observation.getNrChannels() * observation.getNrStations() * observation.getNrBeams() * 2);
	weights->setCLContext(oclContext);
	weights->setCLQueue(&(oclQueues->at(clDeviceID)[0]));
	weights->setDeviceReadOnly();
	weights->allocateDeviceData();

	// Find the parameters
	vector< unsigned int > samplesPerBlock;
	for ( unsigned int samples = minThreads; samples <= maxThreadsPerBlock; samples += minThreads ) {
		if ( (observation.getNrSamplesPerPaddedSecond() % samples) == 0 ) {
			samplesPerBlock.push_back(samples);
		}
	}
	vector< unsigned int > beamsBlock;
	for ( unsigned int beams = 1; beams <= maxItemsPerThread; beams++ ) {
		if ( (observation.getNrBeams() % beams) == 0 ) {
			beamsBlock.push_back(beams);
		}
	}

	for ( vector< unsigned int >::iterator samples = samplesPerBlock.begin(); samples != samplesPerBlock.end(); samples++ ) {
		for ( vector< unsigned int >::iterator beams = beamsBlock.begin(); beams != beamsBlock.end(); beams++ ) {
			BeamFormer< float > * beamFormer = new BeamFormer< float >("BeamFormer", "float");

			try {
				beamFormer->bindOpenCL(oclContext, &(oclDevices->at(clDeviceID)), &(oclQueues->at(clDeviceID)[0]));
				beamFormer->setBeamsBlock(*beams);
				beamFormer->setNrSamplesPerBlock(*samples);
				beamFormer->setObservation(&observation);
				beamFormer->setAveragingFactor(1.0f / observation.getNrStations());

				// Warm-up
				(*beamFormer)(input, output, weights);
				beamFormer->getTimer().reset();
				beamFormer->resetStats();

				for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
					(*beamFormer)(input, output, weights);
				}

				cout << observation.getNrStations() << " " << observation.getNrBeams() << " " << observation.getNrSamplesPerSecond() << " " << observation.getNrChannels() << " " << *samples << " " << *beams << " " << setprecision(3) << beamFormer->getGFLOPs() << " " << beamFormer->getGFLOPsErr() << " " << setprecision(6) << beamFormer->getTimer().getAverageTime() << " " << beamFormer->getTimer().getStdDev() << endl;
			} catch ( OpenCLError err ) {
				cerr << err.what() << endl;
				continue;
			}
		}
	}

	cout << endl;
	return 0;
}

