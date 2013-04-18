/*
 * Copyright (C) 2012
 * Alessio Sclocco <a.sclocco@vu.nl>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <vector>
#include <iostream>
#include <iomanip>

using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

#include <CLData.hpp>
#include <BeamFormer.hpp>
#include <InitializeOpenCL.hpp>
#include <Exceptions.hpp>

using isa::OpenCL::CLData;
using LOFAR::RTCP::BeamFormer;
using isa::OpenCL::initializeOpenCL;
using isa::Exceptions::OpenCLError;

const unsigned int oclPlatformID = 0;
const unsigned int oclDeviceID = 0;
const unsigned int nrIterations = 10;

const unsigned int nrChannels = 256;
const unsigned int nrSamplesPerSecond = 768;
const unsigned int nrPolarizations = 2;

const unsigned int nrStations = 512;
const unsigned int nrBeams = 32;


int main(int argc, char *argv[]) {
	CLData< float > *input = 0;
	CLData< float > *output = 0;
	CLData< float > *weights = 0;
	BeamFormer< float > *beamFormer = 0;

	vector< cl::Platform > *oclPlatforms = new vector< cl::Platform >();
	cl::Context *oclContext = new cl::Context();
	vector< cl::Device > *oclDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > *oclQueues = new vector< vector< cl::CommandQueue > >();
	
	try {
		initializeOpenCL(oclPlatformID, 1, oclPlatforms, oclContext, oclDevices, oclQueues);
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	cout << fixed << setprecision(3);

	for ( unsigned int stations = 2; stations <= nrStations; stations *= 2 ) {
		try {
			input = new CLData< float >("Input", true);
			input->allocateHostData(stations * nrChannels * (nrSamplesPerSecond | 2) * nrPolarizations * 2);
			input->setCLContext(oclContext);
			input->setCLQueue(&(oclQueues->at(oclDeviceID)[0]));
			input->setDeviceReadOnly();
			input->allocateDeviceData();
		} catch ( OpenCLError err ) {
			cerr << err.what() << endl;
			continue;
		}
		
		for ( unsigned int beams = 1; beams <= nrBeams; beams++ ) {
			try {
				output = new CLData< float >("Output", true);
				output->allocateHostData(nrBeams * nrChannels * (nrSamplesPerSecond | 2) * nrPolarizations * 2);
				output->setCLContext(oclContext);
				output->setCLQueue(&(oclQueues->at(oclDeviceID)[0]));
				output->setDeviceWriteOnly();
				output->allocateDeviceData();
				
				weights = new CLData< float >("Weights", true);
				weights->allocateHostData(nrChannels * nrStations * nrBeams * 2);
				weights->setCLContext(oclContext);
				weights->setCLQueue(&(oclQueues->at(oclDeviceID)[0]));
				weights->setDeviceReadOnly();
				weights->allocateDeviceData();
			} catch ( OpenCLError err ) {
				cerr << err.what() << endl;
				continue;
			}
			
			beamFormer = new BeamFormer< float >("BeamFormer", "float");
			beamFormer->setBeamsBlock(beams);
			beamFormer->setNrThreadsPerBlock(nrSamplesPerSecond / 3);
			beamFormer->setNrStations(stations);
			beamFormer->setNrPencilBeams(beams);
			beamFormer->setNrChannels(nrChannels);
			beamFormer->setNrSamplesPerSecond(nrSamplesPerSecond);
			beamFormer->setAveragingFactor(1.0f / stations);

			try {
				beamFormer->bindOpenCL(oclContext, &(oclDevices->at(oclDeviceID)), &(oclQueues->at(oclDeviceID)[0]));
				beamFormer->generateCode();
				for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
					input->copyHostToDevice(true);
					weights->copyHostToDevice();
					(*beamFormer)(input, output, weights);
					output->copyDeviceToHost();
				}
			} catch ( OpenCLError err ) {
				cerr << err.what() << endl;
				continue;
			}

			cout << stations << " " << beams << " " << beamFormer->getGFLOP() / (beamFormer->getTimer()).getAverageTime() << endl;
			
			delete weights;
			delete output;
			delete beamFormer;
		}
		cout << endl << endl;
		delete input;
	}
	
	cout << endl;
	return 0;
}

