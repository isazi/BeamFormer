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

#include <GPUData.hpp>
#include <BeamFormer.hpp>
#include <InitializeOpenCL.hpp>
#include <Exceptions.hpp>

using isa::OpenCL::GPUData;
using LOFAR::RTCP::BeamFormer;
using isa::OpenCL::initializeOpenCL;
using isa::Exceptions::OpenCLError;

const unsigned int oclPlatformID = 0;
const unsigned int oclDeviceID = 0;
const unsigned int nrIterations = 10;
// NVIDIA GTX580
const float memoryBandwidth = 192.384f;
const float peakGFLOPS = 1581.1f;

const unsigned int nrChannels = 256;
const unsigned int nrSamplesPerSecond = 768;
const unsigned int nrStations = 64;
const unsigned int nrPolarizations = 2;
const unsigned int nrBeams = 160;
const unsigned int beamsBlock = 10;


int main(int argc, char *argv[]) {
	GPUData< float > *input = 0;
	GPUData< float > *output = 0;
	GPUData< float > *weights = 0;
	BeamFormer< float > *beamFormer = 0;

	cout << endl << "LOFAR Beam Former" << endl << endl;

	vector< cl::Platform > *oclPlatforms = new vector< cl::Platform >();
	cl::Context *oclContext = new cl::Context();
	vector< cl::Device > *oclDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > *oclQueues = new vector< vector< cl::CommandQueue > >();
	try {
		input = new GPUData< float >("Input", true);
		input->allocateHostData(nrStations * nrChannels * (nrSamplesPerSecond + (nrSamplesPerSecond & 0x00000003)) * nrPolarizations * 2);
		initializeOpenCL(oclPlatformID, 1, oclPlatforms, oclContext, oclDevices, oclQueues);
		input->setCLContext(oclContext);
		input->setCLQueue(&(oclQueues->at(oclDeviceID)[0]));
		input->allocateDeviceData();
		
		output = new GPUData< float >("Output", true);
		output->allocateHostData(nrBeams * nrChannels * (nrSamplesPerSecond + (nrSamplesPerSecond & 0x00000003)) * nrPolarizations * 2);
		output->setCLContext(oclContext);
		output->setCLQueue(&(oclQueues->at(oclDeviceID)[0]));
		output->allocateDeviceData();
		
		weights = new GPUData< float >("Weights", true);
		weights->allocateHostData(nrChannels * nrStations * nrBeams * 2);
		weights->setCLContext(oclContext);
		weights->setCLQueue(&(oclQueues->at(oclDeviceID)[0]));
		weights->allocateDeviceData();
	}
	catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	beamFormer = new BeamFormer< float >("BeamFormer", "float");
	beamFormer->setBeamsBlock(beamsBlock);
	beamFormer->setNrThreadsPerBlock(nrSamplesPerSecond / 3);
	beamFormer->setNrStations(nrStations);
	beamFormer->setNrPencilBeams(nrBeams);
	beamFormer->setNrChannels(nrChannels);
	beamFormer->setNrSamplesPerSecond(nrSamplesPerSecond);
	beamFormer->setAveragingFactor(1.0f / nrStations);

	try {
		beamFormer->bindOpenCL(oclContext, &(oclDevices->at(oclDeviceID)), &(oclQueues->at(oclDeviceID)[0]));
		beamFormer->generateCode();

		for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
			input->copyHostToDevice(true);
			weights->copyHostToDevice();
			(*beamFormer)(input, output, weights);
			output->copyDeviceToHost();
		}
	}
	catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	cout << fixed;
	cout << setprecision(6);
	cout << "Time \t\t" << (beamFormer->getTimer()).getAverageTime() << endl;
	cout << "Time (t) \t" << (beamFormer->getTimer()).getAverageTime() + (input->getH2DTimer()).getAverageTime() + (weights->getH2DTimer()).getAverageTime() + (output->getD2HTimer()).getAverageTime() << endl;
	cout << endl;
	cout << setprecision(3);
	cout << "GFLOP/s \t" << beamFormer->getGFLOP() / (beamFormer->getTimer()).getAverageTime() << endl;
	cout << "GFLOP/s (t) \t" << beamFormer->getGFLOP() / ((beamFormer->getTimer()).getAverageTime() + (input->getH2DTimer()).getAverageTime() + (weights->getH2DTimer()).getAverageTime() + (output->getD2HTimer()).getAverageTime()) << endl;
	cout << "GB/s \t\t" << beamFormer->getGB() / (beamFormer->getTimer()).getAverageTime() << endl;
	cout << endl;
	cout << "AI \t\t" << beamFormer->getArithmeticIntensity() << endl;
	cout << "Roofline \t" << beamFormer->getArithmeticIntensity() * memoryBandwidth << endl;
	cout << "Difference \t" << (beamFormer->getGFLOP() / (beamFormer->getTimer()).getAverageTime()) / (beamFormer->getArithmeticIntensity() * memoryBandwidth) << endl;
	cout << endl;
	cout << setprecision(1);
	cout << "Peak \t\t" << ((beamFormer->getGFLOP() / (beamFormer->getTimer()).getAverageTime()) / peakGFLOPS) * 100 << "%" << endl;

	cout << endl;
	return 0;
}

