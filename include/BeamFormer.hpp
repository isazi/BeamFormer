//
// Copyright (C) 2011
// Alessio Sclocco <a.sclocco@vu.nl>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#include <string>
using std::string;
#include <vector>
using std::vector;

#include <Kernel.hpp>
using isa::OpenCL::Kernel;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
#include <utils.hpp>
using isa::utils::replace;
using isa::utils::toString;
using isa::utils::toStringValue;
using isa::utils::giga;
#include <Observation.hpp>
using AstroData::Observation;


#ifndef BEAM_FORMER_HPP
#define BEAM_FORMER_HPP

namespace LOFAR {
namespace RTCP {


template< typename T > class BeamFormer : public Kernel< T > {
public:
	BeamFormer(string name, string dataType);

	void generateCode() throw (OpenCLError);
	void operator()(CLData< T > * input, CLData< T > * output, CLData< float > * weights) throw (OpenCLError);

	inline void setBeamsBlock(unsigned int block);
	inline void setNrSamplesPerBlock(unsigned int threads);

	inline void setObservation(Observation< T > * obs);
	inline void setAveragingFactor(float factor);
	inline void setStokesI();
	inline void setStokesIQUV();
	inline void setNoStokes();

private:
	cl::NDRange	globalSize;
	cl::NDRange	localSize;

	unsigned int beamsBlock;
	unsigned int nrSamplesPerBlock;

	Observation< T > * observation;
	float averagingFactor;
	bool stokesI;
	bool stokesIQUV;
};


template< typename T > BeamFormer< T >::BeamFormer(string name, string dataType) : Kernel< T >(name, dataType), globalSize(cl::NDRange(1, 1, 1)), localSize(cl::NDRange(1, 1, 1)), beamsBlock(0), nrSamplesPerBlock(0), observation(0), averagingFactor(0), stokesI(false), stokesIQUV(false) {}

template< typename T > void BeamFormer< T >::generateCode() throw (OpenCLError) {
	long long unsigned int ops = 0;
	long long unsigned int memOps = 0;

	if ( stokesI ) {
		ops = (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * observation->getNrBeams() * observation->getNrStations() * 16) + (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * observation->getNrBeams() * 11);
		memOps = (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * (observation->getNrBeams() / beamsBlock) * observation->getNrStations() * 16) + (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * observation->getNrBeams() * observation->getNrStations() * 8) + (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * observation->getNrBeams() * 4);
	}
	else if ( stokesIQUV ) {
		ops = (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * observation->getNrBeams() * observation->getNrStations() * 16) + (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * observation->getNrBeams() * 24);
		memOps = (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * (observation->getNrBeams() / beamsBlock) * observation->getNrStations() * 16) + (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * observation->getNrBeams() * observation->getNrStations() * 8) + (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * observation->getNrBeams() * 16);
	}
	else {
		ops = (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * observation->getNrBeams() * observation->getNrStations() * 16) + (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * observation->getNrBeams() * 4);
		memOps = (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * (observation->getNrBeams() / beamsBlock) * observation->getNrStations() * 16) + (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * observation->getNrBeams() * observation->getNrStations() * 8) + (static_cast< long long unsigned int >(observation->getNrChannels()) * observation->getNrSamplesPerSecond() * observation->getNrBeams() * 16);
	}
	this->arInt = ops / static_cast< double >(memOps);
	this->gflop = giga(ops);
	this->gb = giga(memOps);

	// Begin kernel's template
	if ( this->code != 0 ) {
		delete this->code;
	}
	this->code = new string();
	string beamsBlock_s = toStringValue< unsigned int >(beamsBlock);
	string nrStations_s = toStringValue< unsigned int >(observation->getNrStations());
	string nrBeams_s = toStringValue< unsigned int >(observation->getNrBeams());
	string nrSamplesPerSecond_s = toStringValue< unsigned int >(observation->getNrSamplesPerSecond());
	string nrSamplesPerPaddedSecond_s = toStringValue< unsigned int >(observation->getNrSamplesPerPaddedSecond());
	string nrChannels_s = toStringValue< unsigned int >(observation->getNrChannels());
	string averagingFactor_s = toStringValue< float >(averagingFactor);
	if ( averagingFactor_s.find('.') == string::npos ) {
		averagingFactor_s.append(".0f");
	}
	else {
		averagingFactor_s.append("f");
	}

	if ( stokesI ) {
		*(this->code) += "__kernel void " + this->name + "(__global " + this->dataType + "4 * samples, __global " + this->dataType + " * results, __global float2 * weights) {\n";
	}
	else {
		*(this->code) += "__kernel void " + this->name + "(__global " + this->dataType + "4 * samples, __global " + this->dataType + "4 * results, __global float2 * weights) {\n";
	}
	*(this->code) += "const unsigned int sample = (get_group_id(0) * get_local_size(0)) + get_local_id(0);\n"
	"const unsigned	int channel = get_group_id(1);\n"
	"unsigned int item = 0;\n"
	"\n"
	+ this->dataType + "4 cSample = (" + this->dataType + "4)(0.0f);\n"
	"float2 weight = (float2)(0.0f);\n"
	"\n"
	"for ( unsigned int beam = 0; beam < " + nrBeams_s + "; beam += " + beamsBlock_s + " ) {\n"
	"<%DEFINITIONS%>"
	"\n"
	"for ( unsigned int station = 0; station < " + nrStations_s + "; station++ ) {\n"
	"item = (channel * " + nrStations_s + " * " + nrBeams_s + ") + (station * " + nrBeams_s + ") + beam;\n"
	"cSample = samples[(channel * " + nrStations_s + " * " + nrSamplesPerPaddedSecond_s + ") + (station * " + nrSamplesPerPaddedSecond_s + ") + sample];\n"
	"\n"
	"<%BEAMS%>"
	"}\n"
	"<%AVERAGE%>"
	"item = (channel * " + nrSamplesPerSecond_s + ") + sample;\n"
	"<%STORE%>"
	"}\n"
	"}\n";
	
	string definitionsTemplate = Kernel< T >::getDataType() + "4 beam<%NUM%> = (" + Kernel< T >::getDataType() + "4)(0.0f);\n";
	
	string beamsTemplate = "weight = weights[item++];\n"
	"beam<%NUM%>.x += (cSample.x * weight.x) - (cSample.y * weight.y);\n"
	"beam<%NUM%>.y += (cSample.x * weight.y) + (cSample.y * weight.x);\n"
	"beam<%NUM%>.z += (cSample.z * weight.x) - (cSample.w * weight.y);\n"
	"beam<%NUM%>.w += (cSample.z * weight.y) + (cSample.w * weight.x);\n";
	
	string averageTemplate = "beam<%NUM%> *= " + averagingFactor_s + ";\n";

	string storeTemplate;
	if ( stokesI ) {
		storeTemplate = "results[((beam + <%NUM%>) * " + nrChannels_s + " * " + nrSamplesPerPaddedSecond_s + ") + item] = ((beam<%NUM%>.x * beam<%NUM%>.x) + (beam<%NUM%>.y * beam<%NUM%>.y)) + ((beam<%NUM%>.z * beam<%NUM%>.z) + (beam<%NUM%>.w * beam<%NUM%>.w));\n";
	}
	else if ( stokesIQUV ) {
		storeTemplate = "cSample.x = ((beam<%NUM%>.x * beam<%NUM%>.x) + (beam<%NUM%>.y * beam<%NUM%>.y)) + ((beam<%NUM%>.z * beam<%NUM%>.z) + (beam<%NUM%>.w * beam<%NUM%>.w));\n"
		"cSample.y = ((beam<%NUM%>.x * beam<%NUM%>.x) + (beam<%NUM%>.y * beam<%NUM%>.y)) - ((beam<%NUM%>.z * beam<%NUM%>.z) + (beam<%NUM%>.w * beam<%NUM%>.w));\n"
		"cSample.z = 2.0f * ((beam<%NUM%>.x * beam<%NUM%>.z) + (beam<%NUM%>.y * beam<%NUM%>.w));\n"
		"cSample.w = 2.0f * ((beam<%NUM%>.y * beam<%NUM%>.z) - (beam<%NUM%>.x * beam<%NUM%>.w));\n"
		"results[((beam + <%NUM%>) * " + nrChannels_s + " * " + nrSamplesPerPaddedSecond_s + ") + item] = cSample;\n";
	}
	else {
		storeTemplate = "results[((beam + <%NUM%>) * " + nrChannels_s + " * " + nrSamplesPerPaddedSecond_s + ") + item] = beam<%NUM%>;\n";
	}
	// End kernel's template

	string * definitions = new string();
	string * beams = new string();
	string * averages = new string();
	string * stores = new string();
	for ( unsigned int beam = 0; beam < beamsBlock; beam++ ) {
		string *beam_s = toString< unsigned int >(beam);
		string *temp;

		temp = replace(&definitionsTemplate, "<%NUM%>", *beam_s);
		definitions->append(*temp);
		delete temp;
		temp = replace(&beamsTemplate, "<%NUM%>",*beam_s);
		beams->append(*temp);
		delete temp;
		temp = replace(&averageTemplate, "<%NUM%>", *beam_s);
		averages->append(*temp);
		delete temp;
		temp = replace(&storeTemplate, "<%NUM%>", *beam_s);
		stores->append(*temp);
		delete temp;
		
		delete beam_s;
	}
	this->code = replace(this->code, "<%DEFINITIONS%>", *definitions, true);
	this->code = replace(this->code, "<%BEAMS%>", *beams, true);
	this->code = replace(this->code, "<%AVERAGE%>", *averages, true);
	this->code = replace(this->code, "<%STORE%>", *stores, true);
	delete definitions;
	delete beams;
	delete averages;
	delete stores;

	globalSize = cl::NDRange(observation->getNrSamplesPerPaddedSecond(), observation->getNrChannels());
	localSize = cl::NDRange(nrSamplesPerBlock, 1);

	this->compile();
}


template< typename T > void BeamFormer< T >::operator()(CLData< T > *input, CLData< T > *output, CLData< float > *weights) throw (OpenCLError) {

	this->setArgument(0, *(input->getDeviceData()));
	this->setArgument(1, *(output->getDeviceData()));
	this->setArgument(2, *(weights->getDeviceData()));

	this->run(globalSize, localSize);
}

template< typename T > inline void BeamFormer< T >::setBeamsBlock(unsigned int block) {
	beamsBlock = block;
}

template< typename T > inline void BeamFormer< T >::setNrSamplesPerBlock(unsigned int threads) {
	nrSamplesPerBlock = threads;
}

template< typename T > inline void BeamFormer< T >::setObservation(Observation< T > *obs) {
	observation = obs;
}

template< typename T > inline void BeamFormer< T >::setAveragingFactor(float factor) {
	averagingFactor = factor;
}

template< typename T > inline void BeamFormer< T >::setStokesI() {
	stokesI = true;
	stokesIQUV = false;
}

template< typename T > inline void BeamFormer< T >::setStokesIQUV() {
	stokesI = false;
	stokesIQUV = true;
}

template< typename T > inline void BeamFormer< T >::setNoStokes() {
	stokesI = false;
	stokesIQUV = false;
}

} // RTCP
} // LOFAR

#endif // BEAM_FORMER_HPP

