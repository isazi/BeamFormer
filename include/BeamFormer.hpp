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

#include <string>

#include <utils.hpp>
#include <Observation.hpp>


#ifndef BEAM_FORMER_HPP
#define BEAM_FORMER_HPP

namespace RadioAstronomy {

// OpenCL beam forming algorithm
std::string * getBeamFormerOpenCL(const unsigned int nrSamplesPerBlock, const unsigned int nrBeamsPerBlock, const unsigned int nrSamplesPerThread, const unsigned int nrBeamsPerThread, const std::string & dataType, const AstroData::Observation & observation);

// Implementations
std::string * getBeamFormerOpenCL(const unsigned int nrSamplesPerBlock, const unsigned int nrBeamsPerBlock, const unsigned int nrSamplesPerThread, const unsigned int nrBeamsPerThread, const std::string & dataType, const AstroData::Observation & observation) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void beamFormer(__global const " + dataType + "4 * restrict const samples, __global " + dataType + "4 * restrict const output, __global const float2 * restrict const weights) {\n"
    "const unsigned int channel = get_group_id(2);\n"
    "const unsigned int beam = (get_group_id(1) * " + isa::utils::toString(nrBeamsPerBlock * nrBeamsPerThread) + ") + (get_local_id(1) * " + isa::utils::toString(nrBeamsPerThread) + ");\n"
    "unsigned int itemGlobal = 0;\n"
    "unsigned int itemLocal = 0;\n"
    "<%DEF_SAMPLES%>"
    "<%DEF_SUMS%>"
    + dataType + "4 sample = (" + dataType + "4)(0);\n"
    "__local float2 localWeights[" + isa::utils::toString(nrBeamsPerBlock * nrBeamsPerThread) + "];\n"
    "float2 weight = (float2)(0);\n"
    "\n"
    "for ( unsigned int station = 0; station < " + isa::utils::toString(observation.getNrStations()) + "; station++ ) {\n"
    "<%LOAD_COMPUTE%>"
    "}\n"
    "<%AVERAGE%>"
    "<%STORE%>"
    "}\n";
  std::string defSamplesTemplate = "const unsigned int sample<%SNUM%> = (get_group_id(0) * " + isa::utils::toString(nrSamplesPerBlock * nrSamplesPerThread) + ") + get_local_id(0) + <%OFFSET%>;\n";
  std::string defSumsTemplate = dataType + "4 beam<%BNUM%>s<%SNUM%> = (" + dataType + "4)(0);\n";
  std::string loadComputeTemplate = "itemGlobal = (channel * " + isa::utils::toString(observation.getNrStations() * observation.getNrPaddedBeams()) + ") + (station * " + isa::utils::toString(observation.getNrPaddedBeams()) + ") + (get_local_id(1) * " + isa::utils::toString(nrSamplesPerBlock) + ") + get_local_id(0);\n"
    "itemLocal = get_local_id(0);\n"
    "while ( itemLocal < " + isa::utils::toString(nrBeamsPerBlock * nrBeamsPerThread) + ") {\n"
    "localWeights[itemLocal] = weights[itemGlobal];\n"
    "itemLocal += " + isa::utils::toString(nrSamplesPerBlock * nrBeamsPerBlock) + ";\n"
    "itemGlobal += " + isa::utils::toString(nrSamplesPerBlock * nrBeamsPerBlock) + ";\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "sample = samples[(channel * " + isa::utils::toString(observation.getNrStations() * observation.getNrSamplesPerPaddedSecond()) + ") + (station * " + isa::utils::toString(observation.getNrSamplesPerPaddedSecond()) + ") + sample<%SNUM%>];\n"
    "<%SUMS%>";
  std::string sumsTemplate = "weight = localWeights[(get_local_id(1) * " + isa::utils::toString(nrBeamsPerThread) + ") + <%BNUM%>];\n"
    "beam<%BNUM%>s<%SNUM%>.x += (sample.x * weight.x) - (sample.y * weight.y);\n"
    "beam<%BNUM%>s<%SNUM%>.y += (sample.x * weight.y) + (sample.y * weight.x);\n"
    "beam<%BNUM%>s<%SNUM%>.z += (sample.z * weight.x) - (sample.w * weight.y);\n"
    "beam<%BNUM%>s<%SNUM%>.w += (sample.z * weight.y) + (sample.w * weight.x);\n";
  std::string averageTemplate = "beam<%BNUM%>s<%SNUM%> *= " + isa::utils::toString(1.0f / observation.getNrStations()) + "f;\n";
  std::string storeTemplate = "output[((beam + <%BNUM%>) * " + isa::utils::toString(observation.getNrChannels() * observation.getNrSamplesPerPaddedSecond()) + ") + (channel * " + isa::utils::toString(observation.getNrSamplesPerPaddedSecond()) + ") + sample<%SNUM%>] = beam<%BNUM%>s<%SNUM%>;\n";
  // End kernel's template

  std::string * defSamples_s = new std::string();
  std::string * defSums_s = new std::string();
  std::string * loadCompute_s = new std::string();
  std::string * average_s = new std::string();
  std::string * store_s = new std::string();

  for ( unsigned int sample = 0; sample < nrSamplesPerThread; sample++ ) {
    std::string sample_s = isa::utils::toString(sample);
    std::string offset_s = isa::utils::toString(sample * nrSamplesPerBlock);
    std::string * sums_s = new std::string();
    std::string * temp_s = 0;

    temp_s = isa::utils::replace(&defSamplesTemplate, "<%SNUM%>", sample_s);
    temp_s = isa::utils::replace(temp_s, "<%OFFSET%>", offset_s, true);
    defSamples_s->append(*temp_s);
    delete temp_s;

    for ( unsigned int beam = 0; beam < nrBeamsPerThread; beam++ ) {
      std::string beam_s = isa::utils::toString(beam);
      std::string * temp_s = 0;

      temp_s = isa::utils::replace(&defSumsTemplate, "<%BNUM%>", beam_s);
      defSums_s->append(*temp_s);
      delete temp_s;
      temp_s = isa::utils::replace(&sumsTemplate, "<%BNUM%>", beam_s);
      sums_s->append(*temp_s);
      delete temp_s;
      temp_s = isa::utils::replace(&averageTemplate, "<%BNUM%>", beam_s);
      average_s->append(*temp_s);
      delete temp_s;
      temp_s = isa::utils::replace(&storeTemplate, "<%BNUM%>", beam_s);
      store_s->append(*temp_s);
      delete temp_s;
    }
    defSums_s = isa::utils::replace(defSums_s, "<%SNUM%>", sample_s, true);
    temp_s = isa::utils::replace(&loadComputeTemplate, "<%SNUM%>", sample_s);
    temp_s = isa::utils::replace(temp_s, "<%SUMS%>", *sums_s, true);
    temp_s = isa::utils::replace(temp_s, "<%SNUM%>", sample_s, true);
    loadCompute_s->append(*temp_s);
    delete temp_s;
    average_s = isa::utils::replace(average_s, "<%SNUM%>", sample_s, true);
    store_s = isa::utils::replace(store_s, "<%SNUM%>", sample_s, true);
  }

  code = isa::utils::replace(code, "<%DEF_SAMPLES%>", *defSamples_s, true);
  code = isa::utils::replace(code, "<%DEF_SUMS%>", *defSums_s, true);
  code = isa::utils::replace(code, "<%LOAD_COMPUTE%>", *loadCompute_s, true);
  code = isa::utils::replace(code, "<%AVERAGE%>", *average_s, true);
  code = isa::utils::replace(code, "<%STORE%>", *store_s, true);

  return code;
}

} // RadioAstronomy

#endif // BEAM_FORMER_HPP

