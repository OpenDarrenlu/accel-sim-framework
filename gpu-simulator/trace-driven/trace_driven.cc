// Copyright (c) 2018-2021, Mahmoud Khairy, Vijay Kandiah, Timothy Rogers, Tor
// M. Aamodt, Nikos Hardavellas
// Northwestern University, Purdue University, The University of British
// Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of Northwestern University, Purdue University,
//    The University of British Columbia nor the names of their contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <bits/stdc++.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../ISA_Def/accelwattch_component_mapping.h"
#include "../ISA_Def/ampere_opcode.h"
#include "../ISA_Def/kepler_opcode.h"
#include "../ISA_Def/pascal_opcode.h"
#include "../ISA_Def/trace_opcode.h"
#include "../ISA_Def/turing_opcode.h"
#include "../ISA_Def/volta_opcode.h"
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_parser.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu_context.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "trace_driven.h"

const trace_warp_inst_t *trace_shd_warp_t::get_next_trace_inst() {
  if (trace_pc < warp_traces.size()) {
    trace_warp_inst_t *new_inst =
        new trace_warp_inst_t(get_shader()->get_config());
    new_inst->parse_from_trace_struct(
        warp_traces[trace_pc], m_kernel_info->OpcodeMap,
        m_kernel_info->m_tconfig, m_kernel_info->m_kernel_trace_info);
    trace_pc++;
    return new_inst;
  } else
    return NULL;
}

void trace_shd_warp_t::clear() {
  trace_pc = 0;
  warp_traces.clear();
}

// functional_done
bool trace_shd_warp_t::trace_done() { return trace_pc == (warp_traces.size()); }

address_type trace_shd_warp_t::get_start_trace_pc() {
  assert(warp_traces.size() > 0);
  return warp_traces[0].m_pc;
}

address_type trace_shd_warp_t::get_pc() {
  assert(warp_traces.size() > 0);
  assert(trace_pc < warp_traces.size());
  return warp_traces[trace_pc].m_pc;
}

trace_kernel_info_t::trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                                         trace_function_info *m_function_info,
                                         trace_parser *parser,
                                         class trace_config *config,
                                         kernel_trace_t *kernel_trace_info)
    : kernel_info_t(gridDim, blockDim, m_function_info,
                    kernel_trace_info->cuda_stream_id) {
  m_parser = parser;
  m_tconfig = config;
  m_kernel_trace_info = kernel_trace_info;
  m_was_launched = false;

  // resolve the binary version
  if (kernel_trace_info->binary_verion == AMPERE_RTX_BINART_VERSION ||
      kernel_trace_info->binary_verion == AMPERE_A100_BINART_VERSION)
    OpcodeMap = &Ampere_OpcodeMap;
  else if (kernel_trace_info->binary_verion == VOLTA_BINART_VERSION)
    OpcodeMap = &Volta_OpcodeMap;
  else if (kernel_trace_info->binary_verion == PASCAL_TITANX_BINART_VERSION ||
           kernel_trace_info->binary_verion == PASCAL_P100_BINART_VERSION)
    OpcodeMap = &Pascal_OpcodeMap;
  else if (kernel_trace_info->binary_verion == KEPLER_BINART_VERSION)
    OpcodeMap = &Kepler_OpcodeMap;
  else if (kernel_trace_info->binary_verion == TURING_BINART_VERSION)
    OpcodeMap = &Turing_OpcodeMap;
  else {
    printf("unsupported binary version: %d\n",
           kernel_trace_info->binary_verion);
    fflush(stdout);
    exit(0);
  }
}

void trace_kernel_info_t::get_next_threadblock_traces(
    std::vector<std::vector<inst_trace_t> *> threadblock_traces) {
  m_parser->get_next_threadblock_traces(
      threadblock_traces, m_kernel_trace_info->trace_verion,
      m_kernel_trace_info->enable_lineinfo, m_kernel_trace_info->pipeReader);
}

types_of_operands get_oprnd_type(op_type op, special_ops sp_op) {
  switch (op) {
    case SP_OP:
    case SFU_OP:
    case SPECIALIZED_UNIT_2_OP:
    case SPECIALIZED_UNIT_3_OP:
    case DP_OP:
    case LOAD_OP:
    case STORE_OP:
      return FP_OP;
    case INTP_OP:
    case SPECIALIZED_UNIT_4_OP:
      return INT_OP;
    case ALU_OP:
      if ((sp_op == FP__OP) || (sp_op == TEX__OP) || (sp_op == OTHER_OP))
        return FP_OP;
      else if (sp_op == INT__OP)
        return INT_OP;
    default:
      return UN_OP;
  }
}

bool trace_warp_inst_t::parse_from_trace_struct(
    const inst_trace_t &trace,
    const std::unordered_map<std::string, OpcodeChar> *OpcodeMap,
    const class trace_config *tconfig,
    const class kernel_trace_t *kernel_trace_info) {
  // fill the inst_t and warp_inst_t params

  // fill active mask
  active_mask_t active_mask = trace.mask;
  set_active(active_mask);

  // fill and initialize common params
  m_decoded = true;
  pc = (address_type)trace.m_pc;

  isize =
      16;  // starting from MAXWELL isize=16 bytes (including the control bytes)
  for (unsigned i = 0; i < MAX_OUTPUT_VALUES; i++) {
    out[i] = 0;
  }
  for (unsigned i = 0; i < MAX_INPUT_VALUES; i++) {
    in[i] = 0;
  }

  is_vectorin = 0;
  is_vectorout = 0;
  pred = 0;
  ar1 = 0;
  ar2 = 0;
  memory_op = no_memory_op;
  data_size = 0;
  op = ALU_OP;
  sp_op = OTHER_OP;
  mem_op = NOT_TEX;
  const_cache_operand = 0;
  oprnd_type = UN_OP;

  // get the opcode
  std::vector<std::string> opcode_tokens = trace.get_opcode_tokens();
  std::string opcode1 = opcode_tokens[0];

  std::unordered_map<std::string, OpcodeChar>::const_iterator it =
      OpcodeMap->find(opcode1);
  if (it != OpcodeMap->end()) {
    m_opcode = it->second.opcode;
    op = (op_type)(it->second.opcode_category);
    const std::unordered_map<unsigned, unsigned> *OpcPowerMap = &OpcodePowerMap;
    std::unordered_map<unsigned, unsigned>::const_iterator it2 =
        OpcPowerMap->find(m_opcode);
    if (it2 != OpcPowerMap->end()) sp_op = (special_ops)(it2->second);
    oprnd_type = get_oprnd_type(op, sp_op);
  } else {
    std::cout << "ERROR:  undefined instruction : " << trace.opcode
              << " Opcode: " << opcode1 << std::endl;
    assert(0 && "undefined instruction");
  }
  std::string opcode = trace.opcode;
  if (opcode1 == "MUFU") {  // Differentiate between different MUFU operations
                            // for power model
    if ((opcode == "MUFU.SIN") || (opcode == "MUFU.COS")) sp_op = FP_SIN_OP;
    if ((opcode == "MUFU.EX2") || (opcode == "MUFU.RCP")) sp_op = FP_EXP_OP;
    if (opcode == "MUFU.RSQ") sp_op = FP_SQRT_OP;
    if (opcode == "MUFU.LG2") sp_op = FP_LG_OP;
  }

  if (opcode1 == "IMAD") {  // Differentiate between different IMAD operations
                            // for power model
    if ((opcode == "IMAD.MOV") || (opcode == "IMAD.IADD")) sp_op = INT__OP;
  }

  // fill regs information
  num_regs = trace.reg_srcs_num + trace.reg_dsts_num;
  num_operands = num_regs;
  outcount = trace.reg_dsts_num;
  for (unsigned m = 0; m < trace.reg_dsts_num; ++m) {
    out[m] =
        trace.reg_dest[m] + 1;  // Increment by one because GPGPU-sim starts
                                // from R1, while SASS starts from R0
    arch_reg.dst[m] = trace.reg_dest[m] + 1;
  }

  incount = trace.reg_srcs_num;
  for (unsigned m = 0; m < trace.reg_srcs_num; ++m) {
    in[m] = trace.reg_src[m] + 1;  // Increment by one because GPGPU-sim starts
                                   // from R1, while SASS starts from R0
    arch_reg.src[m] = trace.reg_src[m] + 1;
  }

  // fill latency and initl
  tconfig->set_latency(op, latency, initiation_interval);

  // fill addresses
  if (trace.memadd_info != NULL) {
    data_size = trace.memadd_info->width;
    for (unsigned i = 0; i < warp_size(); ++i)
      set_addr(i, trace.memadd_info->addrs[i]);
  }

  // handle special cases and fill memory space
  switch (m_opcode) {
    case OP_LDC:  // handle Load from Constant
      data_size = 4;
      memory_op = memory_load;
      const_cache_operand = 1;
      space.set_type(const_space);
      cache_op = CACHE_ALL;
      break;
    case OP_LDG:
    // LDGSTS is loading the values needed directly from the global memory to
    // shared memory. Before this feature, the values need to be loaded to
    // registers first, then store to the shared memory.
    case OP_LDGSTS:  // Add for memcpy_async
    case OP_LDL:
      assert(data_size > 0);
      memory_op = memory_load;
      cache_op = CACHE_ALL;
      if (m_opcode == OP_LDL)
        space.set_type(local_space);
      else
        space.set_type(global_space);
      // Add for LDGSTS instruction
      if (m_opcode == OP_LDGSTS) m_is_ldgsts = true;
      // check the cache scope, if its strong GPU, then bypass L1
      if ((trace.check_opcode_contain(opcode_tokens, "STRONG") &&
           trace.check_opcode_contain(opcode_tokens, "GPU")) ||
          trace.check_opcode_contain(opcode_tokens, "BYPASS")) {
        cache_op = CACHE_GLOBAL;
      }
      break;
    case OP_STG:
    case OP_STL:
      assert(data_size > 0);
      memory_op = memory_store;
      cache_op = CACHE_ALL;
      if (m_opcode == OP_STL)
        space.set_type(local_space);
      else
        space.set_type(global_space);
      break;
    case OP_ATOMG:
    case OP_RED:
    case OP_ATOM:
      assert(data_size > 0);
      memory_op = memory_load;
      op = LOAD_OP;
      space.set_type(global_space);
      m_isatomic = true;
      cache_op = CACHE_GLOBAL;  // all the atomics should be done at L2
      break;
    case OP_LDS:
      assert(data_size > 0);
      memory_op = memory_load;
      space.set_type(shared_space);
      break;
    case OP_STS:
      assert(data_size > 0);
      memory_op = memory_store;
      space.set_type(shared_space);
      break;
    case OP_ATOMS:
      assert(data_size > 0);
      m_isatomic = true;
      memory_op = memory_load;
      space.set_type(shared_space);
      break;
    case OP_LDSM:
      assert(data_size > 0);
      space.set_type(shared_space);
      break;
    case OP_ST:
    case OP_LD:
      assert(data_size > 0);
      if (m_opcode == OP_LD)
        memory_op = memory_load;
      else
        memory_op = memory_store;
      // resolve generic loads
      if (kernel_trace_info->shmem_base_addr == 0 ||
          kernel_trace_info->local_base_addr == 0) {
        // shmem and local addresses are not set
        // assume all the mem reqs are shared by default
        space.set_type(shared_space);
      } else {
        // check the first active address
        for (unsigned i = 0; i < warp_size(); ++i)
          if (active_mask.test(i)) {
            if (trace.memadd_info->addrs[i] >=
                    kernel_trace_info->shmem_base_addr &&
                trace.memadd_info->addrs[i] <
                    kernel_trace_info->local_base_addr)
              space.set_type(shared_space);
            else if (trace.memadd_info->addrs[i] >=
                         kernel_trace_info->local_base_addr &&
                     trace.memadd_info->addrs[i] <
                         kernel_trace_info->local_base_addr +
                             LOCAL_MEM_SIZE_MAX) {
              space.set_type(local_space);
              cache_op = CACHE_ALL;
            } else {
              space.set_type(global_space);
              cache_op = CACHE_ALL;
            }
            break;
          }
      }

      break;
    case OP_BAR:
      // TO DO: fill this correctly
      bar_id = 0;
      bar_count = (unsigned)-1;
      bar_type = SYNC;
      // TO DO
      // if bar_type = RED;
      // set bar_type
      // barrier_type bar_type;
      // reduction_type red_type;
      break;
    // LDGDEPBAR is to form a group containing the previous LDGSTS instructions
    // that have not been grouped yet. In the implementation, a group number
    // will be assigned once the instruction is met.
    case OP_LDGDEPBAR:
      m_is_ldgdepbar = true;
      break;
    // DEPBAR is served as a warp-wise barrier that is only effective for LDGSTS
    // instructions. It is associated with a immediate value. The immediate
    // value indicates the last N LDGDEPBAR groups to not wait once the
    // instruction is met. For example, if the immediate value is 1, then the
    // last group is able to proceed even with DEPBAR present; if the immediate
    // value is 0, then all of the groups need to finish before proceed.
    case OP_DEPBAR:
      m_is_depbar = true;
      m_depbar_group_no = trace.imm;
      break;
    case OP_HADD2:
    case OP_HADD2_32I:
    case OP_HFMA2:
    case OP_HFMA2_32I:
    case OP_HMUL2_32I:
    case OP_HSET2:
    case OP_HSETP2:
      initiation_interval =
          initiation_interval / 2;  // FP16 has 2X throughput than FP32
      if (initiation_interval <
          1)  // Make sure initiaion interval never goes below 1
        initiation_interval = 1;
      break;
    case OP_LMMA:
      // LMMA (LUT-based MMA) has its own latency/initiation interval
      // configured via -trace_opcode_latency_initiation_lmma.
      // The opcode format is: LMMA.{M}{N}{K}.{A_dtype}{W_dtype}{Accum_dtype}{O_dtype}
      // e.g., LMMA.16.8.16.F16.I2.F16.F16
      tconfig->get_lmma_latency(latency, initiation_interval);
      m_is_lmma = true;  // Mark this instruction as LMMA for functional sim
      break;
    default:
      break;
  }

  return true;
}

trace_config::trace_config() {}

void trace_config::reg_options(option_parser_t opp) {
  option_parser_register(opp, "-trace", OPT_CSTR, &g_traces_filename,
                         "traces kernel file"
                         "traces kernel file directory",
                         "./traces/kernelslist.g");

  option_parser_register(opp, "-trace_opcode_latency_initiation_int", OPT_CSTR,
                         &trace_opcode_latency_initiation_int,
                         "Opcode latencies and initiation for integers in "
                         "trace driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_sp", OPT_CSTR,
                         &trace_opcode_latency_initiation_sp,
                         "Opcode latencies and initiation for sp in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_dp", OPT_CSTR,
                         &trace_opcode_latency_initiation_dp,
                         "Opcode latencies and initiation for dp in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_sfu", OPT_CSTR,
                         &trace_opcode_latency_initiation_sfu,
                         "Opcode latencies and initiation for sfu in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_tensor",
                         OPT_CSTR, &trace_opcode_latency_initiation_tensor,
                         "Opcode latencies and initiation for tensor in trace "
                         "driven mode <latency,initiation>",
                         "4,1");

  option_parser_register(opp, "-trace_opcode_latency_initiation_lmma",
                         OPT_CSTR, &trace_opcode_latency_initiation_lmma,
                         "Opcode latencies and initiation for LMMA (LUT-based "
                         "MMA) in trace driven mode <latency,initiation>",
                         "8,8");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-trace_opcode_latency_initiation_spec_op_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &trace_opcode_latency_initiation_specialized_op[j],
                           "specialized unit config"
                           " <latency,initiation>",
                           "4,4");
  }
}

void trace_config::parse_config() {
  sscanf(trace_opcode_latency_initiation_int, "%u,%u", &int_latency, &int_init);
  sscanf(trace_opcode_latency_initiation_sp, "%u,%u", &fp_latency, &fp_init);
  sscanf(trace_opcode_latency_initiation_dp, "%u,%u", &dp_latency, &dp_init);
  sscanf(trace_opcode_latency_initiation_sfu, "%u,%u", &sfu_latency, &sfu_init);
  sscanf(trace_opcode_latency_initiation_tensor, "%u,%u", &tensor_latency,
         &tensor_init);

  sscanf(trace_opcode_latency_initiation_lmma, "%u,%u", &lmma_latency,
         &lmma_init);

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    sscanf(trace_opcode_latency_initiation_specialized_op[j], "%u,%u",
           &specialized_unit_latency[j], &specialized_unit_initiation[j]);
  }
}
void trace_config::set_latency(unsigned category, unsigned &latency,
                               unsigned &initiation_interval) const {
  initiation_interval = latency = 1;

  switch (category) {
    case ALU_OP:
    case INTP_OP:
    case BRANCH_OP:
    case CALL_OPS:
    case RET_OPS:
      latency = int_latency;
      initiation_interval = int_init;
      break;
    case SP_OP:
      latency = fp_latency;
      initiation_interval = fp_init;
      break;
    case DP_OP:
      latency = dp_latency;
      initiation_interval = dp_init;
      break;
    case SFU_OP:
      latency = sfu_latency;
      initiation_interval = sfu_init;
      break;
    case TENSOR_CORE_OP:
      latency = tensor_latency;
      initiation_interval = tensor_init;
      break;
    default:
      break;
  }
  // for specialized units
  if (category >= SPEC_UNIT_START_ID) {
    unsigned spec_id = category - SPEC_UNIT_START_ID;
    assert(spec_id >= 0 && spec_id < SPECIALIZED_UNIT_NUM);
    latency = specialized_unit_latency[spec_id];
    initiation_interval = specialized_unit_initiation[spec_id];
  }
}

void trace_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new trace_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                    m_shader_stats, m_memory_stats);
}

void trace_simt_core_cluster::create_shader_core_ctx() {
  m_core = new shader_core_ctx *[m_config->n_simt_cores_per_cluster];
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
    m_core[i] = new trace_shader_core_ctx(m_gpu, this, sid, m_cluster_id,
                                          m_config, m_mem_config, m_stats);
    m_core_sim_order.push_back(i);
  }
}

void trace_shader_core_ctx::create_shd_warp() {
  m_warp.resize(m_config->max_warps_per_shader);
  for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
    m_warp[k] = new trace_shd_warp_t(this, m_config->warp_size);
  }
}

void trace_shader_core_ctx::get_pdom_stack_top_info(unsigned warp_id,
                                                    const warp_inst_t *pI,
                                                    unsigned *pc,
                                                    unsigned *rpc) {
  // In trace-driven mode, we assume no control hazard
  assert(pI != NULL && "Unexpexted behaviour , inst should not be null");
  *pc = pI->pc;
  *rpc = pI->pc;
}

const active_mask_t &trace_shader_core_ctx::get_active_mask(
    unsigned warp_id, const warp_inst_t *pI) {
  // For Trace-driven, the active mask already set in traces, so
  // just read it from the inst
  return pI->get_active_mask();
}

unsigned trace_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  if (kernel.no_more_ctas_to_run()) {
    return 0;  // finished!
  }

  if (kernel.more_threads_in_cta()) {
    kernel.increment_thread_id();
  }

  if (!kernel.more_threads_in_cta()) kernel.increment_cta_id();

  return 1;
}

void trace_shader_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                       unsigned end_thread, unsigned ctaid,
                                       int cta_size, kernel_info_t &kernel) {
  // call base class
  shader_core_ctx::init_warps(cta_id, start_thread, end_thread, ctaid, cta_size,
                              kernel);

  // then init traces
  unsigned start_warp = start_thread / m_config->warp_size;
  unsigned end_warp = end_thread / m_config->warp_size +
                      ((end_thread % m_config->warp_size) ? 1 : 0);

  init_traces(start_warp, end_warp, kernel);
}

const warp_inst_t *trace_shader_core_ctx::get_next_inst(unsigned warp_id,
                                                        address_type pc) {
  // read the inst from the traces
  trace_shd_warp_t *m_trace_warp =
      static_cast<trace_shd_warp_t *>(m_warp[warp_id]);
  const trace_warp_inst_t *ret = m_trace_warp->get_next_trace_inst();
  if (ret == NULL && m_trace_warp->trace_done()) {
    if (!m_warp[warp_id]->inst_in_pipeline() &&
        m_warp[warp_id]->stores_done() &&
        !m_scoreboard->pendingWrites(warp_id)) {
      for (unsigned t = 0; t < m_warp_size; t++) {
        if (m_warp[warp_id]->test_active(t)) {
          m_warp[warp_id]->set_completed(t);
        }
      }
      m_barriers.warp_exit(warp_id);

      // Dump LMMA functional simulation results when this warp finishes.
      // For multi-warp kernels, each warp dumps its own result.
      dump_lmma_func_sim_results();
    }
  }
  return ret;
}

void trace_shader_core_ctx::updateSIMTStack(unsigned warpId,
                                            warp_inst_t *inst) {
  // No SIMT-stack in trace-driven  mode
}

void trace_shader_core_ctx::init_traces(unsigned start_warp, unsigned end_warp,
                                        kernel_info_t &kernel) {
  std::vector<std::vector<inst_trace_t> *> threadblock_traces;
  for (unsigned i = start_warp; i < end_warp; ++i) {
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
    m_trace_warp->clear();
    threadblock_traces.push_back(&(m_trace_warp->warp_traces));
  }
  trace_kernel_info_t &trace_kernel =
      static_cast<trace_kernel_info_t &>(kernel);
  trace_kernel.get_next_threadblock_traces(threadblock_traces);

  // set the pc from the traces and ignore the functional model
  for (unsigned i = start_warp; i < end_warp; ++i) {
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
    m_trace_warp->set_next_pc(m_trace_warp->get_start_trace_pc());
    m_trace_warp->set_kernel(&trace_kernel);
  }

  // Initialize LMMA functional simulation with kernel dimensions
  kernel_trace_t *kinfo = trace_kernel.get_trace_info();
  m_kernel_grid_x = kinfo->grid_dim_x;
  m_kernel_grid_y = kinfo->grid_dim_y;
  m_kernel_grid_z = kinfo->grid_dim_z;
  m_kernel_block_x = kinfo->tb_dim_x;
  m_kernel_block_y = kinfo->tb_dim_y;
  m_kernel_block_z = kinfo->tb_dim_z;
  init_lmma_func_sim();
}

void trace_shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst,
                                                          unsigned t,
                                                          unsigned tid) {
  if (inst.isatomic()) m_warp[inst.warp_id()]->inc_n_atomic();

  if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
    new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
    unsigned num_addrs;
    num_addrs = translate_local_memaddr(
        inst.get_addr(t), tid,
        m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
        inst.data_size, (new_addr_type *)localaddrs);
    inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
  }
}

void trace_shader_core_ctx::func_exec_inst(warp_inst_t &inst) {
  for (unsigned t = 0; t < m_warp_size; t++) {
    if (inst.active(t)) {
      unsigned warpId = inst.warp_id();
      unsigned tid = m_warp_size * warpId + t;

      // virtual function
      checkExecutionStatusAndUpdate(inst, t, tid);
    }
  }
  // here, we generate memory acessess and set the status if thread (done?)
  if (inst.is_load() || inst.is_store()) {
    inst.generate_mem_accesses();
  }

  // LMMA functional simulation: when we encounter an LMMA instruction,
  // trigger the LUT-based computation for this warp.
  trace_warp_inst_t *trace_inst = dynamic_cast<trace_warp_inst_t *>(&inst);
  if (trace_inst && trace_inst->m_is_lmma) {
    unsigned warpId = inst.warp_id();
    // For simple test kernels with 1 block, warp_id maps directly.
    // In general we'd need block coordinates from the CTA context.
    execute_lmma_func_sim(warpId, 0, 0, 0);
  }
}

void trace_shader_core_ctx::issue_warp(register_set &warp,
                                       const warp_inst_t *pI,
                                       const active_mask_t &active_mask,
                                       unsigned warp_id, unsigned sch_id) {
  shader_core_ctx::issue_warp(warp, pI, active_mask, warp_id, sch_id);

  // delete warp_inst_t class here, it is not required anymore by gpgpu-sim
  // after issue
  delete pI;
}

// ------------------------------------------------------------------
// LMMA Functional Simulation
// ------------------------------------------------------------------
//
// PURPOSE: Numerical correctness validation for the LUT-based MMA path.
// This is NOT a general functional simulator for arbitrary GEMM kernels.
//
// CONSTRAINTS:
//   - Fixed dimensions M=8, N=16, K=32 (must match proxy kernel)
//   - Fixed INT2 weight quantization ({-2, -1, 0, 1})
//   - K_GROUP=4 grouped LUT lookup with 256 entries
//   - FP16 quantization of LUT entries (10-bit mantissa)
//
// VALID TARGET: verify_lutsage/sageattn_lut_proxy.cu (single-tile GEMM)
// INVALID TARGET: Real multi-tile kernels (e.g., Triton sageattn_lut);
//   functional sim runs but result comparison is meaningless.
//
// DESIGN: Each warp computes the same C tile with identical fixed test data.
// For multi-warp proxy kernels, only the first warp's result is checked.

void trace_shader_core_ctx::init_lmma_func_sim() {
  m_lmma_func_sim_enabled = true;
  m_lmma_results_dumped = false;
  m_test_A.resize(LMMA_M * LMMA_K);
  m_test_W.resize(LMMA_N * LMMA_K);
  m_test_C_ref.resize(LMMA_M * LMMA_N);

  // Simple LCG random number generator (seed=42, same as Python script)
  unsigned seed = 42;
  auto lcg_rand = [&seed]() -> float {
    seed = seed * 1103515245u + 12345u;
    return (float)(seed & 0x7FFFu) / 32768.0f;  // [0, 1)
  };

  // Generate A: random in [-5, 5]
  for (int i = 0; i < LMMA_M * LMMA_K; i++) {
    m_test_A[i] = lcg_rand() * 10.0f - 5.0f;
  }

  // Generate W: INT2 quantized values {-2, -1, 0, 1}
  for (int i = 0; i < LMMA_N * LMMA_K; i++) {
    unsigned r = (unsigned)(lcg_rand() * 4.0f);  // {0, 1, 2, 3}
    m_test_W[i] = (int)r - 2;  // map to {-2, -1, 0, 1}
  }

  // Compute reference C = A @ W.T (FP32 accumulation)
  for (int m = 0; m < LMMA_M; m++) {
    for (int n = 0; n < LMMA_N; n++) {
      float acc = 0.0f;
      for (int k = 0; k < LMMA_K; k++) {
        acc += m_test_A[m * LMMA_K + k] * (float)m_test_W[n * LMMA_K + k];
      }
      m_test_C_ref[m * LMMA_N + n] = acc;
    }
  }

  printf("[LMMA-FuncSim] Initialized test data: M=%d, N=%d, K=%d, W_BITS=%d\n",
         LMMA_M, LMMA_N, LMMA_K, LMMA_W_BITS);
}

// Helper: simulate FP16 quantization for LUT table entries.
// FP16 has 10 bits of mantissa. We quantize the mantissa to 10 bits
// to mimic on-chip SRAM storage precision.
static float quantize_fp16(float val) {
  if (val == 0.0f) return 0.0f;
  int exponent;
  float mantissa = frexpf(val, &exponent);  // val = mantissa * 2^exponent
  // Round mantissa to 10 bits (1 implicit + 10 explicit)
  float q_mantissa = roundf(mantissa * 1024.0f) / 1024.0f;
  return ldexpf(q_mantissa, exponent);
}

void trace_shader_core_ctx::execute_lmma_func_sim(unsigned warp_id,
                                                   unsigned tb_x, unsigned tb_y,
                                                   unsigned tb_z) {
  if (!m_lmma_func_sim_enabled) return;
  if (m_warp_lmma_done[warp_id]) return;

  // LUT-based mpGEMM using grouped table lookup (matching paper Figure 3).
  // For each group of K_GROUP activations and weights, we precompute a LUT
  // indexed by the weight values. The LUT entries are quantized to FP16 to
  // simulate on-chip SRAM storage.
  const int K_GROUP = 4;
  const int NUM_GROUPS = LMMA_K / K_GROUP;  // 8 groups for K=32
  const int NUM_WEIGHT_VALUES = 1 << LMMA_W_BITS;  // 4 for INT2
  const int LUT_SIZE = 256;  // NUM_WEIGHT_VALUES^K_GROUP = 4^4 = 256

  std::vector<float> &C = m_warp_C[warp_id];
  C.resize(LMMA_M * LMMA_N, 0.0f);

  for (int m = 0; m < LMMA_M; m++) {
    for (int n = 0; n < LMMA_N; n++) {
      float acc = 0.0f;

      for (int g = 0; g < NUM_GROUPS; g++) {
        // 1. Extract activation tile A_tile[K_GROUP]
        float A_tile[K_GROUP];
        for (int k = 0; k < K_GROUP; k++) {
          A_tile[k] = m_test_A[m * LMMA_K + g * K_GROUP + k];
        }

        // 2. Extract weight tile W_tile[K_GROUP] (INT2 values {-2,-1,0,1})
        int W_tile[K_GROUP];
        for (int k = 0; k < K_GROUP; k++) {
          W_tile[k] = m_test_W[n * LMMA_K + g * K_GROUP + k];
        }

        // 3. Precompute LUT for this activation tile.
        //    LUT is indexed by the encoded weight pattern.
        //    For INT2, each weight has 4 possible values, so with K_GROUP=4
        //    the LUT has 4^4 = 256 entries.
        float lut[LUT_SIZE];
        for (int idx = 0; idx < LUT_SIZE; idx++) {
          // Decode index into K_GROUP weight values
          int tmp = idx;
          float sum = 0.0f;
          for (int k = 0; k < K_GROUP; k++) {
            int w_enc = tmp % NUM_WEIGHT_VALUES;  // {0,1,2,3}
            tmp /= NUM_WEIGHT_VALUES;
            // Map encoded value back to original weight: {-2,-1,0,1}
            int w_val = w_enc - (NUM_WEIGHT_VALUES / 2);
            sum += A_tile[k] * (float)w_val;
          }
          // Simulate FP16 table quantization (on-chip LUT storage)
          lut[idx] = quantize_fp16(sum);
        }

        // 4. Lookup: encode W_tile into LUT index.
        //    The loop order must match the decoding order in precompute
        //    (k=0 is the least significant digit).
        int lut_idx = 0;
        for (int k = K_GROUP - 1; k >= 0; k--) {
          int w_enc = W_tile[k] + (NUM_WEIGHT_VALUES / 2);
          lut_idx = lut_idx * NUM_WEIGHT_VALUES + w_enc;
        }

        // 5. Accumulate
        acc += lut[lut_idx];
      }
      C[m * LMMA_N + n] = acc;
    }
  }

  m_warp_lmma_done[warp_id] = true;
  printf("[LMMA-FuncSim] Warp %u executed LUT-based functional simulation.\n",
         warp_id);
}

void trace_shader_core_ctx::dump_lmma_func_sim_results() {
  if (!m_lmma_func_sim_enabled) return;
  if (m_warp_C.empty()) return;
  if (m_lmma_results_dumped) return;  // Prevent duplicate output
  m_lmma_results_dumped = true;

  printf("\n========== LMMA Functional Simulation Results ==========\n");
  printf("NOTE: Results below are valid ONLY for the proxy kernel\n");
  printf("      (verify_lutsage/sageattn_lut_proxy.cu, M=8,N=16,K=32).\n");
  printf("      Real multi-tile kernels are NOT validated here.\n\n");

  // For multi-warp proxy kernels (e.g., cuBLAS GEMM with 4 warps), each
  // warp computes the SAME C tile because the proxy has only one tile.
  // Use only the first warp's result to avoid over-counting.
  // WARNING: For real multi-tile kernels, different warps compute different
  // tiles, so this simplification does NOT apply.
  std::vector<float> C_final = m_warp_C.begin()->second;

  printf("C matrix (M=%d, N=%d):\n", LMMA_M, LMMA_N);
  for (int m = 0; m < LMMA_M; m++) {
    printf("  Row %2d: ", m);
    for (int n = 0; n < LMMA_N; n++) {
      printf("%8.4f ", C_final[m * LMMA_N + n]);
    }
    printf("\n");
  }

  printf("\nReference C matrix:\n");
  for (int m = 0; m < LMMA_M; m++) {
    printf("  Row %2d: ", m);
    for (int n = 0; n < LMMA_N; n++) {
      printf("%8.4f ", m_test_C_ref[m * LMMA_N + n]);
    }
    printf("\n");
  }

  // Compute max absolute error
  float max_err = 0.0f;
  float avg_err = 0.0f;
  for (int i = 0; i < LMMA_M * LMMA_N; i++) {
    float err = fabsf(C_final[i] - m_test_C_ref[i]);
    if (err > max_err) max_err = err;
    avg_err += err;
  }
  avg_err /= (LMMA_M * LMMA_N);

  printf("\nMax absolute error: %.6f\n", max_err);
  printf("Avg absolute error: %.6f\n", avg_err);

  if (max_err < 0.1f) {
    printf("LMMA FUNCTIONAL VERIFICATION: PASSED\n");
  } else {
    printf("LMMA FUNCTIONAL VERIFICATION: FAILED (error %.6f >= 0.1)\n",
           max_err);
  }
  printf("========================================================\n\n");
  fflush(stdout);
}
