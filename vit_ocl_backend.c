#include "vit_ocl_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_SOURCE_SIZE 65536

static const char* read_kernel_source(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) fp = fopen("vit_kernels.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open vit_kernels.cl\n");
        return NULL;
    }
    char* source = (char*)malloc(MAX_SOURCE_SIZE);
    size_t size = fread(source, 1, MAX_SOURCE_SIZE - 1, fp);
    source[size] = '\0';
    fclose(fp);
    return source;
}

static int init_opencl(VitBackend* backend) {
    cl_int err;
    err = clGetPlatformIDs(1, &backend->platform, NULL);
    if (err != CL_SUCCESS) return 0;
    err = clGetDeviceIDs(backend->platform, CL_DEVICE_TYPE_GPU, 1, &backend->device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(backend->platform, CL_DEVICE_TYPE_CPU, 1, &backend->device, NULL);
        if (err != CL_SUCCESS) return 0;
    }
    backend->context = clCreateContext(NULL, 1, &backend->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->queue = clCreateCommandQueue(backend->context, backend->device, 0, &err);
    if (err != CL_SUCCESS) return 0;
    return 1;
}

static int build_program(VitBackend* backend) {
    cl_int err;
    const char* source = read_kernel_source("vit_kernels.cl");
    if (!source) return 0;
    backend->program = clCreateProgramWithSource(backend->context, 1, &source, NULL, &err);
    free((void*)source);
    if (err != CL_SUCCESS) return 0;
    err = clBuildProgram(backend->program, 1, &backend->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(backend->program, backend->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(backend->program, backend->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "OpenCL build error:\n%s\n", log);
        free(log);
        return 0;
    }
    backend->kernel_embed = clCreateKernel(backend->program, "vit_embed", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_layernorm = clCreateKernel(backend->program, "vit_layernorm", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_qkv_proj = clCreateKernel(backend->program, "vit_qkv_proj", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_attention = clCreateKernel(backend->program, "vit_attention", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_ffn = clCreateKernel(backend->program, "vit_ffn", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_head = clCreateKernel(backend->program, "vit_head", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_add = clCreateKernel(backend->program, "vit_add", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_clip_grad_norm = clCreateKernel(backend->program, "vit_clip_grad_norm", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_scale_grads = clCreateKernel(backend->program, "vit_scale_grads", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_adamw_update = clCreateKernel(backend->program, "vit_adamw_update", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_mpp_forward = clCreateKernel(backend->program, "vit_mpp_forward", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_mpp_backward = clCreateKernel(backend->program, "vit_mpp_backward", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_embed_bwd = clCreateKernel(backend->program, "vit_embed_bwd", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_layernorm_bwd = clCreateKernel(backend->program, "vit_layernorm_bwd", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_qkv_proj_bwd = clCreateKernel(backend->program, "vit_qkv_proj_bwd", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_attention_bwd = clCreateKernel(backend->program, "vit_attention_bwd", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_ffn_bwd = clCreateKernel(backend->program, "vit_ffn_bwd", &err);
    if (err != CL_SUCCESS) return 0;
    backend->kernel_head_bwd = clCreateKernel(backend->program, "vit_head_bwd", &err);
    if (err != CL_SUCCESS) return 0;
    return 1;
}

static void compute_offsets(int embed_dim, int num_layers, int num_heads, int mlp_dim, int seq_len, int in_channels,
                            int* offsets, int* bias_offsets) {
    int off = 0, b_off = 0;
    offsets[0] = off; off += in_channels * embed_dim;
    offsets[1] = off; off += embed_dim;
    offsets[2] = off; off += (seq_len + 1) * embed_dim;
    bias_offsets[0] = 0;
    for (int l = 0; l < num_layers; l++) {
        offsets[3 + l * 8 + 0] = off; off += embed_dim;
        bias_offsets[1 + l * 8 + 0] = b_off; b_off += embed_dim;
        offsets[3 + l * 8 + 1] = off; off += embed_dim * (3 * embed_dim);
        bias_offsets[1 + l * 8 + 1] = b_off; b_off += 3 * embed_dim;
        offsets[3 + l * 8 + 2] = off; off += embed_dim * embed_dim;
        bias_offsets[1 + l * 8 + 2] = b_off; b_off += embed_dim;
        offsets[3 + l * 8 + 3] = off; off += embed_dim;
        bias_offsets[1 + l * 8 + 3] = b_off; b_off += embed_dim;
        offsets[3 + l * 8 + 4] = off; off += embed_dim * mlp_dim;
        bias_offsets[1 + l * 8 + 4] = b_off; b_off += mlp_dim;
        offsets[3 + l * 8 + 5] = off; off += mlp_dim * embed_dim;
        bias_offsets[1 + l * 8 + 5] = b_off; b_off += embed_dim;
    }
    offsets[3 + num_layers * 8 + 0] = off; off += embed_dim;
    bias_offsets[1 + num_layers * 8 + 0] = b_off; b_off += embed_dim;
    offsets[3 + num_layers * 8 + 1] = off; off += embed_dim;
    bias_offsets[1 + num_layers * 8 + 1] = b_off; b_off += 1;
}

static void truncated_normal(float* arr, int n, float std, long seed) {
    srand((unsigned int)seed);
    for (int i = 0; i < n; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.1415926535f * u2);
        if (z < -2.0f) z = -2.0f;
        if (z > 2.0f) z = 2.0f;
        arr[i] = z * std;
    }
}

static void release_buffers(VitBackend* backend) {
    if (backend->d_input) clReleaseMemObject(backend->d_input);
    if (backend->d_output) clReleaseMemObject(backend->d_output);
    if (backend->d_weights) clReleaseMemObject(backend->d_weights);
    if (backend->d_biases) clReleaseMemObject(backend->d_biases);
    if (backend->d_gradWeights) clReleaseMemObject(backend->d_gradWeights);
    if (backend->d_gradBiases) clReleaseMemObject(backend->d_gradBiases);
    if (backend->d_mWeights) clReleaseMemObject(backend->d_mWeights);
    if (backend->d_vWeights) clReleaseMemObject(backend->d_vWeights);
    if (backend->d_mBiases) clReleaseMemObject(backend->d_mBiases);
    if (backend->d_vBiases) clReleaseMemObject(backend->d_vBiases);
    if (backend->d_token_embeds) clReleaseMemObject(backend->d_token_embeds);
    if (backend->d_layernorm_input) clReleaseMemObject(backend->d_layernorm_input);
    if (backend->d_qkv_proj) clReleaseMemObject(backend->d_qkv_proj);
    if (backend->d_attn_output) clReleaseMemObject(backend->d_attn_output);
    if (backend->d_ffn_output) clReleaseMemObject(backend->d_ffn_output);
    if (backend->d_head_input) clReleaseMemObject(backend->d_head_input);
    if (backend->d_layernorm_mean) clReleaseMemObject(backend->d_layernorm_mean);
    if (backend->d_layernorm_inv_std) clReleaseMemObject(backend->d_layernorm_inv_std);
    if (backend->d_attn_weights) clReleaseMemObject(backend->d_attn_weights);
    if (backend->d_pre_gelu) clReleaseMemObject(backend->d_pre_gelu);
    if (backend->d_ffn_hidden) clReleaseMemObject(backend->d_ffn_hidden);
    if (backend->d_cls_normed) clReleaseMemObject(backend->d_cls_normed);
    if (backend->d_cls_mean) clReleaseMemObject(backend->d_cls_mean);
    if (backend->d_cls_inv_std) clReleaseMemObject(backend->d_cls_inv_std);
    if (backend->d_grad_qkv) clReleaseMemObject(backend->d_grad_qkv);
    if (backend->d_grad_norm) clReleaseMemObject(backend->d_grad_norm);
    if (backend->d_mpp_logits) clReleaseMemObject(backend->d_mpp_logits);
    if (backend->d_mpp_loss) clReleaseMemObject(backend->d_mpp_loss);
    if (backend->d_mask_weights) clReleaseMemObject(backend->d_mask_weights);
    if (backend->d_mask_biases) clReleaseMemObject(backend->d_mask_biases);
    if (backend->d_grad_mask_weights) clReleaseMemObject(backend->d_grad_mask_weights);
    if (backend->d_grad_mask_biases) clReleaseMemObject(backend->d_grad_mask_biases);
    if (backend->d_m_mask_weights) clReleaseMemObject(backend->d_m_mask_weights);
    if (backend->d_v_mask_weights) clReleaseMemObject(backend->d_v_mask_weights);
    if (backend->d_m_mask_biases) clReleaseMemObject(backend->d_m_mask_biases);
    if (backend->d_v_mask_biases) clReleaseMemObject(backend->d_v_mask_biases);
    if (backend->d_layer_inputs) clReleaseMemObject(backend->d_layer_inputs);
    if (backend->d_ln1_outputs) clReleaseMemObject(backend->d_ln1_outputs);
    if (backend->d_ln2_inputs) clReleaseMemObject(backend->d_ln2_inputs);
    if (backend->d_attn_temp) clReleaseMemObject(backend->d_attn_temp);
    if (backend->d_scores_temp) clReleaseMemObject(backend->d_scores_temp);
    if (backend->d_ffn_temp) clReleaseMemObject(backend->d_ffn_temp);
    if (backend->d_normed_temp) clReleaseMemObject(backend->d_normed_temp);
}

static int reallocate_buffers(VitBackend* backend, int batch_size) {
    cl_int err;
    int seq_len = backend->seq_len;
    int embed_dim = backend->embed_dim;
    int total_tokens = seq_len + 1;
    int mlp_dim = backend->mlp_dim;
    int num_layers = backend->num_layers;

    if (backend->d_input) clReleaseMemObject(backend->d_input);
    if (backend->d_output) clReleaseMemObject(backend->d_output);
    if (backend->d_token_embeds) clReleaseMemObject(backend->d_token_embeds);
    if (backend->d_layernorm_input) clReleaseMemObject(backend->d_layernorm_input);
    if (backend->d_qkv_proj) clReleaseMemObject(backend->d_qkv_proj);
    if (backend->d_attn_output) clReleaseMemObject(backend->d_attn_output);
    if (backend->d_ffn_output) clReleaseMemObject(backend->d_ffn_output);
    if (backend->d_head_input) clReleaseMemObject(backend->d_head_input);
    if (backend->d_layernorm_mean) clReleaseMemObject(backend->d_layernorm_mean);
    if (backend->d_layernorm_inv_std) clReleaseMemObject(backend->d_layernorm_inv_std);
    if (backend->d_attn_weights) clReleaseMemObject(backend->d_attn_weights);
    if (backend->d_pre_gelu) clReleaseMemObject(backend->d_pre_gelu);
    if (backend->d_ffn_hidden) clReleaseMemObject(backend->d_ffn_hidden);
    if (backend->d_cls_normed) clReleaseMemObject(backend->d_cls_normed);
    if (backend->d_cls_mean) clReleaseMemObject(backend->d_cls_mean);
    if (backend->d_cls_inv_std) clReleaseMemObject(backend->d_cls_inv_std);
    if (backend->d_grad_qkv) clReleaseMemObject(backend->d_grad_qkv);
    if (backend->d_grad_norm) clReleaseMemObject(backend->d_grad_norm);
    if (backend->d_mpp_logits) clReleaseMemObject(backend->d_mpp_logits);
    if (backend->d_mpp_loss) clReleaseMemObject(backend->d_mpp_loss);
    if (backend->d_layer_inputs) clReleaseMemObject(backend->d_layer_inputs);
    if (backend->d_ln1_outputs) clReleaseMemObject(backend->d_ln1_outputs);
    if (backend->d_ln2_inputs) clReleaseMemObject(backend->d_ln2_inputs);
    if (backend->d_attn_temp) clReleaseMemObject(backend->d_attn_temp);
    if (backend->d_scores_temp) clReleaseMemObject(backend->d_scores_temp);
    if (backend->d_ffn_temp) clReleaseMemObject(backend->d_ffn_temp);
    if (backend->d_normed_temp) clReleaseMemObject(backend->d_normed_temp);

    backend->d_input = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * seq_len * backend->in_channels * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_output = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_token_embeds = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_layernorm_input = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_qkv_proj = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * 3 * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_attn_output = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_ffn_output = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_head_input = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_layernorm_mean = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_layernorm_inv_std = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_attn_weights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * backend->num_heads * total_tokens * total_tokens * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_pre_gelu = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * mlp_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_ffn_hidden = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * mlp_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_cls_normed = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_cls_mean = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_cls_inv_std = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_grad_qkv = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * 3 * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_grad_norm = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_mpp_logits = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * backend->mpp_num_classes * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_mpp_loss = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_layer_inputs = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        num_layers * batch_size * total_tokens * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_ln1_outputs = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        num_layers * batch_size * total_tokens * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_ln2_inputs = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        num_layers * batch_size * total_tokens * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_attn_temp = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_scores_temp = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * total_tokens * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_ffn_temp = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * total_tokens * mlp_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;
    backend->d_normed_temp = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        batch_size * embed_dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) return 0;

    backend->max_allocated_batch = batch_size;
    return 1;
}

static int ensure_buffers(VitBackend* backend, int batch_size) {
    if (batch_size <= backend->max_allocated_batch) return 1;
    return reallocate_buffers(backend, batch_size);
}

static int init_weights(VitBackend* backend, long seed) {
    truncated_normal(backend->host_weights, backend->total_weights, backend->init_std, seed);
    for (int i = 0; i < backend->total_biases; i++) {
        backend->host_biases[i] = 0.0f;
    }
    cl_int err;
    err = clEnqueueWriteBuffer(backend->queue, backend->d_weights, CL_TRUE, 0,
        backend->total_weights * sizeof(float), backend->host_weights, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;
    err = clEnqueueWriteBuffer(backend->queue, backend->d_biases, CL_TRUE, 0,
        backend->total_biases * sizeof(float), backend->host_biases, 0, NULL, NULL);
    if (err != CL_SUCCESS) return 0;
    return 1;
}

VitBackend* vit_backend_create(int embed_dim, int num_layers, int num_heads, int mlp_dim, int seq_len, int in_channels, long seed, int mppNumClasses) {
    VitBackend* backend = (VitBackend*)calloc(1, sizeof(VitBackend));
    if (!backend) return NULL;
    backend->embed_dim = embed_dim;
    backend->num_layers = num_layers;
    backend->num_heads = num_heads;
    backend->mlp_dim = mlp_dim;
    backend->seq_len = seq_len;
    backend->in_channels = in_channels;
    backend->mpp_num_classes = mppNumClasses;
    backend->max_batch_size = 65536;
    backend->workgroup_size = 256;
    backend->init_std = 0.02f;
    int total_weights = 0, total_biases = 0;
    total_weights += in_channels * embed_dim;
    total_weights += embed_dim;
    total_weights += (seq_len + 1) * embed_dim;
    for (int l = 0; l < num_layers; l++) {
        total_weights += embed_dim;
        total_biases += embed_dim;
        total_weights += embed_dim * (3 * embed_dim);
        total_biases += 3 * embed_dim;
        total_weights += embed_dim * embed_dim;
        total_biases += embed_dim;
        total_weights += embed_dim;
        total_biases += embed_dim;
        total_weights += embed_dim * mlp_dim;
        total_biases += mlp_dim;
        total_weights += mlp_dim * embed_dim;
        total_biases += embed_dim;
    }
    total_weights += embed_dim;
    total_biases += embed_dim;
    total_weights += embed_dim;
    total_biases += 1;
    backend->total_weights = total_weights;
    backend->total_biases = total_biases;
    if (!init_opencl(backend)) { vit_backend_destroy(backend); return NULL; }
    if (!build_program(backend)) { vit_backend_destroy(backend); return NULL; }
    backend->host_weights = (float*)malloc(backend->total_weights * sizeof(float));
    if (!backend->host_weights) { vit_backend_destroy(backend); return NULL; }
    backend->host_biases = (float*)malloc(backend->total_biases * sizeof(float));
    if (!backend->host_biases) { vit_backend_destroy(backend); return NULL; }
    backend->host_mWeights = (float*)calloc(backend->total_weights, sizeof(float));
    if (!backend->host_mWeights) { vit_backend_destroy(backend); return NULL; }
    backend->host_vWeights = (float*)calloc(backend->total_weights, sizeof(float));
    if (!backend->host_vWeights) { vit_backend_destroy(backend); return NULL; }
    backend->host_mBiases = (float*)calloc(backend->total_biases, sizeof(float));
    if (!backend->host_mBiases) { vit_backend_destroy(backend); return NULL; }
    backend->host_vBiases = (float*)calloc(backend->total_biases, sizeof(float));
    if (!backend->host_vBiases) { vit_backend_destroy(backend); return NULL; }

    // 创建参数缓冲区（权重、偏置、梯度、Adam状态）——只创建一次
    cl_int err;
    backend->d_weights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->total_weights * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_biases = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->total_biases * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_gradWeights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->total_weights * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_gradBiases = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->total_biases * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_mWeights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->total_weights * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_vWeights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->total_weights * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_mBiases = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->total_biases * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_vBiases = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->total_biases * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }

    // MPP 相关参数缓冲区（如有需要）
    backend->d_mask_weights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->embed_dim * backend->mpp_num_classes * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_mask_biases = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->mpp_num_classes * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_grad_mask_weights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->embed_dim * backend->mpp_num_classes * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_grad_mask_biases = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->mpp_num_classes * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_m_mask_weights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->embed_dim * backend->mpp_num_classes * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_v_mask_weights = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->embed_dim * backend->mpp_num_classes * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_m_mask_biases = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->mpp_num_classes * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }
    backend->d_v_mask_biases = clCreateBuffer(backend->context, CL_MEM_READ_WRITE,
        backend->mpp_num_classes * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) { vit_backend_destroy(backend); return NULL; }

    // 初始化权重并写入设备
    if (!init_weights(backend, seed)) { vit_backend_destroy(backend); return NULL; }

    // 分配动态缓冲区（输入、中间激活、输出等），batch=1 初始
    if (!reallocate_buffers(backend, 1)) { vit_backend_destroy(backend); return NULL; }

    backend->initialized = 1;
    return backend;
}

VitBackend* vit_backend_create_with_weights(int embed_dim, int num_layers, int num_heads, int mlp_dim, int seq_len, int in_channels, const float* weights, const float* biases, int mppNumClasses) {
    VitBackend* backend = vit_backend_create(embed_dim, num_layers, num_heads, mlp_dim, seq_len, in_channels, 0, mppNumClasses);
    if (!backend) return NULL;
    if (weights) vit_backend_set_weights(backend, weights);
    if (biases) vit_backend_set_biases(backend, biases);
    return backend;
}

void vit_backend_destroy(VitBackend* backend) {
    if (!backend) return;
    if (backend->kernel_embed) clReleaseKernel(backend->kernel_embed);
    if (backend->kernel_layernorm) clReleaseKernel(backend->kernel_layernorm);
    if (backend->kernel_qkv_proj) clReleaseKernel(backend->kernel_qkv_proj);
    if (backend->kernel_attention) clReleaseKernel(backend->kernel_attention);
    if (backend->kernel_ffn) clReleaseKernel(backend->kernel_ffn);
    if (backend->kernel_head) clReleaseKernel(backend->kernel_head);
    if (backend->kernel_add) clReleaseKernel(backend->kernel_add);
    if (backend->kernel_clip_grad_norm) clReleaseKernel(backend->kernel_clip_grad_norm);
    if (backend->kernel_scale_grads) clReleaseKernel(backend->kernel_scale_grads);
    if (backend->kernel_adamw_update) clReleaseKernel(backend->kernel_adamw_update);
    if (backend->kernel_mpp_forward) clReleaseKernel(backend->kernel_mpp_forward);
    if (backend->kernel_mpp_backward) clReleaseKernel(backend->kernel_mpp_backward);
    if (backend->kernel_embed_bwd) clReleaseKernel(backend->kernel_embed_bwd);
    if (backend->kernel_layernorm_bwd) clReleaseKernel(backend->kernel_layernorm_bwd);
    if (backend->kernel_qkv_proj_bwd) clReleaseKernel(backend->kernel_qkv_proj_bwd);
    if (backend->kernel_attention_bwd) clReleaseKernel(backend->kernel_attention_bwd);
    if (backend->kernel_ffn_bwd) clReleaseKernel(backend->kernel_ffn_bwd);
    if (backend->kernel_head_bwd) clReleaseKernel(backend->kernel_head_bwd);
    if (backend->program) clReleaseProgram(backend->program);
    if (backend->queue) clReleaseCommandQueue(backend->queue);
    if (backend->context) clReleaseContext(backend->context);
    release_buffers(backend);
    free(backend->host_weights);
    free(backend->host_biases);
    free(backend->host_mWeights);
    free(backend->host_vWeights);
    free(backend->host_mBiases);
    free(backend->host_vBiases);
    free(backend);
}

void vit_backend_forward(VitBackend* backend, const float* input, float* output, int batch_size) {
    if (!backend || !backend->initialized) return;
    cl_int err;
    float* debug_w1 = (float*)malloc(5 * sizeof(float));
    if (debug_w1) {
        clFinish(backend->queue);
        clEnqueueReadBuffer(backend->queue, backend->d_weights, CL_TRUE, 0, 5 * sizeof(float), debug_w1, 0, NULL, NULL);
        free(debug_w1);
    }
    if (!ensure_buffers(backend, batch_size)) return;
    float* debug_w2 = (float*)malloc(5 * sizeof(float));
    if (debug_w2) {
        clFinish(backend->queue);
        clEnqueueReadBuffer(backend->queue, backend->d_weights, CL_TRUE, 0, 5 * sizeof(float), debug_w2, 0, NULL, NULL);
        free(debug_w2);
    }
    int seq_len = backend->seq_len;
    int embed_dim = backend->embed_dim;
    int total_tokens = seq_len + 1;
    int in_channels = backend->in_channels;
    int num_layers = backend->num_layers;
    int num_heads = backend->num_heads;
    int mlp_dim = backend->mlp_dim;

    err = clEnqueueWriteBuffer(backend->queue, backend->d_input, CL_FALSE, 0,
        batch_size * seq_len * in_channels * sizeof(float), input, 0, NULL, NULL);
    if (err != CL_SUCCESS) return;

    int offsets[100], bias_offsets[100];
    compute_offsets(embed_dim, num_layers, num_heads, mlp_dim, seq_len, in_channels, offsets, bias_offsets);

    clSetKernelArg(backend->kernel_embed, 0, sizeof(cl_mem), &backend->d_input);
    clSetKernelArg(backend->kernel_embed, 1, sizeof(cl_mem), &backend->d_weights);
    clSetKernelArg(backend->kernel_embed, 2, sizeof(cl_mem), &backend->d_biases);
    clSetKernelArg(backend->kernel_embed, 3, sizeof(cl_mem), &backend->d_token_embeds);
    clSetKernelArg(backend->kernel_embed, 4, sizeof(int), &batch_size);
    clSetKernelArg(backend->kernel_embed, 5, sizeof(int), &seq_len);
    clSetKernelArg(backend->kernel_embed, 6, sizeof(int), &in_channels);
    clSetKernelArg(backend->kernel_embed, 7, sizeof(int), &embed_dim);
    clSetKernelArg(backend->kernel_embed, 8, sizeof(int), &total_tokens);
    clSetKernelArg(backend->kernel_embed, 9, sizeof(int), &offsets[0]);
    clSetKernelArg(backend->kernel_embed, 10, sizeof(int), &offsets[1]);
    clSetKernelArg(backend->kernel_embed, 11, sizeof(int), &offsets[2]);
    size_t global_embed = batch_size * total_tokens;
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_embed, 1, NULL, &global_embed, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) return;

    cl_mem current = backend->d_token_embeds;
    for (int l = 0; l < num_layers; l++) {
        clEnqueueCopyBuffer(backend->queue, current, backend->d_layer_inputs,
            0, l * batch_size * total_tokens * embed_dim * sizeof(float),
            batch_size * total_tokens * embed_dim * sizeof(float), 0, NULL, NULL);

        int ln1_gamma_off = offsets[3 + l * 8 + 0];
        int ln1_beta_off = bias_offsets[1 + l * 8 + 0];
        int qkv_off = offsets[3 + l * 8 + 1];
        int qkv_b_off = bias_offsets[1 + l * 8 + 1];
        int proj_off = offsets[3 + l * 8 + 2];
        int proj_b_off = bias_offsets[1 + l * 8 + 2];
        int ln2_gamma_off = offsets[3 + l * 8 + 3];
        int ln2_beta_off = bias_offsets[1 + l * 8 + 3];
        int ffn1_off = offsets[3 + l * 8 + 4];
        int ffn1_b_off = bias_offsets[1 + l * 8 + 4];
        int ffn2_off = offsets[3 + l * 8 + 5];
        int ffn2_b_off = bias_offsets[1 + l * 8 + 5];

        clSetKernelArg(backend->kernel_layernorm, 0, sizeof(cl_mem), &current);
        clSetKernelArg(backend->kernel_layernorm, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_layernorm, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_layernorm, 3, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_layernorm, 4, sizeof(cl_mem), &backend->d_layernorm_mean);
        clSetKernelArg(backend->kernel_layernorm, 5, sizeof(cl_mem), &backend->d_layernorm_inv_std);
        clSetKernelArg(backend->kernel_layernorm, 6, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_layernorm, 7, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_layernorm, 8, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_layernorm, 9, sizeof(int), &ln1_gamma_off);
        clSetKernelArg(backend->kernel_layernorm, 10, sizeof(int), &ln1_beta_off);
        size_t global_ln = batch_size * total_tokens;
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_layernorm, 1, NULL, &global_ln, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;
        clEnqueueCopyBuffer(backend->queue, backend->d_layernorm_input, backend->d_ln1_outputs,
            0, l * batch_size * total_tokens * embed_dim * sizeof(float),
            batch_size * total_tokens * embed_dim * sizeof(float), 0, NULL, NULL);

        clSetKernelArg(backend->kernel_qkv_proj, 0, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_qkv_proj, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_qkv_proj, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_qkv_proj, 3, sizeof(cl_mem), &backend->d_qkv_proj);
        clSetKernelArg(backend->kernel_qkv_proj, 4, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_qkv_proj, 5, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_qkv_proj, 6, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_qkv_proj, 7, sizeof(int), &qkv_off);
        clSetKernelArg(backend->kernel_qkv_proj, 8, sizeof(int), &qkv_b_off);
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_qkv_proj, 1, NULL, &global_ln, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;

        clSetKernelArg(backend->kernel_attention, 0, sizeof(cl_mem), &backend->d_qkv_proj);
        clSetKernelArg(backend->kernel_attention, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_attention, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_attention, 3, sizeof(cl_mem), &backend->d_attn_output);
        clSetKernelArg(backend->kernel_attention, 4, sizeof(cl_mem), &backend->d_attn_weights);
        clSetKernelArg(backend->kernel_attention, 5, sizeof(cl_mem), &backend->d_attn_temp);
        clSetKernelArg(backend->kernel_attention, 6, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_attention, 7, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_attention, 8, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_attention, 9, sizeof(int), &num_heads);
        clSetKernelArg(backend->kernel_attention, 10, sizeof(int), &proj_off);
        clSetKernelArg(backend->kernel_attention, 11, sizeof(int), &proj_b_off);
        size_t global_attn = batch_size * total_tokens;
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_attention, 1, NULL, &global_attn, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;

        clSetKernelArg(backend->kernel_add, 0, sizeof(cl_mem), &current);
        clSetKernelArg(backend->kernel_add, 1, sizeof(cl_mem), &backend->d_attn_output);
        clSetKernelArg(backend->kernel_add, 2, sizeof(cl_mem), &current);
        clSetKernelArg(backend->kernel_add, 3, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_add, 4, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_add, 5, sizeof(int), &embed_dim);
        size_t global_add = batch_size * total_tokens * embed_dim;
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_add, 1, NULL, &global_add, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;
        clEnqueueCopyBuffer(backend->queue, current, backend->d_ln2_inputs,
            0, l * batch_size * total_tokens * embed_dim * sizeof(float),
            batch_size * total_tokens * embed_dim * sizeof(float), 0, NULL, NULL);

        clSetKernelArg(backend->kernel_layernorm, 0, sizeof(cl_mem), &current);
        clSetKernelArg(backend->kernel_layernorm, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_layernorm, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_layernorm, 3, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_layernorm, 4, sizeof(cl_mem), &backend->d_layernorm_mean);
        clSetKernelArg(backend->kernel_layernorm, 5, sizeof(cl_mem), &backend->d_layernorm_inv_std);
        clSetKernelArg(backend->kernel_layernorm, 6, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_layernorm, 7, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_layernorm, 8, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_layernorm, 9, sizeof(int), &ln2_gamma_off);
        clSetKernelArg(backend->kernel_layernorm, 10, sizeof(int), &ln2_beta_off);
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_layernorm, 1, NULL, &global_ln, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;

        clSetKernelArg(backend->kernel_ffn, 0, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_ffn, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_ffn, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_ffn, 3, sizeof(cl_mem), &backend->d_ffn_output);
        clSetKernelArg(backend->kernel_ffn, 4, sizeof(cl_mem), &backend->d_pre_gelu);
        clSetKernelArg(backend->kernel_ffn, 5, sizeof(cl_mem), &backend->d_ffn_hidden);
        clSetKernelArg(backend->kernel_ffn, 6, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_ffn, 7, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_ffn, 8, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_ffn, 9, sizeof(int), &mlp_dim);
        clSetKernelArg(backend->kernel_ffn, 10, sizeof(int), &ffn1_off);
        clSetKernelArg(backend->kernel_ffn, 11, sizeof(int), &ffn1_b_off);
        clSetKernelArg(backend->kernel_ffn, 12, sizeof(int), &ffn2_off);
        clSetKernelArg(backend->kernel_ffn, 13, sizeof(int), &ffn2_b_off);
        size_t global_ffn = batch_size * total_tokens;
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_ffn, 1, NULL, &global_ffn, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;

        clSetKernelArg(backend->kernel_add, 0, sizeof(cl_mem), &current);
        clSetKernelArg(backend->kernel_add, 1, sizeof(cl_mem), &backend->d_ffn_output);
        clSetKernelArg(backend->kernel_add, 2, sizeof(cl_mem), &current);
        clSetKernelArg(backend->kernel_add, 3, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_add, 4, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_add, 5, sizeof(int), &embed_dim);
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_add, 1, NULL, &global_add, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;
    }

    int final_ln_gamma = offsets[3 + num_layers * 8 + 0];
    int final_ln_beta = bias_offsets[1 + num_layers * 8 + 0];
    int head_weight = offsets[3 + num_layers * 8 + 1];
    int head_bias = bias_offsets[1 + num_layers * 8 + 1];

    clSetKernelArg(backend->kernel_head, 0, sizeof(cl_mem), &current);
    clSetKernelArg(backend->kernel_head, 1, sizeof(cl_mem), &backend->d_weights);
    clSetKernelArg(backend->kernel_head, 2, sizeof(cl_mem), &backend->d_biases);
    clSetKernelArg(backend->kernel_head, 3, sizeof(cl_mem), &backend->d_output);
    clSetKernelArg(backend->kernel_head, 4, sizeof(int), &batch_size);
    clSetKernelArg(backend->kernel_head, 5, sizeof(int), &total_tokens);
    clSetKernelArg(backend->kernel_head, 6, sizeof(int), &embed_dim);
    clSetKernelArg(backend->kernel_head, 7, sizeof(int), &head_weight);
    clSetKernelArg(backend->kernel_head, 8, sizeof(int), &head_bias);
    size_t global_head = batch_size * total_tokens;
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_head, 1, NULL, &global_head, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) return;

    err = clEnqueueReadBuffer(backend->queue, backend->d_output, CL_TRUE, 0,
        batch_size * total_tokens * sizeof(float), output, 0, NULL, NULL);
    if (err != CL_SUCCESS) return;
}

void vit_backend_backward(VitBackend* backend, const float* input, const float* label, const float* grad_output, int batch_size) {
    if (!backend || !backend->initialized) return;
    if (!ensure_buffers(backend, batch_size)) return;
    cl_int err;
    int seq_len = backend->seq_len;
    int embed_dim = backend->embed_dim;
    int total_tokens = seq_len + 1;
    int in_channels = backend->in_channels;
    int num_layers = backend->num_layers;
    int num_heads = backend->num_heads;
    int mlp_dim = backend->mlp_dim;

    err = clEnqueueWriteBuffer(backend->queue, backend->d_input, CL_FALSE, 0,
        batch_size * seq_len * in_channels * sizeof(float), input, 0, NULL, NULL);
    if (err != CL_SUCCESS) return;

    err = clEnqueueWriteBuffer(backend->queue, backend->d_output, CL_FALSE, 0,
        batch_size * sizeof(float), grad_output, 0, NULL, NULL);
    if (err != CL_SUCCESS) return;

    int offsets[100], bias_offsets[100];
    compute_offsets(embed_dim, num_layers, num_heads, mlp_dim, seq_len, in_channels, offsets, bias_offsets);

    int final_ln_gamma = offsets[3 + num_layers * 8 + 0];
    int final_ln_beta = bias_offsets[1 + num_layers * 8 + 0];
    int head_weight = offsets[3 + num_layers * 8 + 1];
    int head_bias = bias_offsets[1 + num_layers * 8 + 1];

    cl_mem current_grad = backend->d_head_input;

    clSetKernelArg(backend->kernel_head_bwd, 0, sizeof(cl_mem), &backend->d_output);
    clSetKernelArg(backend->kernel_head_bwd, 1, sizeof(cl_mem), &backend->d_weights);
    clSetKernelArg(backend->kernel_head_bwd, 2, sizeof(cl_mem), &backend->d_biases);
    clSetKernelArg(backend->kernel_head_bwd, 3, sizeof(cl_mem), &backend->d_gradWeights);
    clSetKernelArg(backend->kernel_head_bwd, 4, sizeof(cl_mem), &backend->d_gradBiases);
    clSetKernelArg(backend->kernel_head_bwd, 5, sizeof(cl_mem), &current_grad);
    clSetKernelArg(backend->kernel_head_bwd, 6, sizeof(cl_mem), &backend->d_cls_normed);
    clSetKernelArg(backend->kernel_head_bwd, 7, sizeof(cl_mem), &backend->d_token_embeds);
    clSetKernelArg(backend->kernel_head_bwd, 8, sizeof(cl_mem), &backend->d_cls_mean);
    clSetKernelArg(backend->kernel_head_bwd, 9, sizeof(cl_mem), &backend->d_cls_inv_std);
    clSetKernelArg(backend->kernel_head_bwd, 10, sizeof(cl_mem), &backend->d_normed_temp);
    clSetKernelArg(backend->kernel_head_bwd, 11, sizeof(int), &batch_size);
    clSetKernelArg(backend->kernel_head_bwd, 12, sizeof(int), &total_tokens);
    clSetKernelArg(backend->kernel_head_bwd, 13, sizeof(int), &embed_dim);
    clSetKernelArg(backend->kernel_head_bwd, 14, sizeof(int), &final_ln_gamma);
    clSetKernelArg(backend->kernel_head_bwd, 15, sizeof(int), &final_ln_beta);
    clSetKernelArg(backend->kernel_head_bwd, 16, sizeof(int), &head_weight);
    clSetKernelArg(backend->kernel_head_bwd, 17, sizeof(int), &head_bias);
    size_t global_head = batch_size;
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_head_bwd, 1, NULL, &global_head, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) return;

    for (int l = num_layers - 1; l >= 0; l--) {
        int ln2_gamma_off = offsets[3 + l * 8 + 3];
        int ln2_beta_off = bias_offsets[1 + l * 8 + 3];
        int ffn1_off = offsets[3 + l * 8 + 4];
        int ffn1_b_off = bias_offsets[1 + l * 8 + 4];
        int ffn2_off = offsets[3 + l * 8 + 5];
        int ffn2_b_off = bias_offsets[1 + l * 8 + 5];
        int ln1_gamma_off = offsets[3 + l * 8 + 0];
        int ln1_beta_off = bias_offsets[1 + l * 8 + 0];
        int qkv_off = offsets[3 + l * 8 + 1];
        int qkv_b_off = bias_offsets[1 + l * 8 + 1];
        int proj_off = offsets[3 + l * 8 + 2];
        int proj_b_off = bias_offsets[1 + l * 8 + 2];

        size_t layer_offset = l * batch_size * total_tokens * embed_dim * sizeof(float);

        clEnqueueCopyBuffer(backend->queue, backend->d_ln2_inputs, backend->d_layernorm_input,
            layer_offset, 0, batch_size * total_tokens * embed_dim * sizeof(float), 0, NULL, NULL);

        clSetKernelArg(backend->kernel_ffn_bwd, 0, sizeof(cl_mem), &current_grad);
        clSetKernelArg(backend->kernel_ffn_bwd, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_ffn_bwd, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_ffn_bwd, 3, sizeof(cl_mem), &backend->d_gradWeights);
        clSetKernelArg(backend->kernel_ffn_bwd, 4, sizeof(cl_mem), &backend->d_gradBiases);
        clSetKernelArg(backend->kernel_ffn_bwd, 5, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_ffn_bwd, 6, sizeof(cl_mem), &backend->d_pre_gelu);
        clSetKernelArg(backend->kernel_ffn_bwd, 7, sizeof(cl_mem), &backend->d_ffn_hidden);
        clSetKernelArg(backend->kernel_ffn_bwd, 8, sizeof(cl_mem), &backend->d_ffn_temp);
        clSetKernelArg(backend->kernel_ffn_bwd, 9, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_ffn_bwd, 10, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_ffn_bwd, 11, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_ffn_bwd, 12, sizeof(int), &mlp_dim);
        clSetKernelArg(backend->kernel_ffn_bwd, 13, sizeof(int), &ffn1_off);
        clSetKernelArg(backend->kernel_ffn_bwd, 14, sizeof(int), &ffn1_b_off);
        clSetKernelArg(backend->kernel_ffn_bwd, 15, sizeof(int), &ffn2_off);
        clSetKernelArg(backend->kernel_ffn_bwd, 16, sizeof(int), &ffn2_b_off);
        size_t global_ffn_bwd = batch_size * total_tokens;
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_ffn_bwd, 1, NULL, &global_ffn_bwd, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;

        clSetKernelArg(backend->kernel_layernorm_bwd, 0, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_layernorm_bwd, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_layernorm_bwd, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_layernorm_bwd, 3, sizeof(cl_mem), &backend->d_gradWeights);
        clSetKernelArg(backend->kernel_layernorm_bwd, 4, sizeof(cl_mem), &backend->d_gradBiases);
        clSetKernelArg(backend->kernel_layernorm_bwd, 5, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_layernorm_bwd, 6, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_layernorm_bwd, 7, sizeof(cl_mem), &backend->d_layernorm_mean);
        clSetKernelArg(backend->kernel_layernorm_bwd, 8, sizeof(cl_mem), &backend->d_layernorm_inv_std);
        clSetKernelArg(backend->kernel_layernorm_bwd, 9, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_layernorm_bwd, 10, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_layernorm_bwd, 11, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_layernorm_bwd, 12, sizeof(int), &ln2_gamma_off);
        clSetKernelArg(backend->kernel_layernorm_bwd, 13, sizeof(int), &ln2_beta_off);
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_layernorm_bwd, 1, NULL, &global_ffn_bwd, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;

        clSetKernelArg(backend->kernel_add, 0, sizeof(cl_mem), &current_grad);
        clSetKernelArg(backend->kernel_add, 1, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_add, 2, sizeof(cl_mem), &current_grad);
        clSetKernelArg(backend->kernel_add, 3, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_add, 4, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_add, 5, sizeof(int), &embed_dim);
        size_t global_add = batch_size * total_tokens * embed_dim;
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_add, 1, NULL, &global_add, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;

        float zero = 0.0f;
        clEnqueueFillBuffer(backend->queue, backend->d_grad_qkv, &zero, sizeof(float), 0,
            batch_size * total_tokens * 3 * embed_dim * sizeof(float), 0, NULL, NULL);

        clSetKernelArg(backend->kernel_attention_bwd, 0, sizeof(cl_mem), &current_grad);
        clSetKernelArg(backend->kernel_attention_bwd, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_attention_bwd, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_attention_bwd, 3, sizeof(cl_mem), &backend->d_gradWeights);
        clSetKernelArg(backend->kernel_attention_bwd, 4, sizeof(cl_mem), &backend->d_gradBiases);
        clSetKernelArg(backend->kernel_attention_bwd, 5, sizeof(cl_mem), &backend->d_attn_weights);
        clSetKernelArg(backend->kernel_attention_bwd, 6, sizeof(cl_mem), &backend->d_qkv_proj);
        clSetKernelArg(backend->kernel_attention_bwd, 7, sizeof(cl_mem), &backend->d_grad_qkv);
        clSetKernelArg(backend->kernel_attention_bwd, 8, sizeof(cl_mem), &backend->d_attn_temp);
        clSetKernelArg(backend->kernel_attention_bwd, 9, sizeof(cl_mem), &backend->d_scores_temp);
        clSetKernelArg(backend->kernel_attention_bwd, 10, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_attention_bwd, 11, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_attention_bwd, 12, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_attention_bwd, 13, sizeof(int), &num_heads);
        clSetKernelArg(backend->kernel_attention_bwd, 14, sizeof(int), &proj_off);
        clSetKernelArg(backend->kernel_attention_bwd, 15, sizeof(int), &proj_b_off);
        size_t global_attn_bwd = batch_size * total_tokens;
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_attention_bwd, 1, NULL, &global_attn_bwd, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;

        clEnqueueCopyBuffer(backend->queue, backend->d_ln1_outputs, backend->d_layernorm_input,
            layer_offset, 0, batch_size * total_tokens * embed_dim * sizeof(float), 0, NULL, NULL);

        clSetKernelArg(backend->kernel_qkv_proj_bwd, 0, sizeof(cl_mem), &backend->d_grad_qkv);
        clSetKernelArg(backend->kernel_qkv_proj_bwd, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_qkv_proj_bwd, 2, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_qkv_proj_bwd, 3, sizeof(cl_mem), &backend->d_gradWeights);
        clSetKernelArg(backend->kernel_qkv_proj_bwd, 4, sizeof(cl_mem), &backend->d_gradBiases);
        clSetKernelArg(backend->kernel_qkv_proj_bwd, 5, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_qkv_proj_bwd, 6, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_qkv_proj_bwd, 7, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_qkv_proj_bwd, 8, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_qkv_proj_bwd, 9, sizeof(int), &qkv_off);
        clSetKernelArg(backend->kernel_qkv_proj_bwd, 10, sizeof(int), &qkv_b_off);
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_qkv_proj_bwd, 1, NULL, &global_attn_bwd, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;

        clEnqueueCopyBuffer(backend->queue, backend->d_layer_inputs, backend->d_layernorm_input,
            layer_offset, 0, batch_size * total_tokens * embed_dim * sizeof(float), 0, NULL, NULL);

        clSetKernelArg(backend->kernel_layernorm_bwd, 0, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_layernorm_bwd, 1, sizeof(cl_mem), &backend->d_weights);
        clSetKernelArg(backend->kernel_layernorm_bwd, 2, sizeof(cl_mem), &backend->d_biases);
        clSetKernelArg(backend->kernel_layernorm_bwd, 3, sizeof(cl_mem), &backend->d_gradWeights);
        clSetKernelArg(backend->kernel_layernorm_bwd, 4, sizeof(cl_mem), &backend->d_gradBiases);
        clSetKernelArg(backend->kernel_layernorm_bwd, 5, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_layernorm_bwd, 6, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_layernorm_bwd, 7, sizeof(cl_mem), &backend->d_layernorm_mean);
        clSetKernelArg(backend->kernel_layernorm_bwd, 8, sizeof(cl_mem), &backend->d_layernorm_inv_std);
        clSetKernelArg(backend->kernel_layernorm_bwd, 9, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_layernorm_bwd, 10, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_layernorm_bwd, 11, sizeof(int), &embed_dim);
        clSetKernelArg(backend->kernel_layernorm_bwd, 12, sizeof(int), &ln1_gamma_off);
        clSetKernelArg(backend->kernel_layernorm_bwd, 13, sizeof(int), &ln1_beta_off);
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_layernorm_bwd, 1, NULL, &global_attn_bwd, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;

        clSetKernelArg(backend->kernel_add, 0, sizeof(cl_mem), &current_grad);
        clSetKernelArg(backend->kernel_add, 1, sizeof(cl_mem), &backend->d_layernorm_input);
        clSetKernelArg(backend->kernel_add, 2, sizeof(cl_mem), &current_grad);
        clSetKernelArg(backend->kernel_add, 3, sizeof(int), &batch_size);
        clSetKernelArg(backend->kernel_add, 4, sizeof(int), &total_tokens);
        clSetKernelArg(backend->kernel_add, 5, sizeof(int), &embed_dim);
        err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_add, 1, NULL, &global_add, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) return;

        if (l == 0) {
            clSetKernelArg(backend->kernel_embed_bwd, 0, sizeof(cl_mem), &current_grad);
            clSetKernelArg(backend->kernel_embed_bwd, 1, sizeof(cl_mem), &backend->d_weights);
            clSetKernelArg(backend->kernel_embed_bwd, 2, sizeof(cl_mem), &backend->d_gradWeights);
            clSetKernelArg(backend->kernel_embed_bwd, 3, sizeof(cl_mem), &backend->d_input);
            clSetKernelArg(backend->kernel_embed_bwd, 4, sizeof(int), &batch_size);
            clSetKernelArg(backend->kernel_embed_bwd, 5, sizeof(int), &seq_len);
            clSetKernelArg(backend->kernel_embed_bwd, 6, sizeof(int), &in_channels);
            clSetKernelArg(backend->kernel_embed_bwd, 7, sizeof(int), &embed_dim);
            clSetKernelArg(backend->kernel_embed_bwd, 8, sizeof(int), &total_tokens);
            clSetKernelArg(backend->kernel_embed_bwd, 9, sizeof(int), &offsets[0]);
            clSetKernelArg(backend->kernel_embed_bwd, 10, sizeof(int), &offsets[1]);
            clSetKernelArg(backend->kernel_embed_bwd, 11, sizeof(int), &offsets[2]);
            size_t global_embed_bwd = batch_size * total_tokens;
            err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_embed_bwd, 1, NULL, &global_embed_bwd, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS) return;
        }
    }
}

void vit_backend_adamw_update(VitBackend* backend, int batch_size, float lr, float beta1, float beta2, float epsilon, float weight_decay, int step) {
    if (!backend || !backend->initialized) return;
    int total = backend->total_weights + backend->total_biases;
    clSetKernelArg(backend->kernel_adamw_update, 0, sizeof(cl_mem), &backend->d_weights);
    clSetKernelArg(backend->kernel_adamw_update, 1, sizeof(cl_mem), &backend->d_biases);
    clSetKernelArg(backend->kernel_adamw_update, 2, sizeof(cl_mem), &backend->d_gradWeights);
    clSetKernelArg(backend->kernel_adamw_update, 3, sizeof(cl_mem), &backend->d_gradBiases);
    clSetKernelArg(backend->kernel_adamw_update, 4, sizeof(cl_mem), &backend->d_mWeights);
    clSetKernelArg(backend->kernel_adamw_update, 5, sizeof(cl_mem), &backend->d_vWeights);
    clSetKernelArg(backend->kernel_adamw_update, 6, sizeof(cl_mem), &backend->d_mBiases);
    clSetKernelArg(backend->kernel_adamw_update, 7, sizeof(cl_mem), &backend->d_vBiases);
    clSetKernelArg(backend->kernel_adamw_update, 8, sizeof(int), &batch_size);
    clSetKernelArg(backend->kernel_adamw_update, 9, sizeof(float), &lr);
    clSetKernelArg(backend->kernel_adamw_update, 10, sizeof(float), &beta1);
    clSetKernelArg(backend->kernel_adamw_update, 11, sizeof(float), &beta2);
    clSetKernelArg(backend->kernel_adamw_update, 12, sizeof(float), &epsilon);
    clSetKernelArg(backend->kernel_adamw_update, 13, sizeof(float), &weight_decay);
    clSetKernelArg(backend->kernel_adamw_update, 14, sizeof(int), &backend->total_weights);
    clSetKernelArg(backend->kernel_adamw_update, 15, sizeof(int), &backend->total_biases);
    clSetKernelArg(backend->kernel_adamw_update, 16, sizeof(int), &step);
    size_t global = total;
    clEnqueueNDRangeKernel(backend->queue, backend->kernel_adamw_update, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFlush(backend->queue);
}

void vit_backend_clip_gradients(VitBackend* backend, float max_norm) {
    if (!backend || !backend->initialized) return;
    float zero = 0.0f;
    clEnqueueFillBuffer(backend->queue, backend->d_grad_norm, &zero, sizeof(float), 0, sizeof(float), 0, NULL, NULL);
    int total = backend->total_weights + backend->total_biases;
    clSetKernelArg(backend->kernel_clip_grad_norm, 0, sizeof(cl_mem), &backend->d_gradWeights);
    clSetKernelArg(backend->kernel_clip_grad_norm, 1, sizeof(cl_mem), &backend->d_gradBiases);
    clSetKernelArg(backend->kernel_clip_grad_norm, 2, sizeof(cl_mem), &backend->d_grad_norm);
    clSetKernelArg(backend->kernel_clip_grad_norm, 3, sizeof(int), &backend->total_weights);
    clSetKernelArg(backend->kernel_clip_grad_norm, 4, sizeof(int), &backend->total_biases);
    clSetKernelArg(backend->kernel_clip_grad_norm, 5, sizeof(float), &max_norm);
    size_t global = total;
    size_t local = backend->workgroup_size;
    if (global < local) local = global;
    clEnqueueNDRangeKernel(backend->queue, backend->kernel_clip_grad_norm, 1, NULL, &global, &local, 0, NULL, NULL);
    float norm_val;
    clEnqueueReadBuffer(backend->queue, backend->d_grad_norm, CL_TRUE, 0, sizeof(float), &norm_val, 0, NULL, NULL);
    float scale = (norm_val > 0.0f) ? (max_norm / sqrtf(norm_val)) : 1.0f;
    if (scale < 1.0f) {
        clSetKernelArg(backend->kernel_scale_grads, 0, sizeof(cl_mem), &backend->d_gradWeights);
        clSetKernelArg(backend->kernel_scale_grads, 1, sizeof(cl_mem), &backend->d_gradBiases);
        clSetKernelArg(backend->kernel_scale_grads, 2, sizeof(cl_mem), &backend->d_grad_norm);
        clSetKernelArg(backend->kernel_scale_grads, 3, sizeof(int), &backend->total_weights);
        clSetKernelArg(backend->kernel_scale_grads, 4, sizeof(int), &backend->total_biases);
        clSetKernelArg(backend->kernel_scale_grads, 5, sizeof(float), &max_norm);
        clEnqueueNDRangeKernel(backend->queue, backend->kernel_scale_grads, 1, NULL, &global, NULL, 0, NULL, NULL);
        clFlush(backend->queue);
    }
}

void vit_backend_zero_gradients(VitBackend* backend) {
    if (!backend || !backend->initialized) return;
    float zero = 0.0f;
    clEnqueueFillBuffer(backend->queue, backend->d_gradWeights, &zero, sizeof(float), 0,
        backend->total_weights * sizeof(float), 0, NULL, NULL);
    clEnqueueFillBuffer(backend->queue, backend->d_gradBiases, &zero, sizeof(float), 0,
        backend->total_biases * sizeof(float), 0, NULL, NULL);
    clFlush(backend->queue);
}

void vit_backend_get_weights(VitBackend* backend, float* out) {
    if (!backend || !out) return;
    clFinish(backend->queue);
    clEnqueueReadBuffer(backend->queue, backend->d_weights, CL_TRUE, 0,
        backend->total_weights * sizeof(float), out, 0, NULL, NULL);
}

void vit_backend_get_biases(VitBackend* backend, float* out) {
    if (!backend || !out) return;
    clFinish(backend->queue);
    clEnqueueReadBuffer(backend->queue, backend->d_biases, CL_TRUE, 0,
        backend->total_biases * sizeof(float), out, 0, NULL, NULL);
}

void vit_backend_set_weights(VitBackend* backend, const float* weights) {
    if (!backend || !weights) return;
    clEnqueueWriteBuffer(backend->queue, backend->d_weights, CL_TRUE, 0,
        backend->total_weights * sizeof(float), weights, 0, NULL, NULL);
}

void vit_backend_set_biases(VitBackend* backend, const float* biases) {
    if (!backend || !biases) return;
    clEnqueueWriteBuffer(backend->queue, backend->d_biases, CL_TRUE, 0,
        backend->total_biases * sizeof(float), biases, 0, NULL, NULL);
}

int vit_backend_get_total_weights(VitBackend* backend) {
    return backend ? backend->total_weights : 0;
}

int vit_backend_get_total_biases(VitBackend* backend) {
    return backend ? backend->total_biases : 0;
}

float vit_backend_mpp_forward(VitBackend* backend, const int* mask_indices, const int* targets, int batch_size, int num_masked, int num_classes) {
    if (!backend || !backend->initialized) return 0.0f;
    if (!ensure_buffers(backend, batch_size)) return 0.0f;
    cl_int err;
    float zero = 0.0f;
    clEnqueueFillBuffer(backend->queue, backend->d_mpp_loss, &zero, sizeof(float), 0, sizeof(float), 0, NULL, NULL);
    cl_mem d_mask_indices = clCreateBuffer(backend->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        batch_size * num_masked * sizeof(int), (void*)mask_indices, &err);
    if (err != CL_SUCCESS) return 0.0f;
    cl_mem d_targets = clCreateBuffer(backend->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        batch_size * num_masked * sizeof(int), (void*)targets, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_mask_indices); return 0.0f; }
    clSetKernelArg(backend->kernel_mpp_forward, 0, sizeof(cl_mem), &backend->d_token_embeds);
    clSetKernelArg(backend->kernel_mpp_forward, 1, sizeof(cl_mem), &backend->d_mask_weights);
    clSetKernelArg(backend->kernel_mpp_forward, 2, sizeof(cl_mem), &backend->d_mask_biases);
    clSetKernelArg(backend->kernel_mpp_forward, 3, sizeof(cl_mem), &backend->d_mpp_logits);
    clSetKernelArg(backend->kernel_mpp_forward, 4, sizeof(cl_mem), &d_mask_indices);
    clSetKernelArg(backend->kernel_mpp_forward, 5, sizeof(cl_mem), &d_targets);
    clSetKernelArg(backend->kernel_mpp_forward, 6, sizeof(cl_mem), &backend->d_mpp_loss);
    clSetKernelArg(backend->kernel_mpp_forward, 7, sizeof(int), &batch_size);
    int total_tokens_mpp = backend->seq_len + 1;
    clSetKernelArg(backend->kernel_mpp_forward, 8, sizeof(int), &total_tokens_mpp);
    clSetKernelArg(backend->kernel_mpp_forward, 9, sizeof(int), &backend->embed_dim);
    clSetKernelArg(backend->kernel_mpp_forward, 10, sizeof(int), &backend->mpp_num_classes);
    clSetKernelArg(backend->kernel_mpp_forward, 11, sizeof(int), &num_masked);
    size_t global = batch_size * num_masked;
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_mpp_forward, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_mask_indices); clReleaseMemObject(d_targets); return 0.0f; }
    float loss = 0.0f;
    clEnqueueReadBuffer(backend->queue, backend->d_mpp_loss, CL_TRUE, 0, sizeof(float), &loss, 0, NULL, NULL);
    clReleaseMemObject(d_mask_indices);
    clReleaseMemObject(d_targets);
    return loss / (float)(batch_size * num_masked);
}

void vit_backend_mpp_backward(VitBackend* backend, const int* mask_indices, const int* targets, int batch_size, int num_masked, int num_classes, float loss_scale) {
    if (!backend || !backend->initialized) return;
    if (!ensure_buffers(backend, batch_size)) return;
    cl_int err;
    cl_mem d_mask_indices = clCreateBuffer(backend->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        batch_size * num_masked * sizeof(int), (void*)mask_indices, &err);
    if (err != CL_SUCCESS) return;
    cl_mem d_targets = clCreateBuffer(backend->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        batch_size * num_masked * sizeof(int), (void*)targets, &err);
    if (err != CL_SUCCESS) { clReleaseMemObject(d_mask_indices); return; }
    clSetKernelArg(backend->kernel_mpp_backward, 0, sizeof(cl_mem), &backend->d_mpp_logits);
    clSetKernelArg(backend->kernel_mpp_backward, 1, sizeof(cl_mem), &d_mask_indices);
    clSetKernelArg(backend->kernel_mpp_backward, 2, sizeof(cl_mem), &d_targets);
    clSetKernelArg(backend->kernel_mpp_backward, 3, sizeof(cl_mem), &backend->d_token_embeds);
    clSetKernelArg(backend->kernel_mpp_backward, 4, sizeof(cl_mem), &backend->d_mask_weights);
    clSetKernelArg(backend->kernel_mpp_backward, 5, sizeof(cl_mem), &backend->d_token_embeds);
    clSetKernelArg(backend->kernel_mpp_backward, 6, sizeof(cl_mem), &backend->d_grad_mask_weights);
    clSetKernelArg(backend->kernel_mpp_backward, 7, sizeof(cl_mem), &backend->d_grad_mask_biases);
    clSetKernelArg(backend->kernel_mpp_backward, 8, sizeof(int), &batch_size);
    int total_tokens_mpp_bwd = backend->seq_len + 1;
    clSetKernelArg(backend->kernel_mpp_backward, 9, sizeof(int), &total_tokens_mpp_bwd);
    clSetKernelArg(backend->kernel_mpp_backward, 10, sizeof(int), &backend->embed_dim);
    clSetKernelArg(backend->kernel_mpp_backward, 11, sizeof(int), &backend->mpp_num_classes);
    clSetKernelArg(backend->kernel_mpp_backward, 12, sizeof(int), &num_masked);
    clSetKernelArg(backend->kernel_mpp_backward, 13, sizeof(float), &loss_scale);
    size_t global = batch_size * num_masked;
    clEnqueueNDRangeKernel(backend->queue, backend->kernel_mpp_backward, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFlush(backend->queue);
    clReleaseMemObject(d_mask_indices);
    clReleaseMemObject(d_targets);
}

void vit_backend_get_gradients(VitBackend* backend, float* out) {
    if (!backend || !out) return;
    clFinish(backend->queue);
    clEnqueueReadBuffer(backend->queue, backend->d_gradWeights, CL_TRUE, 0,
        backend->total_weights * sizeof(float), out, 0, NULL, NULL);
}