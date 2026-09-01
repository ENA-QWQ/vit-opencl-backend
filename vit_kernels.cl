#define FLT_MAX_VAL 3.402823466e+38f

inline void atomic_add_float(volatile __global float* addr, float val) {
    union { unsigned int u; float f; } old, new_val;
    do {
        old.f = *addr;
        new_val.f = old.f + val;
    } while (atom_cmpxchg((volatile __global unsigned int*)addr, old.u, new_val.u) != old.u);
}

__kernel void vit_embed(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    int batch_size,
    int seq_len,
    int in_channels,
    int embed_dim,
    int total_tokens,
    int embed_weight_off,
    int cls_off,
    int pos_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    int b = idx / total_tokens;
    int t = idx % total_tokens;
    __global float* out_ptr = output + idx * embed_dim;
    if (t == 0) {
        for (int d = 0; d < embed_dim; d++) {
            out_ptr[d] = weights[cls_off + d];
        }
    } else {
        int patch_idx = t - 1;
        __global const float* in_ptr = input + b * seq_len * in_channels + patch_idx * in_channels;
        for (int d = 0; d < embed_dim; d++) {
            float sum = 0.0f;
            for (int c = 0; c < in_channels; c++) {
                sum += in_ptr[c] * weights[embed_weight_off + c * embed_dim + d];
            }
            out_ptr[d] = sum;
        }
    }
    for (int d = 0; d < embed_dim; d++) {
        out_ptr[d] += weights[pos_off + t * embed_dim + d];
    }
}

__kernel void vit_layernorm(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    __global float* mean_out,
    __global float* inv_std_out,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int gamma_off,
    int beta_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    __global const float* in = input + idx * embed_dim;
    __global float* out = output + idx * embed_dim;
    float mean = 0.0f;
    for (int d = 0; d < embed_dim; d++) mean += in[d];
    mean /= (float)embed_dim;
    float var = 0.0f;
    for (int d = 0; d < embed_dim; d++) {
        float diff = in[d] - mean;
        var += diff * diff;
    }
    var /= (float)embed_dim;
    float inv_std = 1.0f / sqrt(var + 1e-5f);
    mean_out[idx] = mean;
    inv_std_out[idx] = inv_std;
    for (int d = 0; d < embed_dim; d++) {
        out[d] = (in[d] - mean) * inv_std * weights[gamma_off + d] + biases[beta_off + d];
    }
}

__kernel void vit_qkv_proj(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int qkv_off,
    int qkv_b_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    __global const float* in = input + idx * embed_dim;
    __global float* out = output + idx * 3 * embed_dim;
    int three_ed = 3 * embed_dim;
    for (int j = 0; j < three_ed; j++) {
        float sum = 0.0f;
        for (int k = 0; k < embed_dim; k++) {
            sum += in[k] * weights[qkv_off + k * three_ed + j];
        }
        out[j] = sum + biases[qkv_b_off + j];
    }
}

__kernel void vit_attention(
    __global const float* qkv,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    __global float* attn_weights_out,
    __global float* attn_temp,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int num_heads,
    int proj_off,
    int proj_b_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    int b = idx / total_tokens;
    int t = idx % total_tokens;
    int head_dim = embed_dim / num_heads;
    float scale = 1.0f / sqrt((float)head_dim);
    __global const float* qkv_base = qkv + idx * 3 * embed_dim;
    __global const float* q_ptr = qkv_base;
    __global const float* k_ptr = qkv_base + embed_dim;
    __global const float* v_ptr = qkv_base + 2 * embed_dim;
    for (int h = 0; h < num_heads; h++) {
        float max_val = -FLT_MAX_VAL;
        for (int t2 = 0; t2 < total_tokens; t2++) {
            __global const float* k2 = qkv + (b * total_tokens + t2) * 3 * embed_dim + embed_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_ptr[h * head_dim + d] * k2[h * head_dim + d];
            }
            score *= scale;
            attn_weights_out[((b * num_heads + h) * total_tokens + t) * total_tokens + t2] = score;
            if (score > max_val) max_val = score;
        }
        float sum_exp = 0.0f;
        for (int t2 = 0; t2 < total_tokens; t2++) {
            int aw_idx = ((b * num_heads + h) * total_tokens + t) * total_tokens + t2;
            float e = exp(attn_weights_out[aw_idx] - max_val);
            attn_weights_out[aw_idx] = e;
            sum_exp += e;
        }
        float inv_sum = (sum_exp > 1e-12f) ? (1.0f / sum_exp) : 0.0f;
        for (int t2 = 0; t2 < total_tokens; t2++) {
            int aw_idx = ((b * num_heads + h) * total_tokens + t) * total_tokens + t2;
            attn_weights_out[aw_idx] *= inv_sum;
        }
    }
    __global float* attn_out = attn_temp + idx * embed_dim;
    for (int i = 0; i < embed_dim; i++) attn_out[i] = 0.0f;
    for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int t2 = 0; t2 < total_tokens; t2++) {
                float w = attn_weights_out[((b * num_heads + h) * total_tokens + t) * total_tokens + t2];
                __global const float* v2 = qkv + (b * total_tokens + t2) * 3 * embed_dim + 2 * embed_dim;
                val += w * v2[h * head_dim + d];
            }
            attn_out[h * head_dim + d] = val;
        }
    }
    __global float* out = output + idx * embed_dim;
    for (int d = 0; d < embed_dim; d++) {
        float sum = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            sum += attn_out[j] * weights[proj_off + j * embed_dim + d];
        }
        out[d] = sum + biases[proj_b_off + d];
    }
}

__kernel void vit_ffn(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    __global float* pre_gelu_out,
    __global float* hidden_out,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int mlp_dim,
    int ffn1_off,
    int ffn1_b_off,
    int ffn2_off,
    int ffn2_b_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    __global const float* in = input + idx * embed_dim;
    __global float* out = output + idx * embed_dim;
    __global float* pre_ptr = pre_gelu_out + idx * mlp_dim;
    __global float* hid_ptr = hidden_out + idx * mlp_dim;
    for (int i = 0; i < mlp_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            sum += in[j] * weights[ffn1_off + j * mlp_dim + i];
        }
        float val = sum + biases[ffn1_b_off + i];
        pre_ptr[i] = val;
        float x = val;
        float gelu = 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)));
        hid_ptr[i] = gelu;
    }
    for (int i = 0; i < embed_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < mlp_dim; j++) {
            sum += hid_ptr[j] * weights[ffn2_off + j * embed_dim + i];
        }
        out[i] = sum + biases[ffn2_b_off + i];
    }
}

__kernel void vit_head(
    __global const float* input,
    __global const float* weights,
    __global const float* biases,
    __global float* output,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int head_weight_off,
    int head_bias_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    __global const float* token = input + idx * embed_dim;
    float sum = 0.0f;
    for (int d = 0; d < embed_dim; d++) {
        sum += token[d] * weights[head_weight_off + d];
    }
    output[idx] = sum + biases[head_bias_off];
}

__kernel void vit_add(
    __global const float* a,
    __global const float* b,
    __global float* out,
    int batch_size,
    int total_tokens,
    int embed_dim
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens * embed_dim;
    if (idx >= total) return;
    out[idx] = a[idx] + b[idx];
}

__kernel void vit_clip_grad_norm(
    __global float* grad_weights,
    __global float* grad_biases,
    __global float* grad_norm_out,
    int total_weights,
    int total_biases,
    float max_norm
) {
    int idx = get_global_id(0);
    int total = total_weights + total_biases;
    if (idx >= total) return;
    float g = (idx < total_weights) ? grad_weights[idx] : grad_biases[idx - total_weights];
    float sq = g * g;
    __local float local_sum[256];
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    local_sum[lid] = sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) local_sum[lid] += local_sum[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) atomic_add_float(grad_norm_out, local_sum[0]);
}

__kernel void vit_scale_grads(
    __global float* grad_weights,
    __global float* grad_biases,
    __global const float* grad_norm,
    int total_weights,
    int total_biases,
    float max_norm
) {
    int idx = get_global_id(0);
    int total = total_weights + total_biases;
    if (idx >= total) return;
    float norm = sqrt(grad_norm[0]);
    float scale = (norm > max_norm) ? (max_norm / norm) : 1.0f;
    if (idx < total_weights) {
        grad_weights[idx] *= scale;
    } else {
        grad_biases[idx - total_weights] *= scale;
    }
}

__kernel void vit_adamw_update(
    __global float* weights,
    __global float* biases,
    __global const float* grad_weights,
    __global const float* grad_biases,
    __global float* m_weights,
    __global float* v_weights,
    __global float* m_biases,
    __global float* v_biases,
    int batch_size,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    int total_weights,
    int total_biases,
    int step
) {
    int idx = get_global_id(0);
    float bias_correction1 = 1.0f - pow(beta1, (float)(step + 1));
    float bias_correction2 = 1.0f - pow(beta2, (float)(step + 1));
    if (idx < total_weights) {
        float g = grad_weights[idx] / (float)batch_size;
        float w = weights[idx];
        weights[idx] = w - lr * weight_decay * w;
        float m = beta1 * m_weights[idx] + (1.0f - beta1) * g;
        float v = beta2 * v_weights[idx] + (1.0f - beta2) * g * g;
        m_weights[idx] = m;
        v_weights[idx] = v;
        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;
        weights[idx] -= lr * m_hat / (sqrt(v_hat) + epsilon);
    } else if (idx < total_weights + total_biases) {
        int b_idx = idx - total_weights;
        float g = grad_biases[b_idx] / (float)batch_size;
        float m = beta1 * m_biases[b_idx] + (1.0f - beta1) * g;
        float v = beta2 * v_biases[b_idx] + (1.0f - beta2) * g * g;
        m_biases[b_idx] = m;
        v_biases[b_idx] = v;
        float m_hat = m / bias_correction1;
        float v_hat = v / bias_correction2;
        biases[b_idx] -= lr * m_hat / (sqrt(v_hat) + epsilon);
    }
}

__kernel void vit_mpp_forward(
    __global const float* token_embeds,
    __global const float* mask_weights,
    __global const float* mask_biases,
    __global float* logits,
    __global const int* mask_indices,
    __global const int* targets,
    __global float* loss_out,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int num_classes,
    int num_masked
) {
    int idx = get_global_id(0);
    if (idx >= batch_size * num_masked) return;
    int b = idx / num_masked;
    int m = idx % num_masked;
    int token_idx = mask_indices[idx];
    __global const float* emb = token_embeds + (b * total_tokens + token_idx) * embed_dim;
    __global float* logit = logits + idx * num_classes;
    float max_val = -FLT_MAX_VAL;
    for (int c = 0; c < num_classes; c++) {
        float sum = 0.0f;
        for (int d = 0; d < embed_dim; d++) {
            sum += emb[d] * mask_weights[d * num_classes + c];
        }
        logit[c] = sum + mask_biases[c];
        if (logit[c] > max_val) max_val = logit[c];
    }
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        logit[c] = exp(logit[c] - max_val);
        sum_exp += logit[c];
    }
    float inv_sum = (sum_exp > 1e-12f) ? (1.0f / sum_exp) : 0.0f;
    for (int c = 0; c < num_classes; c++) {
        logit[c] *= inv_sum;
    }
    int target = targets[idx];
    float prob = logit[target];
    float loss = -log(max(prob, 1e-12f));
    atomic_add_float(loss_out, loss);
}

__kernel void vit_mpp_backward(
    __global const float* logits,
    __global const int* mask_indices,
    __global const int* targets,
    __global const float* token_embeds,
    __global const float* mask_weights,
    __global float* grad_token_embeds,
    __global float* grad_mask_weights,
    __global float* grad_mask_biases,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int num_classes,
    int num_masked,
    float loss_scale
) {
    int idx = get_global_id(0);
    if (idx >= batch_size * num_masked) return;
    int b = idx / num_masked;
    int m = idx % num_masked;
    int token_idx = mask_indices[idx];
    __global const float* logit = logits + idx * num_classes;
    __global const float* emb = token_embeds + (b * total_tokens + token_idx) * embed_dim;
    __global float* g_emb = grad_token_embeds + (b * total_tokens + token_idx) * embed_dim;
    int target = targets[idx];
    for (int c = 0; c < num_classes; c++) {
        float d = logit[c] * loss_scale;
        if (c == target) d -= loss_scale;
        if (d == 0.0f) continue;
        for (int d_i = 0; d_i < embed_dim; d_i++) {
            atomic_add_float(&grad_mask_weights[d_i * num_classes + c], d * emb[d_i]);
            atomic_add_float(&g_emb[d_i], d * mask_weights[d_i * num_classes + c]);
        }
        atomic_add_float(&grad_mask_biases[c], d);
    }
}

__kernel void vit_embed_bwd(
    __global const float* grad_in,
    __global const float* weights,
    __global float* grad_weights,
    __global const float* input,
    int batch_size,
    int seq_len,
    int in_channels,
    int embed_dim,
    int total_tokens,
    int embed_weight_off,
    int cls_off,
    int pos_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    int b = idx / total_tokens;
    int t = idx % total_tokens;
    int emb_off = idx * embed_dim;
    if (t == 0) {
        for (int d = 0; d < embed_dim; d++) {
            atomic_add_float(&grad_weights[cls_off + d], grad_in[emb_off + d]);
        }
        return;
    }
    int patch_idx = t - 1;
    __global const float* in_ptr = input + b * seq_len * in_channels + patch_idx * in_channels;
    for (int d = 0; d < embed_dim; d++) {
        float g = grad_in[emb_off + d];
        for (int c = 0; c < in_channels; c++) {
            atomic_add_float(&grad_weights[embed_weight_off + c * embed_dim + d], g * in_ptr[c]);
        }
    }
    for (int d = 0; d < embed_dim; d++) {
        atomic_add_float(&grad_weights[pos_off + t * embed_dim + d], grad_in[emb_off + d]);
    }
}

__kernel void vit_layernorm_bwd(
    __global const float* grad_out,
    __global const float* weights,
    __global const float* biases,
    __global float* grad_weights,
    __global float* grad_biases,
    __global float* grad_in,
    __global const float* input,
    __global const float* mean,
    __global const float* inv_std,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int gamma_off,
    int beta_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    __global const float* in = input + idx * embed_dim;
    __global const float* go = grad_out + idx * embed_dim;
    __global float* gi = grad_in + idx * embed_dim;
    float m = mean[idx];
    float istd = inv_std[idx];
    float sum_gw = 0.0f, sum_gwx = 0.0f;
    for (int d = 0; d < embed_dim; d++) {
        float x_hat = (in[d] - m) * istd;
        float gw = go[d] * weights[gamma_off + d];
        sum_gw += gw;
        sum_gwx += gw * x_hat;
    }
    float norm_factor = istd / (float)embed_dim;
    for (int d = 0; d < embed_dim; d++) {
        float x_hat = (in[d] - m) * istd;
        float gw = go[d] * weights[gamma_off + d];
        gi[d] = istd * (gw - norm_factor * (sum_gw + x_hat * sum_gwx));
        atomic_add_float(&grad_weights[gamma_off + d], go[d] * x_hat);
        atomic_add_float(&grad_biases[beta_off + d], go[d]);
    }
}

__kernel void vit_qkv_proj_bwd(
    __global const float* grad_qkv,
    __global const float* weights,
    __global const float* input,
    __global float* grad_weights,
    __global float* grad_biases,
    __global float* grad_in,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int qkv_off,
    int qkv_b_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    __global const float* gqkv = grad_qkv + idx * 3 * embed_dim;
    __global const float* in = input + idx * embed_dim;
    __global float* gi = grad_in + idx * embed_dim;
    int three_ed = 3 * embed_dim;
    for (int k = 0; k < embed_dim; k++) {
        float sum = 0.0f;
        for (int j = 0; j < three_ed; j++) {
            sum += gqkv[j] * weights[qkv_off + k * three_ed + j];
            atomic_add_float(&grad_weights[qkv_off + k * three_ed + j], gqkv[j] * in[k]);
        }
        gi[k] = sum;
    }
    for (int j = 0; j < three_ed; j++) {
        atomic_add_float(&grad_biases[qkv_b_off + j], gqkv[j]);
    }
}

__kernel void vit_attention_bwd(
    __global const float* grad_out,
    __global const float* weights,
    __global const float* biases,
    __global float* grad_weights,
    __global float* grad_biases,
    __global const float* attn_weights,
    __global const float* qkv,
    __global float* grad_qkv,
    __global float* attn_temp,
    __global float* scores_temp,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int num_heads,
    int proj_off,
    int proj_b_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    int b = idx / total_tokens;
    int t = idx % total_tokens;
    int head_dim = embed_dim / num_heads;
    __global const float* go = grad_out + idx * embed_dim;
    __global float* d_attn_out = attn_temp + idx * embed_dim;
    for (int d = 0; d < embed_dim; d++) {
        float sum = 0.0f;
        for (int j = 0; j < embed_dim; j++) {
            sum += go[j] * weights[proj_off + d * embed_dim + j];
        }
        d_attn_out[d] = sum;
    }
    __global const float* qkv_cur = qkv + idx * 3 * embed_dim;
    for (int d = 0; d < embed_dim; d++) {
        for (int j = 0; j < embed_dim; j++) {
            atomic_add_float(&grad_weights[proj_off + d * embed_dim + j], go[d] * qkv_cur[j]);
        }
        atomic_add_float(&grad_biases[proj_b_off + d], go[d]);
    }
    __global float* gqkv_self = grad_qkv + idx * 3 * embed_dim;
    __global float* d_scores = scores_temp + idx * total_tokens;
    for (int h = 0; h < num_heads; h++) {
        float dot_sum = 0.0f;
        for (int t2 = 0; t2 < total_tokens; t2++) {
            float w = attn_weights[((b * num_heads + h) * total_tokens + t) * total_tokens + t2];
            __global const float* v2 = qkv + (b * total_tokens + t2) * 3 * embed_dim + 2 * embed_dim;
            float dv_dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dv_dot += d_attn_out[h * head_dim + d] * v2[h * head_dim + d];
            }
            d_scores[t2] = w * dv_dot;
            dot_sum += d_scores[t2];
        }
        for (int t2 = 0; t2 < total_tokens; t2++) {
            float w = attn_weights[((b * num_heads + h) * total_tokens + t) * total_tokens + t2];
            d_scores[t2] = w * (d_scores[t2] - dot_sum);
        }
        float scale = 1.0f / sqrt((float)head_dim);
        for (int t2 = 0; t2 < total_tokens; t2++) {
            float ds = d_scores[t2] * scale;
            __global const float* k2 = qkv + (b * total_tokens + t2) * 3 * embed_dim + embed_dim;
            __global const float* v2 = qkv + (b * total_tokens + t2) * 3 * embed_dim + 2 * embed_dim;
            float w_attn = attn_weights[((b * num_heads + h) * total_tokens + t) * total_tokens + t2];
            for (int d = 0; d < head_dim; d++) {
                float dq = ds * k2[h * head_dim + d];
                float dk = ds * qkv_cur[h * head_dim + d];
                float dv = w_attn * d_attn_out[h * head_dim + d];
                atomic_add_float(&gqkv_self[h * head_dim + d], dq);
                atomic_add_float(&grad_qkv[(b * total_tokens + t2) * 3 * embed_dim + embed_dim + h * head_dim + d], dk);
                atomic_add_float(&grad_qkv[(b * total_tokens + t2) * 3 * embed_dim + 2 * embed_dim + h * head_dim + d], dv);
            }
        }
    }
}

__kernel void vit_ffn_bwd(
    __global const float* grad_out,
    __global const float* weights,
    __global const float* biases,
    __global float* grad_weights,
    __global float* grad_biases,
    __global float* grad_in,
    __global const float* input,
    __global const float* pre_gelu,
    __global const float* hidden,
    __global float* ffn_temp,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int mlp_dim,
    int ffn1_off,
    int ffn1_b_off,
    int ffn2_off,
    int ffn2_b_off
) {
    int idx = get_global_id(0);
    int total = batch_size * total_tokens;
    if (idx >= total) return;
    __global const float* go = grad_out + idx * embed_dim;
    __global float* gi = grad_in + idx * embed_dim;
    __global const float* in = input + idx * embed_dim;
    __global const float* pre = pre_gelu + idx * mlp_dim;
    __global const float* hid = hidden + idx * mlp_dim;
    __global float* d_hidden = ffn_temp + idx * mlp_dim;
    for (int j = 0; j < mlp_dim; j++) {
        float sum = 0.0f;
        for (int d = 0; d < embed_dim; d++) {
            sum += go[d] * weights[ffn2_off + j * embed_dim + d];
        }
        d_hidden[j] = sum;
    }
    for (int j = 0; j < mlp_dim; j++) {
        float x = pre[j];
        float tanh_val = tanh(0.79788456f * (x + 0.044715f * x * x * x));
        float gelu_deriv = 0.5f * (1.0f + tanh_val) +
                           0.5f * x * (1.0f - tanh_val * tanh_val) *
                           (0.79788456f + 0.10703222f * x * x);
        d_hidden[j] *= gelu_deriv;
    }
    for (int d = 0; d < embed_dim; d++) {
        for (int j = 0; j < mlp_dim; j++) {
            atomic_add_float(&grad_weights[ffn2_off + j * embed_dim + d], go[d] * hid[j]);
        }
        atomic_add_float(&grad_biases[ffn2_b_off + d], go[d]);
    }
    for (int j = 0; j < mlp_dim; j++) {
        atomic_add_float(&grad_biases[ffn1_b_off + j], d_hidden[j]);
        for (int i = 0; i < embed_dim; i++) {
            atomic_add_float(&grad_weights[ffn1_off + i * mlp_dim + j], d_hidden[j] * in[i]);
        }
    }
    for (int d = 0; d < embed_dim; d++) {
        float s = 0.0f;
        for (int j = 0; j < mlp_dim; j++) {
            s += d_hidden[j] * weights[ffn1_off + d * mlp_dim + j];
        }
        gi[d] = s;
    }
}

__kernel void vit_head_bwd(
    __global const float* grad_out,
    __global const float* weights,
    __global const float* biases,
    __global float* grad_weights,
    __global float* grad_biases,
    __global float* grad_input,
    __global const float* cls_normed,
    __global const float* cls_raw,
    __global const float* cls_mean,
    __global const float* cls_inv_std,
    __global float* normed_temp,
    int batch_size,
    int total_tokens,
    int embed_dim,
    int ln_gamma_off,
    int ln_beta_off,
    int head_weight_off,
    int head_bias_off
) {
    int b = get_global_id(0);
    if (b >= batch_size) return;
    float g = grad_out[b];
    atomic_add_float(&grad_biases[head_bias_off], g);
    __global const float* normed = cls_normed + b * embed_dim;
    for (int d = 0; d < embed_dim; d++) {
        atomic_add_float(&grad_weights[head_weight_off + d], g * normed[d]);
    }
    __global float* d_normed = normed_temp + b * embed_dim;
    for (int d = 0; d < embed_dim; d++) {
        d_normed[d] = g * weights[head_weight_off + d];
    }
    __global const float* raw = cls_raw + b * total_tokens * embed_dim;
    float m = cls_mean[b];
    float istd = cls_inv_std[b];
    float sum_gw = 0.0f, sum_gwx = 0.0f;
    for (int d = 0; d < embed_dim; d++) {
        float x_hat = (raw[d] - m) * istd;
        float gw = d_normed[d] * weights[ln_gamma_off + d];
        sum_gw += gw;
        sum_gwx += gw * x_hat;
    }
    float norm_factor = istd / (float)embed_dim;
    __global float* gi = grad_input + b * total_tokens * embed_dim;
    for (int d = 0; d < embed_dim; d++) {
        float x_hat = (raw[d] - m) * istd;
        float gw = d_normed[d] * weights[ln_gamma_off + d];
        gi[d] = istd * (gw - norm_factor * (sum_gw + x_hat * sum_gwx));
        atomic_add_float(&grad_weights[ln_gamma_off + d], d_normed[d] * x_hat);
        atomic_add_float(&grad_biases[ln_beta_off + d], d_normed[d]);
    }
}