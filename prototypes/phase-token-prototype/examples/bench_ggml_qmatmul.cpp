#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

namespace {

constexpr int64_t GEMM_M = 1024;
constexpr int64_t GEMM_N = 1024;
constexpr int64_t GEMM_K = 1024;
constexpr int64_t GEMV_M = 1;
constexpr int64_t GEMV_N = 4096;
constexpr int64_t GEMV_K = 4096;
constexpr int WARMUP = 10;
constexpr int MEASURED = 100;

struct Format {
    const char * name;
    ggml_type type;
};

struct Shape {
    const char * mode;
    int64_t m;
    int64_t n;
    int64_t k;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

bool contains(const char * haystack, const char * needle) {
    return haystack != nullptr && std::strstr(haystack, needle) != nullptr;
}

ggml_backend_t init_metal_backend() {
    ggml_backend_load_all_from_path("/opt/homebrew/Cellar/ggml/0.9.11/libexec");

    ggml_backend_dev_t fallback = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = ggml_backend_reg_name(reg);
        const char * dev_name = ggml_backend_dev_name(dev);
        const char * desc = ggml_backend_dev_description(dev);
        const auto type = ggml_backend_dev_type(dev);
        std::fprintf(stderr, "ggml device[%zu]: reg=%s name=%s desc=%s type=%d\n",
                     i,
                     reg_name ? reg_name : "",
                     dev_name ? dev_name : "",
                     desc ? desc : "",
                     static_cast<int>(type));

        if (contains(reg_name, "Metal") || contains(reg_name, "MTL") ||
            contains(dev_name, "Metal") || contains(dev_name, "MTL") ||
            contains(desc, "Metal")) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend != nullptr) {
                return backend;
            }
        }
        if (fallback == nullptr &&
            (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU)) {
            fallback = dev;
        }
    }

    if (fallback != nullptr) {
        ggml_backend_t backend = ggml_backend_dev_init(fallback, nullptr);
        if (backend != nullptr) {
            return backend;
        }
    }

    fail("could not initialize a ggml Metal/GPU backend");
}

std::vector<float> make_weights(int64_t rows, int64_t cols) {
    std::vector<float> values(rows * cols);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            values[r * cols + c] = 1.0f;
        }
    }
    return values;
}

std::vector<float> make_activations(int64_t m, int64_t k) {
    std::vector<float> values(m * k);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            values[row * k + col] = static_cast<float>((col % 17) - 8) / 16.0f;
        }
    }
    return values;
}

double percentile(const std::vector<double> & sorted, double p) {
    const size_t idx = static_cast<size_t>(std::llround((sorted.size() - 1) * p));
    return sorted[idx];
}

std::vector<Format> selected_formats(int argc, char ** argv, int start) {
    std::vector<Format> all = {
        {"Q4_0", GGML_TYPE_Q4_0},
        {"Q4_1", GGML_TYPE_Q4_1},
        {"Q5_0", GGML_TYPE_Q5_0},
        {"Q5_1", GGML_TYPE_Q5_1},
        {"Q8_0", GGML_TYPE_Q8_0},
        {"Q8_1", GGML_TYPE_Q8_1},
        {"Q2K", GGML_TYPE_Q2_K},
        {"Q3K", GGML_TYPE_Q3_K},
        {"Q4K", GGML_TYPE_Q4_K},
        {"Q5K", GGML_TYPE_Q5_K},
        {"Q6K", GGML_TYPE_Q6_K},
        {"Q8K", GGML_TYPE_Q8_K},
    };
    if (argc <= start) {
        return all;
    }

    std::vector<Format> out;
    std::string raw = argv[start];
    size_t pos = 0;
    while (pos <= raw.size()) {
        size_t comma = raw.find(',', pos);
        std::string item = raw.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        for (const auto & format : all) {
            std::string name = format.name;
            std::string lower = name;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            std::transform(item.begin(), item.end(), item.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (item == lower || item == name) {
                out.push_back(format);
                break;
            }
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    if (out.empty()) {
        fail("no recognized quantization formats");
    }
    return out;
}

void run_one(ggml_backend_t backend, ggml_backend_t cpu_backend, const Shape & shape, const Format & format) {
    if (ggml_quantize_requires_imatrix(format.type)) {
        fail(std::string(format.name) + " requires an importance matrix in this ggml build");
    }

    ggml_init_params params = {};
    params.mem_size = 64ull * 1024ull * 1024ull;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        fail("ggml_init failed");
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, format.type, shape.k, shape.n);
    ggml_tensor * act = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape.k, shape.m);
    ggml_tensor * out = ggml_mul_mat(ctx, weights, act);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        std::printf("ggml_%s: %s not supported by %s\n\n", shape.mode, format.name,
                    ggml_backend_name(backend));
        return;
    }

    ggml_backend_t backends[] = {backend, cpu_backend};
    ggml_backend_sched_t sched =
        ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    if (sched == nullptr) {
        ggml_free(ctx);
        fail("ggml_backend_sched_new failed");
    }
    ggml_backend_sched_set_tensor_backend(sched, weights, backend);
    ggml_backend_sched_set_tensor_backend(sched, act, backend);
    ggml_backend_sched_set_tensor_backend(sched, out, backend);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        fail("ggml_backend_sched_alloc_graph failed");
    }

    std::vector<float> w_f32 = make_weights(shape.n, shape.k);
    std::vector<uint8_t> w_quant(ggml_nbytes(weights));
    const size_t quantized = ggml_quantize_chunk(
        format.type, w_f32.data(), w_quant.data(), 0, shape.n, shape.k, nullptr);
    if (quantized == 0 || quantized > w_quant.size()) {
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        fail("ggml_quantize_chunk failed");
    }
    std::vector<float> x = make_activations(shape.m, shape.k);

    ggml_backend_tensor_set(weights, w_quant.data(), 0, w_quant.size());
    ggml_backend_tensor_set(act, x.data(), 0, x.size() * sizeof(float));

    for (int i = 0; i < WARMUP; ++i) {
        const ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            fail(std::string("warmup failed: ") + ggml_status_to_string(status));
        }
        ggml_backend_sched_synchronize(sched);
    }

    std::vector<double> samples;
    samples.reserve(MEASURED);
    for (int i = 0; i < MEASURED; ++i) {
        auto start = std::chrono::steady_clock::now();
        const ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            fail(std::string("compute failed: ") + ggml_status_to_string(status));
        }
        ggml_backend_sched_synchronize(sched);
        auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double>(end - start).count());
    }

    std::sort(samples.begin(), samples.end());
    double mean = 0.0;
    for (double sample : samples) {
        mean += sample;
    }
    mean /= samples.size();
    const double flops = 2.0 * static_cast<double>(shape.m) * shape.n * shape.k;
    const double weight_bytes = static_cast<double>(w_quant.size());

    std::printf("ggml_%s: %s A=%lldx%lld B=%lldx%lld -> Y=%lldx%lld backend=%s\n",
                shape.mode, format.name,
                static_cast<long long>(shape.m), static_cast<long long>(shape.k),
                static_cast<long long>(shape.k), static_cast<long long>(shape.n),
                static_cast<long long>(shape.m), static_cast<long long>(shape.n),
                ggml_backend_name(backend));
    std::printf("dispatches: %d measured, %d warmup\n", MEASURED, WARMUP);
    std::printf("mean_dispatch_time_us: %.3f\n", mean * 1.0e6);
    std::printf("p50_dispatch_time_us: %.3f\n", percentile(samples, 0.50) * 1.0e6);
    std::printf("p90_dispatch_time_us: %.3f\n", percentile(samples, 0.90) * 1.0e6);
    std::printf("min_dispatch_time_us: %.3f\n", samples.front() * 1.0e6);
    std::printf("max_dispatch_time_us: %.3f\n", samples.back() * 1.0e6);
    std::printf("effective_gflops: %.6f\n", flops / mean / 1.0e9);
    std::printf("effective_tflops: %.6f\n", flops / mean / 1.0e12);
    std::printf("packed_weight_bandwidth_gb_s: %.6f\n\n", weight_bytes / mean / 1.0e9);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "gemm";
        Shape shape = {};
        int format_arg = 2;
        if (mode == "gemm") {
            shape = {"Gemm", GEMM_M, GEMM_N, GEMM_K};
        } else if (mode == "gemv") {
            shape = {"Gemv", GEMV_M, GEMV_N, GEMV_K};
        } else {
            format_arg = 1;
            shape = {"Gemm", GEMM_M, GEMM_N, GEMM_K};
        }

        ggml_time_init();
        ggml_backend_t backend = init_metal_backend();
        ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (cpu_backend == nullptr) {
            ggml_backend_free(backend);
            fail("could not initialize ggml CPU backend");
        }
        std::fprintf(stderr, "selected ggml backend: %s\n", ggml_backend_name(backend));
        for (const auto & format : selected_formats(argc, argv, format_arg)) {
            run_one(backend, cpu_backend, shape, format);
        }
        ggml_backend_free(cpu_backend);
        ggml_backend_free(backend);
        ggml_quantize_free();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
