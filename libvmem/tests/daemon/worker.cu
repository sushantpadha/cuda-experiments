// worker: one tenant of the multi-process tests, started by workers_check.py (make stress, make test-workers-*)
//   bin/worker --workload churn|gemm|image --name NAME [--backend vmem|vmem-local|cuda|uvm|raw] [--socket PATH]
//              [--ops N] [--seed S] [--n MATRIX_N] [--image-mib MIB] [--images K] [--passes P] [--first-base B]
//              [--resident] [--data DIR]
// churn: random calls with checks after each (the stress test); gemm: one matrix multiply; image: sharpen on images
// vmem talks to the vmemd at --socket; vmem-local, cuda, uvm and raw (plain cudaMallocManaged, no libvmem, no hints;
// gemm and image only) run without a broker, as baselines
// prints "hint moved|noop|refused UNIX_TIME" after every image evict/prefetch, so a tracer can place hints in time
// protocol with the runner: prints READY once it is connected, waits for a line on stdin, runs, prints one RESULT line
#include "../workload.cuh"

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string name = "worker", socket, data = "tests/data", backend = "vmem", workload;
    int n = 4096, images = 4, passes = 1, first_base = 0, ops = 2000;
    unsigned seed = 1;
    size_t image_mib = 256;
    bool resident = false, bad_arg = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        bool more = i + 1 < argc;
        if (a == "--resident") resident = true;
        else if (a == "--name" && more) name = argv[++i];
        else if (a == "--workload" && more) workload = argv[++i];
        else if (a == "--backend" && more) backend = argv[++i];
        else if (a == "--socket" && more) socket = argv[++i];
        else if (a == "--ops" && more) ops = atoi(argv[++i]);
        else if (a == "--seed" && more) seed = atoi(argv[++i]);
        else if (a == "--n" && more) n = atoi(argv[++i]);
        else if (a == "--image-mib" && more) image_mib = atoi(argv[++i]);
        else if (a == "--images" && more) images = atoi(argv[++i]);
        else if (a == "--passes" && more) passes = atoi(argv[++i]);
        else if (a == "--first-base" && more) first_base = atoi(argv[++i]);
        else if (a == "--data" && more) data = argv[++i];
        else bad_arg = true;
    }
    bool daemon = backend == "vmem", gemm = workload == "gemm", churn = workload == "churn";
    raw = backend == "raw";
    if (bad_arg || (!churn && !gemm && workload != "image") || (daemon && socket.empty()) || (raw && churn) ||
        (!daemon && !raw && backend != "vmem-local" && backend != "cuda" && backend != "uvm") || n % TILE) {
        fprintf(stderr, "usage: worker --workload churn|gemm|image --name NAME [--backend vmem|vmem-local|cuda|uvm|raw] [--socket PATH] "
                        "[--ops N] [--seed S] [--n MATRIX_N (multiple of %d)] [--image-mib MIB] [--images K] [--passes P] "
                        "[--first-base B] [--resident] [--data DIR]\n", TILE);
        return 2;
    }
    std::vector<Base> bases = workload != "image" ? std::vector<Base>() : load_bases(data);
    if (workload == "image" && bases.empty()) { fprintf(stderr, "no .pgm images in %s (run tests/data/make_images.py)\n", data.c_str()); return 2; }

    int rc = 0, nulls = 0;
    double t = 0;
    unsigned long long total = 0;
    try {
        unsetenv("VMEM_BACKEND");
        unsetenv("VMEM_DAEMON");
        CHECK(cudaFree(0));
        log_hints = !churn;
        vmem::Kind kind = backend == "cuda" ? vmem::Kind::Cuda : backend == "uvm" ? vmem::Kind::Uvm : vmem::Kind::Vmem;
        if (!raw) vmem::init({.backend = kind, .daemon = daemon ? socket.c_str() : nullptr, .name = name.c_str()});
        if (daemon && vmem::get().info().link != vmem::Link::Connected) {
            printf("FAIL: no link to vmemd at %s\n", socket.c_str());
            rc = 2;
        } else {
            const size_t W = 16384, H = (image_mib << 20) / W;
            if (churn) printf("%s: churn, %d calls, seed %u\n", name.c_str(), ops, seed);
            else if (gemm) printf("%s: gemm, matrices %dx%d, 3 x %zu MiB\n", name.c_str(), n, n, ((size_t)n * n * 4) >> 20);
            else printf("%s: image, %d images of %zu MiB (%zux%zu), %d passes, first base %d%s\n", name.c_str(), images,
                        (W * H) >> 20, W, H, passes, first_base, resident ? ", resident" : "");
            printf("READY\n");
            char line[64];
            if (!fgets(line, sizeof line, stdin)) return 2;   // the runner went away

            double t0 = now();
            if (churn) rc |= churn_phase(ops, seed, nulls);
            else if (gemm) rc |= matmul_phase(n, &total);
            else rc |= sharpen_phase(resident, images, W, H, bases, total, first_base, passes);
            t = now() - t0;
        }
    } catch (const vmem::Error &e) {
        printf("FAIL: %s (%s)\n", e.what(), vmem::status_name(e.status));
        rc = 2;
    }
    if (!raw) vmem::shutdown();
    printf("RESULT %s %d %llx %.2f %d %d\n", name.c_str(), rc, total, t, hints_ok, hints_no);
    return rc;
}
