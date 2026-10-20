// worker: one tenant of the multi-process stress test. runs the stress.cu workload (matrix multiply, then sharpen
// on images) through libvmem on the vmem backend, against a running vmemd. started by workers_check.py: make test-workers
//   bin/worker --name NAME --socket PATH [--n MATRIX_N] [--image-mib MIB] [--images K] [--first-base B] [--resident] [--data DIR]
// protocol with the runner: prints READY once it is connected, waits for a line on stdin, runs, prints one RESULT line
#include "../workload.cuh"

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string name = "worker", socket, data = "tests/data";
    int n = 4096, images = 4, first_base = 0;
    size_t image_mib = 256;
    bool resident = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        bool more = i + 1 < argc;
        if (a == "--resident") resident = true;
        else if (a == "--name" && more) name = argv[++i];
        else if (a == "--socket" && more) socket = argv[++i];
        else if (a == "--n" && more) n = atoi(argv[++i]);
        else if (a == "--image-mib" && more) image_mib = atoi(argv[++i]);
        else if (a == "--images" && more) images = atoi(argv[++i]);
        else if (a == "--first-base" && more) first_base = atoi(argv[++i]);
        else if (a == "--data" && more) data = argv[++i];
        else socket.clear();
    }
    if (socket.empty() || n % TILE) {
        fprintf(stderr, "usage: worker --name NAME --socket PATH [--n MATRIX_N (multiple of %d)] [--image-mib MIB] [--images K] "
                        "[--first-base B] [--resident] [--data DIR]\n", TILE);
        return 2;
    }
    std::vector<Base> bases = load_bases(data);
    if (bases.empty()) { fprintf(stderr, "no .pgm images in %s (run tests/data/make_images.py)\n", data.c_str()); return 2; }

    int rc = 0;
    double t_mm = 0, t_img = 0;
    unsigned long long total = 0;
    try {
        unsetenv("VMEM_BACKEND");
        unsetenv("VMEM_DAEMON");
        CHECK(cudaFree(0));
        vmem::init({.backend = vmem::Kind::Vmem, .daemon = socket.c_str(), .name = name.c_str()});
        if (vmem::get().info().link != vmem::Link::Connected) {
            printf("FAIL: no link to vmemd at %s\n", socket.c_str());
            rc = 2;
        } else {
            const size_t W = 16384, H = (image_mib << 20) / W;
            printf("%s: matrices %dx%d, %d images of %zu MiB (%zux%zu), first base %d%s\n", name.c_str(), n, n, images,
                   (W * H) >> 20, W, H, first_base, resident ? ", resident" : "");
            printf("READY\n");
            char line[64];
            if (!fgets(line, sizeof line, stdin)) return 2;   // the runner went away

            double t0 = now();
            rc |= matmul_phase(n);
            double t1 = now();
            rc |= sharpen_phase(resident, images, W, H, bases, total, first_base);
            t_mm = t1 - t0;
            t_img = now() - t1;
        }
    } catch (const vmem::Error &e) {
        printf("FAIL: %s (%s)\n", e.what(), vmem::status_name(e.status));
        rc = 2;
    }
    vmem::shutdown();
    printf("RESULT %s %d %llx %.2f %.2f %d %d\n", name.c_str(), rc, total, t_mm, t_img, hints_ok, hints_no);
    return rc;
}
