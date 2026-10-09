// action runner: check a .vm file (python-like calls on one vmem::Manager), then run it line by line
//
//   program.cuh  values, statements, the parsed program
//   calls.cuh    what each call does (table: name -> argument shape, result kind, code), fill/check kernels
//   parser.cuh   one pass over the file: syntax, names, kinds
//   log.cuh      log files, --stable, the block format
//   run.cuh      runs a program, writes the logs
//   main.cu      command line
#include "run.cuh"

#include <cstdio>
#include <string>

static int usage(const char *argv0) {
    fprintf(stderr, "usage: %s [-v] [-d] [--stable] [--pid] [--parse] [-o logdir] file.vm\n"
                    "  -v        also print the log to stdout\n"
                    "  -d        also write <logdir>/<name>.debug.log with the state after every line\n"
                    "  --stable  replace addresses and times so two runs can be diffed\n"
                    "  --pid     name logs <name>.<pid>.log, for many runners on one file\n"
                    "  --parse   only check the file\n"
                    "  -o dir    log folder (default logs)\n", argv0);
    return 2;
}

int main(int argc, char **argv) {
    runner::Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-v") o.verbose = true;
        else if (a == "-d") o.debug = true;
        else if (a == "--stable") o.stable = true;
        else if (a == "--pid") o.pid = true;
        else if (a == "--parse") o.parse_only = true;
        else if (a == "-o" && i + 1 < argc) o.logdir = argv[++i];
        else if (!a.starts_with('-') && o.file.empty()) o.file = a;
        else return usage(argv[0]);
    }
    return o.file.empty() ? usage(argv[0]) : runner::run(o);
}
