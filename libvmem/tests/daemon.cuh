// shared by the tests that run bin/vmemd (1d, 2d): start it with a config text, stop it
// always with --debug, so the log holds the broker's counts after every message for a human to read
#pragma once

#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>

// socket paths hold at most 107 characters, so test files live in /tmp
inline const char *SOCK = "/tmp/vmemd-test.sock", *LOG = "/tmp/vmemd-test.log", *CONF = "/tmp/vmemd-test.toml";

// starts bin/vmemd with this config text; returns its pid, or -1 if it exited (exit code in *code)
inline pid_t start(const std::string &conf, const char *config_path = CONF, int *code = nullptr) {
    if (FILE *f = fopen(CONF, "w")) fputs(conf.c_str(), f), fclose(f);
    pid_t pid = fork();
    if (pid == 0) {
        if (!freopen("/dev/null", "w", stderr)) _exit(127);
        execl("bin/vmemd", "vmemd", "--socket", SOCK, "--log", LOG, "--config", config_path, "--debug", (char *)nullptr);
        _exit(127);
    }
    sockaddr_un a = {};
    a.sun_family = AF_UNIX;
    strcpy(a.sun_path, SOCK);
    for (int i = 0; i < 100; i++) {   // up to 1 s for the socket to come up
        int st;
        if (waitpid(pid, &st, WNOHANG) == pid) {
            if (code) *code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
            return -1;
        }
        int s = socket(AF_UNIX, SOCK_SEQPACKET, 0);
        bool up = connect(s, (sockaddr *)&a, sizeof a) == 0;
        close(s);
        if (up) return pid;
        usleep(10000);
    }
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    return -1;
}

inline void stop(pid_t pid) {
    kill(pid, SIGTERM);
    waitpid(pid, nullptr, 0);
    unlink(SOCK);
}
