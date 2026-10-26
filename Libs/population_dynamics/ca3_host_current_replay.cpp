// Host-current replay of the pinned CA3 retrieval main, without CARLsim.
// snn_manager.cpp: SNNinit seeds drand48(10); generateGroupRuntime calls
// resetNeuron once per regular Izhikevich unit, consuming nine draws even
// when parameter standard deviations are zero. No Poisson/homeostasis here.
// The source's nine engines AND distribution caches survive all intervals.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kGroups = 8;
constexpr int kDurationMs = 1001;
constexpr int kResetDrawsPerUnit = 9;
constexpr long kCarlsimSeed = 10;
constexpr float kSetCurrent = 10000.0f;
unsigned long seedArgument(const char* raw) {
    const std::string value(raw);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("invalid noise seed");
    size_t used = 0;
    const unsigned long seed = std::stoul(value, &used);
    if (used != value.size() || seed < 1 || seed > 2147483646UL)
        throw std::runtime_error("noise seed outside qualified engine support");
    return seed;
}
// Pyramidal, AAC, Basket, CCK, Bistratified, Ivy, MFAO, QuadD.
std::array<std::default_random_engine, 9> engines;
using Lognormal = std::lognormal_distribution<float>;
std::array<Lognormal, 9> distributions = {{
    Lognormal(4, 1.5), Lognormal(4, 1.25), Lognormal(4, 1.75),
    Lognormal(4, 1), Lognormal(4, 1), Lognormal(4, 1.75),
    Lognormal(4, 1.25), Lognormal(4, 1.25), Lognormal(4, 1)
}};

bool cue(int t) {
    return (520 <= t && t < 540) || (750 <= t && t < 770)
        || (980 <= t && t < 1000);
}
}

int main(int argc, char** argv) {
    try {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                      "IEEE float32 required");
        const uint32_t endian = 1;
        if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
            throw std::runtime_error("little-endian host required");
        const bool block = argc >= 2 && std::string(argv[1]) == "--noise-block";
        unsigned long circuitSeed = kCarlsimSeed;
        unsigned long uniformSeed = kCarlsimSeed;
        unsigned long engineSeed = std::default_random_engine::default_seed;
        const bool identity = argc >= 2 && std::string(argv[argc-1]) == "--identity";
        int end = argc - (identity ? 1 : 0);
        int durationMs = kDurationMs;
        if (end >= 3 && std::string(argv[end-2]) == "--duration-ms") {
            const std::string value(argv[end-1]);
            if (value != "1001" && value != "1210")
                throw std::runtime_error("unsupported retrieval duration");
            durationMs = std::stoi(value);
            end -= 2;
        }
        if (block) {
            if (end != 4 && !(end == 6 && std::string(argv[4]) == "--circuit-seed"))
                throw std::runtime_error("noise block requires uniform and common engine seeds");
            uniformSeed = seedArgument(argv[2]);
            engineSeed = seedArgument(argv[3]);
            if (end == 6) circuitSeed = seedArgument(argv[5]);
            if (std::default_random_engine::max() != 2147483646UL)
                throw std::runtime_error("unqualified default engine seed support");
        } else if (end != 1) {
            throw std::runtime_error("unexpected argument");
        }
        if (identity) {
            std::cout << "{\"compiler\":\"" << __VERSION__
                      << "\",\"libstdcxx_date\":" << __GLIBCXX__
                      << ",\"default_engine_seed\":"
                      << std::default_random_engine::default_seed
                      << ",\"carlsim_seed\":" << circuitSeed << ",\"reset_draws_per_unit\":9"
                      << ",\"explicit_noise_block\":" << (block ? "true" : "false")
                      << ",\"uniform_seed\":" << uniformSeed
                      << ",\"common_engine_seed\":" << engineSeed
                      << ",\"duration_ms\":" << durationMs
                      << ",\"post_seed_discard_draws_per_unit\":" << (block ? 0 : 9)
                      << "}" << std::endl;
            return 0;
        }
        std::array<int, kGroups> starts, sizes;
        int units = 0;
        for (int g = 0; g < kGroups; ++g) {
            if (!(std::cin >> starts[g] >> sizes[g]) || starts[g] < 0
                    || sizes[g] <= 0 || starts[g] > 1000000 || sizes[g] > 1000000)
                throw std::runtime_error("invalid population map");
            units += sizes[g];
        }
        if (units > 1000000) throw std::runtime_error("unsupported unit count");
        std::vector<bool> covered(units, false);
        for (int g = 0; g < kGroups; ++g) {
            if (starts[g] + sizes[g] > units)
                throw std::runtime_error("population outside map");
            for (int j = starts[g]; j < starts[g] + sizes[g]; ++j) {
                if (covered[j]) throw std::runtime_error("overlapping population map");
                covered[j] = true;
            }
        }
        int command_count;
        if (!(std::cin >> command_count) || command_count < 0 || command_count > 1000000)
            throw std::runtime_error("invalid command count");
        std::vector<std::vector<int>> commands(durationMs);
        int previous = -1;
        for (int i = 0; i < command_count; ++i) {
            int t, unit;
            if (!(std::cin >> t >> unit) || t < previous || !cue(t)
                    || unit < starts[0] || unit >= starts[0] + sizes[0])
                throw std::runtime_error("invalid ordered cue command");
            commands[t].push_back(unit);
            previous = t;
        }
        std::string extra;
        if (std::cin >> extra) throw std::runtime_error("trailing input");
        srand48(static_cast<long>(uniformSeed));
        if (block) {
            for (auto& engine : engines) engine.seed(engineSeed);
        } else {
            for (int64_t i = 0; i < int64_t(kResetDrawsPerUnit) * units; ++i)
                (void)drand48();
        }
        std::vector<float> currents(units, 0.0f);
        for (int t = 0; t < durationMs; ++t) {
            const bool recall = cue(t);
            for (int g = 0; g < (recall ? 1 : kGroups); ++g) {
                const int stream = recall ? 8 : g;
                for (int j = starts[g]; j < starts[g] + sizes[g]; ++j) {
                    float value = distributions[stream](engines[stream]) * (float)drand48();
                    if (!recall && value >= kSetCurrent) value = kSetCurrent;
                    currents[j] = value;
                }
            }
            // Source assignments replace currents; repeated assignments do not add.
            for (int unit : commands[t]) currents[unit] = kSetCurrent;
            if (std::fwrite(currents.data(), sizeof(float), units, stdout) != size_t(units))
                throw std::runtime_error("output write failed");
        }
        if (std::fflush(stdout)) throw std::runtime_error("output flush failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CA3_HOST_CURRENT_REPLAY: " << error.what() << std::endl;
        return 2;
    }
}
