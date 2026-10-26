// Native API diagnostic, not a biological network or production episode.
// k=a=b=d=0 removes intrinsic currents: consecutive voltage samples must
// differ by external_current / C per millisecond. Exercise both setter forms.
#include <carlsim.h>
#include <neuron_monitor.h>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) throw std::invalid_argument("Expected output CSV path");
    std::ofstream output(argv[1]);
    if (!output) throw std::runtime_error("Cannot create response CSV");
    output << "solver,capacitance,phase,neuron,sample,requested_current,voltage,recovery\n";
    output << std::setprecision(9);
    const float capacitances[] = {100.f, 200.f};
    const std::vector<float> vector_current = {0.f, 25.f, 100.f};
    for (int solver = 0; solver < 2; ++solver) {
        const std::string label = solver == 0 ? "euler1" : "rk4_5";
        CARLsim sim("current_response_" + label, CPU_MODE, SILENT, 0, 10);
        std::vector<int> groups;
        for (float capacitance : capacitances) {
            const int group = sim.createGroup("C" + std::to_string(capacitance), 3, EXCITATORY_NEURON);
            sim.setNeuronParameters(group, capacitance, 0.f, -60.f, -40.f, 0.f, 0.f, 1000.f, -60.f, 0.f);
            groups.push_back(group);
        }
        sim.setConductances(true);
        sim.setIntegrationMethod(solver == 0 ? FORWARD_EULER : RUNGE_KUTTA4, solver == 0 ? 1 : 5);
        std::vector<NeuronMonitor*> monitors;
        for (int group : groups) monitors.push_back(sim.setNeuronMonitor(group, "NULL"));
        sim.setupNetwork();
        for (int phase = 0; phase < 3; ++phase) {
            for (std::size_t group = 0; group < groups.size(); ++group) {
                if (phase == 0) sim.setExternalCurrent(groups[group], vector_current);
                else sim.setExternalCurrent(groups[group], phase == 1 ? 50.f : 0.f);
                monitors[group]->startRecording();
            }
            sim.runNetwork(0, 3, false);
            for (std::size_t group = 0; group < groups.size(); ++group) {
                monitors[group]->stopRecording();
                const auto voltage = monitors[group]->getVectorV();
                const auto recovery = monitors[group]->getVectorU();
                if (voltage.size() < 3 || recovery.size() < 3)
                    throw std::runtime_error("Native monitor omitted neurons");
                for (int neuron = 0; neuron < 3; ++neuron) {
                    if (voltage[neuron].size() != 3 || recovery[neuron].size() != 3)
                        throw std::runtime_error("Native monitor omitted millisecond samples");
                    const float current = phase == 0 ? vector_current[neuron] : phase == 1 ? 50.f : 0.f;
                    for (int sample = 0; sample < 3; ++sample)
                        output << label << ',' << capacitances[group] << ',' << phase << ',' << neuron << ','
                               << sample << ',' << current << ',' << voltage[neuron][sample] << ','
                               << recovery[neuron][sample] << '\n';
                }
            }
        }
    }
    return output ? 0 : 1;
}
