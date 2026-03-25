#include "solver.hpp"
#include "simulation_oracle.hpp"
#include "openalex_oracle.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <filesystem>
#include "gurobi_c++.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        cerr << "Usage (Simulation): " << argv[0] << " <edge.csv> <q_node_string_id> <k_target> [out.csv]" << endl;
        cerr << "Usage (OpenAlex):   " << argv[0] << " openalex <q_node_string_id> <k_target> [out.csv]" << endl;
        return 1;
    }

    string mode_arg = argv[1];
    bool is_openalex = (mode_arg == "openalex");

    string q_node_str = argv[2];
    int k_target = stoi(argv[3]);
    string output_filename = (argc == 5) ? argv[4] : "";

    try {
        GRBEnv env = GRBEnv(true);
        env.set("OutputFlag", "0");
        env.set("Method", "1"); 
        env.start();

        unique_ptr<IGraphOracle> oracle_ptr;
        auto t_io_start = chrono::high_resolution_clock::now();

        if (is_openalex) {
            cout << "==================================================" << endl;
            cout << "K-DENSEST NEIGHBORHOOD (OPENALEX LIVE API)" << endl;
            cout << "==================================================" << endl;
            oracle_ptr = make_unique<OpenAlexOracle>();
        } else {
            cout << "==================================================" << endl;
            cout << "K-DENSEST NEIGHBORHOOD (LAZY API SIMULATOR)" << endl;
            cout << "==================================================" << endl;
            
            auto sim_oracle = make_unique<SimulationOracle>();
            ifstream infile(mode_arg); // mode_arg is edge.csv here
            if (!infile.is_open()) {
                cerr << "Error: Could not open file " << mode_arg << endl;
                return 1;
            }

            string line;
            size_t edge_count = 0;
            if (getline(infile, line)) { 
                while (getline(infile, line)) {
                    size_t comma_pos = line.find(',');
                    if (comma_pos != string::npos) {
                        string u_str = line.substr(0, comma_pos);
                        string v_str = line.substr(comma_pos + 1);
                        if (!v_str.empty() && v_str.back() == '\r') v_str.pop_back();
                        sim_oracle->add_db_edge(u_str, v_str);
                        edge_count++;
                    }
                }
            }
            infile.close();
            auto t_io_end = chrono::high_resolution_clock::now();
            cout << "[" << get_timestamp() << "] Sim DB Loaded     | Edges: " << edge_count << " | IO Time: " 
                 << fixed << setprecision(2) << chrono::duration<double>(t_io_end - t_io_start).count() << "s" << endl;
            cout << "--------------------------------------------------" << endl;
            
            oracle_ptr = std::move(sim_oracle);
        }

        int q_node_int = oracle_ptr->mapper.get_or_create_id(q_node_str);

        // Run solver with the polymorphic oracle
        FullBranchAndPriceSolver solver(*oracle_ptr, q_node_int, k_target, env);
        auto [best_nodes, final_density] = solver.solve();
        
        // --- PRINT STATS (Identical to your provided code) ---
        cout << "==================================================" << endl;
        cout << "OPTIMIZATION STATISTICS" << endl;
        cout << "==================================================" << endl;
        cout << left << setw(25) << "B&B Nodes Explored" << ": " << solver.stats.total_bb_nodes << endl;
        cout << left << setw(25) << "Total LP Solves" << ": " << solver.stats.total_lp_solves << endl;
        cout << left << setw(25) << "Columns Generated" << ": " << solver.stats.total_columns_added << endl;
        cout << left << setw(25) << "BQP Cuts Added" << ": " << solver.stats.total_cuts_added << endl;
        cout << left << setw(25) << "API Queries Made" << ": " << oracle_ptr->queries_made << endl;
        cout << left << setw(25) << "Unique Nodes Mapped" << ": " << oracle_ptr->mapper.size() << endl;
        cout << "--------------------------------------------------" << endl;
        cout << "TIMING BREAKDOWN" << endl;
        cout << left << setw(25) << "Model Sync Time" << ": " << fixed << setprecision(3) << solver.stats.t_sync << "s" << endl;
        cout << left << setw(25) << "Gurobi LP Time" << ": " << fixed << setprecision(3) << solver.stats.t_lp_solve << "s" << endl;
        cout << left << setw(25) << "Pricing Time" << ": " << fixed << setprecision(3) << solver.stats.t_pricing << "s" << endl;
        cout << left << setw(25) << "Separation Time" << ": " << fixed << setprecision(3) << solver.stats.t_separation << "s" << endl;
        cout << "--------------------------------------------------" << endl;
        cout << left << setw(25) << "Total Solver Time" << ": " << fixed << setprecision(3) << solver.stats.t_total << "s" << endl;
        cout << "==================================================" << endl;
        cout << "FINAL SOLUTION" << endl;
        cout << "==================================================" << endl;
        cout << left << setw(25) << "Density" << ": " << fixed << setprecision(6) << final_density << endl;
        cout << left << setw(25) << "Size" << ": " << best_nodes.size() << endl;

        // Execute Optional Output Logic
        if (!output_filename.empty()) {
            filesystem::path out_path(output_filename);
            if (out_path.has_parent_path() && !filesystem::exists(out_path.parent_path())) {
                filesystem::create_directories(out_path.parent_path());
            }
            ofstream outfile(output_filename);
            if (outfile.is_open()) {
                outfile << "node_id\n";
                for (int node : best_nodes) outfile << oracle_ptr->mapper.get_str(node) << "\n";
                outfile.close();
                cout << "[" << get_timestamp() << "] Solution saved to " << output_filename << endl;
            } else {
                cerr << "[" << get_timestamp() << "] Error: Could not write to " << output_filename << endl;
            }
        } else {
            cout << "Nodes:" << endl;
            for (int node : best_nodes) cout << oracle_ptr->mapper.get_str(node) << " ";
            cout << endl;
        }

    } catch(const GRBException& e) {
        cerr << "[" << get_timestamp() << "] Gurobi Error code = " << e.getErrorCode() << "\n" << e.getMessage() << endl;
    } catch(const std::exception& e) {
        cerr << "[" << get_timestamp() << "] Exception: " << e.what() << endl;
    }

    return 0;
}