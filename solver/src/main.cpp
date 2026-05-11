#include "simulation_oracle.hpp"
#include "openalex_oracle.hpp"
#include "solver.hpp"
#include "average_degree_solver.hpp"
#include "bfs_solver.hpp"
#include "subgraph_quality.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <filesystem>
#include <string>
#include <stdexcept>
#include "gurobi_c++.h"

using namespace std;

void print_usage(const char *prog_name)
{
    cout << "Usage: " << prog_name << " [OPTIONS]\n\n"
         << "Required Arguments:\n"
         << "  --mode <sim|openalex>     Mode of operation (local CSV simulation or live API)\n"
         << "  --query <node_id>         The string ID of the target query node\n\n"
         << "Conditionally Required:\n"
         << "  --input <edge.csv>        Path to the local edge list (REQUIRED if --mode sim)\n\n"
         << "Optional I/O:\n"
         << "  --output <out.csv>        Path to save the resulting subgraph node IDs\n\n"
         << "  --compute-qualities       Compute final density/internal-edge metrics (may issue extra oracle queries)\n\n"
         << "Solver Hyperparameters (Defaults shown):\n"
         << "  --time-limit <float>      Max Branch-and-Bound time in seconds; -1 disables (default: -1)\n"
         << "  --node-limit <int>        Max Branch-and-Bound nodes to explore; -1 disables (default: -1)\n"
         << "  --max-in-edges <int>      Max incoming edges to fetch per node (default: 0)\n"
         << "  --gap-tol <float>         Early stopping relative gap tolerance; -1 disables (default: -1)\n"
         << "  --dinkelbach-iter <int>   Max Dinkelbach iterations; -1 disables (default: -1)\n"
         << "  --cg-batch-frac <float>   Fraction of active set to price per iteration (default: 1.0)\n"
         << "  --cg-min-batch <int>      Minimum columns to add per pricing round (default: 0)\n"
         << "  --cg-max-batch <int>      Maximum columns to add per pricing round (default: 50)\n"
         << "  --tol <float>             Numerical tolerance for zero-checks (default: 1e-6)\n"
         << "  --k <int>                 Target subgraph size (REQUIRED for --bp; k >= 2)\n"
         << "  --kappa <int>             Edge-connectivity threshold for --bp (default: 0; 0 disables)\n"
         << "  --bfs-depth <int>         Max BFS depth for --bfs (default: 1)\n\n"
         << "Solver Variants:\n"
         << "  --bp                      Run BP; uses --k and --kappa\n"
         << "  --avgdeg                  Run exact query-anchored avgdeg; no method-specific CLI parameter\n"
         << "  --bfs                     Run BFS; uses --bfs-depth\n"
         << "  --help, -h                Print this help menu and exit\n";
}

// Loads a two-column (source,target) CSV into the oracle. Returns the edge count.
static size_t load_edge_csv(SimulationOracle &oracle, const string &path)
{
    ifstream infile(path);
    if (!infile.is_open())
        throw runtime_error("Could not open file " + path);

    string line;
    size_t edge_count = 0;
    getline(infile, line); // skip header
    while (getline(infile, line))
    {
        size_t comma_pos = line.find(',');
        if (comma_pos != string::npos)
        {
            string u = line.substr(0, comma_pos);
            string v = line.substr(comma_pos + 1);
            if (!v.empty() && v.back() == '\r')
                v.pop_back();
            oracle.add_db_edge(u, v);
            ++edge_count;
        }
    }
    return edge_count;
}

int main(int argc, char *argv[])
{
    // ---------------------------------------------------------
    // 1. Default Parameter Initialization
    // ---------------------------------------------------------
    string mode = "";
    string input_file = "";
    string query_node = "";
    int k = -1;
    string output_file = "";
    bool compute_qualities = false;

    double time_limit = -1.0;
    int node_limit = -1;
    int max_in_edges = 0;
    double gap_tol = -1.0;
    int dinkelbach_iter = -1;
    double cg_batch_frac = 1.0;
    int cg_min_batch = 0;
    int cg_max_batch = 50;
    double tol = 1e-6;
    int kappa = 0;

    // Baseline parameters
    bool run_bp = false;
    bool run_avgdeg = false;
    bool run_bfs = false;
    int bfs_depth = 1;
    bool k_provided = false;
    bool kappa_provided = false;
    bool bfs_depth_provided = false;

    // ---------------------------------------------------------
    // 2. Command Line Argument Parsing
    // ---------------------------------------------------------
    if (argc == 1)
    {
        print_usage(argv[0]);
        return 1;
    }

    try
    {
        for (int i = 1; i < argc; ++i)
        {
            string arg = argv[i];

            if (arg == "--help" || arg == "-h")
            {
                print_usage(argv[0]);
                return 0;
            }

            if (arg == "--compute-qualities")
            {
                compute_qualities = true;
                continue;
            }

            if (arg == "--bp")
            {
                run_bp = true;
                continue;
            }

            if (arg == "--avgdeg")
            {
                run_avgdeg = true;
                continue;
            }

            if (arg == "--bfs")
            {
                run_bfs = true;
                continue;
            }

            if (i + 1 >= argc)
            {
                cerr << "Error: Argument " << arg << " requires a value.\n";
                return 1;
            }

            if (arg == "--mode")
                mode = argv[++i];
            else if (arg == "--input")
                input_file = argv[++i];
            else if (arg == "--query")
                query_node = argv[++i];
            else if (arg == "--output")
                output_file = argv[++i];
            else if (arg == "--k")
            {
                k = stoi(argv[++i]);
                k_provided = true;
            }
            else if (arg == "--time-limit")
                time_limit = stod(argv[++i]);
            else if (arg == "--node-limit")
                node_limit = stoi(argv[++i]);
            else if (arg == "--max-in-edges")
                max_in_edges = stoi(argv[++i]);
            else if (arg == "--gap-tol")
                gap_tol = stod(argv[++i]);
            else if (arg == "--dinkelbach-iter")
                dinkelbach_iter = stoi(argv[++i]);
            else if (arg == "--cg-batch-frac")
                cg_batch_frac = stod(argv[++i]);
            else if (arg == "--cg-min-batch")
                cg_min_batch = stoi(argv[++i]);
            else if (arg == "--cg-max-batch")
                cg_max_batch = stoi(argv[++i]);
            else if (arg == "--tol")
                tol = stod(argv[++i]);
            else if (arg == "--kappa")
            {
                kappa = stoi(argv[++i]);
                kappa_provided = true;
            }
            else if (arg == "--bfs-depth")
            {
                bfs_depth = stoi(argv[++i]);
                bfs_depth_provided = true;
            }
            else
            {
                cerr << "Error: Unknown argument '" << arg << "'\n";
                print_usage(argv[0]);
                return 1;
            }
        }
    }
    catch (const invalid_argument &)
    {
        cerr << "Error: Invalid numeric type provided for one of the arguments.\n";
        return 1;
    }
    catch (const out_of_range &)
    {
        cerr << "Error: Numeric value provided is out of range.\n";
        return 1;
    }

    // ---------------------------------------------------------
    // 3. Validation
    // ---------------------------------------------------------
    if (mode != "sim" && mode != "openalex")
    {
        cerr << "Error: --mode must be either 'sim' or 'openalex'.\n";
        return 1;
    }
    if (query_node.empty())
    {
        cerr << "Error: --query is required.\n";
        return 1;
    }
    int solver_count = (run_bp ? 1 : 0) + (run_avgdeg ? 1 : 0) + (run_bfs ? 1 : 0);
    if (solver_count > 1)
    {
        cerr << "Error: Specify at most one solver variant.\n";
        return 1;
    }

    const bool uses_k = run_bp || solver_count == 0;
    if (uses_k && k < 2)
    {
        cerr << "Error: --k must be specified and >= 2 for this solver.\n";
        return 1;
    }
    if (!uses_k && k_provided)
    {
        cerr << "Error: --k is only valid for --bp.\n";
        return 1;
    }
    if (mode == "sim" && input_file.empty())
    {
        cerr << "Error: --input edge list is required when using 'sim' mode.\n";
        return 1;
    }
    if (kappa < 0)
    {
        cerr << "Error: --kappa must be >= 0.\n";
        return 1;
    }
    if (!run_bp && solver_count != 0 && kappa_provided)
    {
        cerr << "Error: --kappa is only valid for --bp.\n";
        return 1;
    }
    if (!run_bfs && bfs_depth_provided)
    {
        cerr << "Error: --bfs-depth is only valid for --bfs.\n";
        return 1;
    }

    // ---------------------------------------------------------
    // 4. Engine Execution
    // ---------------------------------------------------------
    try
    {
        GRBEnv env = GRBEnv(true);
        env.set("OutputFlag", "0");
        env.set("Method", "1");
        env.start();

        unique_ptr<IGraphOracle> oracle_ptr;

        if (mode == "openalex")
        {
            cout << "==================================================" << endl;
            cout << "K-DENSEST NEIGHBORHOOD (OPENALEX LIVE API)" << endl;
            cout << "==================================================" << endl;
            oracle_ptr = make_unique<OpenAlexOracle>(max_in_edges);
        }
        else
        {
            cout << "==================================================" << endl;
            cout << "K-DENSEST NEIGHBORHOOD (LAZY API SIMULATOR)" << endl;
            cout << "==================================================" << endl;

            auto sim_oracle = make_unique<SimulationOracle>(max_in_edges);
            auto t_io_start = chrono::high_resolution_clock::now();
            size_t edge_count = load_edge_csv(*sim_oracle, input_file);
            auto t_io_end = chrono::high_resolution_clock::now();

            cout << "[" << get_timestamp() << "] Sim DB Loaded     | Edges: " << edge_count
                 << " | IO Time: " << fixed << setprecision(2)
                 << chrono::duration<double>(t_io_end - t_io_start).count() << "s" << endl;
            cout << "--------------------------------------------------" << endl;

            oracle_ptr = std::move(sim_oracle);
        }

        int q_node_int = oracle_ptr->mapper.get_or_create_id(query_node);

        vector<int> best_nodes;

        if (run_bp)
        {
            cout << "==================================================" << endl;
            cout << "[" << get_timestamp() << "] Running BP solver..." << endl;
            cout << "==================================================" << endl;

            auto t_start = chrono::high_resolution_clock::now();
            FullBranchAndPriceSolver solver(*oracle_ptr, q_node_int, k, env,
                                            tol, node_limit, time_limit, gap_tol,
                                            dinkelbach_iter, cg_batch_frac, cg_min_batch, cg_max_batch, kappa);

            auto bp_result = solver.solve();
            best_nodes.assign(bp_result.first.begin(), bp_result.first.end());
            auto t_end = chrono::high_resolution_clock::now();
            cout << left << setw(25) << "API Queries Made" << ": " << oracle_ptr->queries_made << endl;
            cout << left << setw(25) << "Unique Nodes Mapped" << ": " << oracle_ptr->mapper.size() << endl;
            cout << "--------------------------------------------------" << endl;
            cout << left << setw(25) << "Total Solver Time" << ": " << fixed << setprecision(3)
                 << chrono::duration<double>(t_end - t_start).count() << "s" << endl;
        }
        else if (run_avgdeg)
        {
            cout << "==================================================" << endl;
            cout << "[" << get_timestamp() << "] Running Avgdeg solver..." << endl;
            cout << "==================================================" << endl;

            auto t_start = chrono::high_resolution_clock::now();
            AverageDegreeSolver baseline_solver(oracle_ptr.get());

            vector<int> res = baseline_solver.solve(q_node_int, -1);
            auto t_end = chrono::high_resolution_clock::now();

            best_nodes = res;

            cout << left << setw(25) << "API Queries Made" << ": " << oracle_ptr->queries_made << endl;
            cout << left << setw(25) << "Unique Nodes Mapped" << ": " << oracle_ptr->mapper.size() << endl;
            cout << "--------------------------------------------------" << endl;
            cout << left << setw(25) << "Total Solver Time" << ": " << fixed << setprecision(3)
                 << chrono::duration<double>(t_end - t_start).count() << "s" << endl;
        }
        else if (run_bfs)
        {
            cout << "==================================================" << endl;
            cout << "[" << get_timestamp() << "] Running BFS solver..." << endl;
            cout << "==================================================" << endl;

            auto t_start = chrono::high_resolution_clock::now();
            BFSSolver bfs_solver(oracle_ptr.get());
            vector<int> res = bfs_solver.solve(q_node_int, bfs_depth);
            auto t_end = chrono::high_resolution_clock::now();

            best_nodes = res;

            cout << left << setw(25) << "API Queries Made" << ": " << oracle_ptr->queries_made << endl;
            cout << left << setw(25) << "Unique Nodes Mapped" << ": " << oracle_ptr->mapper.size() << endl;
            cout << "--------------------------------------------------" << endl;
            cout << left << setw(25) << "Total Solver Time" << ": " << fixed << setprecision(3)
                 << chrono::duration<double>(t_end - t_start).count() << "s" << endl;
        }
        else
        {
            FullBranchAndPriceSolver solver(*oracle_ptr, q_node_int, k, env,
                                            tol, node_limit, time_limit, gap_tol,
                                            dinkelbach_iter, cg_batch_frac, cg_min_batch, cg_max_batch, kappa);

            auto bp_result = solver.solve();
            best_nodes.assign(bp_result.first.begin(), bp_result.first.end());

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
            cout << left << setw(25) << "Network Wait Time" << ": " << fixed << setprecision(3) << oracle_ptr->cumulative_network_time << "s" << endl;
            cout << "--------------------------------------------------" << endl;
            cout << left << setw(25) << "Total Solver Time" << ": " << fixed << setprecision(3) << solver.stats.t_total << "s" << endl;
        }

        cout << "==================================================" << endl;
        cout << "FINAL SOLUTION" << endl;
        cout << "==================================================" << endl;
        int solve_queries_made = oracle_ptr->queries_made;
        if (compute_qualities)
        {
            SubgraphQualities final_metrics = compute_subgraph_qualities(best_nodes, oracle_ptr.get());
            cout << left << setw(25) << "Average Degree Density" << ": " << fixed << setprecision(6) << final_metrics.avg_degree_density << endl;
            cout << left << setw(25) << "Avg Total Int Degree" << ": " << fixed << setprecision(6) << final_metrics.avg_total_internal_degree << endl;
            cout << left << setw(25) << "Edge Density (Target)" << ": " << fixed << setprecision(6) << final_metrics.edge_density << endl;
            cout << left << setw(25) << "Internal Edges" << ": " << final_metrics.num_edges << endl;
            cout << left << setw(25) << "Boundary Out Edges" << ": " << final_metrics.boundary_edges_out << endl;
            cout << left << setw(25) << "Outgoing Conductance" << ": " << fixed << setprecision(6) << final_metrics.outgoing_conductance << endl;
            cout << left << setw(25) << "Expansion" << ": " << fixed << setprecision(6) << final_metrics.expansion << endl;
            cout << left << setw(25) << "Weak Components" << ": " << final_metrics.weak_components << endl;
            cout << left << setw(25) << "Largest Weak Comp Ratio" << ": " << fixed << setprecision(6) << final_metrics.weak_component_ratio << endl;
            cout << left << setw(25) << "Reciprocity" << ": " << fixed << setprecision(6) << final_metrics.reciprocity << endl;
            cout << left << setw(25) << "Size" << ": " << final_metrics.num_nodes << endl;
            cout << left << setw(25) << "Quality Extra Queries" << ": " << (oracle_ptr->queries_made - solve_queries_made) << endl;
        }
        else
        {
            cout << left << setw(25) << "Average Degree Density" << ": skipped" << endl;
            cout << left << setw(25) << "Edge Density (Target)" << ": skipped" << endl;
            cout << left << setw(25) << "Internal Edges" << ": skipped" << endl;
            cout << left << setw(25) << "Size" << ": " << best_nodes.size() << endl;
        }

        if (!output_file.empty())
        {
            filesystem::path out_path(output_file);
            if (out_path.has_parent_path() && !filesystem::exists(out_path.parent_path()))
                filesystem::create_directories(out_path.parent_path());

            ofstream outfile(output_file);
            if (outfile.is_open())
            {
                outfile << "node_id\n";
                for (int node : best_nodes)
                    outfile << oracle_ptr->mapper.get_str(node) << "\n";
                outfile.close();
                cout << "[" << get_timestamp() << "] Solution saved to " << output_file << endl;
            }
            else
            {
                cerr << "[" << get_timestamp() << "] Error: Could not write to " << output_file << endl;
            }
        }
        else
        {
            cout << "Nodes:" << endl;
            for (int node : best_nodes)
                cout << oracle_ptr->mapper.get_str(node) << " ";
            cout << endl;
        }
    }
    catch (const GRBException &e)
    {
        cerr << "[" << get_timestamp() << "] Gurobi Error code = " << e.getErrorCode() << "\n"
             << e.getMessage() << endl;
    }
    catch (const std::exception &e)
    {
        cerr << "[" << get_timestamp() << "] Exception: " << e.what() << endl;
    }

    return 0;
}
