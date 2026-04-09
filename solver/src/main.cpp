#include "solver.hpp"
#include "simulation_oracle.hpp"
#include "openalex_oracle.hpp"
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
         << "  --query <node_id>         The string ID of the target query node\n"
         << "  --k <int>                 The target subgraph size (k >= 2)\n\n"
         << "Conditionally Required:\n"
         << "  --input <edge.csv>        Path to the local edge list (REQUIRED if --mode sim)\n\n"
         << "Optional I/O:\n"
         << "  --output <out.csv>        Path to save the resulting subgraph node IDs\n\n"
         << "Solver Hyperparameters (Defaults shown):\n"
         << "  --time-limit <float>      Max Branch-and-Bound time in seconds (default: 60.0)\n"
         << "  --node-limit <int>        Max Branch-and-Bound nodes to explore (default: 100000)\n"
         << "  --max-in-edges <int>      Max incoming edges to fetch per node (default: 1500)\n"
         << "  --gap-tol <float>         Early stopping relative gap tolerance (default: 1e-4)\n"
         << "  --dinkelbach-iter <int>   Max Dinkelbach (fractional programming) iterations (default: 50)\n"
         << "  --cg-batch-frac <float>   Fraction of active set to price per iteration (default: 0.1)\n"
         << "  --cg-min-batch <int>      Minimum columns to add per pricing round (default: 5)\n"
         << "  --cg-max-batch <int>      Maximum columns to add per pricing round (default: 50)\n"
         << "  --tol <float>             Numerical tolerance for zero-checks (default: 1e-6)\n"
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

    double time_limit = 60.0;
    int node_limit = 100000;
    int max_in_edges = 1500;
    double gap_tol = 1e-4;
    int dinkelbach_iter = 50;
    double cg_batch_frac = 0.1;
    int cg_min_batch = 5;
    int cg_max_batch = 50;
    double tol = 1e-6;

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
                k = stoi(argv[++i]);
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
    if (k < 2)
    {
        cerr << "Error: --k must be specified and >= 2.\n";
        return 1;
    }
    if (mode == "sim" && input_file.empty())
    {
        cerr << "Error: --input edge list is required when using 'sim' mode.\n";
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

        FullBranchAndPriceSolver solver(*oracle_ptr, q_node_int, k, env,
                                        tol, node_limit, time_limit, gap_tol,
                                        dinkelbach_iter, cg_batch_frac, cg_min_batch, cg_max_batch);

        auto [best_nodes, final_density] = solver.solve();

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
        cout << "==================================================" << endl;
        cout << "FINAL SOLUTION" << endl;
        cout << "==================================================" << endl;
        cout << left << setw(25) << "Density" << ": " << fixed << setprecision(6) << final_density << endl;
        cout << left << setw(25) << "Size" << ": " << best_nodes.size() << endl;

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
