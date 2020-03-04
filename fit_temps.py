import logging.config
import os
import sys

import glob

sys.path.append('.')
sys.path.append("./temperature")
import numpy as np
from l2l.utils.environment import Environment
from l2l.paths import Paths
from l2l.logging_tools import create_shared_logger_data, configure_loggers
from l2l.utils import JUBE_runner
from l2l import dict_to_list
from temperature.optimizee import FitOptimizee, config
from omnigloter.evolution_optimizer import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters
from omnigloter.utils import load_last_trajs, trajectories_to_individuals

logger = logging.getLogger("bin.temperature")
GRADDESC, EVOSTRAT, GENALG = range(3)
#OPTIMIZER = EVOSTRAT
#OPTIMIZER = GRADDESC
OPTIMIZER = GENALG
ON_JEWELS = bool(0)
USE_MPI = bool(1)
MULTIPROCESSING = (ON_JEWELS or USE_MPI or bool(0))

def main():

    name = "FIT-TEMPS"
    root_dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    paths = Paths(name, dict(run_num="test"), root_dir_path=root_dir_path)

    print("All output logs can be found in directory ", paths.logs_path)

    traj_file = os.path.join(paths.output_dir_path, "data.h5")
    print(traj_file)
    os.makedirs(paths.output_dir_path, exist_ok=True)
    print("Trajectory file is: {}".format(traj_file))

    # Create an environment that handles running our simulation
    # This initializes an environment
    env = Environment(trajectory=name,
                      filename=traj_file,
                      file_title="{} data".format(name),
                      comment="{} data".format(name),
                      add_time=bool(1),
                      automatic_storing=bool(1),
                      log_stdout=bool(0),  # Sends stdout to logs
                      multiprocessing=MULTIPROCESSING,
                      )
    create_shared_logger_data(logger_names=["bin", "optimizers"],
                              log_levels=["INFO", "INFO"],
                              log_to_consoles=[True, True],
                              sim_name=name,
                              log_directory=paths.logs_path)
    configure_loggers()

    trajectories = load_last_trajs(os.path.join(paths.output_dir_path, 'per_gen_trajectories'))
    # trajectories = load_last_trajs(os.path.join(paths.root_dir_path,'trajectories'))

    # env.trajectory.individuals[0] = trajectories

    # Get the trajectory from the environment
    traj = env.trajectory

    # Set JUBE params
    traj.f_add_parameter_group("JUBE_params", "Contains JUBE parameters")

    # Scheduler parameters
    # Name of the scheduler
    # traj.f_add_parameter_to_group("JUBE_params", "scheduler", "Slurm")

    # Command to submit jobs to the schedulers
    # traj.f_add_parameter_to_group("JUBE_params", "submit_cmd", "sbatch")

    # Template file for the particular scheduler
    traj.f_add_parameter_to_group("JUBE_params", "job_file", "job.run")
    # Number of nodes to request for each run
    traj.f_add_parameter_to_group("JUBE_params", "nodes", "1")
    # Requested time for the compute resources
    traj.f_add_parameter_to_group("JUBE_params", "walltime", "00:10:00")
    # MPI Processes per node
    traj.f_add_parameter_to_group("JUBE_params", "ppn", "1")
    # CPU cores per MPI process
    traj.f_add_parameter_to_group("JUBE_params", "cpu_pp", "1")
    # Threads per process
    traj.f_add_parameter_to_group("JUBE_params", "threads_pp", "1")
    # Type of emails to be sent from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_mode", "ALL")
    # Email to notify events from the scheduler
    traj.f_add_parameter_to_group("JUBE_params", "mail_address", "g.pineda-garcia@sussex.ac.uk")
    # Error file for the job
    traj.f_add_parameter_to_group("JUBE_params", "err_file", "stderr")
    # Output file for the job
    traj.f_add_parameter_to_group("JUBE_params", "out_file", "stdout")
    # JUBE parameters for multiprocessing. Relevant even without scheduler.
    # MPI Processes per job
    traj.f_add_parameter_to_group("JUBE_params", "tasks_per_job", "1")


    # The execution command
    run_filename = os.path.join(paths.root_dir_path, "run_files/run_optimizee.py")
    command = "python3 {}".format(run_filename)
    if ON_JEWELS and not USE_MPI:
        # -N num nodes
        # -t exec time (mins)
        # -n num sub-procs
        command = "srun -t 15 -N 1 -n 4 -c 1 --gres=gpu:1 {}".format(command)
    elif USE_MPI:
        command = "MPIEXEC_TIMEOUT={} mpiexec -bind-to none -np 1 {}".format(60, command)

    traj.f_add_parameter_to_group("JUBE_params", "exec", command)

    # Ready file for a generation
    traj.f_add_parameter_to_group("JUBE_params", "ready_file",
                                  os.path.join(paths.root_dir_path, "readyfiles/ready_w_"))
    # Path where the job will be executed
    traj.f_add_parameter_to_group("JUBE_params", "work_path", paths.root_dir_path)

    ### Maybe we should pass the Paths object to avoid defining paths here and there
    traj.f_add_parameter_to_group("JUBE_params", "paths_obj", paths)

    csv = open('./temperature/temperature-anomaly.csv', 'r')
    temps = []
    years = []
    for i, line in enumerate(csv):
        sp = line.split(',')
        if sp[0] == 'Global':
            temps.append(float(sp[3]))
            years.append(float(sp[2]))
        if sp[0].startswith('Northern'):
            break
    csv.close()

    traj.f_add_parameter_group("simulation", "Contains JUBE parameters")
    traj.f_add_parameter_to_group("simulation", 'target', temps)  # ms
    traj.f_add_parameter_to_group("simulation", 'years', years)
    ## Innerloop simulator
    optimizee = FitOptimizee(traj, 1234)

    # Prepare optimizee for jube runs
    JUBE_runner.prepare_optimizee(optimizee, paths.root_dir_path)

    _, dict_spec = dict_to_list(optimizee.create_individual(), get_dict_spec=True)
    # step_size = np.asarray([config.ATTR_STEPS[k] for (k, spec, length) in dict_spec])


    fit_weights = [1.0,]# 0.1]
    num_generations = 50#000
    population_size = 20
    # population_size = 5


    if len(trajectories):
        traj.individuals = trajectories_to_individuals(
                                trajectories, population_size, optimizee)

    parameters = GeneticAlgorithmParameters(seed=0,
                    popsize=population_size,
                    CXPB=0.5, # probability of mating 2 individuals
                    MUTPB=0.8, # probability of individual to mutate
                    NGEN=num_generations,
                    indpb=0.1, # probability of "gene" to mutate
                    tournsize=100, # number of best individuals to mate
                    matepar=0.5, # how much to mix two genes when mating
                    mutpar=2.0/4.0, #standard deviations for normal distribution
                    )

    optimizer = GeneticAlgorithmOptimizer(traj,
                  optimizee_create_individual=optimizee.create_individual,
                  optimizee_fitness_weights=fit_weights,
                  parameters=parameters,
                  optimizee_bounding_func=optimizee.bounding_func,
                  percent_hall_of_fame=0.3,
                  percent_elite=0.5,
                  )

    # Add post processing
    ### guess this is where we want to split results from multiple runs?
    env.add_postprocessing(optimizer.post_process)

    # Run the simulation with all parameter combinations
    env.run(optimizee.simulate)

    ## Outerloop optimizer end
    optimizer.end(traj)

    # Finally disable logging and close all log-files
    env.disable_logging()






if __name__ == '__main__':
    main()
