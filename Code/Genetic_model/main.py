import logging
from optimizer import Optimizer
from tqdm import tqdm

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

def train_networks(networks):
    """Train each network."""
    
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train()
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks."""
    
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_choices):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" % (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            networks = optimizer.evolve(networks)    # Do the evolution.

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    logging.info('-'*80)
    for network in networks[:5]:
        network.print_network()

#def print_networks(networks):
#    """Print a list of networks.
#
#    Args:
#        networks (list): The population of networks
#
#    """
#    logging.info('-'*80)
#    for network in networks:
#        network.print_network()
    
def main():
    """Evolve a network."""
    generations = 20  # Number of times to evole the population.
    population = 30  # Number of networks in each generation.

    nn_param_choices = {
        'nb_layers': [3, 4],
        'activation': ['relu', 'selu'],
        'kernel1' : [ 7, 9, 11, 13],
        'kernel2' : [ 7, 9, 11],
        'kernel3' : [ 7, 9, 11],
        'filter1' : [ 16, 24 ],
        'filter2' : [ 16, 32, 64],
        'filter3' : [ 32, 64],
        'dropout' : [.25, .3, .35],
        'batch_size' : [32, 40, 48]
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))
    generate(generations, population, nn_param_choices)

if __name__ == '__main__':
    main()
