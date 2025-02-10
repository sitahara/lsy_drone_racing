import toml
import pprint


def load_config_from_toml(file_path):
    with open(file_path, "r") as file:
        config = toml.load(file)
    return config


def update_config_with_hyperparameters(config, hyperparameter_dict):
    updated_config = config.copy()
    for section, params in hyperparameter_dict.items():
        if section in updated_config:
            updated_config[section].update(params)
        else:
            updated_config[section] = params
    return updated_config


# Load the entire configuration from the TOML file
config = load_config_from_toml("lsy_drone_racing/mpc_utils/config.toml")
pprint.pprint(config)
# Define the hyperparameters to be tuned
tuned_hyperparameters = {
    "dynamics_info": {"ts": 0.02, "n_horizon": 50},
    "cost_info": {"Qs_pos": 5, "Qs_vel": 0.1},
}

# Update the configuration with the tuned hyperparameters
updated_config = update_config_with_hyperparameters(config, tuned_hyperparameters)

pprint.pprint(updated_config)
