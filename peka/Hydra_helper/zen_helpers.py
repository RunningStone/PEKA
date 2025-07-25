from hydra_zen import store, to_yaml
import os

def write_config_to_yaml(config_from_builds_fn, 
                         group, name, all_configs_path):
    # save to ConfigStore
    store(group=group, name=name, node=config_from_builds_fn)

    # print to yaml
    yaml_output = to_yaml(config_from_builds_fn)

    yaml_file_path = f"{all_configs_path}/{group}/{name}.yaml" # ensure path correct

    if not os.path.exists(os.path.dirname(yaml_file_path)):
        os.makedirs(os.path.dirname(yaml_file_path))
    # save to file
    with open(yaml_file_path, "w") as f:
        f.write(yaml_output)


