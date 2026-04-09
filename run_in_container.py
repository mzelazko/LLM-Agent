from pathlib import Path
import os
import shutil
import traceback
from docker import DockerClient, from_env
from docker.errors import ImageNotFound
from docker.models.containers import Container
from config import Config

def _repo_paths():
    showcase_dir = Path(__file__).resolve().parent
    workspace_root = showcase_dir.parent
    return showcase_dir, workspace_root


def prepare_custom_container() -> Container:
    """
    
    """
    docker_client: DockerClient = from_env()
    
    showcase_dir, workspace_root = _repo_paths()
    venv_dir = workspace_root / ".venv"
    script_file = showcase_dir / "agent.py"
    output_dir = workspace_root / "output"
    config_file = showcase_dir / "config.py"
    output_dir.mkdir(exist_ok=True)
    
    volumes = {
        script_file.absolute().as_posix(): {"bind": "/workspace/agent.py", "mode": "ro"},
        venv_dir.absolute().as_posix(): {"bind": "/workspace/.venv", "mode": "ro"},
        config_file.absolute().as_posix(): {"bind": "/workspace/config.py", "mode": "ro"},
        (showcase_dir / "prompts_v1.py").absolute().as_posix(): {"bind": "/workspace/prompts_v1.py", "mode": "ro"},
        (showcase_dir / "prompts_v2.py").absolute().as_posix(): {"bind": "/workspace/prompts_v2.py", "mode": "ro"},
        output_dir.absolute().as_posix(): {"bind": "/workspace/output", "mode": "rw"}
    }
    
    base_image_name = "ubuntu:22.04"
    custom_image_name = "ubuntu-python3:22.04"

    try:
        image = docker_client.images.get(custom_image_name)
        print(f"Using existing custom image: {custom_image_name}")
    except ImageNotFound:
        print(f"Custom image not found. Creating {custom_image_name} from {base_image_name}...")
        
        try:
            base_image = docker_client.images.get(base_image_name)
        except ImageNotFound:
            print(f"Pulling base image: {base_image_name}")
            base_image = docker_client.images.pull(base_image_name)
        
        print("Creating temporary container to install Python3...")
        temp_container = docker_client.containers.run(
            image=base_image,
            command="/bin/bash",
            detach=True,
            tty=True,
            stdin_open=True,
        )
        setup_commands = [
            "apt-get update -qq",
            "apt-get install -y -qq python3 python3-pip",
            "pip3 install --quiet openai",
        ]
        print("Installing Python3 and packages...")
        for command in setup_commands:
            try:
                new_command = f'/bin/bash -c "{command}"'
                return_code, output = temp_container.exec_run(cmd=new_command, stream=False)
                output_text = output.decode('utf-8') if isinstance(output, bytes) else str(output)
                if return_code is not None and return_code != 0:
                    print(f"Docker exec error for '{command}'. Exit code: {return_code}")
                    print(f"Error message: {output_text}")
            except Exception as e:
                print(f"{command} failed.")
                print(traceback.format_exc())
                temp_container.stop()
                temp_container.remove()
                raise
        
        print(f"Committing container to image: {custom_image_name}")
        temp_container.stop()
        image = temp_container.commit(repository="ubuntu-python3", tag="22.04")
        temp_container.remove()
        print(f"Custom image {custom_image_name} created successfully!")

    model = Config.MODEL_NAME
    provider = model.split("/", 1)[0].lower() if "/" in model else "openai"

    key = f"{provider.upper()}_API_KEY"
    env_vars = {}
    if model:
        env_vars["MODEL_NAME"] = model
    if not os.environ.get(key):
        raise ValueError(f"Missing {key} for MODEL_NAME='{model}' (provider='{provider}').")
    env_vars[key] = os.environ[key]
    
    container: Container = docker_client.containers.run(
        image = image,
        command="/bin/bash",
        detach=True,
        tty=True,
        stdin_open=True,
        volumes=volumes,
        working_dir="/workspace",
        environment=env_vars if env_vars else None,
        stream=True,
    )
    
    container_id = container.id
    container_name = container.name
    print(f"\n{'='*80}")
    print(f"Container prepared")
    print(f"Container ID: {container_id}")
    print(f"Container Name: {container_name}")
    print(f"\nTo explore the container manually, run:")
    print(f"  docker exec -it {container_id} /bin/bash")
    print(f"  OR")
    print(f"  docker exec -it {container_name} /bin/bash")
    print(f"{'='*80}\n")
    
    return container

def run_in_custom_container(container):
    command = "python3 /workspace/agent.py"
    
    exit_code, output = container.exec_run(cmd=command, stream=False)
    
    output_text = output.decode('utf-8') if isinstance(output, bytes) else str(output)
    
    print(f"\n{'='*80}")
    print("CONTAINER OUTPUT:")
    print(f"{'='*80}")
    print(output_text)
    print(f"{'='*80}")
    print(f"Exit code: {exit_code}\n")


def prepare_swe_container() -> Container:
    """
    
    """
    docker_client: DockerClient = from_env()
    
    showcase_dir, workspace_root = _repo_paths()
    venv_dir = workspace_root / ".venv"
    script_file = showcase_dir / "agent.py"
    output_dir = workspace_root / "output"
    config_file = showcase_dir / "config.py"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    volumes = {
        script_file.absolute().as_posix(): {"bind": "/workspace/agent.py", "mode": "ro"},
        venv_dir.absolute().as_posix(): {"bind": "/workspace/.venv", "mode": "ro"},
        config_file.absolute().as_posix(): {"bind": "/workspace/config.py", "mode": "ro"},
        (showcase_dir / "prompts_v1.py").absolute().as_posix(): {"bind": "/workspace/prompts_v1.py", "mode": "ro"},
        (showcase_dir / "prompts_v2.py").absolute().as_posix(): {"bind": "/workspace/prompts_v2.py", "mode": "ro"},
        output_dir.absolute().as_posix(): {"bind": "/workspace/output", "mode": "rw"}
    }
    
    image_name = "swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-26194:latest"

    model = Config.MODEL_NAME
    provider = model.split("/", 1)[0].lower() if "/" in model else "openai"

    key = f"{provider.upper()}_API_KEY"
    env_vars = {}
    if model:
        env_vars["MODEL_NAME"] = model
    if not os.environ.get(key):
        raise ValueError(f"Missing {key} for MODEL_NAME='{model}' (provider='{provider}').")
    env_vars[key] = os.environ[key]

    container: Container = docker_client.containers.run(
        image = image_name,
        command="/bin/bash",
        detach=True,
        tty=True,
        stdin_open=True,
        volumes=volumes,
        working_dir="/workspace",
        environment=env_vars if env_vars else None,
        stream=True,
    )
    
    container_id = container.id
    container_name = container.name
    print(f"\n{'='*80}")
    print(f"Container prepared")
    print(f"Container ID: {container_id}")
    print(f"Container Name: {container_name}")
    print(f"\nTo explore the container manually, run:")
    print(f"  docker exec -it {container_id} /bin/bash")
    print(f"  OR")
    print(f"  docker exec -it {container_name} /bin/bash")
    print(f"{'='*80}\n")
    
    return container



if __name__ == "__main__":
    container = prepare_custom_container()

    print("\nExecuting script in container...")
    run_in_custom_container(container)

    container.stop()
    container.remove()
    print("Container stopped and removed.")
