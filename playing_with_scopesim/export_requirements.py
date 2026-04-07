import subprocess
import json
import re

def get_conda_packages():
    result = subprocess.run(
        ["conda", "list", "--json"],
        capture_output=True,
        text=True,
        check=True
    )
    packages = json.loads(result.stdout)
    
    conda_pkgs = []
    for pkg in packages:
        # Only include packages installed via conda
        if pkg.get("channel") and pkg.get("name") and pkg.get("version"):
            conda_pkgs.append(f"{pkg['name']}=={pkg['version']}")
    
    return conda_pkgs


def get_pip_packages():
    result = subprocess.run(
        ["pip", "freeze"],
        capture_output=True,
        text=True,
        check=True
    )
    
    pip_pkgs = []
    for line in result.stdout.splitlines():
        # Skip local paths or editable installs
        if "@ file://" in line or line.startswith("-e "):
            continue
        
        # Keep only standard package==version lines
        if "==" in line:
            pip_pkgs.append(line.strip())
    
    return pip_pkgs


def main():
    conda_pkgs = get_conda_packages()
    pip_pkgs = get_pip_packages()
    
    # Combine and deduplicate
    all_pkgs = sorted(set(conda_pkgs + pip_pkgs))
    
    with open("requirements.txt", "w") as f:
        for pkg in all_pkgs:
            f.write(pkg + "\n")
    
    print(f"Exported {len(all_pkgs)} packages to requirements.txt")


if __name__ == "__main__":
    main()