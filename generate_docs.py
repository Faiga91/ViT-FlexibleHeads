import os
import subprocess

os.environ["PDOC_DISPLAY_ENV_VARS"] = "1"


def generate_docs(package_dir, output_dir):
    """Generate Markdown documentation for all Python files in a package directory."""
    os.makedirs(output_dir, exist_ok=True)
    os.environ["PYTHONPATH"] = os.path.abspath(package_dir)

    for root, dirs, files in os.walk(package_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Construct the module name
                module_path = os.path.join(root, file)
                relative_path = os.path.relpath(module_path, package_dir)
                module_name = os.path.splitext(relative_path.replace(os.sep, "."))[0]

                try:
                    # Generate HTML documentation using pdoc
                    subprocess.run(
                        ["pdoc", module_name, "--output-dir", output_dir],
                        check=True,
                    )
                except subprocess.CalledProcessError:
                    print(f"Failed to generate documentation for module: {module_name}")


if __name__ == "__main__":
    generate_docs("vit_flexible_heads", "docs")
