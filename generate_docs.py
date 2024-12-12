import os
import subprocess
import shutil


def generate_docs(package_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set the PYTHONPATH to the package directory
    os.environ["PYTHONPATH"] = os.path.abspath(package_dir)

    for root, dirs, files in os.walk(package_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Construct the module name
                module_path = os.path.join(root, file)
                relative_path = os.path.relpath(module_path, package_dir)
                module_name = os.path.splitext(relative_path.replace(os.sep, "."))[0]

                try:
                    # Generate HTML documentation using pydoc
                    subprocess.run(
                        ["python", "-m", "pydoc", "-w", module_name], check=True
                    )

                    # Move the generated HTML file to the output directory
                    html_file = f"{module_name}.html"
                    if os.path.exists(html_file):
                        shutil.move(html_file, os.path.join(output_dir, html_file))
                except subprocess.CalledProcessError:
                    print(f"Failed to generate documentation for module: {module_name}")


if __name__ == "__main__":
    generate_docs("vit_flexible_heads", "docs")
