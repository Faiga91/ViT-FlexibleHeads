import os
import subprocess
import shutil


def generate_index(output_dir):
    """Create an index.html file linking to all module documentation."""
    index_file = os.path.join(output_dir, "index.html")
    with open(index_file, "w") as f:
        f.write("<html><head><title>Module Documentation</title></head><body>\n")
        f.write("<h1>Module Documentation Index</h1>\n<ul>\n")

        for file in sorted(os.listdir(output_dir)):
            if file.endswith(".html") and file != "index.html":
                module_name = os.path.splitext(file)[0]
                f.write(f'<li><a href="{file}">{module_name}</a></li>\n')

        f.write("</ul>\n</body></html>\n")


def generate_docs(package_dir, output_dir):
    """Generate pydoc documentation for all Python files in a package directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Add the package_dir to the PYTHONPATH
    package_path = os.path.abspath(package_dir)
    os.environ["PYTHONPATH"] = package_path
    for root, dirs, files in os.walk(package_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Construct the module name
                module_path = os.path.join(root, file)
                relative_path = os.path.relpath(module_path, package_dir)
                module_name = os.path.splitext(relative_path.replace(os.sep, "."))[0]

                try:
                    # Generate HTML documentation using pydoc
                    subprocess.run(["pydoc", "-w", module_name], check=True)

                    # Move the generated HTML file to the output directory
                    html_file = f"{module_name}.html"
                    if os.path.exists(html_file):
                        shutil.move(html_file, os.path.join(output_dir, html_file))
                except subprocess.CalledProcessError:
                    print(f"Failed to generate documentation for module: {module_name}")
                except FileNotFoundError:
                    print(f"HTML file not found for module: {module_name}")

    # Generate the index.html file
    generate_index(output_dir)


if __name__ == "__main__":
    # Replace 'vit_flexible_heads' with your package directory
    generate_docs("vit_flexible_heads", "docs")
