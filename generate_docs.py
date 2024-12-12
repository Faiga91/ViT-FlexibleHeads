import os
import subprocess

os.environ["PDOC_DISPLAY_ENV_VARS"] = "1"

def generate_index(output_dir):
    """Create an index.html file linking to all generated HTML files."""
    index_file = os.path.join(output_dir, "index.html")
    with open(index_file, "w") as f:
        f.write("<html><head><title>Module Documentation Index</title></head><body>\n")
        f.write("<h1>Module Documentation</h1>\n<ul>\n")

        # Recursively find all HTML files except index.html
        for root, _, files in os.walk(output_dir):
            for file in sorted(files):
                if file.endswith(".html") and file != "index.html":
                    relative_path = os.path.relpath(os.path.join(root, file), output_dir)
                    module_name = os.path.splitext(file)[0]  # Extract the module name
                    f.write(f'<li><a href="{relative_path}">{module_name}</a></li>\n')

        f.write("</ul>\n</body></html>\n")

def generate_docs(package_dir, output_dir):
    """Generate HTML documentation for all Python files in a package directory."""
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

    # Generate a custom index.html after processing all modules
    generate_index(output_dir)

if __name__ == "__main__":
    generate_docs("vit_flexible_heads", "docs")
